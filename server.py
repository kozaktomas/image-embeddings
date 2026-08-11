import io
import os
from typing import List

import numpy as np
import torch
import open_clip
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Form
from insightface.app import FaceAnalysis
import warnings

import ocr

warnings.filterwarnings(
    "ignore",
    message="`rcond` parameter will change",
    category=FutureWarning
)

app = FastAPI(title="Embeddings API (OpenCLIP + InsightFace)")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# === MODEL CONFIG ===
# SigLIP 2 so400m at 378px. It beats the previous ViT-L-14/laion2b on every metric
# that matters here -- text->image R@1 on COCO goes 46.1 -> 55.8 -- at the cost of a
# 768 -> 1152 dimension change, so swapping it back is not just a restart: consumers
# store the vectors and their column width has to match.
#
# Rollback without a redeploy: set CLIP_MODEL/CLIP_PRETRAINED in the unit file back to
# ViT-L-14 / laion2b_s32b_b82k (and re-embed, because the stored vectors are 1152-dim).
MODEL_NAME = os.environ.get("CLIP_MODEL", "ViT-SO400M-14-SigLIP2-378")
PRETRAINED = os.environ.get("CLIP_PRETRAINED", "webli")

# fp16 is not an optimisation here, it is what makes the model fit. The weights are
# 4331 MiB in fp32 and 2167 MiB in fp16 (measured), against roughly 4.7 GB of the
# 3070 left over once the co-resident photo-enhancer has taken its share. Quality
# cost is nil in practice: SigLIP 2 was trained in bf16, and consumers store these
# vectors as fp16 anyway. CPU stays fp32 -- half() on CPU is slower, not faster.
PRECISION = os.environ.get("CLIP_PRECISION", "auto").strip().lower()
USE_HALF = PRECISION == "fp16" or (PRECISION == "auto" and DEVICE == "cuda")

print(f"Loading CLIP model {MODEL_NAME} ({PRETRAINED}), half={USE_HALF}...")
# precision and device are passed to open_clip rather than applied afterwards, because
# .to(cuda) followed by .half() would stage the full fp32 model on the GPU first and
# peak at 4334 MiB during load -- more than the card has free, so it OOMs before the
# conversion that would have made it fit. Converting before the transfer keeps the peak
# at the fp16 size.
model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=PRETRAINED,
    precision="fp16" if USE_HALF else "fp32",
    device=DEVICE,
)
model = model.eval()
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

# The dtype the towers expect. preprocess() always produces float32, so image tensors
# are cast to this before the forward pass; token ids are integers and need no cast.
MODEL_DTYPE = next(model.parameters()).dtype

# Embedding width, reported by /health so a consumer can verify its column matches
# before it starts writing vectors it cannot store. Read from the model config rather
# than a projection attribute, which SigLIP-style models (custom_text) do not expose.
EMBED_DIM = open_clip.get_model_config(MODEL_NAME)["embed_dim"]

# Era estimation prompts: (era_label, representative_date, prompt)
ERA_PROMPTS = [
    ("1920s-1930s", "1930-06-15", "a vintage black and white photograph from the 1920s or 1930s"),
    ("1940s-1950s", "1950-06-15", "a photograph from the 1940s or 1950s, mid-century style"),
    ("1960s", "1965-06-15", "a photograph from the 1960s"),
    ("1970s", "1975-06-15", "a photograph from the 1970s, vintage color photo"),
    ("1980s", "1985-06-15", "a photograph from the 1980s"),
    ("1990s", "1995-06-15", "a photograph from the 1990s"),
    ("2000s", "2005-06-15", "a photograph from the 2000s, early digital camera era"),
    ("2010s", "2015-06-15", "a photograph from the 2010s, modern smartphone photo"),
    ("2020s", "2022-06-15", "a photograph from the 2020s, recent high quality photo"),
]

# Temperature for turning era similarities into a distribution. The old hard-coded 100
# happened to match CLIP's learned scale; SigLIP 2 has its own, so read it off the model
# instead of assuming. Falls back to the historical constant if a model lacks the field.
LOGIT_SCALE = (
    float(model.logit_scale.detach().exp()) if hasattr(model, "logit_scale") else 100.0
)

# Pre-compute text embeddings for eras
with torch.inference_mode():
    era_texts = [prompt for _, _, prompt in ERA_PROMPTS]
    era_tokens = tokenizer(era_texts).to(DEVICE)
    era_features = model.encode_text(era_tokens).float()
    era_features = era_features / era_features.norm(dim=-1, keepdim=True)

# === INSIGHTFACE CONFIG ===
# buffalo_l uses ResNet100 for face recognition (512-dim embeddings)
print("Loading InsightFace model...")
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0 if DEVICE == "cuda" else -1, det_size=(1600, 1600))
print("Loading OCR engine...")
ocr.load_engine()

print("All models loaded. Starting server...")

@app.get("/health")
def health():
    result = {"device": DEVICE, "cuda": torch.cuda.is_available()}
    if torch.cuda.is_available():
        result["gpu_name"] = torch.cuda.get_device_name(0)
        result["gpu_memory_total"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    result["ocr"] = ocr.engine_info()
    # Consumers store these vectors in a fixed-width column, so let them check the
    # width they are about to receive instead of discovering it on a failed insert.
    result["clip"] = {
        "model": MODEL_NAME,
        "pretrained": PRETRAINED,
        "dim": EMBED_DIM,
        "precision": "fp16" if USE_HALF else "fp32",
    }
    return result

@app.post("/embed/image", response_model=dict)
async def embed_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image/* file.")

    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    x = preprocess(img).unsqueeze(0).to(DEVICE, dtype=MODEL_DTYPE)

    with torch.inference_mode():
        # Back to fp32 before normalising: the forward pass may run in fp16, but the
        # norm and the vector handed out should not carry that precision.
        feat = model.encode_image(x).float()
        feat = feat / feat.norm(dim=-1, keepdim=True)  # normalising makes cosine similarity straightforward
        vec: List[float] = feat[0].detach().cpu().tolist()

    return {"dim": len(vec), "embedding": vec, "model": MODEL_NAME, "pretrained": PRETRAINED}

@app.post("/embed/text", response_model=dict)
async def embed_text(text: str = Body(..., embed=True)):
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    tokens = tokenizer([text]).to(DEVICE)

    with torch.inference_mode():
        feat = model.encode_text(tokens).float()
        feat = feat / feat.norm(dim=-1, keepdim=True)
        vec: List[float] = feat[0].detach().cpu().tolist()

    return {"dim": len(vec), "embedding": vec, "model": MODEL_NAME, "pretrained": PRETRAINED}

@app.post("/embed/face", response_model=dict)
async def embed_face(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image/* file.")

    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img_array = np.array(img)

    faces = face_app.get(img_array)

    results = []
    for i, face in enumerate(faces):
        embedding = face.normed_embedding.tolist()
        bbox = face.bbox.tolist()
        results.append({
            "face_index": i,
            "dim": len(embedding),
            "embedding": embedding,
            "bbox": bbox,
            "det_score": float(face.det_score)
        })

    return {"faces_count": len(results), "faces": results, "model": "buffalo_l (ResNet100)"}

@app.post("/estimate/era", response_model=dict)
async def estimate_era(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image/* file.")

    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    x = preprocess(img).unsqueeze(0).to(DEVICE, dtype=MODEL_DTYPE)

    with torch.inference_mode():
        img_features = model.encode_image(x).float()
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)

        similarities = (img_features @ era_features.T).squeeze(0)
        probs = torch.softmax(similarities * LOGIT_SCALE, dim=0)

    results = [
        {"era": ERA_PROMPTS[i][0], "date": ERA_PROMPTS[i][1], "confidence": float(probs[i])}
        for i in range(len(ERA_PROMPTS))
    ]
    results.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "estimated_date": results[0]["date"],
        "era": results[0]["era"],
        "confidence": results[0]["confidence"],
        "all_eras": results
    }

@app.post("/ocr/image", response_model=dict)
async def ocr_image(
    file: UploadFile = File(...),
    min_confidence: float = Form(ocr.DEFAULT_MIN_CONFIDENCE),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image/* file.")

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read the image: {exc}")

    return ocr.extract_text(img, min_confidence=min_confidence)
