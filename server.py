import io
from typing import List

import numpy as np
import torch
import open_clip
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from insightface.app import FaceAnalysis
import warnings

warnings.filterwarnings(
    "ignore",
    message="`rcond` parameter will change",
    category=FutureWarning
)

app = FastAPI(title="Embeddings API (OpenCLIP + InsightFace)")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# === MODEL CONFIG ===
# 1) Default (rychlé, kvalitní, stabilní):
#    MODEL_NAME, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"
# 2) Lepší kvalita (pořád OK na 3070):
#    MODEL_NAME, PRETRAINED = "ViT-B-16", "laion2b_s34b_b88k"
# 3) Nejlepší kvalita (vyšší latency/VRAM):
#    MODEL_NAME, PRETRAINED = "ViT-L-14", "laion2b_s32b_b82k"

MODEL_NAME, PRETRAINED = "ViT-L-14", "laion2b_s32b_b82k"

print(f"Loading CLIP model {MODEL_NAME}...")
model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=PRETRAINED
)
model = model.to(DEVICE).eval()
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

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

# Pre-compute text embeddings for eras
with torch.inference_mode():
    era_texts = [prompt for _, _, prompt in ERA_PROMPTS]
    era_tokens = tokenizer(era_texts).to(DEVICE)
    era_features = model.encode_text(era_tokens)
    era_features = era_features / era_features.norm(dim=-1, keepdim=True)

# === INSIGHTFACE CONFIG ===
# buffalo_l uses ResNet100 for face recognition (512-dim embeddings)
print("Loading InsightFace model...")
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0 if DEVICE == "cuda" else -1, det_size=(1600, 1600))
print("All models loaded. Starting server...")

@app.get("/health")
def health():
    result = {"device": DEVICE, "cuda": torch.cuda.is_available()}
    if torch.cuda.is_available():
        result["gpu_name"] = torch.cuda.get_device_name(0)
        result["gpu_memory_total"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    return result

@app.post("/embed/image", response_model=dict)
async def embed_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploadni prosím image/* soubor.")

    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    x = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        feat = model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)  # normalizace je praktická pro cosine similarity
        vec: List[float] = feat[0].detach().float().cpu().tolist()

    return {"dim": len(vec), "embedding": vec, "model": MODEL_NAME, "pretrained": PRETRAINED}

@app.post("/embed/face", response_model=dict)
async def embed_face(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploadni prosím image/* soubor.")

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
        raise HTTPException(status_code=400, detail="Uploadni prosím image/* soubor.")

    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    x = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        img_features = model.encode_image(x)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)

        similarities = (img_features @ era_features.T).squeeze(0)
        probs = torch.softmax(similarities * 100, dim=0)

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
