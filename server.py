import gc
import io
import os
from typing import List

import numpy as np
import torch
import open_clip
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Form, Query
import warnings

# Safe to import unconditionally even in the text-only image, which ships neither
# onnxruntime nor rapidocr: ocr.py imports both inside functions, not at module level.
import ocr
# Same rule as ocr: face_attributes imports cv2 inside the function that needs it.
import face_attributes
# And again: age_gender imports torch, timm, scipy and ultralytics only once a model is built.
import age_gender

warnings.filterwarnings(
    "ignore",
    message="`rcond` parameter will change",
    category=FutureWarning
)

app = FastAPI(title="Embeddings API (OpenCLIP + InsightFace)")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# === MODE CONFIG ===
# "full" is the box: GPU, every endpoint. "text" is the VPS, which has no GPU and exists
# for one reason -- kukátko's search box. Embedding a query is the only interactive call
# in the system, and pointing it at a machine that is usually powered off is why semantic
# search silently degrades to full-text. Text-only keeps the 707.8M-parameter text tower
# and drops the rest: no visual tower, no face model, no OCR engine.
EMBED_MODE = os.environ.get("EMBED_MODE", "full").strip().lower()
if EMBED_MODE not in {"full", "text"}:
    raise SystemExit(f"EMBED_MODE must be 'full' or 'text', got {EMBED_MODE!r}")
TEXT_ONLY = EMBED_MODE == "text"
print(f"Mode: {EMBED_MODE}")

# torch sizes its thread pool from the host CPU count, not from the cgroup quota, so in a
# container limited to fewer CPUs than the host has it starts too many threads and they
# contend with each other. Set this to the container's CPU limit.
TORCH_THREADS = os.environ.get("TORCH_NUM_THREADS", "").strip()
if TORCH_THREADS:
    torch.set_num_threads(int(TORCH_THREADS))
    print(f"torch threads: {torch.get_num_threads()}")

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

# The visual tower is 428.2M of the model's 1136M parameters, about 1.7 GB in fp32, and
# encode_text() never touches it -- it goes through model.text. open_clip has no way to
# build one tower without the other, so the load still peaks at the full 4.3 GB; this
# gives most of it straight back. Sizing a memory limit for this service means clearing
# the peak, not the steady state.
if TEXT_ONLY:
    model.visual = None
    gc.collect()

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

# Pre-compute text embeddings for eras. Skipped in text-only mode: these exist solely to
# be compared against an image embedding, and /estimate/era is not served there.
era_features = None
if not TEXT_ONLY:
    with torch.inference_mode():
        era_texts = [prompt for _, _, prompt in ERA_PROMPTS]
        era_tokens = tokenizer(era_texts).to(DEVICE)
        era_features = model.encode_text(era_tokens).float()
        era_features = era_features / era_features.norm(dim=-1, keepdim=True)

# === INSIGHTFACE CONFIG ===
# buffalo_l uses ResNet100 for face recognition (512-dim embeddings). Both this and the
# OCR engine stay unloaded in text-only mode, and insightface is imported here rather than
# at the top of the module so the text-only image does not have to ship it at all.
face_app = None
if not TEXT_ONLY:
    from insightface.app import FaceAnalysis

    print("Loading InsightFace model...")
    # These are all five buffalo_l models, i.e. exactly what FaceAnalysis loads when
    # allowed_modules is left out -- listed so what runs per face is visible here:
    # detection (bbox, det_score, 5 keypoints), the 106-point 2D landmarks (eye
    # openness), the 68-point 3D landmarks (head pose), genderage, and recognition (the
    # 512-dim embedding).
    #
    # Age and gender are coarse: the model is poor on children, the elderly, scanned
    # prints and blurry faces. Treat them as a weak signal, never as a fact about a person.
    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
        allowed_modules=[
            "detection",
            "landmark_2d_106",
            "landmark_3d_68",
            "genderage",
            "recognition",
        ],
    )
    face_app.prepare(ctx_id=0 if DEVICE == "cuda" else -1, det_size=(1600, 1600))
    print("Loading OCR engine...")
    ocr.load_engine()

# === MIVOLO CONFIG ===
# MiVOLO v2 estimates age and gender from the face and the body of the person it belongs
# to, next to InsightFace's genderage (see age_gender.py). It needs a person detector, a
# YOLOv8x, on top of buffalo_l. Loaded after every other model on purpose: importing
# ultralytics changes process-wide settings, which age_gender contains, and nothing that
# is already loaded can be affected by it.
#
# MIVOLO_ENABLED=0 skips both models (about 250 MB of weights, and ~0.1 s per photo on
# the GPU) without a rebuild; every face then reports `mivolo: null`. The weights are
# CC-BY-NC-4.0 and the detector AGPL-3.0, which is another reason to be able to turn it off.
MIVOLO_ENABLED = os.environ.get("MIVOLO_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
MIVOLO_MODELS_DIR = os.environ.get(
    "MIVOLO_MODELS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "mivolo")
)
age_gender_model = None
if not TEXT_ONLY and MIVOLO_ENABLED:
    print(f"Loading MiVOLO v2 and the person detector from {MIVOLO_MODELS_DIR}...")
    age_gender_model = age_gender.AgeGenderEstimator(MIVOLO_MODELS_DIR, DEVICE)

print("All models loaded. Starting server...")


# Models that see the image in the channel order this service has always given them.
# The face endpoint hands insightface RGB (straight from PIL), while every insightface
# model was trained on BGR (cv2) and swaps channels itself on the way in. Detection and
# recognition keep getting RGB anyway: the stored embeddings and bboxes were produced
# that way, and feeding them BGR would move every vector a consumer already has.
FACE_RGB_TASKS = {"detection", "recognition"}


def analyze_faces(img_rgb: np.ndarray, img_bgr: np.ndarray = None) -> list:
    """FaceAnalysis.get(), except the attribute models see the channel order they expect.

    The same loop insightface's FaceAnalysis.get() runs (detect, then every other model
    per face), so nothing runs twice. The difference is the input: genderage and the two
    landmark models get BGR. On RGB they degrade badly and silently -- on insightface's
    own t1 group photo, genderage called all six faces "female", three of them men, while
    the BGR copy gets all six right. Detection and recognition still get RGB, so
    `embedding`, `bbox`, `det_score` and `kps` are exactly what face_app.get() returns
    (tests/test_api_face.py checks it).
    """
    from insightface.app.common import Face

    if img_bgr is None:
        img_bgr = np.ascontiguousarray(img_rgb[:, :, ::-1])
    bboxes, kpss = face_app.det_model.detect(img_rgb, max_num=0, metric="default")
    faces = []
    for i in range(bboxes.shape[0]):
        face = Face(
            bbox=bboxes[i, 0:4],
            kps=kpss[i] if kpss is not None else None,
            det_score=bboxes[i, 4],
        )
        for taskname, face_model in face_app.models.items():
            if taskname == "detection":
                continue
            face_model.get(img_rgb if taskname in FACE_RGB_TASKS else img_bgr, face)
        faces.append(face)
    return faces


def require_full_mode(capability: str) -> None:
    """Reject an image-side call when the service is running text-only.

    503 rather than 404: the route exists and the service is healthy, it simply has no
    visual tower, face model or OCR engine loaded. A 404 would read as a wrong URL or a
    stale image and send the caller looking in the wrong place.
    """
    if TEXT_ONLY:
        raise HTTPException(
            status_code=503,
            detail=f"{capability} is unavailable: EMBED_MODE=text serves /embed/text only.",
        )

def age_gender_info() -> dict:
    """What /health says about MiVOLO and its person detector."""
    if age_gender_model is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "mivolo": {
            "loaded": True,
            "model": age_gender.MIVOLO_MODEL_NAME,
            "source": f"{age_gender.MIVOLO_REPO}@{age_gender.MIVOLO_REVISION}",
            "precision": "fp16" if age_gender_model.half else "fp32",
        },
        "person_detector": {
            "loaded": True,
            "model": age_gender.DETECTOR_MODEL_NAME,
            "source": f"{age_gender.DETECTOR_REPO}@{age_gender.DETECTOR_REVISION}",
        },
    }


@app.get("/health")
def health():
    result = {"mode": EMBED_MODE, "device": DEVICE, "cuda": torch.cuda.is_available()}
    if torch.cuda.is_available():
        result["gpu_name"] = torch.cuda.get_device_name(0)
        result["gpu_memory_total"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    # engine_info() loads the engine lazily, so it must not be reached in text-only mode:
    # that image ships neither onnxruntime nor rapidocr and /health would raise instead of
    # answering -- turning the one endpoint that reports trouble into trouble of its own.
    result["ocr"] = {"enabled": False} if TEXT_ONLY else ocr.engine_info()
    if not TEXT_ONLY:
        result["face_attributes_version"] = face_attributes.ATTRIBUTES_VERSION
    result["age_gender"] = age_gender_info()
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
    require_full_mode("Image embedding")
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
async def embed_face(
    file: UploadFile = File(...),
    # Off by default: 106 points per face would multiply the response size for a field
    # most callers never read.
    landmarks: bool = Query(False, description="Add landmarks_106 to every face."),
):
    require_full_mode("Face embedding")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image/* file.")

    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img_array = np.array(img)
    # BGR for the models trained on it (insightface's attribute models, the person
    # detector and MiVOLO); detection and recognition keep getting RGB, see analyze_faces.
    img_bgr = np.ascontiguousarray(img_array[:, :, ::-1])

    faces = analyze_faces(img_array, img_bgr)
    # One detector pass and one MiVOLO forward pass for all faces of the photo.
    if age_gender_model is not None:
        estimates = age_gender_model.estimate(img_bgr, [face.bbox for face in faces])
    else:
        estimates = [None] * len(faces)

    results = []
    for i, face in enumerate(faces):
        embedding = face.normed_embedding.tolist()
        bbox = face.bbox.tolist()
        entry = {
            "face_index": i,
            "dim": len(embedding),
            "embedding": embedding,
            "bbox": bbox,
            "det_score": float(face.det_score)
        }
        # Additive only: the fields above stay exactly as they were.
        entry.update(face_attributes.describe(face, img_array, include_landmarks=landmarks))
        entry["mivolo"] = estimates[i]
        results.append(entry)

    # `model` must not change: the client re-detects every face when it does, and the
    # embeddings are the same as before. New attributes bump attributes_version instead.
    return {
        "faces_count": len(results),
        "faces": results,
        "model": "buffalo_l (ResNet100)",
        "attributes_version": face_attributes.ATTRIBUTES_VERSION,
    }

@app.post("/estimate/era", response_model=dict)
async def estimate_era(file: UploadFile = File(...)):
    require_full_mode("Era estimation")
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
    require_full_mode("OCR")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image/* file.")

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read the image: {exc}")

    return ocr.extract_text(img, min_confidence=min_confidence)
