#!/usr/bin/env bash
# Build the development venv for image-embeddings on the GPU machine.
#
# The install order is deliberate. insightface pulls in CPU onnxruntime and full
# opencv-python as dependencies, so installing it after them would shadow
# onnxruntime-gpu and opencv-python-headless (each pair shares an import namespace).
# insightface therefore goes in first and its versions are overwritten afterwards.
set -euo pipefail

cd "$(dirname "$0")/.."
python3 -m venv venv
./venv/bin/pip install --upgrade pip

# PyTorch with CUDA (same as the production venv)
./venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# transformers/sentencepiece/protobuf are for the SigLIP 2 tokenizer: unlike the old
# ViT-L-14, it is a HuggingFace SentencePiece tokenizer (256k vocab) that open_clip
# loads through transformers, so without these get_tokenizer() raises ModuleNotFoundError
# at import time and the service never starts.
./venv/bin/pip install open_clip_torch fastapi uvicorn python-multipart \
    "numpy<2" insightface pytest httpx PyYAML Pillow \
    transformers sentencepiece protobuf

# rapidocr depends on opencv_python -> install without dependencies, add the rest by hand.
./venv/bin/pip install --no-deps rapidocr
./venv/bin/pip install pyclipper "Shapely>=1.7.1" "omegaconf!=2.2.1" colorlog six tqdm requests

# Only now overwrite what insightface dragged in: CPU onnxruntime -> GPU, full opencv ->
# headless (the server has no GUI libraries).
#
# The onnxruntime-gpu version is pinned on purpose: 1.28 is built against CUDA 13, while
# this machine (and torch cu128) provide CUDA 12.8. With 1.28 the CUDAExecutionProvider
# reports as available but fails to load (missing libcublasLt.so.13) and inference
# silently falls back to CPU. 1.22 is a CUDA 12 build and runs on the libraries torch
# already brought in.
#
# --force-reinstall is required: each pair shares an import namespace (cv2,
# onnxruntime), so uninstalling one sibling also deletes files owned by its
# replacement. Without it you get "ModuleNotFoundError: No module named 'cv2'" or
# "module 'onnxruntime' has no attribute '__version__'".
./venv/bin/pip uninstall -y onnxruntime opencv-python
./venv/bin/pip install --force-reinstall --no-deps "onnxruntime-gpu==1.22.0" "opencv-python-headless<4.10"
./venv/bin/pip install coloredlogs

./scripts/fetch_models.sh

# Runnable check: if this passes, the environment is genuinely usable.
./venv/bin/python - <<'PY'
import sys

import cv2
import numpy
import onnxruntime

if hasattr(onnxruntime, "preload_dlls"):
    onnxruntime.preload_dlls()

providers = onnxruntime.get_available_providers()
print("onnxruntime:", onnxruntime.__version__, providers)
print("cv2:", cv2.__version__)
print("numpy:", numpy.__version__)

problems = []
if "CUDAExecutionProvider" not in providers:
    problems.append(f"CUDAExecutionProvider missing (only got {providers})")
if not numpy.__version__.startswith("1."):
    problems.append(f"numpy must be 1.x for insightface, got {numpy.__version__}")

# "Available" is not enough: on a CUDA version mismatch the provider fails to load and
# inference silently falls back to CPU. Actually creating a session is the only
# reliable check.
session = onnxruntime.InferenceSession(
    "models/PP-OCRv5_mobile_det.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
active = session.get_providers()
print("session providers:", active)
if "CUDAExecutionProvider" not in active:
    problems.append(f"session is running on {active}, CUDA did not load")

from insightface.app import FaceAnalysis  # noqa: F401
from rapidocr import RapidOCR  # noqa: F401
print("rapidocr + insightface imports OK")

if problems:
    sys.exit("ENVIRONMENT ERROR:\n  - " + "\n  - ".join(problems))
print("environment OK")
PY

echo "done"
