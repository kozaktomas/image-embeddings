#!/usr/bin/env bash
# Postavi dev venv pro image-embeddings na boxu (GPU).
#
# Poradi instalace je zamerne. insightface si tahne onnxruntime (CPU) a
# opencv-python jako zavislosti, takze kdyby se instaloval az po nich, prebil by
# onnxruntime-gpu i opencv-python-headless (oba pary sdili stejny import namespace).
# Proto se nejdriv nainstaluje insightface a teprve pak se jeho verze prepisou.
set -euo pipefail

cd "$(dirname "$0")/.."
python3 -m venv venv
./venv/bin/pip install --upgrade pip

# PyTorch s CUDA (stejne jako produkcni venv)
./venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

./venv/bin/pip install open_clip_torch fastapi uvicorn python-multipart \
    "numpy<2" insightface pytest httpx PyYAML Pillow

# rapidocr zavisi na opencv_python -> instalace bez zavislosti, zbytek rucne.
./venv/bin/pip install --no-deps rapidocr
./venv/bin/pip install pyclipper "Shapely>=1.7.1" "omegaconf!=2.2.1" colorlog six tqdm requests

# Az ted prepsat to, co si pritahl insightface: CPU onnxruntime -> GPU,
# plne opencv -> headless (kvuli chybejicim GUI knihovnam na serveru).
#
# --force-reinstall je nutny: kazdy par sdili stejny import namespace (cv2,
# onnxruntime), takze odinstalace jednoho sourozence smaze i soubory, ktere
# vlastni jeho nahrada. Bez toho skonci "ModuleNotFoundError: No module named
# 'cv2'" resp. "module 'onnxruntime' has no attribute '__version__'".
#
# Verze onnxruntime-gpu je pripnuta zamerne: 1.28 je postavene na CUDA 13, ale
# tenhle stroj (i torch cu128) ma CUDA 12.8. S 1.28 se CUDAExecutionProvider
# tvari jako dostupny, ale nenacte se (chybi libcublasLt.so.13) a inference tise
# spadne na CPU. 1.22 je CUDA 12 build a bezi na knihovnach, ktere pritahl torch.
./venv/bin/pip uninstall -y onnxruntime opencv-python
./venv/bin/pip install --force-reinstall --no-deps "onnxruntime-gpu==1.22.0" "opencv-python-headless<4.10"
./venv/bin/pip install coloredlogs

./scripts/fetch_models.sh

# Runnable check: kdyz tohle projde, prostredi je opravdu pouzitelne.
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
    problems.append(f"chybi CUDAExecutionProvider (mam jen {providers})")
if not numpy.__version__.startswith("1."):
    problems.append(f"numpy musi byt 1.x kvuli insightface, mam {numpy.__version__}")

# "Available" nestaci: pri nesouladu verzi CUDA se provider nenacte a inference
# tise spadne na CPU. Jedina spolehliva kontrola je opravdu vytvorit session.
session = onnxruntime.InferenceSession(
    "models/PP-OCRv5_mobile_det.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
active = session.get_providers()
print("session providers:", active)
if "CUDAExecutionProvider" not in active:
    problems.append(f"session bezi na {active}, CUDA se nenacetla")

from insightface.app import FaceAnalysis  # noqa: F401
from rapidocr import RapidOCR  # noqa: F401
print("importy rapidocr + insightface OK")

if problems:
    sys.exit("CHYBA prostredi:\n  - " + "\n  - ".join(problems))
print("prostredi OK")
PY

echo "hotovo"
