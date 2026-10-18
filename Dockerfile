# Two images out of one file:
#
#   --target text   CPU-only, text tower only. Runs on the VPS and answers kukátko's
#                   search box, which is the one interactive call in the system.
#   --target full   Everything, on the GPU: CUDA torch (cu128) and onnxruntime-gpu.
#                   What the box publishes and what a bare `docker build` with no
#                   --target still produces, which is why `full` is last. Run it with
#                   `docker run --gpus all`; without a GPU it still starts, on CPU.
#
# Both get the SigLIP 2 checkpoint from `base`, so it is downloaded once at build time.
# `full` copies the HuggingFace cache out of `base` instead of inheriting the stage,
# because inheriting would also inherit the CPU torch that `full` has to replace.

FROM python:3.10-slim AS base

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --upgrade pip

# numpy has to stay on 1.x: the opencv and onnxruntime wheels are built against the 1.x
# ABI and fail at import with "numpy.core.multiarray failed to import" under numpy 2.
# This used to hold by accident, because one pip command installed numpy, opencv and
# insightface together and the resolver saw the pin. Splitting the install across stages
# broke that -- `pip install insightface` pulled numpy 2 back in -- so the pin is a
# constraint file that applies to every later pip command instead.
RUN echo "numpy<2" > /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

# PyTorch CPU-only (much smaller)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# transformers/sentencepiece/protobuf back the SigLIP 2 tokenizer (a HuggingFace
# SentencePiece tokenizer with a 256k vocab, loaded by open_clip through transformers).
# Without them get_tokenizer() fails at import and the service never starts.
RUN pip install --no-cache-dir \
    open_clip_torch \
    Pillow \
    fastapi \
    uvicorn \
    python-multipart \
    numpy \
    transformers \
    sentencepiece \
    protobuf

# Pre-download the weights and the tokenizer. This is the layer that matters: ~4.5 GB
# that neither target ever changes, so a code-only deploy pulls megabytes. Baking them in
# also means a container start does not depend on HuggingFace being reachable -- this
# network has already had DNS4EU block ModelScope out from under RapidOCR.
RUN python -c "import open_clip; open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP2-378', pretrained='webli'); open_clip.get_tokenizer('ViT-SO400M-14-SigLIP2-378')"

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]


# === text-only ===
# No insightface, no onnxruntime, no rapidocr, no opencv, no OCR models, no build
# toolchain to compile any of them. server.py still imports ocr, which is safe: ocr.py
# imports its heavy dependencies inside functions, and text mode never calls them.
FROM base AS text

COPY server.py ocr.py ./

ENV EMBED_MODE=text


# === full ===
FROM python:3.10-slim AS full

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --upgrade pip

RUN echo "numpy<2" > /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

# PyTorch with CUDA 12.8. The wheels bring the CUDA runtime and cuDNN as nvidia-* pip
# packages, so a plain python base image is enough: the container only needs the host
# driver, which the NVIDIA container toolkit mounts in at `--gpus all`.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir \
    open_clip_torch \
    Pillow \
    fastapi \
    uvicorn \
    python-multipart \
    numpy \
    transformers \
    sentencepiece \
    protobuf

COPY --from=base /root/.cache/huggingface /root/.cache/huggingface

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

# Build tools are needed to compile insightface; libglib2.0-0 is what opencv wants at
# runtime; curl fetches the OCR models.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    build-essential \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "opencv-python-headless<4.10" \
    insightface

# RapidOCR depends on opencv_python, which would collide with opencv-python-headless
# (same cv2 namespace) -> install without dependencies and add the rest by hand.
# Pinned: 3.10 loads det/rec lazily on first inference, so ocr.py finds no session at
# startup, reports CPUExecutionProvider and refuses to start with OCR_USE_CUDA=1.
RUN pip install --no-cache-dir --no-deps "rapidocr==3.9.2" && \
    pip install --no-cache-dir pyclipper "Shapely>=1.7.1" "omegaconf!=2.2.1" \
    colorlog six tqdm requests PyYAML

# insightface pulls in full opencv-python, which shadows the headless build and then
# needs libGL.so.1, and CPU onnxruntime, which shadows onnxruntime-gpu. Uninstall both and
# force-reinstall the replacements -- uninstalling a sibling also deletes files owned by
# its replacement.
# onnxruntime-gpu is pinned on purpose: 1.28 is built against CUDA 13, while torch cu128
# provides CUDA 12.8. With 1.28 the CUDA provider looks available but fails to load, and
# inference silently falls back to CPU.
RUN pip uninstall -y opencv-python onnxruntime && \
    pip install --no-cache-dir --force-reinstall --no-deps \
    "onnxruntime-gpu==1.22.0" "opencv-python-headless<4.10" && \
    pip install --no-cache-dir coloredlogs flatbuffers

# buffalo_l comes from a GitHub release, which has dropped the connection mid-build
# before and failed the whole pipeline. Retry instead of failing on the first try.
RUN for i in 1 2 3 4 5; do \
    python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])" && exit 0; \
    echo "buffalo_l download failed (attempt $i), retrying"; rm -rf /root/.insightface/models/buffalo_l*; sleep 15; \
    done; exit 1

COPY server.py ocr.py ./
COPY scripts/fetch_models.sh scripts/

# OCR models from HuggingFace (OCR runs on CPU inside the container)
ENV OCR_MODELS_DIR=/app/models
ENV OCR_USE_CUDA=0
RUN chmod +x scripts/fetch_models.sh && ./scripts/fetch_models.sh
