FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    build-essential \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# PyTorch CPU-only (much smaller)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# ONNX Runtime CPU
RUN pip install --no-cache-dir onnxruntime

RUN pip install --no-cache-dir \
    open_clip_torch \
    Pillow \
    fastapi \
    uvicorn \
    python-multipart \
    "numpy<2" \
    "opencv-python-headless<4.10" \
    insightface

# RapidOCR depends on opencv_python, which would collide with opencv-python-headless
# (same cv2 namespace) -> install without dependencies and add the rest by hand.
RUN pip install --no-cache-dir --no-deps rapidocr && \
    pip install --no-cache-dir pyclipper "Shapely>=1.7.1" "omegaconf!=2.2.1" \
    colorlog six tqdm requests PyYAML

# insightface pulls in full opencv-python, which shadows the headless build and then
# needs libGL.so.1. Uninstall it and force-reinstall headless -- uninstalling a sibling
# also deletes files owned by its replacement.
RUN pip uninstall -y opencv-python && \
    pip install --no-cache-dir --force-reinstall --no-deps "opencv-python-headless<4.10"

# Pre-download models during build
RUN python -c "import open_clip; open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')"
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])"

COPY server.py ocr.py ./
COPY scripts/fetch_models.sh scripts/

# OCR models from HuggingFace (OCR runs on CPU inside the container)
ENV OCR_MODELS_DIR=/app/models
ENV OCR_USE_CUDA=0
RUN chmod +x scripts/fetch_models.sh && ./scripts/fetch_models.sh

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
