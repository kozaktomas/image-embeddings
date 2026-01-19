FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# PyTorch CPU-only (much smaller)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

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

# Pre-download models during build
RUN python -c "import open_clip; open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')"
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])"

COPY server.py .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
