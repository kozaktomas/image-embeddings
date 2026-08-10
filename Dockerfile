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

# RapidOCR zavisi na opencv_python, ktere by kolidovalo s opencv-python-headless
# (stejny cv2 namespace) -> instalace bez zavislosti a rucne doinstalovany zbytek.
RUN pip install --no-cache-dir --no-deps rapidocr && \
    pip install --no-cache-dir pyclipper "Shapely>=1.7.1" "omegaconf!=2.2.1" \
    colorlog six tqdm requests PyYAML

# insightface si sam tahne plne opencv-python, ktere prebije headless variantu
# a pak chybi libGL.so.1. Odinstalovat a headless nasadit znovu natvrdo --
# odinstalace sourozence smaze i soubory, ktere vlastni jeho nahrada.
RUN pip uninstall -y opencv-python && \
    pip install --no-cache-dir --force-reinstall --no-deps "opencv-python-headless<4.10"

# Pre-download models during build
RUN python -c "import open_clip; open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')"
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])"

COPY server.py ocr.py ./
COPY scripts/fetch_models.sh scripts/

# OCR modely z HuggingFace (v kontejneru bezi OCR na CPU)
ENV OCR_MODELS_DIR=/app/models
ENV OCR_USE_CUDA=0
RUN chmod +x scripts/fetch_models.sh && ./scripts/fetch_models.sh

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
