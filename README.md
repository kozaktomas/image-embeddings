# Embeddings API

FastAPI service for image and face embeddings using OpenCLIP and InsightFace.

## Features

- **Image embeddings** - CLIP ViT-L-14 (768-dim vectors)
- **Text embeddings** - CLIP ViT-L-14 (768-dim vectors, same space as image embeddings)
- **Face embeddings** - InsightFace buffalo_l (512-dim vectors)
- **Era estimation** - Estimate photo decade using CLIP
- **OCR** - text from photos via PP-OCRv5 (Latin script incl. Czech), with bounding boxes and confidence

## Build & Run

```bash
podman build -t emb .
podman run -p 8000:8000 emb
```

## API Endpoints

### Health Check
```
GET /health
```

### Image Embedding
```
POST /embed/image
Content-Type: multipart/form-data
Body: file=<image>
```

### Text Embedding
```
POST /embed/text
Content-Type: application/json
Body: {"text": "a photo of a cat"}
```

### Face Embedding
```
POST /embed/face
Content-Type: multipart/form-data
Body: file=<image>
```

### Era Estimation
```
POST /estimate/era
Content-Type: multipart/form-data
Body: file=<image>
```

### OCR
```
POST /ocr/image
Content-Type: multipart/form-data
Body: file=<image>, min_confidence=<float, default 0.5>
```

Returns the merged text plus one entry per detected block:

```json
{
  "text": "BEZPEČNOSTNÍ\nZÓNA",
  "blocks_count": 2,
  "blocks": [
    {"text": "BEZPEČNOSTNÍ", "bbox": [1988.1, 567.0, 2919.5, 870.2], "confidence": 0.95},
    {"text": "ZÓNA", "bbox": [2100.0, 900.0, 2700.0, 1180.0], "confidence": 0.99}
  ],
  "min_confidence": 0.5,
  "lang": "latin",
  "model": "PP-OCRv5_mobile"
}
```

`bbox` is `[x_min, y_min, x_max, y_max]` in pixels of the original image, the same
shape `/embed/face` returns. Blocks come back in reading order. A photo with no text
is a normal 200 with an empty result, not an error.

## Examples

```bash
# Image embedding
curl -X POST http://localhost:8000/embed/image -F "file=@photo.jpg"

# Text embedding (same vector space as image)
curl -X POST http://localhost:8000/embed/text -H "Content-Type: application/json" -d '{"text": "a photo of a cat"}'

# OCR
curl -X POST http://localhost:8000/ocr/image -F "file=@photo.jpg"
```

## Native Deployment (GPU)

```bash
# Clone and set up venv
sudo mkdir -p /opt/image-embeddings
git clone <repo-url> /opt/image-embeddings
cd /opt/image-embeddings
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
pip install open_clip_torch fastapi uvicorn python-multipart \
    "numpy<2" insightface

# OCR: rapidocr depends on opencv_python, which collides with the headless build
# (same cv2 namespace), so install it without dependencies.
pip install --no-deps rapidocr
pip install pyclipper "Shapely>=1.7.1" "omegaconf!=2.2.1" colorlog six tqdm requests PyYAML coloredlogs

# onnxruntime-gpu is pinned on purpose: 1.28 is built against CUDA 13, while this
# machine (and torch cu128) provide CUDA 12.8. With 1.28 the CUDA provider looks
# available but fails to load, and inference silently falls back to CPU.
# --force-reinstall is required because uninstalling a sibling that shares an
# import namespace deletes files its replacement owns.
pip uninstall -y onnxruntime opencv-python
pip install --force-reinstall --no-deps "onnxruntime-gpu==1.22.0" "opencv-python-headless<4.10"

# Pre-download models
python -c "import open_clip; open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')"
python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])"
OCR_MODELS_DIR=/opt/image-embeddings/models ./scripts/fetch_models.sh

# Install and start systemd service
sudo cp deploy/image-embeddings.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now image-embeddings
```

Check status:
```bash
sudo systemctl status image-embeddings
curl http://localhost:8000/health
```

## Development

Development and tests run on a GPU machine (`box`), not on the Raspberry Pi.

```bash
./scripts/sync-box.sh                                     # push the repo to the box
ssh box 'cd ~/dev/image-embeddings && ./scripts/setup_dev_box.sh'

# Full suite. CUDA is disabled here on purpose: the production service already
# holds a CLIP ViT-L-14 on the GPU and a second copy does not fit in 8 GB.
ssh box 'cd ~/dev/image-embeddings && CUDA_VISIBLE_DEVICES="" OCR_USE_CUDA=0 ./venv/bin/python -m pytest tests/ -v'

# OCR tests again on the GPU, to cover the CUDA path (small footprint, fits).
ssh box 'cd ~/dev/image-embeddings && ./venv/bin/python -m pytest tests/test_ocr_engine.py tests/test_ocr_helpers.py -v'
```

### OCR models

`./scripts/fetch_models.sh` downloads PP-OCRv5 detection and Latin recognition models
from HuggingFace and extracts the character dictionary. ModelScope, which RapidOCR uses
by default, is blocked by the DNS4EU Protective resolver on this network, so HuggingFace
is the source and the models are pinned by SHA256.

Environment variables: `OCR_MODELS_DIR` (where the models live), `OCR_USE_CUDA`
(`auto` / `1` / `0`), `OCR_MIN_CONFIDENCE` (default threshold).

### Measured throughput

RTX 3070, 7 real photos (1024×576 to 4928×3264) plus synthetic images, 3 runs each:

| Provider | Median | p95 | Throughput |
|---|---|---|---|
| CUDA | 0.226 s/photo | 0.407 s | 4.43 photos/s |
| CPU (24 cores) | 0.687 s/photo | 0.816 s | 1.46 photos/s |

CUDA is ~3× faster, so production runs on it. Adding OCR raised the service's GPU
memory from 1922 MiB to 2576 MiB.

### PP-OCRv6

RapidOCR defaults to `PP-OCRv6`, so it was compared against the pinned PP-OCRv5 Latin
model on the same photos. It won on some Czech words (`státního`, `ŠOŠUVKA`, `HASICI`)
and lost on others (`PAMÁTKOVY` without the acute, `15OSOB` run together), while adding
spurious blocks; speed was identical. No demonstrable win, so PP-OCRv5 Latin stays.

## License

MIT
