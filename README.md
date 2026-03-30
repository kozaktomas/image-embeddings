# Embeddings API

FastAPI service for image and face embeddings using OpenCLIP and InsightFace.

## Features

- **Image embeddings** - CLIP ViT-L-14 (768-dim vectors)
- **Text embeddings** - CLIP ViT-L-14 (768-dim vectors, same space as image embeddings)
- **Face embeddings** - InsightFace buffalo_l (512-dim vectors)
- **Era estimation** - Estimate photo decade using CLIP

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

## Examples

```bash
# Image embedding
curl -X POST http://localhost:8000/embed/image -F "file=@photo.jpg"

# Text embedding (same vector space as image)
curl -X POST http://localhost:8000/embed/text -H "Content-Type: application/json" -d '{"text": "a photo of a cat"}'
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
    "numpy<2" "opencv-python-headless<4.10" insightface onnxruntime

# Pre-download models
python -c "import open_clip; open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')"
python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])"

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

## License

MIT
