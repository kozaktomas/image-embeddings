# Embeddings API

FastAPI service for image and face embeddings using OpenCLIP and InsightFace.

## Features

- **Image embeddings** - CLIP ViT-L-14 (768-dim vectors)
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

## Example

```bash
curl -X POST http://localhost:8000/embed/image -F "file=@photo.jpg"
```

## License

MIT
