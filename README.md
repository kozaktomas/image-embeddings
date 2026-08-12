# Embeddings API

FastAPI service for image and face embeddings using OpenCLIP and InsightFace.

## Features

- **Image embeddings** - SigLIP 2 so400m/14 @378 (1152-dim vectors)
- **Text embeddings** - SigLIP 2 so400m/14 @378 (1152-dim vectors, same space as image embeddings)
- **Face embeddings** - InsightFace buffalo_l (512-dim vectors)
- **Era estimation** - Estimate photo decade using CLIP
- **OCR** - text from photos via PP-OCRv5 (Latin script incl. Czech), with bounding boxes and confidence
- **Text-only mode** - `EMBED_MODE=text` serves `/embed/text` alone, on CPU, for a machine with no GPU

## Build & Run

```bash
podman build -t emb .
podman run -p 8000:8000 emb
```

Two targets come out of the one Dockerfile. `full` is the default and is what the
command above builds; `text` is the CPU-only image described in
[Text-only mode](#text-only-mode).

```bash
podman build --target text -t emb-text .
podman run -p 8000:8000 emb-text
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
# transformers/sentencepiece/protobuf back the SigLIP 2 tokenizer, which unlike the old
# ViT-L-14 one is a HuggingFace SentencePiece tokenizer with a 256k vocabulary. Without
# them open_clip.get_tokenizer() raises ModuleNotFoundError and the service never starts.
pip install open_clip_torch fastapi uvicorn python-multipart \
    "numpy<2" insightface transformers sentencepiece protobuf

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
python -c "import open_clip; open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP2-378', pretrained='webli'); open_clip.get_tokenizer('ViT-SO400M-14-SigLIP2-378')"
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
# holds the CLIP model on the GPU and a second copy does not fit in 8 GB.
ssh box 'cd ~/dev/image-embeddings && CUDA_VISIBLE_DEVICES="" OCR_USE_CUDA=0 ./venv/bin/python -m pytest tests/ -v'

# OCR tests again on the GPU, to cover the CUDA path (small footprint, fits).
ssh box 'cd ~/dev/image-embeddings && ./venv/bin/python -m pytest tests/test_ocr_engine.py tests/test_ocr_helpers.py -v'

# Text-only mode. A separate invocation because the mode is read when server.py is
# imported: flipping it inside a running process would mean rebuilding the model.
ssh box 'cd ~/dev/image-embeddings && EMBED_MODE=text CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/ -v'
```

Both invocations skip what does not apply to their mode, so the counts differ and
neither should report zero skips:

```
full   31 passed, 7 skipped
text   29 passed, 9 skipped
```

### CLIP model

The image/text tower is **SigLIP 2 so400m/14 @378** (`ViT-SO400M-14-SigLIP2-378`,
`webli`, Apache 2.0), replacing the previous `ViT-L-14` / `laion2b_s32b_b82k`. On the
benchmark that matches what this service is used for — type words, get photos — it moves
COCO text→image R@1 from 46.1 to 55.8 (+21 % relative); zero-shot ImageNet goes 75.3 % →
84.1 %. Numbers are from the [SigLIP 2 paper](https://arxiv.org/abs/2502.14786), Table 1.

Environment variables: `CLIP_MODEL`, `CLIP_PRETRAINED`, `CLIP_PRECISION`
(`auto` / `fp16` / `fp32`; `auto` means fp16 on CUDA and fp32 on CPU).

**fp16 is a requirement, not a tuning knob.** Measured weight sizes:

| Model | Params | fp32 | fp16 | dim |
|---|---|---|---|---|
| ViT-L-14 (previous) | 427.6M (visual 304.0M / text 123.7M) | 1631 MiB | 816 MiB | 768 |
| SigLIP 2 so400m @378 | 1136.0M (visual 428.2M / text 707.8M) | 4334 MiB | 2167 MiB | 1152 |

The 3070 has 8 GB shared with a co-resident `photo-enhancer` (3442 MiB), leaving ~4.3 GB.
In fp32 this model does not fit — that is measured, not predicted: loading it in fp32
dies with `torch.OutOfMemoryError` after reaching 4.20 GiB. In fp16 it fits with room to
spare. Quality cost is nil in practice — SigLIP 2 was trained in bf16, and the consumer
stores these vectors as fp16 `halfvec` anyway.

Because of that ceiling, precision and device are handed to `open_clip` at construction
rather than applied afterwards. `.to(cuda)` followed by `.half()` stages the full fp32
model on the card first and OOMs during load, before the conversion that would have made
it fit.

Note the shape of the model: most of it is the *text* tower (708M of 1136M, mostly the
256k-token vocabulary), while images are the hot path. If VRAM ever gets tight, moving the
text tower to CPU frees ~1.3 GB and costs only the interactive query, which already has a
timeout and a full-text fallback.

### Measured: new model vs old

Service footprint and end-to-end `POST /embed/image` (JPEG decode included), same 7 real
photos, 3 runs each, on the RTX 3070:

| | ViT-L-14 fp32 (previous) | SigLIP 2 fp16 (current) |
|---|---|---|
| VRAM, whole service | 1972 MiB | 2516 MiB |
| VRAM peak under load | 1986 MiB | 2530 MiB |
| Median | 0.084 s/photo | 0.091 s/photo |
| p95 | 0.224 s | 0.202 s |
| Throughput | 11.88 photos/s | 11.00 photos/s |
| Interactive text query | 6.3 ms | 9.1 ms |

The bigger model costs +544 MiB and 8 % throughput, not the 2× that parameter counts
suggest: JPEG decode dominates a real request, so a heavier tower barely moves the total.
Re-embedding a 20 664-photo library lands around 31 minutes against 29.

**Changing the model is not just a restart.** The embedding width is part of the contract:
consumers store the vectors in a fixed-width column, so a model change means a schema
migration plus a full re-embed on their side. `/health` reports `clip.dim` so a consumer
can verify the width before it starts writing.

### Text-only mode

`EMBED_MODE=text` loads the text tower and nothing else — no visual tower, no InsightFace,
no OCR engine. It exists for `prodvps`, which has no GPU and runs this for one reason:
kukátko's search box. Embedding a query is the only call a person waits on, and the box
that used to serve it is usually powered off, so semantic search quietly degraded to
full-text. Queue work (image, face, OCR) stays on the box and still wakes it over
Wake-on-LAN.

It fits on a CPU because the text side is small work: a fixed 64-token context over 27
layers of width 1152, roughly 53 GFLOPs per query, against kukátko's 5-second budget for
an interactive search.

| | |
|---|---|
| Text tower | 707.8M of the model's 1136M params, 2.7 GB fp32 |
| Visual tower, dropped at load | 428.2M params, ~1.7 GB fp32 |
| Peak during load | the full 4.3 GB |

open_clip cannot build one tower without the other, so the load peaks at the full model
before `model.visual` is released. A memory limit has to clear that peak, not the steady
state.

`/embed/image`, `/embed/face`, `/estimate/era` and `/ocr/image` answer **503** naming the
mode. Not 404: the route exists and the service is healthy, the capability is simply
switched off, while a 404 reads as a wrong URL or a stale image. `/health` reports `mode`,
and still reports `clip.dim` so a consumer can verify the width in either mode.

Environment variables: `EMBED_MODE` (`full` / `text`), `TORCH_NUM_THREADS`. The second is
not a tuning knob — torch sizes its thread pool from the host CPU count rather than the
cgroup quota, so inside a container limited to fewer CPUs than the host has it starts too
many threads and they contend with each other. Set it to the container's CPU limit.

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
