# Embeddings API

FastAPI service for image and face embeddings using OpenCLIP and InsightFace.

## Features

- **Image embeddings** - SigLIP 2 so400m/14 @378 (1152-dim vectors)
- **Text embeddings** - SigLIP 2 so400m/14 @378 (1152-dim vectors, same space as image embeddings)
- **Face embeddings** - InsightFace buffalo_l (512-dim vectors)
- **Face attributes** - per face: 5 keypoints, age, gender, head pose (pitch/yaw/roll), eye openness,
  sharpness and size; optionally the 106 landmarks. Age and gender are a weak signal, see
  [Face Embedding](#face-embedding)
- **Age and gender from face + body** - MiVOLO v2 per face, using the face crop and the crop of the
  person it belongs to; much better than InsightFace's age, see [MiVOLO v2](#mivolo-v2-age-and-gender)
- **Era estimation** - Estimate photo decade using CLIP
- **OCR** - text from photos via PP-OCRv5 (Latin script incl. Czech), with bounding boxes and confidence
- **Text-only mode** - `EMBED_MODE=text` serves `/embed/text` alone, on CPU, for a machine with no GPU

## Build & Run

```bash
podman build -t emb .
podman run --device nvidia.com/gpu=all -p 8000:8000 emb   # podman (CDI)
docker run --gpus all -p 8000:8000 emb                    # docker
```

The `full` image is built for the GPU: CUDA torch (cu128) and `onnxruntime-gpu` 1.22. The
CUDA runtime comes inside the image as pip wheels, so the host only needs the NVIDIA
driver and the container toolkit. Without a GPU it still starts and runs on CPU. Check
`/health`: `"device": "cuda"` and `"ocr": {"provider": "CUDAExecutionProvider"}`.

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
POST /embed/face[?landmarks=1]
Content-Type: multipart/form-data
Body: file=<image>
```

One entry per detected face. `face_index`, `dim`, `embedding`, `bbox` and `det_score`
are unchanged from before; `kps` to `landmarks_106` were added in `attributes_version` 1 and
come from models buffalo_l was already running for every face (only the eye ratio and the
sharpness are computed on top, 0.2–0.3 ms per face on CPU). `mivolo` was added in
`attributes_version` 2 and comes from two more models, see
[MiVOLO v2](#mivolo-v2-age-and-gender).

```json
{
  "faces_count": 1,
  "faces": [
    {
      "face_index": 0,
      "dim": 512,
      "embedding": [0.012, "…"],
      "bbox": [412.3, 118.9, 530.1, 270.4],
      "det_score": 0.87,
      "kps": [[448.1, 180.2], [500.7, 179.5], [476.0, 210.3], [452.9, 237.8], [497.4, 237.1]],
      "age": 31,
      "gender": "female",
      "pose": {"pitch": -4.2, "yaw": 12.8, "roll": 1.5},
      "eyes": {"left": 0.29, "right": 0.31},
      "sharpness": 412.7,
      "size": {"width": 117.8, "height": 151.5, "rel_area": 0.0217},
      "mivolo": {
        "age": 34.62,
        "gender": "female",
        "gender_probability": 0.997,
        "body_used": true,
        "person_bbox": [301.5, 80.2, 702.9, 1046.0]
      }
    }
  ],
  "model": "buffalo_l (ResNet100)",
  "attributes_version": 2
}
```

| Field | Meaning |
|---|---|
| `kps` | The 5 detection keypoints `[[x, y], …]` in original image pixels: eye, eye, nose tip, mouth corner, mouth corner, as InsightFace orders them. |
| `age` | Integer years from `genderage`. **Coarse**: poor on children, the elderly, scanned prints and blurry faces. A weak signal (e.g. one input to dating an undated photo), never a fact about a person. |
| `gender` | `"female"` / `"male"` from `genderage`, same caveat as `age`. |
| `pose` | Head pose in degrees from the 3D landmarks. `yaw` < 0: the subject turns to their own right (nose toward the image's left); `yaw` > 0: to their own left. `roll` > 0: the head tilts clockwise as seen in the image. Both flip sign on a mirrored photo (checked, and pinned by a test). Pitch and roll are noisier than yaw; use them for "roughly frontal" filters, not geometry. |
| `eyes` | Eye aspect ratio per eye (vertical lid distance / horizontal eye width) from the 106 landmarks. `left`/`right` is the **subject's own** left/right, i.e. `left` is on the image's right in an unmirrored photo. Open eyes read about 0.25–0.35, closed ones under ~0.15, but the level varies by person and pose: compare photos of the same face rather than applying a fixed threshold. |
| `sharpness` | Variance of the Laplacian of the grey face crop (bbox clamped to the image) scaled to 112×112, so faces of different sizes are comparable. Unitless; higher is sharper. Meaningful as a ranking (best frame of a series, best exemplar of a person), not as an absolute threshold across cameras. |
| `size` | `width`/`height` of the bbox in pixels and `rel_area` = bbox area / image area, with the bbox clamped to the image so `rel_area` is in (0, 1]. |
| `landmarks_106` | Only with `?landmarks=1`: the 106 2D landmarks `[[x, y], …]`. Off by default to keep responses small. |
| `mivolo.age` | Years as a float (two decimals) from MiVOLO v2, clamped to the model's 0–122 range. The better age of the two; still an estimate. |
| `mivolo.gender` | `"female"` / `"male"`, the same labels as `gender` (MiVOLO's own class order is the reverse of genderage's; it is mapped). |
| `mivolo.gender_probability` | Softmax probability of that label, 0.5–1. The model is very sure of itself (≈0.997 on most faces), so treat it as a tie-breaker, not a calibrated confidence. |
| `mivolo.body_used` | `true` when the crop of the person's body went into the estimate, `false` when it ran on the face alone (no person box matched, the match was ambiguous, or the body crop was too small or too covered by other people). |
| `mivolo.person_bbox` | `[x1, y1, x2, y2]` of the person box matched to this face, or `null`. Can be set while `body_used` is `false`: the box matched but its crop was rejected. |

`mivolo` is `null` for every face when the service runs with `MIVOLO_ENABLED=0`.

Any attribute a model did not produce comes back as `null` rather than missing. `model` is
deliberately still `buffalo_l (ResNet100)`: consumers re-detect faces when it changes, and
the embeddings are identical. New or changed attributes bump `attributes_version` instead;
`/health` reports it as `face_attributes_version` in full mode.

The attribute models (genderage and both landmark models) are fed the image in the BGR
order insightface was trained on, while detection and recognition keep the RGB input they
have always had, so stored embeddings stay valid. On RGB, genderage called every face in
insightface's own group photo "female", including the three men.

### MiVOLO v2 age and gender

[MiVOLO](https://github.com/WildChlamydia/MiVOLO) (Kuprashevich & Tolstykh,
[arXiv:2307.04616](https://arxiv.org/abs/2307.04616)) estimates age and gender from two
inputs at once: the face crop and the crop of the body it belongs to. The v2 checkpoint
reports an age MAE of 3.65 years on Lagenda with both inputs. It runs next to InsightFace's
`genderage`, which stays in the response unchanged.

Per photo, in `age_gender.py`:

1. A YOLOv8x face + person detector finds the person boxes. Faces still come only from
   buffalo_l; the detector's own face boxes are only used to help assign buffalo_l faces to
   persons (a face buffalo_l missed still claims its own body, so it cannot be handed to a
   neighbour).
2. Faces are assigned to persons with MiVOLO's own `assign_faces()`: Hungarian matching on
   the mean of IoU and the share of the face inside the person box, one-to-one, so a person
   box never goes to two faces. Two rules on top, both of which can only take a body away:
   at least half of the face has to lie inside the box, and an ambiguous match is dropped
   (another free person box fits the face, or another face fits the box, within 10 % of the
   match's own score — a face inside two overlapping people, two faces inside one box).
3. Crops are cut exactly as MiVOLO cuts them: the face box as is; the person box with every
   face (its own too) and every other person blacked out, dropped when under 50 px or less
   than 40 % of it is left. In group photos where people stand close, that last rule drops
   most bodies; that is MiVOLO's behaviour and those faces run face-only.
4. All faces of the photo go through MiVOLO in one batch: fp16 on CUDA, fp32 on CPU.

The detector and MiVOLO get the image in BGR, the order they were trained on (MiVOLO's
preprocessing converts its BGR crops to RGB itself). The port was checked against MiVOLO's
original code on the same detections: assignment, crops and input tensors are
bit-identical, and the network's outputs agree to 1.2e-7 with the original model class
running on its pinned timm 0.8.13.dev0.

| | MiVOLO v2 | Person detector |
|---|---|---|
| Source | [`iitolstykh/mivolo_v2`](https://huggingface.co/iitolstykh/mivolo_v2) @ `e8bdd8ae` | [`iitolstykh/YOLO-Face-Person-Detector`](https://huggingface.co/iitolstykh/YOLO-Face-Person-Detector) @ `b3d071ae` |
| Architecture | `mivolo_d1_384` (VOLO-D1, two 384×384 inputs) | YOLOv8x, classes `face` and `person` |
| Parameters | 28.8M | 68.1M (fused) |
| Weights | 110 MiB (`model.safetensors`, fp32) | 130 MiB (`yolov8x_person_face.pt`) |
| Licence | **CC-BY-NC-4.0** (non-commercial) | **AGPL-3.0**, run through `ultralytics` (AGPL-3.0) |

Both checkpoints are baked into the `full` image at build time from the pinned commits,
SHA-256 checked (`scripts/fetch_mivolo.sh`), so a container start needs no network. The
text image ships neither them nor ultralytics.

The MiVOLO package itself is not installed: it pins timm 0.8.13.dev0, torch 2.5.1 and
transformers 4.51.0, which would break open_clip and the SigLIP 2 tokenizer. The few
hundred lines that build the network and its preprocessing are vendored in
`mivolo_model.py` (Apache-2.0, licence in `third_party/MiVOLO/LICENSE`) and run on the timm
and torch already in the image. Importing ultralytics and calling it changes process-wide
settings (`OMP_NUM_THREADS`, `CUBLAS_WORKSPACE_CONFIG`, OpenCV's and torch's thread counts);
`age_gender.py` restores each of them after every call, and a test checks it.

Environment variables: `MIVOLO_ENABLED` (default `1`; `0` loads neither model and every face
reports `mivolo: null`), `MIVOLO_MODELS_DIR` (default `models/mivolo`; `/app/models/mivolo`
in the image). `/health` reports `age_gender` with both models, their sources and MiVOLO's
precision, or `{"enabled": false}`.

#### Measured

Same four photos through the image before this change and after it, one container at a
time, `/embed/face` median of 5 runs (JPEG decode and buffalo_l included; buffalo_l runs on
CPU in both): insightface's `t1` (6 people standing close), scikit-image's `astronaut`
(one person), the NASA Expedition 64 crew portrait (7 people, 5746×4597) and the 2009
White House family portrait (4 people, 4096×2731), the last two public domain from
Wikimedia Commons.

| Photo | Faces | With a body | RTX 3070 before | after | CPU (24 threads) before | after |
|---|---|---|---|---|---|---|
| t1 | 6 | 1 | 0.630 s | 0.727 s | 0.734 s | 2.375 s |
| astronaut | 1 | 1 | 0.413 s | 0.452 s | 0.399 s | 0.855 s |
| Expedition 64 | 7 | 2 | 1.163 s | 1.196 s | 1.329 s | 3.086 s |
| family portrait | 5 | 1 | 0.807 s | 0.840 s | 0.867 s | 1.988 s |

On the GPU it costs 0.03–0.10 s per photo and **+796 MiB** of VRAM for the whole service
(2474 → 3270 MiB after the run, PyTorch's cache included). On CPU the YOLOv8x detector
dominates and a photo takes 2–3× as long; `MIVOLO_ENABLED=0` is the switch for a CPU
machine that cannot afford it.

Every other output is untouched: between the two images, SigLIP image and text embeddings,
buffalo_l embeddings, `/estimate/era` and every pre-existing face field are identical
(max abs difference 0.0), on CPU and on GPU, and `pip freeze` only gained ultralytics and
its dependencies.

Where the two age estimates part ways the most is where InsightFace is known to be weak.
On the family portrait (dated October 2009) the two daughters were 11 and 8; MiVOLO
says 15.8 and 10.0, InsightFace 27 and 36. On the Expedition 64 crew, roughly 37 to 55 at
the time, MiVOLO ranges 36–50 and InsightFace 45–68. Gender agrees between the two on all
19 detections. (The family portrait's fifth "face" is a figurine on the mantelpiece that
buffalo_l has always detected; MiVOLO gives it an age like any other face.)

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
# (same cv2 namespace), so install it without dependencies. Pinned: 3.10 loads its
# models lazily, which ocr.py reads as "running on CPU".
pip install --no-deps "rapidocr==3.9.2"
pip install pyclipper "Shapely>=1.7.1" "omegaconf!=2.2.1" colorlog six tqdm requests PyYAML coloredlogs

# onnxruntime-gpu is pinned on purpose: 1.28 is built against CUDA 13, while this
# machine (and torch cu128) provide CUDA 12.8. With 1.28 the CUDA provider looks
# available but fails to load, and inference silently falls back to CPU.
# --force-reinstall is required because uninstalling a sibling that shares an
# import namespace deletes files its replacement owns.
# ultralytics runs MiVOLO's person detector; pinned. It also pulls opencv-python, which
# the next step removes again.
pip install "ultralytics==8.4.215"
pip uninstall -y onnxruntime opencv-python
pip install --force-reinstall --no-deps "onnxruntime-gpu==1.22.0" "opencv-python-headless<4.10"

# Pre-download models
python -c "import open_clip; open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP2-378', pretrained='webli'); open_clip.get_tokenizer('ViT-SO400M-14-SigLIP2-378')"
python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])"
OCR_MODELS_DIR=/opt/image-embeddings/models ./scripts/fetch_models.sh
MIVOLO_MODELS_DIR=/opt/image-embeddings/models/mivolo ./scripts/fetch_mivolo.sh

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
full   87 passed, 15 skipped
text   31 passed, 46 skipped
```

(Run inside the `full` and `text` images, which ship no DejaVu font, so the 7 OCR tests
that render text skip in `full`; with the font installed they run.)

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

#### Measured in production

On `prodvps` (8 vCPU Skylake @ 2.0 GHz with AVX-512, `TORCH_NUM_THREADS=6`, `mem_limit: 6g`),
20 sequential queries:

Measured on each host locally, 20 sequential queries:

| | text-only on prodvps | full on the 3070 |
|---|---|---|
| `/embed/text` median | 0.432 s | 0.0086 s |
| p95 | 0.495 s | 0.0102 s |
| RSS, steady | 3.29 GiB (55 % of the limit) | — |

The GPU is ~50× faster, and it does not matter: the real alternative is not the box at 9 ms
but the box asleep and no answer at all. 0.43 s is under a tenth of kukátko's 5-second
budget, which is why this needs neither ONNX nor quantisation. Kukátko runs on the same
machine and calls it over the compose network, so it pays none of the ~20 ms tailnet RTT
that shows up when measuring from elsewhere. RSS confirms the limit is sized for the 4.3 GB
load peak rather than the steady state. Cold start to a `/health` answer is roughly 40 s,
polled rather than instrumented.

**The vectors agree with the GPU's, but are not identical to them.** The box computes in
fp16 and this path in fp32, so the two never match bit for bit. Measured over five Czech and
English queries, cosine ranges from **0.9999992** to **0.9999998**, with a worst per-component
difference of 1.3e-4 — fp16 rounding, an angle of about 0.07°, no effect on ranking. This is
the check that matters: the stored 1152-dim image vectors were written by the fp16 GPU path,
and search compares them against fp32 CPU query vectors.

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

The code in this repository is MIT, except `mivolo_model.py`, which is vendored from
MiVOLO under Apache-2.0 (`third_party/MiVOLO/LICENSE`).

The models it downloads come with their own licences:

- SigLIP 2 (`webli`): Apache-2.0.
- InsightFace buffalo_l: non-commercial research use only, per InsightFace.
- MiVOLO v2 weights: **CC-BY-NC-4.0**, non-commercial use only.
- YOLOv8x face + person detector weights and the `ultralytics` package that runs them:
  **AGPL-3.0**. The `full` image includes ultralytics; anyone distributing that image or
  offering it as a network service takes on AGPL-3.0's obligations for it. The `text` image
  contains neither, and `MIVOLO_ENABLED=0` keeps both models out of a running `full` service.
