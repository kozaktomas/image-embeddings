# OCR endpoint — implementační plán

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Přidat do služby `image-embeddings` endpoint `POST /ocr/image`, který z fotky přečte tištěný text a vrátí ho i s bounding boxy a jistotou.

**Architecture:** Nový modul `ocr.py` obaluje RapidOCR (PP-OCRv5, latinkový rozpoznávací model) běžící přes ONNX Runtime nad lokálně uloženými modely. `server.py` přidává jen routing. Modely se stahují jednorázově skriptem z HuggingFace.

**Tech Stack:** Python 3.12, FastAPI, RapidOCR 3.9.x, ONNX Runtime (GPU na boxu, CPU v Dockeru), Pillow, NumPy, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-10-ocr-endpoint-design.md`.
- **Vývoj a testy běží na boxu**, ne na Pi. Git zůstává na Pi; na box se synchronizuje rsyncem (`ssh box`, uživatel `panbotka`, sudo bez hesla je k dispozici).
- **Dev server na portu 8010.** Port 8000 drží produkční `image-embeddings`, 8001 `photo-enhancer`.
- **Modely jen z HuggingFace.** `modelscope.cn` je blokovaný resolverem DNS4EU Protective a blokace se obcházet nebude.
- `numpy<2` a `opencv-python-headless<4.10` jsou dané InsightFace — nesmí se rozbít.
- InsightFace zůstává na `CPUExecutionProvider`; instalace `onnxruntime-gpu` nesmí změnit chování stávajících endpointů.
- Bez `git push` — vše zůstává na lokální branchi `ocr-endpoint`, dokud uživatel neřekne jinak.
- Fotky uživatele se **necommitují**; leží v `testdata/`, které je v `.gitignore`.

## Struktura souborů

| Soubor | Odpovědnost |
|---|---|
| `ocr.py` | Načtení RapidOCR nad lokálními modely + převod výstupu do tvaru API (nový) |
| `server.py` | Routing a HTTP, přidává `/ocr/image` a rozšiřuje `/health` (úprava) |
| `tests/test_ocr_helpers.py` | Čisté funkce: polygon → bbox, čtecí pořadí, prahování (nový) |
| `tests/test_ocr_engine.py` | RapidOCR nad reálnými modely: diakritika, prázdný obrázek, velká fotka (nový) |
| `tests/test_api_ocr.py` | Kontrakt endpointu přes `TestClient` (nový) |
| `scripts/fetch_models.sh` | Stažení ONNX z HF, ověření SHA256, extrakce slovníku (nový) |
| `scripts/setup_dev_box.sh` | Vytvoření dev venv na boxu (nový) |
| `scripts/sync-box.sh` | rsync repa z Pi na box (nový) |
| `scripts/bench_ocr.py` | Měření propustnosti a kvality, CUDA vs CPU (nový) |
| `Dockerfile` | Přidání rapidocr + předstažení OCR modelů, CPU varianta (úprava) |
| `.gitignore` | Přidat `models/`, `testdata/` (úprava) |
| `README.md` | Dokumentace endpointu a dev postupu (úprava) |

---

### Task 1: Skript pro stažení modelů

**Files:**
- Create: `scripts/fetch_models.sh`
- Modify: `.gitignore`

**Interfaces:**
- Produces: adresář `$OCR_MODELS_DIR` (výchozí `models/`) se soubory `PP-OCRv5_mobile_det.onnx`, `latin_PP-OCRv5_mobile_rec.onnx`, `latin_dict.txt`.

- [ ] **Step 1: Přidat ignorované adresáře**

Do `.gitignore` přidat na konec:

```
models/
testdata/
```

- [ ] **Step 2: Napsat skript**

Vytvořit `scripts/fetch_models.sh`:

```bash
#!/usr/bin/env bash
# Stáhne OCR modely (PP-OCRv5) z HuggingFace a připraví slovník znaků.
# ModelScope, odkud RapidOCR stahuje ve výchozím stavu, je na této síti
# blokovaný resolverem DNS4EU Protective — proto oficiální PaddlePaddle repa na HF.
set -euo pipefail

MODELS_DIR="${OCR_MODELS_DIR:-$(cd "$(dirname "$0")/.." && pwd)/models}"
HF="https://huggingface.co"

DET_URL="$HF/PaddlePaddle/PP-OCRv5_mobile_det_onnx/resolve/main/inference.onnx"
DET_SHA="a431985659dc921974177a95adcfbb90fd9e51989a5e04d70d0b75f597b6e61d"
REC_URL="$HF/PaddlePaddle/latin_PP-OCRv5_mobile_rec_onnx/resolve/main/inference.onnx"
REC_SHA="7888113072263cb471b93f66dd5e2ad70548dc526fa1ace760d0d973dd121498"
YML_URL="$HF/PaddlePaddle/latin_PP-OCRv5_mobile_rec_onnx/resolve/main/inference.yml"
YML_SHA="0bbe984570f597af3638e50bdf2e8276f3ab26a61966096538b3b0d1849f5c84"

mkdir -p "$MODELS_DIR"

fetch() {
    local url="$1" dest="$2" want="$3"
    if [ -f "$dest" ] && [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$want" ]; then
        echo "ok (cached): $(basename "$dest")"
        return
    fi
    echo "stahuji: $(basename "$dest")"
    curl -fsSL -o "$dest.tmp" "$url"
    local got
    got="$(sha256sum "$dest.tmp" | cut -d' ' -f1)"
    if [ "$got" != "$want" ]; then
        rm -f "$dest.tmp"
        echo "CHYBA: SHA256 nesouhlasi pro $url" >&2
        echo "  ocekavano: $want" >&2
        echo "  ziskano:   $got" >&2
        exit 1
    fi
    mv "$dest.tmp" "$dest"
    echo "ok: $(basename "$dest")"
}

fetch "$DET_URL" "$MODELS_DIR/PP-OCRv5_mobile_det.onnx" "$DET_SHA"
fetch "$REC_URL" "$MODELS_DIR/latin_PP-OCRv5_mobile_rec.onnx" "$REC_SHA"
fetch "$YML_URL" "$MODELS_DIR/latin_rec_inference.yml" "$YML_SHA"

python3 - "$MODELS_DIR" <<'PY'
import sys, pathlib, yaml

models_dir = pathlib.Path(sys.argv[1])
cfg = yaml.safe_load((models_dir / "latin_rec_inference.yml").read_text(encoding="utf-8"))
chars = cfg["PostProcess"]["character_dict"]
out = models_dir / "latin_dict.txt"
out.write_text("\n".join(chars) + "\n", encoding="utf-8")
print(f"slovnik: {len(chars)} znaku -> {out}")
missing = [c for c in "ěščřžýáíéůúňťď" if c not in chars]
if missing:
    raise SystemExit(f"CHYBA: ve slovniku chybi ceska diakritika: {missing}")
print("ceska diakritika ve slovniku OK")
PY
```

- [ ] **Step 3: Spustit a ověřit**

```bash
chmod +x scripts/fetch_models.sh && ./scripts/fetch_models.sh
```

Očekávaný výstup: tři řádky `ok:`, pak `slovnik: <N> znaku -> .../latin_dict.txt`
a `ceska diakritika ve slovniku OK`.

Pokud `python3` nemá `yaml`, doinstalovat `pip install PyYAML` (v dev venvu z Tasku 2 už je).

- [ ] **Step 4: Ověřit, že se cache chová správně**

Spustit skript znovu. Očekávané: tři řádky `ok (cached):` a žádné stahování.

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_models.sh .gitignore
git commit -m "Add script to fetch PP-OCRv5 ONNX models from HuggingFace"
```

---

### Task 2: Dev prostředí na boxu

**Files:**
- Create: `scripts/sync-box.sh`, `scripts/setup_dev_box.sh`

**Interfaces:**
- Consumes: `scripts/fetch_models.sh` z Tasku 1.
- Produces: `~/dev/image-embeddings/venv` na boxu s importovatelnými `rapidocr`, `onnxruntime` (s CUDA), `torch`, `open_clip`, `insightface`, `pytest`. Skript `scripts/sync-box.sh` použitelný ve všech dalších taskách.

- [ ] **Step 1: Napsat sync skript**

Vytvořit `scripts/sync-box.sh`:

```bash
#!/usr/bin/env bash
# Synchronizuje repo z Pi na box. Git zustava na Pi, box nema pristup na GitHub.
set -euo pipefail

REMOTE="${BOX_HOST:-box}"
DEST="${BOX_DEST:-/home/panbotka/dev/image-embeddings}"

ssh "$REMOTE" "mkdir -p '$DEST'"
rsync -a --delete \
    --exclude 'venv/' \
    --exclude '.git/' \
    --exclude 'models/' \
    --exclude 'testdata/' \
    --exclude '__pycache__/' \
    --exclude '.pytest_cache/' \
    ./ "$REMOTE:$DEST/"
echo "synchronizovano -> $REMOTE:$DEST"
```

- [ ] **Step 2: Napsat setup skript**

Vytvořit `scripts/setup_dev_box.sh` (spouští se **na boxu**):

```bash
#!/usr/bin/env bash
# Postavi dev venv pro image-embeddings na boxu (GPU).
set -euo pipefail

cd "$(dirname "$0")/.."
python3 -m venv venv
./venv/bin/pip install --upgrade pip

# PyTorch s CUDA (stejne jako produkcni venv)
./venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# ONNX Runtime s CUDA. InsightFace si CPUExecutionProvider vyzaduje explicitne,
# takze se jeho chovani nemeni.
./venv/bin/pip install onnxruntime-gpu

./venv/bin/pip install open_clip_torch fastapi uvicorn python-multipart \
    "numpy<2" "opencv-python-headless<4.10" insightface pytest httpx PyYAML

# rapidocr zavisi na opencv_python, ktere by kolidovalo s nainstalovanym
# opencv-python-headless (stejny cv2 namespace) -> instalace bez zavislosti.
./venv/bin/pip install --no-deps rapidocr
./venv/bin/pip install pyclipper "Shapely>=1.7.1" "omegaconf!=2.2.1" colorlog six tqdm requests Pillow

./scripts/fetch_models.sh
echo "hotovo"
```

- [ ] **Step 3: Spustit sync a setup**

```bash
chmod +x scripts/sync-box.sh scripts/setup_dev_box.sh
./scripts/sync-box.sh
ssh box 'cd ~/dev/image-embeddings && ./scripts/setup_dev_box.sh'
```

- [ ] **Step 4: Ověřit prostředí**

```bash
ssh box 'cd ~/dev/image-embeddings && ./venv/bin/python -c "
import onnxruntime, cv2, numpy
print(\"providers:\", onnxruntime.get_available_providers())
print(\"cv2:\", cv2.__version__)
print(\"numpy:\", numpy.__version__)
from rapidocr import RapidOCR
print(\"rapidocr import OK\")
from insightface.app import FaceAnalysis
print(\"insightface import OK\")
"'
```

Očekávané: `providers:` obsahuje `CUDAExecutionProvider`, `numpy` verze 1.x,
oba importy OK. Kdyby `CUDAExecutionProvider` chyběl, zkontrolovat `nvidia-smi`
a verzi `onnxruntime-gpu` proti CUDA 12.8.

- [ ] **Step 5: Nakopírovat testovací fotky**

```bash
ssh box 'mkdir -p ~/dev/image-embeddings/testdata/photos'
rsync -a /tmp/claude-1000/-home-pi-projects-image-embeddings/2573699d-020a-47d9-9ee0-666d960ae2e8/scratchpad/photos/ box:~/dev/image-embeddings/testdata/photos/
ssh box 'ls -la ~/dev/image-embeddings/testdata/photos | head'
```

Očekávané: 7 souborů `ph*.jpg`.

- [ ] **Step 6: Commit**

```bash
git add scripts/sync-box.sh scripts/setup_dev_box.sh
git commit -m "Add box dev environment and sync scripts"
```

---

### Task 3: Čisté funkce pro tvar výstupu

**Files:**
- Create: `ocr.py`
- Test: `tests/test_ocr_helpers.py`

**Interfaces:**
- Produces:
  - `polygon_to_bbox(polygon) -> list[float]` — ze 4bodového polygonu (`[[x,y],…]`) vrací `[x_min, y_min, x_max, y_max]`.
  - `sort_reading_order(blocks) -> list[dict]` — bloky (`{"text","bbox","confidence"}`) seřazené ve čtecím pořadí.
  - `build_result(polygons, texts, scores, min_confidence) -> dict` — hotové tělo odpovědi bez klíče `min_confidence`.
  - Konstanty `MODEL_LABEL = "PP-OCRv5_mobile"`, `LANG = "latin"`.

- [ ] **Step 1: Napsat padající testy**

Vytvořit `tests/test_ocr_helpers.py`:

```python
import numpy as np
import pytest

from ocr import build_result, polygon_to_bbox, sort_reading_order


def test_polygon_to_bbox_takes_extremes_of_rotated_quad():
    polygon = [[10.0, 20.0], [110.0, 15.0], [112.0, 45.0], [12.0, 50.0]]
    assert polygon_to_bbox(polygon) == [10.0, 15.0, 112.0, 50.0]


def test_polygon_to_bbox_accepts_numpy_array():
    polygon = np.array([[0, 0], [4, 1], [4, 6], [0, 5]], dtype=np.float32)
    assert polygon_to_bbox(polygon) == [0.0, 0.0, 4.0, 6.0]


def test_sort_reading_order_orders_rows_top_down_and_within_row_left_right():
    blocks = [
        {"text": "vpravo", "bbox": [500.0, 10.0, 700.0, 40.0], "confidence": 0.9},
        {"text": "dole", "bbox": [10.0, 200.0, 200.0, 230.0], "confidence": 0.9},
        {"text": "vlevo", "bbox": [10.0, 12.0, 200.0, 42.0], "confidence": 0.9},
    ]
    assert [b["text"] for b in sort_reading_order(blocks)] == ["vlevo", "vpravo", "dole"]


def test_sort_reading_order_tolerates_slightly_tilted_line():
    # Stejny radek, ale kazdy blok o kus niz - nesmi se rozpadnout na tri radky.
    blocks = [
        {"text": "c", "bbox": [400.0, 24.0, 500.0, 54.0], "confidence": 0.9},
        {"text": "a", "bbox": [10.0, 10.0, 110.0, 40.0], "confidence": 0.9},
        {"text": "b", "bbox": [200.0, 17.0, 300.0, 47.0], "confidence": 0.9},
    ]
    assert [b["text"] for b in sort_reading_order(blocks)] == ["a", "b", "c"]


def test_sort_reading_order_handles_empty_input():
    assert sort_reading_order([]) == []


def test_build_result_filters_and_joins():
    polygons = [
        [[10.0, 10.0], [200.0, 10.0], [200.0, 40.0], [10.0, 40.0]],
        [[10.0, 100.0], [200.0, 100.0], [200.0, 130.0], [10.0, 130.0]],
    ]
    result = build_result(polygons, ["HOSPODA", "sum"], [0.97, 0.20], min_confidence=0.5)

    assert result["text"] == "HOSPODA"
    assert result["blocks_count"] == 1
    assert result["blocks"] == [
        {"text": "HOSPODA", "bbox": [10.0, 10.0, 200.0, 40.0], "confidence": 0.97}
    ]
    assert result["lang"] == "latin"
    assert result["model"] == "PP-OCRv5_mobile"


def test_build_result_on_empty_detection():
    result = build_result(None, None, None, min_confidence=0.5)
    assert result["text"] == ""
    assert result["blocks"] == []
    assert result["blocks_count"] == 0


def test_build_result_drops_blank_texts():
    polygons = [[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]]
    result = build_result(polygons, ["   "], [0.99], min_confidence=0.5)
    assert result["text"] == ""
    assert result["blocks"] == []


@pytest.mark.parametrize("min_confidence", [0.0, 0.5, 1.0])
def test_build_result_threshold_applies_to_text_and_blocks_alike(min_confidence):
    polygons = [[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]]
    result = build_result(polygons, ["A"], [0.5], min_confidence=min_confidence)
    assert (result["text"] == "A") == (len(result["blocks"]) == 1)
```

- [ ] **Step 2: Spustit testy, ověřit, že padají**

```bash
./scripts/sync-box.sh && ssh box 'cd ~/dev/image-embeddings && ./venv/bin/python -m pytest tests/test_ocr_helpers.py -v'
```

Očekávané: FAIL s `ModuleNotFoundError: No module named 'ocr'`.

- [ ] **Step 3: Napsat minimální implementaci**

Vytvořit `ocr.py`:

```python
"""OCR nad fotkami: RapidOCR (PP-OCRv5, latinka) pres ONNX Runtime."""

from typing import Any, Dict, List, Optional, Sequence

MODEL_LABEL = "PP-OCRv5_mobile"
LANG = "latin"


def polygon_to_bbox(polygon: Sequence[Sequence[float]]) -> List[float]:
    """Prevede ctyrbodovy polygon na osove zarovnany bbox [x_min, y_min, x_max, y_max]."""
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def sort_reading_order(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Seradi bloky ve ctecim poradi: radky shora dolu, uvnitr radku zleva doprava.

    Bloky se do radku seskupuji podle stredu na ose y s toleranci poloviny medianove
    vysky bloku. Bez te tolerance by mirne nakloneny napis skoncil jako nekolik radku.
    """
    if not blocks:
        return []

    heights = sorted(block["bbox"][3] - block["bbox"][1] for block in blocks)
    median_height = heights[len(heights) // 2]
    tolerance = max(median_height / 2.0, 1.0)

    def center_y(block: Dict[str, Any]) -> float:
        return (block["bbox"][1] + block["bbox"][3]) / 2.0

    rows: List[List[Dict[str, Any]]] = []
    row_anchors: List[float] = []
    for block in sorted(blocks, key=center_y):
        if rows and abs(center_y(block) - row_anchors[-1]) <= tolerance:
            rows[-1].append(block)
        else:
            rows.append([block])
            row_anchors.append(center_y(block))

    ordered: List[Dict[str, Any]] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda block: block["bbox"][0]))
    return ordered


def build_result(
    polygons: Optional[Sequence[Sequence[Sequence[float]]]],
    texts: Optional[Sequence[str]],
    scores: Optional[Sequence[float]],
    min_confidence: float,
) -> Dict[str, Any]:
    """Slozi telo odpovedi z vystupu RapidOCR.

    Prazdny vysledek (fotka bez textu) je normalni stav, ne chyba.
    """
    if not polygons or not texts or scores is None:
        blocks: List[Dict[str, Any]] = []
    else:
        blocks = [
            {
                "text": text.strip(),
                "bbox": polygon_to_bbox(polygon),
                "confidence": float(score),
            }
            for polygon, text, score in zip(polygons, texts, scores)
            if text.strip() and float(score) >= min_confidence
        ]
        blocks = sort_reading_order(blocks)

    return {
        "text": "\n".join(block["text"] for block in blocks),
        "blocks_count": len(blocks),
        "blocks": blocks,
        "lang": LANG,
        "model": MODEL_LABEL,
    }
```

- [ ] **Step 4: Spustit testy, ověřit, že prochází**

```bash
./scripts/sync-box.sh && ssh box 'cd ~/dev/image-embeddings && ./venv/bin/python -m pytest tests/test_ocr_helpers.py -v'
```

Očekávané: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add ocr.py tests/test_ocr_helpers.py
git commit -m "Add OCR result shaping helpers"
```

---

### Task 4: Napojení RapidOCR

**Files:**
- Modify: `ocr.py`
- Test: `tests/test_ocr_engine.py`

**Interfaces:**
- Consumes: `build_result` z Tasku 3, modely z Tasku 1.
- Produces:
  - `get_engine() -> RapidOCR` — líný singleton.
  - `load_engine() -> None` — načte engine dopředu (volá `server.py` při startu).
  - `extract_text(image, min_confidence=None) -> dict` — `image` je `PIL.Image.Image`, vrací tělo odpovědi včetně `min_confidence`.
  - `engine_info() -> dict` — `{"provider": "CUDAExecutionProvider"|"CPUExecutionProvider", "model": ..., "lang": ...}` pro `/health`.
  - `DEFAULT_MIN_CONFIDENCE: float`.

- [ ] **Step 1: Napsat padající testy**

Vytvořit `tests/test_ocr_engine.py`:

```python
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

import ocr

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    pytest.skip("DejaVu font neni k dispozici")


def render_text(text: str, size=(1200, 300), font_size=90) -> Image.Image:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 90), text, fill="black", font=_font(font_size))
    return image


def test_reads_czech_text_with_diacritics():
    result = ocr.extract_text(render_text("PŘÍJEZD DO VESELICE 1978"))

    normalized = result["text"].upper().replace(" ", "")
    assert "PŘÍJEZD" in normalized
    assert "VESELICE" in normalized
    assert "1978" in normalized


def test_blank_image_returns_empty_result_not_error():
    result = ocr.extract_text(Image.new("RGB", (800, 600), "white"))

    assert result["text"] == ""
    assert result["blocks"] == []
    assert result["blocks_count"] == 0


def test_bboxes_stay_within_original_dimensions_for_large_image():
    # Delsi strana > max_side_len (2000), takze RapidOCR obrazek zmensi.
    # Bboxy se musi vratit v souradnicich puvodniho obrazku.
    width, height = 4000, 1000
    image = render_text("VESELICE", size=(width, height), font_size=200)

    result = ocr.extract_text(image)

    assert result["blocks"], "na velkem obrazku se nenasel zadny text"
    for block in result["blocks"]:
        x_min, y_min, x_max, y_max = block["bbox"]
        assert 0 <= x_min < x_max <= width
        assert 0 <= y_min < y_max <= height


def test_min_confidence_filters_everything_when_set_to_one():
    result = ocr.extract_text(render_text("VESELICE"), min_confidence=1.0)

    assert result["text"] == ""
    assert result["blocks"] == []
    assert result["min_confidence"] == 1.0


def test_extract_text_reports_default_min_confidence():
    result = ocr.extract_text(render_text("VESELICE"))
    assert result["min_confidence"] == ocr.DEFAULT_MIN_CONFIDENCE


def test_engine_info_reports_provider_and_model():
    info = ocr.engine_info()
    assert info["provider"] in {"CUDAExecutionProvider", "CPUExecutionProvider"}
    assert info["model"] == "PP-OCRv5_mobile"
    assert info["lang"] == "latin"


def test_grayscale_image_is_accepted():
    image = render_text("VESELICE").convert("L")
    result = ocr.extract_text(image)
    assert "VESELICE" in result["text"].upper()
```

- [ ] **Step 2: Spustit testy, ověřit, že padají**

```bash
./scripts/sync-box.sh && ssh box 'cd ~/dev/image-embeddings && ./venv/bin/python -m pytest tests/test_ocr_engine.py -v'
```

Očekávané: FAIL s `AttributeError: module 'ocr' has no attribute 'extract_text'`.

- [ ] **Step 3: Doplnit implementaci**

Do `ocr.py` přidat nahoru k importům:

```python
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image
```

a pod stávající konstanty:

```python
logger = logging.getLogger(__name__)

MODELS_DIR = Path(os.environ.get("OCR_MODELS_DIR", Path(__file__).parent / "models"))
DEFAULT_MIN_CONFIDENCE = float(os.environ.get("OCR_MIN_CONFIDENCE", "0.5"))
USE_CUDA_SETTING = os.environ.get("OCR_USE_CUDA", "auto").strip().lower()

_engine = None
_provider = None


def _resolve_provider() -> bool:
    """Rozhodne, jestli se pojede na CUDA. Vraci True pro CUDA, False pro CPU."""
    import onnxruntime

    cuda_available = "CUDAExecutionProvider" in onnxruntime.get_available_providers()

    if USE_CUDA_SETTING in {"1", "true", "yes"}:
        if not cuda_available:
            raise RuntimeError(
                "OCR_USE_CUDA=1, ale CUDAExecutionProvider neni k dispozici. "
                "Zkontroluj instalaci onnxruntime-gpu."
            )
        return True
    if USE_CUDA_SETTING in {"0", "false", "no"}:
        return False

    if not cuda_available:
        logger.warning("CUDAExecutionProvider neni k dispozici, OCR pojede na CPU.")
    return cuda_available


def load_engine():
    """Nacte RapidOCR nad lokalnimi modely. Chybejici model ma shodit start sluzby."""
    global _engine, _provider

    if _engine is not None:
        return _engine

    from rapidocr import RapidOCR

    det_path = MODELS_DIR / "PP-OCRv5_mobile_det.onnx"
    rec_path = MODELS_DIR / "latin_PP-OCRv5_mobile_rec.onnx"
    dict_path = MODELS_DIR / "latin_dict.txt"
    for path in (det_path, rec_path, dict_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Chybi OCR model: {path}. Spust ./scripts/fetch_models.sh"
            )

    use_cuda = _resolve_provider()
    _provider = "CUDAExecutionProvider" if use_cuda else "CPUExecutionProvider"

    _engine = RapidOCR(
        params={
            "Det.model_path": str(det_path),
            "Rec.model_path": str(rec_path),
            "Rec.rec_keys_path": str(dict_path),
            "Rec.lang_type": "latin",
            "Rec.ocr_version": "PP-OCRv5",
            "Rec.model_type": "mobile",
            "Global.text_score": 0.05,
            "EngineConfig.onnxruntime.use_cuda": use_cuda,
        }
    )
    logger.info("OCR engine nacten (%s)", _provider)
    return _engine


def get_engine():
    """Vrati nacteny engine, pripadne ho lene nacte."""
    return load_engine()


def engine_info() -> Dict[str, Any]:
    """Informace o OCR pro /health."""
    if _provider is None:
        load_engine()
    return {"provider": _provider, "model": MODEL_LABEL, "lang": LANG}


def extract_text(image: "Image.Image", min_confidence: Optional[float] = None) -> Dict[str, Any]:
    """Precte text z obrazku a vrati telo odpovedi endpointu /ocr/image."""
    threshold = DEFAULT_MIN_CONFIDENCE if min_confidence is None else float(min_confidence)

    array = np.array(image.convert("RGB"))
    output = get_engine()(array, text_score=threshold)

    result = build_result(
        getattr(output, "boxes", None),
        getattr(output, "txts", None),
        getattr(output, "scores", None),
        threshold,
    )
    result["min_confidence"] = threshold
    return result
```

- [ ] **Step 4: Spustit testy, ověřit, že prochází**

```bash
./scripts/sync-box.sh && ssh box 'cd ~/dev/image-embeddings && ./venv/bin/python -m pytest tests/test_ocr_engine.py -v'
```

Očekávané: 7 passed. První běh trvá déle (načítání modelů).

Kdyby `test_reads_czech_text_with_diacritics` selhal na chybějící diakritice, ověřit,
že se načetl latinkový slovník: `wc -l models/latin_dict.txt` a `grep -c "ř" models/latin_dict.txt`.

- [ ] **Step 5: Commit**

```bash
git add ocr.py tests/test_ocr_engine.py
git commit -m "Wire RapidOCR engine with local PP-OCRv5 latin models"
```

---

### Task 5: HTTP endpoint

**Files:**
- Modify: `server.py`
- Test: `tests/test_api_ocr.py`

**Interfaces:**
- Consumes: `ocr.extract_text`, `ocr.engine_info`, `ocr.load_engine`, `ocr.DEFAULT_MIN_CONFIDENCE` z Tasku 4.
- Produces: `POST /ocr/image`; `/health` rozšířený o klíč `ocr`.

- [ ] **Step 1: Napsat padající testy**

Vytvořit `tests/test_api_ocr.py`:

```python
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from server import app

client = TestClient(app)

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    pytest.skip("DejaVu font neni k dispozici")


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def text_png(text: str) -> bytes:
    image = Image.new("RGB", (1200, 300), "white")
    ImageDraw.Draw(image).text((40, 90), text, fill="black", font=_font(90))
    return png_bytes(image)


def test_rejects_non_image_content_type():
    response = client.post(
        "/ocr/image", files={"file": ("data.txt", b"nejsem obrazek", "text/plain")}
    )
    assert response.status_code == 400


def test_rejects_undecodable_file():
    response = client.post(
        "/ocr/image", files={"file": ("broken.png", b"\x89PNG rozbite", "image/png")}
    )
    assert response.status_code == 400


def test_blank_photo_returns_200_with_empty_result():
    payload = png_bytes(Image.new("RGB", (800, 600), "white"))
    response = client.post("/ocr/image", files={"file": ("blank.png", payload, "image/png")})

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == ""
    assert body["blocks"] == []
    assert body["blocks_count"] == 0


def test_reads_text_and_returns_full_contract():
    response = client.post(
        "/ocr/image", files={"file": ("napis.png", text_png("VESELICE 1978"), "image/png")}
    )

    assert response.status_code == 200
    body = response.json()
    assert "VESELICE" in body["text"].upper()
    assert body["blocks_count"] == len(body["blocks"])
    assert body["lang"] == "latin"
    assert body["model"] == "PP-OCRv5_mobile"
    assert body["min_confidence"] == 0.5

    block = body["blocks"][0]
    assert set(block) == {"text", "bbox", "confidence"}
    assert len(block["bbox"]) == 4
    assert 0.0 <= block["confidence"] <= 1.0


def test_min_confidence_is_honoured():
    response = client.post(
        "/ocr/image",
        files={"file": ("napis.png", text_png("VESELICE"), "image/png")},
        data={"min_confidence": "1.0"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["min_confidence"] == 1.0
    assert body["blocks"] == []
    assert body["text"] == ""


def test_health_reports_ocr():
    body = client.get("/health").json()
    assert body["ocr"]["provider"] in {"CUDAExecutionProvider", "CPUExecutionProvider"}
    assert body["ocr"]["model"] == "PP-OCRv5_mobile"
```

- [ ] **Step 2: Spustit testy, ověřit, že padají**

```bash
./scripts/sync-box.sh && ssh box 'cd ~/dev/image-embeddings && ./venv/bin/python -m pytest tests/test_api_ocr.py -v'
```

Očekávané: FAIL — `/ocr/image` vrací 404.

- [ ] **Step 3: Přidat endpoint do `server.py`**

K importům nahoru přidat:

```python
import ocr
```

Za blok načítání InsightFace (řádek s `print("All models loaded. Starting server...")`)
vložit **před** ten print:

```python
print("Loading OCR engine...")
ocr.load_engine()
```

Rozšířit `/health` — do funkce `health()` před `return result` přidat:

```python
    result["ocr"] = ocr.engine_info()
```

Na konec souboru přidat endpoint:

```python
@app.post("/ocr/image", response_model=dict)
async def ocr_image(
    file: UploadFile = File(...),
    min_confidence: float = Form(ocr.DEFAULT_MIN_CONFIDENCE),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploadni prosím image/* soubor.")

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Obrázek se nepodařilo načíst: {exc}")

    return ocr.extract_text(img, min_confidence=min_confidence)
```

A do importu z fastapi přidat `Form`:

```python
from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Form
```

- [ ] **Step 4: Spustit testy, ověřit, že prochází**

```bash
./scripts/sync-box.sh && ssh box 'cd ~/dev/image-embeddings && ./venv/bin/python -m pytest tests/ -v'
```

Očekávané: všechny testy prochází (10 + 7 + 6 = 23 passed).

**Pozor:** `from server import app` načte i CLIP a InsightFace. Při úplně prvním běhu
se stahují váhy ViT-L-14 (~1,7 GB), takže první spuštění trvá řádově minuty.

- [ ] **Step 5: Ověřit doopravdy běžící službu**

```bash
ssh box 'cd ~/dev/image-embeddings && (OCR_USE_CUDA=auto ./venv/bin/uvicorn server:app --host 127.0.0.1 --port 8010 > /tmp/dev8010.log 2>&1 &) ; sleep 90; curl -s http://127.0.0.1:8010/health; echo'
ssh box 'cd ~/dev/image-embeddings && curl -s -X POST http://127.0.0.1:8010/ocr/image -F "file=@testdata/photos/phb6bvub84sshb451vstvbl06c.jpg" | head -c 600; echo'
```

Očekávané: `/health` obsahuje `"ocr"` s providerem; OCR na fotce se SPZ vrátí nějaký text.

Po ověření dev server zase shodit, ať na boxu nedrží VRAM:

```bash
ssh box 'pkill -f "uvicorn server:app --host 127.0.0.1 --port 8010" || true; sleep 2; ss -lnt | grep :8010 || echo "port 8010 uvolnen"'
```

- [ ] **Step 6: Commit**

```bash
git add server.py tests/test_api_ocr.py
git commit -m "Add POST /ocr/image endpoint and report OCR in /health"
```

---

### Task 6: Benchmark na reálných fotkách

**Files:**
- Create: `scripts/bench_ocr.py`

**Interfaces:**
- Consumes: `ocr.extract_text`, fotky v `testdata/photos/`.
- Produces: tabulku časů a přečtených textů na stdout.

- [ ] **Step 1: Napsat skript**

Vytvořit `scripts/bench_ocr.py`:

```python
"""Zmeri propustnost OCR a vypise precteny text, aby sel posoudit rucne.

Spousteni (na boxu):
    OCR_USE_CUDA=1 ./venv/bin/python scripts/bench_ocr.py
    OCR_USE_CUDA=0 ./venv/bin/python scripts/bench_ocr.py
"""

import statistics
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ocr  # noqa: E402

REPEATS = 3
PHOTOS_DIR = Path(__file__).resolve().parent.parent / "testdata" / "photos"
SYNTHETIC_SIZES = [(1200, 900), (2400, 1800), (4000, 3000)]


def synthetic_images():
    from PIL import ImageDraw, ImageFont

    for width, height in SYNTHETIC_SIZES:
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", height // 12
        )
        draw.text((width // 20, height // 3), "VESELICE 1978", fill="black", font=font)
        yield f"syntetic-{width}x{height}", image


def main():
    ocr.load_engine()
    provider = ocr.engine_info()["provider"]
    print(f"provider: {provider}\n")

    items = [(path.name, Image.open(path)) for path in sorted(PHOTOS_DIR.glob("*.jpg"))]
    items.extend(synthetic_images())

    all_times = []
    for name, image in items:
        times = []
        result = None
        for _ in range(REPEATS):
            started = time.perf_counter()
            result = ocr.extract_text(image)
            times.append(time.perf_counter() - started)
        all_times.extend(times)

        preview = result["text"].replace("\n", " | ")[:110]
        print(f"{name:45s} {image.size[0]:5d}x{image.size[1]:<5d} "
              f"median {statistics.median(times):6.3f}s  bloku {result['blocks_count']:3d}")
        print(f"    {preview}")

    print(f"\ncelkem {len(all_times)} behu")
    print(f"median      {statistics.median(all_times):.3f} s/fotka")
    print(f"p95         {sorted(all_times)[int(len(all_times) * 0.95) - 1]:.3f} s/fotka")
    print(f"propustnost {1 / statistics.median(all_times):.2f} fotek/s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Změřit na CUDA**

```bash
./scripts/sync-box.sh && ssh box 'cd ~/dev/image-embeddings && OCR_USE_CUDA=1 ./venv/bin/python scripts/bench_ocr.py 2>&1 | tail -40'
```

- [ ] **Step 3: Změřit na CPU**

```bash
ssh box 'cd ~/dev/image-embeddings && OCR_USE_CUDA=0 ./venv/bin/python scripts/bench_ocr.py 2>&1 | tail -40'
```

- [ ] **Step 4: Posoudit kvalitu a rozhodnout o provideru**

Projít vypsané texty u sedmi reálných fotek a posoudit, jestli dávají smysl.
Porovnat mediány CUDA vs CPU a poznamenat je do `README.md` v Tasku 8.
Do produkce jde ten provider, který je rychlejší; při shodě CPU (méně závislostí).

- [ ] **Step 5: Prověřit PP-OCRv6 (spec: rizika)**

Spec si vyhradil porovnání s novějším `PP-OCRv6`, který má RapidOCR jako výchozí.
Nejdřív zjistit, jestli je vůbec k mání mimo ModelScope:

```bash
curl -s "https://huggingface.co/api/models?search=PP-OCRv6&limit=30" | python3 -c "import json,sys; [print(m['id']) for m in json.load(sys.stdin)]"
```

- Když se objeví ONNX varianta rozpoznávacího modelu pokrývající latinku, stáhnout ji
  stejným způsobem jako v Tasku 1 a projet `bench_ocr.py` s `Rec.model_path` na ni.
  Přepnout jen tehdy, když na sedmi reálných fotkách přečte češtinu prokazatelně lépe.
- Když na HuggingFace není, zapsat do README, že se porovnání neprovedlo, protože jediný
  dostupný zdroj je blokovaný ModelScope. Nedokončené porovnání se nesmí zamlčet.

- [ ] **Step 6: Commit**

```bash
git add scripts/bench_ocr.py
git commit -m "Add OCR benchmark script"
```

---

### Task 7: Nasazení na box a regresní kontrola

**Files:**
- Modify: `deploy/image-embeddings.service`

**Interfaces:**
- Consumes: hotový kód z Tasků 1–6.
- Produces: běžící produkční služba na portu 8000 s `/ocr/image`.

- [ ] **Step 1: Zaznamenat chování před zásahem**

```bash
ssh box 'curl -s http://localhost:8000/health; echo'
ssh box 'cd /tmp && curl -s -X POST http://localhost:8000/embed/image -F "file=@/home/panbotka/dev/image-embeddings/testdata/photos/phih85dpabe50dpapetu27eoiq.jpg" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d[\"dim\"], round(sum(d[\"embedding\"]),6))"'
ssh box 'curl -s -X POST http://localhost:8000/embed/face -F "file=@/home/panbotka/dev/image-embeddings/testdata/photos/phcgj0hre3oolpjp7saeoid4dq.jpg" | python3 -c "import json,sys; d=json.load(sys.stdin); print(\"faces:\", d[\"faces_count\"])"'
```

Výstupy si poznamenat — po nasazení musí být stejné.

- [ ] **Step 2: Přidat proměnné do systemd unitu**

V `deploy/image-embeddings.service` do sekce `[Service]` za `Environment=HOME=/home/box` přidat:

```
Environment=OCR_MODELS_DIR=/opt/image-embeddings/models
Environment=OCR_USE_CUDA=auto
```

- [ ] **Step 3: Nasadit kód a modely**

```bash
./scripts/sync-box.sh
ssh box 'sudo rsync -a --exclude venv/ --exclude .git/ --exclude testdata/ --exclude __pycache__/ /home/panbotka/dev/image-embeddings/ /opt/image-embeddings/ && sudo chown -R box:box /opt/image-embeddings'
ssh box 'sudo -u box /opt/image-embeddings/venv/bin/pip uninstall -y onnxruntime && sudo -u box /opt/image-embeddings/venv/bin/pip install onnxruntime-gpu && sudo -u box /opt/image-embeddings/venv/bin/pip install --no-deps rapidocr && sudo -u box /opt/image-embeddings/venv/bin/pip install pyclipper "Shapely>=1.7.1" "omegaconf!=2.2.1" colorlog six tqdm requests PyYAML'
ssh box 'sudo -u box env OCR_MODELS_DIR=/opt/image-embeddings/models /opt/image-embeddings/scripts/fetch_models.sh'
```

- [ ] **Step 4: Restartovat a ověřit**

```bash
ssh box 'sudo cp /opt/image-embeddings/deploy/image-embeddings.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl restart image-embeddings && sleep 120 && systemctl is-active image-embeddings && curl -s http://localhost:8000/health; echo'
```

Očekávané: `active` a `/health` obsahuje `ocr`.

- [ ] **Step 5: Regresní kontrola všech endpointů**

Zopakovat příkazy z kroku 1 a porovnat výstupy — `dim`, součet embeddingu a `faces_count`
musí být shodné. Navíc:

```bash
ssh box 'curl -s -X POST http://localhost:8000/embed/text -H "Content-Type: application/json" -d "{\"text\":\"a photo of a cat\"}" | python3 -c "import json,sys; print(json.load(sys.stdin)[\"dim\"])"'
ssh box 'curl -s -X POST http://localhost:8000/estimate/era -F "file=@/home/panbotka/dev/image-embeddings/testdata/photos/phih85dpabe50dpapetu27eoiq.jpg" | python3 -c "import json,sys; print(json.load(sys.stdin)[\"era\"])"'
ssh box 'curl -s -X POST http://localhost:8000/ocr/image -F "file=@/home/panbotka/dev/image-embeddings/testdata/photos/phb6bvub84sshb451vstvbl06c.jpg" | head -c 400; echo'
```

Kdyby cokoli z původních endpointů selhalo, vrátit `onnxruntime` zpět
(`pip uninstall onnxruntime-gpu && pip install onnxruntime`) a restartovat.

- [ ] **Step 6: Commit**

```bash
git add deploy/image-embeddings.service
git commit -m "Configure OCR environment in systemd unit"
```

---

### Task 8: Dockerfile a dokumentace

**Files:**
- Modify: `Dockerfile`, `README.md`

- [ ] **Step 1: Doplnit Dockerfile**

V `Dockerfile` za blok `RUN pip install --no-cache-dir open_clip_torch ... insightface` přidat:

```dockerfile
# RapidOCR zavisi na opencv_python, ktere by kolidovalo s opencv-python-headless
# (stejny cv2 namespace) -> instalace bez zavislosti a rucne doinstalovany zbytek.
RUN pip install --no-cache-dir --no-deps rapidocr && \
    pip install --no-cache-dir pyclipper "Shapely>=1.7.1" "omegaconf!=2.2.1" \
    colorlog six tqdm requests PyYAML
```

a za `COPY server.py .` (před `EXPOSE`) přidat:

```dockerfile
COPY ocr.py .
COPY scripts/fetch_models.sh scripts/
RUN ./scripts/fetch_models.sh
ENV OCR_MODELS_DIR=/app/models
ENV OCR_USE_CUDA=0
```

- [ ] **Step 2: Ověřit build**

```bash
./scripts/sync-box.sh && ssh box 'cd ~/dev/image-embeddings && docker build -t emb-ocr-test . 2>&1 | tail -20'
```

Očekávané: build projde, `fetch_models.sh` v něm vypíše `ceska diakritika ve slovniku OK`.

- [ ] **Step 3: Doplnit README**

Do sekce `## Features` přidat řádek:

```markdown
- **OCR** - text z fotek přes PP-OCRv5 (latinka včetně češtiny), bounding boxy a confidence
```

Do `## API Endpoints` přidat:

````markdown
### OCR
```
POST /ocr/image
Content-Type: multipart/form-data
Body: file=<image>, min_confidence=<float, default 0.5>
```
````

Do `## Examples` přidat:

```bash
# OCR
curl -X POST http://localhost:8000/ocr/image -F "file=@photo.jpg"
```

A novou sekci před `## License`:

````markdown
## Development

Vývoj a testy běží na stroji s GPU (box), ne na Raspberry Pi.

```bash
./scripts/sync-box.sh                                    # nahraje repo na box
ssh box 'cd ~/dev/image-embeddings && ./scripts/setup_dev_box.sh'
ssh box 'cd ~/dev/image-embeddings && ./venv/bin/python -m pytest tests/ -v'
```

OCR modely (PP-OCRv5 detekce + latinkové rozpoznávání) se stahují z HuggingFace
skriptem `./scripts/fetch_models.sh`. ModelScope, který RapidOCR používá ve výchozím
stavu, je na této síti blokovaný resolverem DNS4EU Protective.

Proměnné prostředí: `OCR_MODELS_DIR` (kde leží modely), `OCR_USE_CUDA`
(`auto` / `1` / `0`), `OCR_MIN_CONFIDENCE` (výchozí práh).
````

- [ ] **Step 4: Doplnit naměřené hodnoty**

Do sekce `## Development` v README přidat výsledky benchmarku ve tvaru níž, kde `<X>` a `<Y>`
jsou mediány vypsané skriptem `bench_ocr.py` v Tasku 6 kroky 2 a 3 (řádek `median`),
a doplnit výsledek prověření PP-OCRv6 z Tasku 6 kroku 5:

```markdown
Naměřená propustnost (RTX 3070, 7 reálných fotek + syntetické, medián):
CUDA <X> s/fotka, CPU <Y> s/fotka.
```

- [ ] **Step 5: Commit**

```bash
git add Dockerfile README.md
git commit -m "Add OCR to Docker image and document the endpoint"
```

---

## Poznámky k testům

- **Proč syntetický obrázek s diakritikou:** hlavní riziko je, že se místo latinkového modelu načte výchozí čínský. Test `test_reads_czech_text_with_diacritics` to chytí okamžitě.
- **Proč test velkého obrázku:** RapidOCR zmenšuje vstup nad `max_side_len: 2000` a bboxy mapuje zpět. Test ověřuje, že se souřadnice opravdu vrací v rozměrech původní fotky.
- **Proč se testy pouští na boxu:** Pi nemá GPU a testy s modely by tam trvaly neúnosně dlouho.
- Reálné fotky se nepoužívají v `pytest` — nemají referenční přepis a hodnotí se ručně v benchmarku.
