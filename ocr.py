"""OCR for photos: RapidOCR (PP-OCRv5, Latin script) over ONNX Runtime."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

MODEL_LABEL = "PP-OCRv5_mobile"
LANG = "latin"

logger = logging.getLogger(__name__)

MODELS_DIR = Path(os.environ.get("OCR_MODELS_DIR", Path(__file__).parent / "models"))
DEFAULT_MIN_CONFIDENCE = float(os.environ.get("OCR_MIN_CONFIDENCE", "0.5"))
USE_CUDA_SETTING = os.environ.get("OCR_USE_CUDA", "auto").strip().lower()

_engine = None
_provider = None


def polygon_to_bbox(polygon: Sequence[Sequence[float]]) -> List[float]:
    """Convert a four-point polygon to an axis-aligned [x_min, y_min, x_max, y_max]."""
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def sort_reading_order(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort blocks into reading order: rows top to bottom, left to right within a row.

    Blocks are grouped into rows by their vertical centre, with a tolerance of half the
    median block height. Without that tolerance a slightly tilted sign would break apart
    into several rows.
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
    """Assemble the response body from RapidOCR output.

    An empty result (a photo without text) is a normal outcome, not an error.
    """
    if polygons is None or texts is None or scores is None:
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


def _detect_actual_provider(engine) -> str:
    """Report which execution provider the engine actually ended up using.

    When CUDA does not work, RapidOCR only logs a warning and silently falls back to
    CPU, so "CUDA was requested" and "CUDA is running" are not the same thing.
    /health has to report reality.
    """
    for part in (getattr(engine, "text_rec", None), getattr(engine, "text_det", None)):
        session = getattr(getattr(part, "session", None), "session", None)
        if session is not None and hasattr(session, "get_providers"):
            providers = session.get_providers()
            if "CUDAExecutionProvider" in providers:
                return "CUDAExecutionProvider"
            return "CPUExecutionProvider"
    return "CPUExecutionProvider"


def _resolve_provider() -> bool:
    """Decide whether to run on CUDA. Returns True for CUDA, False for CPU."""
    import onnxruntime

    # ONNX Runtime picks up the CUDA/cuDNN libraries from the nvidia-* packages that
    # torch brought into the venv. Without this, CUDAExecutionProvider reports as
    # "available" but fails to load, and inference silently falls back to CPU.
    if hasattr(onnxruntime, "preload_dlls"):
        onnxruntime.preload_dlls()

    cuda_available = "CUDAExecutionProvider" in onnxruntime.get_available_providers()

    if USE_CUDA_SETTING in {"1", "true", "yes"}:
        if not cuda_available:
            raise RuntimeError(
                "OCR_USE_CUDA=1, but CUDAExecutionProvider is not available. "
                "Check the onnxruntime-gpu installation."
            )
        return True
    if USE_CUDA_SETTING in {"0", "false", "no"}:
        return False

    if not cuda_available:
        logger.warning("CUDAExecutionProvider is not available, OCR will run on CPU.")
    return cuda_available


def load_engine():
    """Load RapidOCR over the local models. A missing model must fail service startup."""
    global _engine, _provider

    if _engine is not None:
        return _engine

    from rapidocr import RapidOCR
    from rapidocr.utils.typings import LangRec, ModelType, OCRVersion

    det_path = MODELS_DIR / "PP-OCRv5_mobile_det.onnx"
    rec_path = MODELS_DIR / "latin_PP-OCRv5_mobile_rec.onnx"
    dict_path = MODELS_DIR / "latin_dict.txt"
    for path in (det_path, rec_path, dict_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing OCR model: {path}. Run ./scripts/fetch_models.sh"
            )

    use_cuda = _resolve_provider()

    _engine = RapidOCR(
        params={
            "Det.model_path": str(det_path),
            "Rec.model_path": str(rec_path),
            "Rec.rec_keys_path": str(dict_path),
            # RapidOCR validates these three as enums, not as strings.
            "Rec.lang_type": LangRec.LATIN,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.model_type": ModelType.MOBILE,
            "Global.text_score": 0.05,
            "EngineConfig.onnxruntime.use_cuda": use_cuda,
        }
    )
    _provider = _detect_actual_provider(_engine)
    if use_cuda and _provider != "CUDAExecutionProvider":
        message = (
            "CUDA was requested but RapidOCR is running on CPU. Most common cause: "
            "onnxruntime-gpu built against a different CUDA version than the machine "
            "provides (1.28 requires CUDA 13; on CUDA 12.x use 1.22)."
        )
        if USE_CUDA_SETTING in {"1", "true", "yes"}:
            raise RuntimeError(message)
        logger.warning(message)

    logger.info("OCR engine loaded (%s)", _provider)
    return _engine


def get_engine():
    """Return the loaded engine, loading it lazily if needed."""
    return load_engine()


def engine_info() -> Dict[str, Any]:
    """OCR information for /health."""
    if _provider is None:
        load_engine()
    return {"provider": _provider, "model": MODEL_LABEL, "lang": LANG}


def extract_text(
    image: "Image.Image", min_confidence: Optional[float] = None
) -> Dict[str, Any]:
    """Read text from an image and return the /ocr/image response body."""
    threshold = (
        DEFAULT_MIN_CONFIDENCE if min_confidence is None else float(min_confidence)
    )

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
