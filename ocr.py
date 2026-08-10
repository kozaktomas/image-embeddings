"""OCR nad fotkami: RapidOCR (PP-OCRv5, latinka) pres ONNX Runtime."""

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
    """Zjisti, na cem OCR opravdu bezi.

    RapidOCR pri nefunkcni CUDA jen zaloguje varovani a tise spadne na CPU, takze
    "chtel jsem CUDA" a "bezim na CUDA" nejsou totez. /health musi hlasit realitu.
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
    """Rozhodne, jestli se pojede na CUDA. Vraci True pro CUDA, False pro CPU."""
    import onnxruntime

    # ONNX Runtime si CUDA/cuDNN knihovny natahne z nvidia-* balicku, ktere do venvu
    # prinesl torch. Bez tohohle je CUDAExecutionProvider sice "available", ale
    # nenacte se a inference tise spadne na CPU.
    if hasattr(onnxruntime, "preload_dlls"):
        onnxruntime.preload_dlls()

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
    from rapidocr.utils.typings import LangRec, ModelType, OCRVersion

    det_path = MODELS_DIR / "PP-OCRv5_mobile_det.onnx"
    rec_path = MODELS_DIR / "latin_PP-OCRv5_mobile_rec.onnx"
    dict_path = MODELS_DIR / "latin_dict.txt"
    for path in (det_path, rec_path, dict_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Chybi OCR model: {path}. Spust ./scripts/fetch_models.sh"
            )

    use_cuda = _resolve_provider()

    _engine = RapidOCR(
        params={
            "Det.model_path": str(det_path),
            "Rec.model_path": str(rec_path),
            "Rec.rec_keys_path": str(dict_path),
            # RapidOCR tyhle tri validuje jako Enum, ne jako retezec.
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
            "CUDA byla vyzadana, ale RapidOCR bezi na CPU. Nejcastejsi pricina: "
            "onnxruntime-gpu postavene na jine verzi CUDA, nez je na stroji "
            "(1.28 vyzaduje CUDA 13, tenhle stroj ma 12.8 -> pouzij 1.22)."
        )
        if USE_CUDA_SETTING in {"1", "true", "yes"}:
            raise RuntimeError(message)
        logger.warning(message)

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


def extract_text(
    image: "Image.Image", min_confidence: Optional[float] = None
) -> Dict[str, Any]:
    """Precte text z obrazku a vrati telo odpovedi endpointu /ocr/image."""
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
