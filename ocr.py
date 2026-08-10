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
