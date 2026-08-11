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
        {"text": "right", "bbox": [500.0, 10.0, 700.0, 40.0], "confidence": 0.9},
        {"text": "bottom", "bbox": [10.0, 200.0, 200.0, 230.0], "confidence": 0.9},
        {"text": "left", "bbox": [10.0, 12.0, 200.0, 42.0], "confidence": 0.9},
    ]
    assert [b["text"] for b in sort_reading_order(blocks)] == ["left", "right", "bottom"]


def test_sort_reading_order_tolerates_slightly_tilted_line():
    # Same line, each block a little lower - must not split into three rows.
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
    result = build_result(polygons, ["HOSPODA", "noise"], [0.97, 0.20], min_confidence=0.5)

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
