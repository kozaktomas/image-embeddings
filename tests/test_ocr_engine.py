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
    pytest.skip("DejaVu font is not available")


def render_text(text: str, size=(2000, 300), font_size=90) -> Image.Image:
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
    # Longer side > max_side_len (2000), so RapidOCR downscales the image.
    # Bounding boxes must come back in the original image's coordinates.
    width, height = 4000, 1000
    image = render_text("VESELICE", size=(width, height), font_size=200)

    result = ocr.extract_text(image)

    assert result["blocks"], "no text found in the large image"
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


def test_engine_info_reports_the_provider_actually_in_use():
    # RapidOCR silently falls back to CPU when CUDA fails. /health must not claim otherwise.
    engine = ocr.load_engine()
    providers = engine.text_rec.session.session.get_providers()
    expected = (
        "CUDAExecutionProvider"
        if "CUDAExecutionProvider" in providers
        else "CPUExecutionProvider"
    )
    assert ocr.engine_info()["provider"] == expected


def test_grayscale_image_is_accepted():
    image = render_text("VESELICE").convert("L")
    result = ocr.extract_text(image)
    assert "VESELICE" in result["text"].upper()
