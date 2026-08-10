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
    image = Image.new("RGB", (2000, 300), "white")
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
    response = client.post(
        "/ocr/image", files={"file": ("blank.png", payload, "image/png")}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == ""
    assert body["blocks"] == []
    assert body["blocks_count"] == 0


def test_reads_text_and_returns_full_contract():
    response = client.post(
        "/ocr/image",
        files={"file": ("napis.png", text_png("VESELICE 1978"), "image/png")},
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
