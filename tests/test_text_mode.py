"""Text-only mode: what the VPS runs.

Run with EMBED_MODE=text; everything here skips otherwise. The mode is decided when
server.py is imported, so it cannot be flipped inside a running process without
rebuilding the model — hence a separate pytest invocation rather than a fixture.
"""

import io
import math
import os

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import server
from server import EMBED_DIM, app

client = TestClient(app)

pytestmark = pytest.mark.skipif(
    os.environ.get("EMBED_MODE", "full").strip().lower() != "text",
    reason="run this file with EMBED_MODE=text",
)


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_announces_the_mode_and_still_reports_the_width():
    """A consumer must be able to tell a text-only sidecar apart from a broken full one."""
    body = client.get("/health").json()

    assert body["mode"] == "text"
    assert body["clip"]["dim"] == EMBED_DIM
    # engine_info() loads the OCR engine lazily, so reaching it here would raise inside
    # the image that ships no rapidocr. /health must answer, not explode.
    assert body["ocr"] == {"enabled": False}


def test_text_embedding_still_works():
    response = client.post("/embed/text", json={"text": "a photograph of a dog"})

    assert response.status_code == 200
    vec = response.json()["embedding"]
    assert len(vec) == EMBED_DIM
    assert math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, rel_tol=1e-3)


def test_visual_tower_is_released():
    """Dropping it is what makes the service fit in a container limit worth setting."""
    assert server.model.visual is None


@pytest.mark.parametrize(
    "path",
    ["/embed/image", "/embed/face", "/estimate/era", "/ocr/image"],
)
def test_image_endpoints_answer_503_not_404(path):
    """503 says 'up, capability off'; 404 would send the caller hunting a bad URL."""
    response = client.post(
        path, files={"file": ("red.png", png_bytes(), "image/png")}
    )

    assert response.status_code == 503
    assert "EMBED_MODE=text" in response.json()["detail"]
