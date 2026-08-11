import io
import math

from fastapi.testclient import TestClient
from PIL import Image

from server import EMBED_DIM, app

client = TestClient(app)


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def solid_png(color: str, size: int = 400) -> bytes:
    return png_bytes(Image.new("RGB", (size, size), color))


def embed_image(color: str) -> list:
    response = client.post(
        "/embed/image", files={"file": (f"{color}.png", solid_png(color), "image/png")}
    )
    assert response.status_code == 200
    return response.json()["embedding"]


def embed_text(text: str) -> list:
    response = client.post("/embed/text", json={"text": text})
    assert response.status_code == 200
    return response.json()["embedding"]


def cosine(a: list, b: list) -> float:
    return sum(x * y for x, y in zip(a, b))


def test_health_reports_the_width_the_endpoints_return():
    """A consumer stores these in a fixed-width column, so /health must not lie about it."""
    body = client.get("/health").json()

    assert body["clip"]["dim"] == EMBED_DIM
    assert body["clip"]["precision"] in {"fp16", "fp32"}
    assert len(embed_image("red")) == EMBED_DIM


def test_image_embedding_is_unit_length():
    """Cosine similarity downstream is a plain dot product, which assumes unit vectors."""
    vec = embed_image("red")

    assert math.isclose(math.sqrt(cosine(vec, vec)), 1.0, rel_tol=1e-3)


def test_text_embedding_matches_the_image_width():
    """Image and text must land in one space or nearest-neighbour search is meaningless."""
    vec = embed_text("a photograph of a dog")

    assert len(vec) == EMBED_DIM
    assert math.isclose(math.sqrt(cosine(vec, vec)), 1.0, rel_tol=1e-3)


def test_image_and_text_share_a_space():
    """The whole point of the service: the right caption wins on a dot product.

    Colour is used rather than an object because it survives being drawn as a flat
    synthetic image, which is what a test can generate without shipping photos.
    """
    red = embed_image("red")
    match = embed_text("a solid red colour")
    mismatch = embed_text("a solid blue colour")

    assert cosine(red, match) > cosine(red, mismatch)


def test_rejects_non_image_content_type():
    response = client.post(
        "/embed/image", files={"file": ("data.txt", b"not an image", "text/plain")}
    )

    assert response.status_code == 400


def test_rejects_empty_text():
    response = client.post("/embed/text", json={"text": "   "})

    assert response.status_code == 400
