"""/embed/face: the attributes InsightFace computes per face, on a real group photo."""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import server
from server import TEXT_ONLY, app

client = TestClient(app)

# Text-only mode loads no face model and answers /embed/face with 503, which
# tests/test_text_mode.py asserts. Nothing here applies there.
pytestmark = pytest.mark.skipif(
    TEXT_ONLY, reason="EMBED_MODE=text loads no face model"
)

FACE_MODEL = "buffalo_l (ResNet100)"


def sample_bgr():
    """insightface's bundled group photo (t1), as the BGR array cv2 reads."""
    from insightface.data import get_image

    return get_image("t1")


def jpeg(bgr) -> bytes:
    import cv2

    ok, buffer = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return buffer.tobytes()


def embed_faces(payload: bytes, **params) -> dict:
    response = client.post(
        "/embed/face",
        files={"file": ("t1.jpg", payload, "image/jpeg")},
        params=params,
    )
    assert response.status_code == 200
    return response.json()


@pytest.fixture(scope="module")
def image():
    return sample_bgr()


@pytest.fixture(scope="module")
def body(image):
    return embed_faces(jpeg(image))


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def test_finds_the_group(body):
    assert body["faces_count"] == len(body["faces"]) >= 3


def test_model_string_and_version(body):
    """The client re-detects every face when `model` changes; it must not."""
    assert body["model"] == FACE_MODEL
    assert body["attributes_version"] == 1


def test_old_fields_keep_their_shape(body):
    for i, face in enumerate(body["faces"]):
        assert face["face_index"] == i
        assert face["dim"] == 512 == len(face["embedding"])
        assert len(face["bbox"]) == 4
        assert 0.0 < face["det_score"] <= 1.0


def test_old_fields_are_what_face_app_get_returns(body, image):
    """Byte-for-byte compatibility: detection and recognition run exactly as before.

    The endpoint now runs the per-face loop itself (attribute models get BGR), so check it
    against the plain face_app.get() call it replaced, on the same decoded pixels.
    """
    rgb = np.array(Image.open(io.BytesIO(jpeg(image))).convert("RGB"))
    expected = server.face_app.get(rgb)

    assert len(body["faces"]) == len(expected)
    for face, ref in zip(body["faces"], expected):
        assert face["embedding"] == ref.normed_embedding.tolist()
        assert face["bbox"] == ref.bbox.tolist()
        assert face["det_score"] == float(ref.det_score)
        assert face["kps"] == ref.kps.tolist()


def test_gender_sees_the_channel_order_it_was_trained_on(body):
    """t1 is three women and three men. Fed RGB, genderage called all six "female"."""
    genders = [face["gender"] for face in body["faces"]]

    assert genders.count("male") >= 2
    assert genders.count("female") >= 2


def test_yaw_sign_flips_with_a_mirrored_photo(body, image):
    """Pins the documented convention: mirroring the photo must negate yaw."""
    width = image.shape[1]
    mirrored = embed_faces(jpeg(np.ascontiguousarray(image[:, ::-1])))
    turned = max(body["faces"], key=lambda f: abs(f["pose"]["yaw"]))
    x1, y1, x2, y2 = turned["bbox"]
    flipped_bbox = [width - x2, y1, width - x1, y2]
    twin = max(mirrored["faces"], key=lambda f: iou(f["bbox"], flipped_bbox))

    assert iou(twin["bbox"], flipped_bbox) > 0.5
    assert abs(turned["pose"]["yaw"]) > 20
    assert np.sign(twin["pose"]["yaw"]) == -np.sign(turned["pose"]["yaw"])


def test_every_face_has_sane_attributes(body, image):
    height, width = image.shape[:2]
    for face in body["faces"]:
        assert len(face["kps"]) == 5
        for x, y in face["kps"]:
            assert 0 <= x <= width and 0 <= y <= height

        assert isinstance(face["age"], int) and 1 <= face["age"] <= 100
        assert face["gender"] in {"female", "male"}

        assert set(face["pose"]) == {"pitch", "yaw", "roll"}
        assert abs(face["pose"]["yaw"]) <= 90
        assert abs(face["pose"]["pitch"]) <= 90
        assert abs(face["pose"]["roll"]) <= 180

        assert set(face["eyes"]) == {"left", "right"}
        assert face["eyes"]["left"] >= 0 and face["eyes"]["right"] >= 0

        assert face["sharpness"] > 0

        size = face["size"]
        assert size["width"] > 0 and size["height"] > 0
        assert 0 < size["rel_area"] <= 1


def test_landmarks_are_off_by_default(body):
    assert all("landmarks_106" not in face for face in body["faces"])


def test_landmarks_on_request(image):
    body = embed_faces(jpeg(image), landmarks=1)

    assert body["faces"]
    for face in body["faces"]:
        assert len(face["landmarks_106"]) == 106
        assert all(len(point) == 2 for point in face["landmarks_106"])


def test_blur_lowers_sharpness_of_the_same_face(body, image):
    import cv2

    blurred = embed_faces(jpeg(cv2.GaussianBlur(image, (0, 0), 3)))

    pairs = []
    for face in blurred["faces"]:
        best = max(body["faces"], key=lambda f: iou(f["bbox"], face["bbox"]))
        if iou(best["bbox"], face["bbox"]) > 0.5:
            pairs.append((best["sharpness"], face["sharpness"]))

    assert pairs, "no face found again in the blurred copy"
    for sharp, soft in pairs:
        assert soft < sharp


def test_health_reports_the_attributes_version():
    assert client.get("/health").json()["face_attributes_version"] == 1
