"""/embed/face: the MiVOLO v2 age and gender estimate added to every face."""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import face_attributes
import server
from server import TEXT_ONLY, app

client = TestClient(app)

pytestmark = [
    pytest.mark.skipif(TEXT_ONLY, reason="EMBED_MODE=text loads no face model"),
    pytest.mark.skipif(
        not TEXT_ONLY and server.age_gender_model is None, reason="MIVOLO_ENABLED=0"
    ),
]

MIVOLO_KEYS = {"age", "gender", "gender_probability", "body_used", "person_bbox"}


def jpeg(bgr) -> bytes:
    import cv2

    ok, buffer = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return buffer.tobytes()


def group_photo() -> bytes:
    """insightface's t1: six people, three women and three men, standing close."""
    from insightface.data import get_image

    return jpeg(get_image("t1"))


def single_person() -> bytes:
    """scikit-image's astronaut: one person, head and upper body, public domain."""
    from skimage.data import astronaut

    return jpeg(np.ascontiguousarray(astronaut()[:, :, ::-1]))


def embed_faces(payload: bytes, **params) -> dict:
    response = client.post(
        "/embed/face", files={"file": ("photo.jpg", payload, "image/jpeg")}, params=params
    )
    assert response.status_code == 200
    return response.json()


def without_mivolo(payload: bytes, landmarks: bool = False) -> list:
    """The faces /embed/face built before MiVOLO, on the same decoded pixels."""
    rgb = np.array(Image.open(io.BytesIO(payload)).convert("RGB"))
    entries = []
    for i, face in enumerate(server.analyze_faces(rgb)):
        embedding = face.normed_embedding.tolist()
        entry = {
            "face_index": i,
            "dim": len(embedding),
            "embedding": embedding,
            "bbox": face.bbox.tolist(),
            "det_score": float(face.det_score),
        }
        entry.update(face_attributes.describe(face, rgb, include_landmarks=landmarks))
        entries.append(entry)
    return entries


@pytest.fixture(scope="module")
def group():
    return embed_faces(group_photo())


def test_every_face_has_a_well_typed_estimate(group):
    assert group["faces"]
    for face in group["faces"]:
        est = face["mivolo"]
        assert set(est) == MIVOLO_KEYS
        assert isinstance(est["age"], float) and 0.0 <= est["age"] <= 122.0
        # the same labels as the InsightFace `gender` field
        assert est["gender"] in {"female", "male"}
        assert 0.5 <= est["gender_probability"] <= 1.0
        assert isinstance(est["body_used"], bool)
        if est["person_bbox"] is None:
            assert est["body_used"] is False
        else:
            assert len(est["person_bbox"]) == 4
            assert all(isinstance(v, float) for v in est["person_bbox"])


def test_gender_on_the_group(group):
    """Three women and three men; MiVOLO's 0 = male must not be read as genderage's 0."""
    genders = [face["mivolo"]["gender"] for face in group["faces"]]
    assert genders.count("male") >= 2
    assert genders.count("female") >= 2


def test_ages_are_plausible_adults(group):
    for face in group["faces"]:
        assert 15 <= face["mivolo"]["age"] <= 60


def test_a_person_box_goes_to_one_face_only(group):
    boxes = [tuple(f["mivolo"]["person_bbox"]) for f in group["faces"] if f["mivolo"]["person_bbox"]]
    assert len(boxes) == len(set(boxes))


def test_matched_person_box_holds_the_face(group):
    for face in group["faces"]:
        box = face["mivolo"]["person_bbox"]
        if box is None:
            continue
        x1, y1, x2, y2 = face["bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        assert box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def test_single_person_uses_the_body():
    body = embed_faces(single_person())

    assert body["faces_count"] == 1
    est = body["faces"][0]["mivolo"]
    assert est["body_used"] is True
    assert est["person_bbox"] is not None
    assert est["gender"] == "female"
    assert 20 <= est["age"] <= 60


@pytest.mark.parametrize("landmarks", [False, True])
def test_existing_fields_are_identical_to_a_run_without_mivolo(landmarks):
    """Additive only: every field other than `mivolo` is exactly what it was before."""
    payload = group_photo()
    body = embed_faces(payload, landmarks=int(landmarks))
    expected = without_mivolo(payload, landmarks=landmarks)

    assert len(body["faces"]) == len(expected)
    for face, ref in zip(body["faces"], expected):
        assert {k: v for k, v in face.items() if k != "mivolo"} == ref


def test_turned_off_it_reports_null_and_nothing_else_changes(monkeypatch):
    payload = group_photo()
    with_it = embed_faces(payload)
    monkeypatch.setattr(server, "age_gender_model", None)
    without_it = embed_faces(payload)

    assert all(face["mivolo"] is None for face in without_it["faces"])
    for a, b in zip(with_it["faces"], without_it["faces"]):
        a.pop("mivolo")
        b.pop("mivolo")
        assert a == b
    assert client.get("/health").json()["age_gender"] == {"enabled": False}


def test_no_faces_no_detector_call():
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(buffer, format="PNG")
    response = client.post("/embed/face", files={"file": ("red.png", buffer.getvalue(), "image/png")})

    assert response.status_code == 200
    assert response.json()["faces"] == []


def test_a_request_leaves_process_settings_alone():
    """ultralytics changes thread counts and env vars; age_gender must put them back."""
    import os

    import cv2
    import torch

    def state():
        return (
            torch.get_num_threads(),
            cv2.getNumThreads(),
            os.environ.get("OMP_NUM_THREADS"),
            os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        )

    before = state()
    embed_faces(single_person())
    assert state() == before


def test_health_reports_both_models():
    info = client.get("/health").json()["age_gender"]

    assert info["enabled"] is True
    assert info["mivolo"]["loaded"] is True
    assert info["mivolo"]["model"] == "mivolo_v2 (mivolo_d1_384)"
    assert info["person_detector"]["loaded"] is True
    assert info["person_detector"]["model"] == "yolov8x_person_face"
    assert info["mivolo"]["precision"] == ("fp16" if server.DEVICE == "cuda" else "fp32")
