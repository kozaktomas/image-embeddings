"""Per-face attributes for /embed/face, on top of what InsightFace buffalo_l returns.

Most of what this module hands out InsightFace has already computed: age and gender come
from `genderage`, the head pose from `landmark_3d_68`, the 106 landmarks from
`landmark_2d_106`, the 5 keypoints from detection. Two things are new and computed here:
the eye-openness ratio (from the 106 landmarks) and the sharpness of the face crop. Both
are cheap next to the models that already ran.

Like ocr.py, this is imported unconditionally by server.py, so it must import in the
text-only image, which ships no OpenCV: cv2 is imported inside the function that needs it.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Bump when a field is added, removed or its meaning changes. Reported as
# `attributes_version` by /embed/face and `face_attributes_version` by /health. Kept
# separate from the face `model` string on purpose: the client re-detects faces when that
# string changes, and these attributes do not change the embeddings.
ATTRIBUTES_VERSION = 1

# `face.gender` from genderage is argmax over two logits: 0 = female, 1 = male (that is
# what insightface's own Face.sex maps to "F" / "M").
GENDER_LABELS = {0: "female", 1: "male"}

# Eye contours in the buffalo_l `2d106det` 106-point layout (the layout drawn in
# insightface's alignment/coordinate_reg README). Each eye is 10 points: 8 on the
# contour and 2 near the pupil, which the ratio does not use. The indexes below are the
# six-point eye of the classic 68-point layout picked out of the 106 points, ordered
# like that layout (p1 corner, p2 p3 upper lid, p4 corner, p5 p6 lower lid), via the
# 106 -> 68 mapping used by tools that convert between the two:
#   68-point 36..41 (subject's right eye) = 106-point 35, 41, 42, 39, 37, 36
#   68-point 42..47 (subject's left eye)  = 106-point 89, 95, 96, 93, 91, 90
# The landmark model labels points by facial side, not by image side, so "left" is the
# subject's own left eye -- on the image's right in an ordinary, unmirrored photo.
RIGHT_EYE_106 = (35, 41, 42, 39, 37, 36)
LEFT_EYE_106 = (89, 95, 96, 93, 91, 90)

# Sharpness is measured on the face crop scaled to a fixed size, so a large face and a
# small one of equal focus score alike. 112x112 is the size the recognition model sees.
SHARPNESS_SIZE = 112


def eye_aspect_ratio(points: np.ndarray) -> Optional[float]:
    """Eye openness of one eye: mean vertical lid distance over horizontal eye width.

    `points` are six (x, y) points ordered p1..p6 as in the 68-point layout: p1 and p4 the
    corners, p2/p3 the upper lid, p6/p5 the lower lid below them. This is the eye aspect
    ratio of Soukupova & Cech (2016): (|p2-p6| + |p3-p5|) / (2 |p1-p4|). Roughly 0.25-0.35
    for an open eye and under 0.15 for a closed one, but the absolute level varies by
    person, so it is best compared between photos of the same face. None when the eye has
    no width (degenerate landmarks), rather than a division by zero.
    """
    p = np.asarray(points, dtype=np.float64)
    width = np.linalg.norm(p[0] - p[3])
    if width < 1e-6:
        return None
    vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    return float(vertical / (2.0 * width))


def eye_openness(landmarks_106: Optional[np.ndarray]) -> Optional[Dict[str, Optional[float]]]:
    """Eye aspect ratio of both eyes from the 106 landmarks; None without landmarks."""
    if landmarks_106 is None:
        return None
    lmk = np.asarray(landmarks_106, dtype=np.float64)
    if lmk.shape[0] < 106:
        return None
    return {
        "left": eye_aspect_ratio(lmk[list(LEFT_EYE_106)]),
        "right": eye_aspect_ratio(lmk[list(RIGHT_EYE_106)]),
    }


def clamp_bbox(bbox: Sequence[float], width: int, height: int) -> List[int]:
    """Clamp an [x1, y1, x2, y2] box to the image and round it to whole pixels.

    Detection boxes may reach past the frame for faces at its edge; the crop must not.
    """
    x1 = int(np.clip(np.floor(bbox[0]), 0, width))
    y1 = int(np.clip(np.floor(bbox[1]), 0, height))
    x2 = int(np.clip(np.ceil(bbox[2]), 0, width))
    y2 = int(np.clip(np.ceil(bbox[3]), 0, height))
    return [x1, y1, x2, y2]


def sharpness(image_rgb: np.ndarray, bbox: Sequence[float]) -> Optional[float]:
    """Variance of the Laplacian of the grey face crop, scaled to 112x112.

    Higher is sharper. The crop is the bbox clamped to the image. Scaling to a fixed size
    makes faces of different sizes comparable; downscaling uses INTER_AREA and upscaling
    INTER_LINEAR, because INTER_AREA enlarges like nearest-neighbour and the blocky edges
    would read as detail. None when the clamped crop is empty.
    """
    import cv2

    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(bbox, w, h)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image_rgb[y1:y2, x1:x2]
    grey = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    shrinking = grey.shape[0] * grey.shape[1] >= SHARPNESS_SIZE * SHARPNESS_SIZE
    grey = cv2.resize(
        grey,
        (SHARPNESS_SIZE, SHARPNESS_SIZE),
        interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR,
    )
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def face_size(bbox: Sequence[float], width: int, height: int) -> Dict[str, float]:
    """Face box size in pixels and as a share of the image area.

    Measured on the box clamped to the image (not rounded), so `rel_area` stays within
    (0, 1] even for a face cut off by the frame.
    """
    x1, x2 = (float(np.clip(v, 0, width)) for v in (bbox[0], bbox[2]))
    y1, y2 = (float(np.clip(v, 0, height)) for v in (bbox[1], bbox[3]))
    box_w, box_h = max(x2 - x1, 0.0), max(y2 - y1, 0.0)
    return {
        "width": box_w,
        "height": box_h,
        "rel_area": (box_w * box_h) / float(width * height),
    }


def _points(value: Any) -> Optional[List[List[float]]]:
    if value is None:
        return None
    return [[float(x), float(y)] for x, y in np.asarray(value)[:, :2]]


def describe(face: Any, image_rgb: np.ndarray, include_landmarks: bool = False) -> Dict[str, Any]:
    """The attribute fields of one InsightFace `Face`, ready to merge into the response.

    Every field InsightFace might not have produced (a module disabled or failed) comes
    back as None rather than being left out, so the response shape does not vary.
    """
    h, w = image_rgb.shape[:2]
    bbox = face.bbox

    age = face.get("age")
    gender = face.get("gender")
    pose = face.get("pose")
    landmarks = face.get("landmark_2d_106")

    result: Dict[str, Any] = {
        "kps": _points(face.get("kps")),
        "age": int(age) if age is not None else None,
        "gender": GENDER_LABELS.get(int(gender)) if gender is not None else None,
        # insightface's pose is [pitch, yaw, roll] in degrees, from the 68-point 3D
        # landmarks. Signs, checked on insightface's t1 photo and its mirror image (yaw
        # and roll both flip sign when the photo is mirrored):
        #   yaw < 0  the subject turns to their own right (nose toward the image's left),
        #   yaw > 0  to their own left (nose toward the image's right);
        #   roll > 0 the head tilts clockwise as seen in the image (the eye on the
        #            image's right sits lower), roll < 0 counter-clockwise.
        # Pitch is the least reliable of the three and its sign was not pinned down.
        "pose": (
            {"pitch": float(pose[0]), "yaw": float(pose[1]), "roll": float(pose[2])}
            if pose is not None
            else None
        ),
        "eyes": eye_openness(landmarks),
        "sharpness": sharpness(image_rgb, bbox),
        "size": face_size(bbox, w, h),
    }
    if include_landmarks:
        result["landmarks_106"] = _points(landmarks)
    return result
