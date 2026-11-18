"""Age and gender per face from MiVOLO v2, which looks at the face and the body.

InsightFace's genderage sees a 96x96 face crop and nothing else; its age is poor on
children, the elderly and anything not sharp. MiVOLO v2 (Kuprashevich & Tolstykh,
arXiv:2307.04616) reads the face crop and the crop of the person it belongs to together,
which is where most of its accuracy comes from (age MAE 3.65 years on Lagenda with both
inputs). This module adds its estimate next to InsightFace's, it does not replace it.

Pipeline, following MiVOLO's own (mivolo/predictor.py, mivolo/structures.py):

1. A YOLOv8x detector trained on two classes, face and person, finds the person boxes.
   Faces still come only from buffalo_l: the detector's face boxes are only used to help
   assign buffalo_l faces to persons (a face buffalo_l missed still claims its body).
2. Faces are assigned to persons with MiVOLO's assign_faces(): Hungarian matching on the
   mean of IoU and the share of the face inside the person box, one person per face at
   most (see assign_faces_to_persons for the two safety rules added on top).
3. Crops are cut the way MiVOLO cuts them: the face box as is; the person box with every
   other face and person box blacked out, dropped when too small or too covered.
4. One MiVOLO forward pass for all faces of the image. A face without a usable body runs
   with an empty body input, as in MiVOLO, and says so in `body_used`.

Licences differ from the rest of the service: the MiVOLO v2 weights are CC-BY-NC-4.0
(non-commercial), the detector weights and ultralytics are AGPL-3.0. See the README.

Like ocr.py and face_attributes.py this is imported unconditionally by server.py, so the
module level needs nothing but numpy: torch, timm, OpenCV, scipy and ultralytics are
imported where they are used, and text mode never gets there.
"""

import contextlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# HuggingFace sources, pinned to commits; scripts/fetch_mivolo.sh downloads exactly these
# (and checks their SHA-256) at image build time.
MIVOLO_REPO = "iitolstykh/mivolo_v2"
MIVOLO_REVISION = "e8bdd8aec21ae468cbfe73b41f52550105ad8fbf"
DETECTOR_REPO = "iitolstykh/YOLO-Face-Person-Detector"
DETECTOR_REVISION = "b3d071aedccd46a3b2d4b40609da6880a815f395"

# File names inside MIVOLO_MODELS_DIR.
MIVOLO_WEIGHTS = "mivolo_v2.safetensors"
MIVOLO_CONFIG = "mivolo_v2_config.json"
DETECTOR_WEIGHTS = "yolov8x_person_face.pt"

MIVOLO_MODEL_NAME = "mivolo_v2 (mivolo_d1_384)"
DETECTOR_MODEL_NAME = "yolov8x_person_face"

# Detector thresholds of MiVOLO's own Detector class (mivolo/model/yolo_detector.py).
DETECTOR_CONF = 0.4
DETECTOR_IOU = 0.7

# MiVOLO's assign_faces() accepts any pair whose cost is above this, i.e. any overlap.
ASSIGN_MIN_COST = 0.0001
# Added on top of MiVOLO: a face counts as belonging to a person box only when at least
# half of it lies inside. MiVOLO's own threshold accepts a face that merely touches the
# edge of a box, which in a group photo hands a face the body of the person next to it
# whenever its own body was not detected.
MIN_FACE_INSIDE = 0.5
# Added on top of MiVOLO: a match is ambiguous, and the face runs without a body, when
# another candidate fits nearly as well -- a free person box whose cost for this face, or
# an unmatched face whose cost for this person box, reaches this share of the match's own
# cost. Typical case: a face fully inside two overlapping person boxes, or two faces
# fully inside one (a child held in someone's arms). The cost of a face fully inside a
# box is 0.5 + IoU/2, so two such boxes differ only by their small IoU and are within
# a few percent of each other; a face only partly inside the rival box falls well short.
AMBIGUITY_RATIO = 0.9
# A detector face box duplicates a buffalo_l face (rather than being a face buffalo_l
# missed) when this share of the smaller box lies inside the other.
DUPLICATE_FACE_OVERLAP = 0.5

# Person crop rules of MiVOLO's PersonAndFaceResult.crop_object().
MIN_PERSON_SIZE = 50
MIN_PERSON_CROP_AFTERCUT_RATIO = 0.4
CROP_ROUND_RATE = 0.3
CUT_IOU_THRESH = 0.000001


def _as_boxes(boxes: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(boxes, dtype=np.float64)
    return arr.reshape(-1, 4)


def _truncate(boxes: np.ndarray) -> np.ndarray:
    """Whole pixels the way MiVOLO gets them: `.type(torch.int32)`, toward zero."""
    return boxes.astype(np.int32).astype(np.float64)


def box_iou(box1: np.ndarray, box2: np.ndarray, over_second: bool = False) -> np.ndarray:
    """N x M IoU matrix of [x1, y1, x2, y2] boxes; MiVOLO's box_iou() in numpy.

    With `over_second`, the mean of IoU and intersection / area of the second box: how
    much of box2 (a face) lies inside box1 (a person), blended with how well they match.
    That is the cost MiVOLO assigns faces to persons by.
    """
    box1 = _as_boxes(box1)
    box2 = _as_boxes(box2)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    lt = np.maximum(box1[:, None, :2], box2[None, :, :2])
    rb = np.minimum(box1[:, None, 2:], box2[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = inter / (area1[:, None] + area2[None, :] - inter)
        if over_second:
            iou = (inter / area2[None, :] + iou) / 2
    return np.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0)


def _inside_share(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """N x M share of each `inner` box's area that lies inside each `outer` box."""
    outer = _as_boxes(outer)
    inner = _as_boxes(inner)
    area = (inner[:, 2] - inner[:, 0]) * (inner[:, 3] - inner[:, 1])
    lt = np.maximum(outer[:, None, :2], inner[None, :, :2])
    rb = np.minimum(outer[:, None, 2:], inner[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = inter / area[None, :]
    return np.nan_to_num(share, nan=0.0, posinf=0.0, neginf=0.0)


def extra_faces(
    faces: Sequence[Sequence[float]], detector_faces: Sequence[Sequence[float]]
) -> List[int]:
    """Indexes of detector face boxes that are not one of `faces` (buffalo_l's).

    A detector box duplicates a buffalo_l face when DUPLICATE_FACE_OVERLAP of the smaller
    of the two lies inside the other. The rest are faces buffalo_l missed; they compete
    for person boxes but get no estimate of their own.
    """
    det = _as_boxes(detector_faces)
    own = _as_boxes(faces)
    if len(det) == 0:
        return []
    if len(own) == 0:
        return list(range(len(det)))
    # share of the detector box inside a buffalo box, and of the buffalo box inside it
    det_in_own = _inside_share(own, det)  # own x det
    own_in_det = _inside_share(det, own).T  # own x det
    overlap = np.maximum(det_in_own, own_in_det).max(axis=0)
    return [i for i in range(len(det)) if overlap[i] < DUPLICATE_FACE_OVERLAP]


def assign_faces_to_persons(
    faces: Sequence[Sequence[float]],
    persons: Sequence[Sequence[float]],
    other_faces: Sequence[Sequence[float]] = (),
) -> List[Optional[int]]:
    """For each of `faces`, the index of the person box it belongs to, or None.

    MiVOLO's assign_faces(): cost = box_iou(person, face, over_second=True), Hungarian
    matching that maximises the total cost, pairs at or below ASSIGN_MIN_COST dropped.
    `other_faces` (detector faces buffalo_l missed) take part in the matching so that they
    can claim their own body, but are not reported.

    Two rules on top, both of which only ever take a body away, never hand out a wrong one:
    - a pair counts only if MIN_FACE_INSIDE of the face lies inside the person box;
    - an ambiguous match (see AMBIGUITY_RATIO) is dropped, so the face runs face-only.

    A person box is never assigned to two faces: the matching is one-to-one.
    """
    from scipy.optimize import linear_sum_assignment

    own = _as_boxes(faces)
    n_own = len(own)
    all_faces = np.concatenate([own, _as_boxes(other_faces)]) if len(other_faces) else own
    person_boxes = _as_boxes(persons)
    result: List[Optional[int]] = [None] * n_own
    if n_own == 0 or len(person_boxes) == 0:
        return result

    face_boxes = _truncate(all_faces)
    person_boxes = _truncate(person_boxes)
    cost = box_iou(person_boxes, face_boxes, over_second=True)  # persons x faces
    cost[_inside_share(person_boxes, face_boxes) < MIN_FACE_INSIDE] = 0.0

    person_idx, face_idx = linear_sum_assignment(cost, maximize=True)
    match_of_face: Dict[int, int] = {}
    for p, f in zip(person_idx, face_idx):
        if cost[p, f] > ASSIGN_MIN_COST:
            match_of_face[int(f)] = int(p)
    matched_persons = set(match_of_face.values())
    free_persons = [p for p in range(len(person_boxes)) if p not in matched_persons]
    unmatched_faces = [f for f in range(len(face_boxes)) if f not in match_of_face]

    for f in range(n_own):
        p = match_of_face.get(f)
        if p is None:
            continue
        bar = AMBIGUITY_RATIO * cost[p, f]
        rival_person = any(cost[q, f] >= bar for q in free_persons)
        rival_face = any(cost[p, g] >= bar for g in unmatched_faces)
        if not (rival_person or rival_face):
            result[f] = p
    return result


def _clamped_int_box(box: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    """MiVOLO's get_bbox_by_ind(ind, h, w): truncate to int32, clamp to the last pixel."""
    x1, y1, x2, y2 = (int(v) for v in np.asarray(box, dtype=np.float64).astype(np.int32))
    x1 = min(max(x1, 0), width - 1)
    y1 = min(max(y1, 0), height - 1)
    x2 = min(max(x2, 0), width - 1)
    y2 = min(max(y2, 0), height - 1)
    return x1, y1, x2, y2


def face_crop(image_bgr: np.ndarray, box: Sequence[float]) -> Optional[np.ndarray]:
    """The face box cut out as is, as in crop_object(cut_other_classes=[]). None if empty."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = _clamped_int_box(box, w, h)
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop.copy()


def person_crop(
    image_bgr: np.ndarray,
    person: Sequence[float],
    other_persons: Sequence[Sequence[float]],
    faces: Sequence[Sequence[float]],
) -> Optional[np.ndarray]:
    """The person box with every face and every other person blacked out.

    MiVOLO's crop_object(ind, cut_other_classes=["face", "person"]), including that the
    person's own face is blacked out too: the model was trained on body crops like that
    and gets the face through the other input. Other person boxes that reach within
    CROP_ROUND_RATE of the crop edge are blacked out to that edge. None when the crop is
    under MIN_PERSON_SIZE on a side or less than MIN_PERSON_CROP_AFTERCUT_RATIO of it is
    left after the cuts -- such a body would mislead more than help.
    """
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = _clamped_int_box(person, w, h)
    obj = image_bgr[y1:y2, x1:x2].copy()
    crop_h, crop_w = obj.shape[:2]
    if crop_h < MIN_PERSON_SIZE or crop_w < MIN_PERSON_SIZE:
        return None

    others = [("person", b) for b in _as_boxes(other_persons)] + [("face", b) for b in _as_boxes(faces)]
    if others:
        clamped = np.array([_clamped_int_box(b, w, h) for _, b in others], dtype=np.float64)
        ious = box_iou(np.array([[x1, y1, x2, y2]], dtype=np.float64), clamped)[0]
        for (category, box), iou in zip(others, ious):
            if iou < CUT_IOU_THRESH:
                continue
            o_x1, o_y1, o_x2, o_y2 = (int(v) for v in np.asarray(box).astype(np.int32))
            # remap into the crop's coordinates
            o_x1 = max(o_x1 - x1, 0)
            o_y1 = max(o_y1 - y1, 0)
            o_x2 = min(o_x2 - x1, crop_w)
            o_y2 = min(o_y2 - y1, crop_h)
            if category != "face":
                if (o_y1 / crop_h) < CROP_ROUND_RATE:
                    o_y1 = 0
                if ((crop_h - o_y2) / crop_h) < CROP_ROUND_RATE:
                    o_y2 = crop_h
                if (o_x1 / crop_w) < CROP_ROUND_RATE:
                    o_x1 = 0
                if ((crop_w - o_x2) / crop_w) < CROP_ROUND_RATE:
                    o_x2 = crop_w
            obj[o_y1:o_y2, o_x1:o_x2] = 0

    remain_ratio = np.count_nonzero(obj) / obj.size
    if remain_ratio < MIN_PERSON_CROP_AFTERCUT_RATIO:
        return None
    return obj


@contextlib.contextmanager
def _contained_ultralytics():
    """Undo the process-wide settings ultralytics changes behind the caller's back.

    Measured on ultralytics 8.4.215: importing it sets OMP_NUM_THREADS=1 when unset and
    calls cv2.setNumThreads(0); YOLO() sets CUBLAS_WORKSPACE_CONFIG (MiVOLO's own
    yolo_detector.py unsets it for the same reason); the first predict() on CPU resets
    torch's thread count to min(8, cpus - 1), overriding TORCH_NUM_THREADS. Every one of
    them would change how the rest of the service runs, so restore them after each call.
    Requests are handled one at a time on the event loop, so nothing runs in between.
    """
    import cv2
    import torch

    env = {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG")}
    torch_threads = torch.get_num_threads()
    cv2_threads = cv2.getNumThreads()
    try:
        yield
    finally:
        for key, value in env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if torch.get_num_threads() != torch_threads:
            torch.set_num_threads(torch_threads)
        if cv2.getNumThreads() != cv2_threads:
            cv2.setNumThreads(cv2_threads)


class AgeGenderEstimator:
    """The person detector and MiVOLO v2, loaded once, used per request."""

    def __init__(self, models_dir: str, device: str):
        import torch
        from safetensors.torch import load_file

        import mivolo_model

        self.device = device
        # fp16 on CUDA like MiVOLO's own inference; CPU stays fp32 (half is slower there).
        self.half = device == "cuda"
        self.dtype = torch.float16 if self.half else torch.float32

        with open(os.path.join(models_dir, MIVOLO_CONFIG), encoding="utf-8") as fh:
            config = json.load(fh)
        if config.get("model_name") != "mivolo_d1_384" or not config.get("with_persons_model"):
            raise RuntimeError(f"unexpected MiVOLO checkpoint config: {config.get('model_name')}")
        self.input_size = int(config["input_size"])
        self.min_age = float(config["min_age"])
        self.max_age = float(config["max_age"])
        self.avg_age = float(config["avg_age"])
        self.num_classes_gender = int(config["num_classes_gender"])
        # 0 = male, 1 = female: the opposite of genderage's order, hence a lookup and
        # not an index shared with face_attributes.GENDER_LABELS.
        self.gender_labels = {int(k): v for k, v in config["gender_id2label"].items()}

        model = mivolo_model.mivolo_d1_384(num_classes=int(config["num_classes"]), in_chans=int(config["in_chans"]))
        state = load_file(os.path.join(models_dir, MIVOLO_WEIGHTS))
        prefix = "mivolo.model."
        state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        model.load_state_dict(state, strict=True)
        self.model = model.to(device=device, dtype=self.dtype).eval()
        self.param_count = sum(p.numel() for p in self.model.parameters())

        self.detector = self._load_detector(os.path.join(models_dir, DETECTOR_WEIGHTS))

    def _load_detector(self, weights: str):
        os.environ.setdefault("YOLO_OFFLINE", "1")  # no update checks, no network
        os.environ.setdefault("YOLO_VERBOSE", "False")
        with _contained_ultralytics():
            from ultralytics import YOLO

            yolo = YOLO(weights)
            yolo.fuse()
            if self.half:
                yolo.model = yolo.model.half()
        names = {int(k): v for k, v in yolo.model.names.items()}
        self.person_class = next(k for k, v in names.items() if v == "person")
        self.face_class = next(k for k, v in names.items() if v == "face")
        return yolo

    def detect(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Person boxes and face boxes found by the detector, each N x 4 [x1, y1, x2, y2]."""
        with _contained_ultralytics():
            result = self.detector.predict(
                image_bgr,
                conf=DETECTOR_CONF,
                iou=DETECTOR_IOU,
                half=self.half,
                device="cuda:0" if self.device == "cuda" else "cpu",
                verbose=False,
            )[0]
        boxes = result.boxes
        xyxy = boxes.xyxy.float().cpu().numpy().astype(np.float64).reshape(-1, 4)
        cls = boxes.cls.cpu().numpy().astype(int)
        return xyxy[cls == self.person_class], xyxy[cls == self.face_class]

    def estimate(self, image_bgr: np.ndarray, faces: Sequence[Sequence[float]]) -> List[Optional[Dict[str, Any]]]:
        """One `mivolo` object per buffalo_l face box, in the order given.

        `image_bgr` must be BGR (OpenCV order): the detector and MiVOLO's preprocessing
        both expect it. None for a face with no usable crop at all (an empty box).
        """
        import torch

        import mivolo_model

        face_boxes = _as_boxes(faces)
        if len(face_boxes) == 0:
            return []

        persons, detector_faces = self.detect(image_bgr)
        extra = detector_faces[extra_faces(face_boxes, detector_faces)]
        assignment = assign_faces_to_persons(face_boxes, persons, extra)
        cut_faces = np.concatenate([face_boxes, detector_faces]) if len(detector_faces) else face_boxes

        face_crops: List[Optional[np.ndarray]] = []
        body_crops: List[Optional[np.ndarray]] = []
        for i, box in enumerate(face_boxes):
            face_crops.append(face_crop(image_bgr, box))
            p = assignment[i]
            if p is None:
                body_crops.append(None)
            else:
                others = np.delete(persons, p, axis=0)
                body_crops.append(person_crop(image_bgr, persons[p], others, cut_faces))

        runnable = [i for i in range(len(face_boxes)) if face_crops[i] is not None or body_crops[i] is not None]
        results: List[Optional[Dict[str, Any]]] = [None] * len(face_boxes)
        if not runnable:
            return results

        faces_input = mivolo_model.prepare_classification_images([face_crops[i] for i in runnable], self.input_size)
        bodies_input = mivolo_model.prepare_classification_images([body_crops[i] for i in runnable], self.input_size)
        model_input = torch.cat((faces_input, bodies_input), dim=1).to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            output = self.model(model_input).float().cpu()

        gender_probs = output[:, : self.num_classes_gender].softmax(-1)
        prob, index = gender_probs.max(-1)
        raw_age = output[:, self.num_classes_gender]
        ages = raw_age * (self.max_age - self.min_age) + self.avg_age

        for row, i in enumerate(runnable):
            p = assignment[i]
            # Clamped to the range the model was trained on: a regression head can step
            # slightly outside it (a newborn at -0.4), which no caller should see.
            age = min(max(float(ages[row]), self.min_age), self.max_age)
            results[i] = {
                "age": round(age, 2),
                "gender": self.gender_labels[int(index[row])],
                "gender_probability": float(prob[row]),
                "body_used": body_crops[i] is not None,
                "person_bbox": [float(v) for v in persons[p]] if p is not None else None,
            }
        return results
