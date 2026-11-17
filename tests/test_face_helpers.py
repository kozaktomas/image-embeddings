"""Unit tests for the face attribute helpers, on synthetic inputs (no models)."""

import numpy as np
import pytest

from face_attributes import (
    LEFT_EYE_106,
    RIGHT_EYE_106,
    eye_aspect_ratio,
    eye_openness,
    face_size,
)


def eye(width: float, lid_gap: float, x0: float = 0.0, y0: float = 0.0) -> np.ndarray:
    """Six eye points p1..p6 (68-point order) for an eye of given width and lid gap."""
    half = lid_gap / 2.0
    return np.array(
        [
            [x0, y0],  # p1 corner
            [x0 + width / 3, y0 - half],  # p2 upper lid
            [x0 + 2 * width / 3, y0 - half],  # p3 upper lid
            [x0 + width, y0],  # p4 corner
            [x0 + 2 * width / 3, y0 + half],  # p5 lower lid
            [x0 + width / 3, y0 + half],  # p6 lower lid
        ]
    )


def test_eye_aspect_ratio_is_gap_over_width():
    assert eye_aspect_ratio(eye(30.0, 9.0)) == pytest.approx(0.3)


def test_eye_aspect_ratio_does_not_depend_on_face_size():
    """A ratio, so a face twice as large reads the same."""
    assert eye_aspect_ratio(eye(60.0, 18.0)) == pytest.approx(eye_aspect_ratio(eye(30.0, 9.0)))


def test_closed_eye_reads_lower_than_open_eye():
    assert eye_aspect_ratio(eye(30.0, 1.0)) < eye_aspect_ratio(eye(30.0, 9.0))


def test_eye_without_width_is_none_not_a_division_by_zero():
    assert eye_aspect_ratio(np.zeros((6, 2))) is None


def test_eye_openness_reads_each_eye_from_its_own_indexes():
    landmarks = np.zeros((106, 2))
    landmarks[list(LEFT_EYE_106)] = eye(30.0, 3.0, x0=200.0, y0=100.0)
    landmarks[list(RIGHT_EYE_106)] = eye(30.0, 9.0, x0=100.0, y0=100.0)

    eyes = eye_openness(landmarks)

    assert eyes["left"] == pytest.approx(0.1)
    assert eyes["right"] == pytest.approx(0.3)


def test_eye_indexes_do_not_overlap():
    assert len(set(LEFT_EYE_106) | set(RIGHT_EYE_106)) == 12
    assert all(0 <= i < 106 for i in LEFT_EYE_106 + RIGHT_EYE_106)


def test_eye_openness_without_landmarks_is_none():
    assert eye_openness(None) is None


def test_face_size_is_clamped_to_the_image():
    """A face cut off by the frame must not report more than the whole image."""
    size = face_size([-50.0, -50.0, 150.0, 150.0], 100, 100)

    assert size == {"width": 100.0, "height": 100.0, "rel_area": 1.0}


def test_face_size_relative_area():
    size = face_size([10.0, 20.0, 60.0, 120.0], 200, 400)

    assert size["width"] == 50.0
    assert size["height"] == 100.0
    assert size["rel_area"] == pytest.approx(5000 / 80000)


class TestSharpness:
    """Needs OpenCV, which the text-only image does not ship."""

    @pytest.fixture(autouse=True)
    def _cv2(self):
        self.cv2 = pytest.importorskip("cv2")
        from face_attributes import sharpness

        self.sharpness = sharpness

    @staticmethod
    def checkerboard(size: int, square: int) -> np.ndarray:
        ys, xs = np.mgrid[0:size, 0:size]
        board = (((xs // square) + (ys // square)) % 2 * 255).astype(np.uint8)
        return np.stack([board] * 3, axis=-1)

    def test_flat_crop_has_zero_sharpness(self):
        image = np.full((200, 200, 3), 128, dtype=np.uint8)

        assert self.sharpness(image, [10, 10, 190, 190]) == 0.0

    def test_blur_lowers_sharpness(self):
        image = self.checkerboard(400, 20)
        blurred = self.cv2.GaussianBlur(image, (0, 0), 4)
        bbox = [50.0, 50.0, 350.0, 350.0]

        assert self.sharpness(blurred, bbox) < self.sharpness(image, bbox)

    def test_comparable_across_face_sizes(self):
        """The same pattern at two scales lands close, thanks to the fixed 112x112 crop."""
        small = self.checkerboard(224, 28)
        large = self.checkerboard(896, 112)

        a = self.sharpness(small, [0, 0, 224, 224])
        b = self.sharpness(large, [0, 0, 896, 896])

        assert a == pytest.approx(b, rel=0.25)

    def test_bbox_outside_the_image_is_clamped(self):
        image = self.checkerboard(100, 10)

        assert self.sharpness(image, [-40.0, -40.0, 140.0, 140.0]) == pytest.approx(
            self.sharpness(image, [0, 0, 100, 100])
        )

    def test_empty_crop_is_none(self):
        image = self.checkerboard(100, 10)

        assert self.sharpness(image, [150.0, 150.0, 200.0, 200.0]) is None
