"""Unit tests for the MiVOLO face-to-person association and crops (no models).

Boxes are [x1, y1, x2, y2]. A "person" is a tall box; its face sits in the top part.
"""

import numpy as np
import pytest

from age_gender import (
    MIN_PERSON_SIZE,
    assign_faces_to_persons,
    box_iou,
    extra_faces,
    face_crop,
    person_crop,
)

# The matching itself is scipy's Hungarian solver, which the text-only image does not ship.
scipy = pytest.importorskip("scipy")


def person(x: float, width: float = 200, top: float = 0, height: float = 600):
    return [x, top, x + width, top + height]


def face_of(p, size: float = 60, dx: float = 0.0):
    """A face centred near the top of person box `p`, shifted by `dx`."""
    cx = (p[0] + p[2]) / 2 + dx
    return [cx - size / 2, p[1] + 20, cx + size / 2, p[1] + 20 + size]


class TestBoxIou:
    def test_identical_boxes(self):
        assert box_iou([[0, 0, 10, 10]], [[0, 0, 10, 10]])[0, 0] == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        assert box_iou([[0, 0, 10, 10]], [[20, 20, 30, 30]])[0, 0] == 0.0

    def test_over_second_rewards_a_face_inside_a_person(self):
        """MiVOLO's cost: mean of IoU and the share of the face inside the person."""
        cost = box_iou([[0, 0, 100, 100]], [[10, 10, 20, 20]], over_second=True)[0, 0]
        assert cost == pytest.approx((1.0 + 0.01) / 2)

    def test_degenerate_box_is_zero_not_nan(self):
        assert box_iou([[0, 0, 10, 10]], [[5, 5, 5, 5]], over_second=True)[0, 0] == 0.0


class TestAssignFacesToPersons:
    def test_single_person(self):
        p = person(100)
        assert assign_faces_to_persons([face_of(p)], [p]) == [0]

    def test_face_without_body(self):
        p = person(100)
        far_face = [600, 20, 660, 80]
        assert assign_faces_to_persons([far_face], [p]) == [None]

    def test_no_persons_at_all(self):
        assert assign_faces_to_persons([[0, 0, 10, 10]], []) == [None]

    def test_no_faces(self):
        assert assign_faces_to_persons([], [person(0)]) == []

    def test_two_faces_two_persons(self):
        """Side by side with overlapping boxes, each face gets its own body, in any order."""
        left, right = person(100), person(250)  # overlap by 50 px
        faces = [face_of(right), face_of(left)]
        assert assign_faces_to_persons(faces, [left, right]) == [1, 0]

    def test_face_only_touching_a_box_is_not_matched(self):
        """MiVOLO would take any overlap; under half the face inside is not a match."""
        p = person(100)
        edge_face = [280, 20, 340, 80]  # 20 of 60 px inside
        assert assign_faces_to_persons([edge_face], [p]) == [None]

    def test_ambiguous_overlap_runs_face_only(self):
        """A lone face fully inside two overlapping person boxes could be either body."""
        a, b = person(100), person(130)
        f = [190, 20, 240, 70]  # inside both
        assert assign_faces_to_persons([f], [a, b]) == [None]

    def test_one_person_box_for_two_faces_goes_to_neither(self):
        """Two faces fully inside one box (a child in someone's arms): ambiguous."""
        p = person(100, width=300)
        adult = [200, 20, 260, 80]
        child = [180, 200, 230, 250]
        assert assign_faces_to_persons([adult, child], [p]) == [None, None]

    def test_one_person_box_is_never_assigned_twice(self):
        """However the faces sit, a person box ends up with at most one face."""
        p = person(100, width=300)
        rng = np.random.default_rng(0)
        for _ in range(200):
            faces = []
            for _ in range(3):
                x, y = rng.uniform(50, 400), rng.uniform(0, 500)
                s = rng.uniform(20, 80)
                faces.append([x, y, x + s, y + s])
            result = assign_faces_to_persons(faces, [p, person(350)])
            taken = [r for r in result if r is not None]
            assert len(taken) == len(set(taken))

    def test_clear_winner_keeps_the_body(self):
        """A second face mostly outside the box does not make the first one ambiguous."""
        p = person(100)
        own = face_of(p)
        neighbour = [280, 20, 340, 80]  # 20 of 60 px inside: not a candidate
        assert assign_faces_to_persons([own, neighbour], [p]) == [0, None]

    def test_face_buffalo_missed_still_claims_its_body(self):
        """A detector-only face takes its own person box, which resolves the ambiguity."""
        a, b = person(100), person(130)
        f = [190, 20, 240, 70]  # inside both a and b
        other = [305, 300, 328, 323]  # inside b only (lower down, e.g. a child in front)
        assert assign_faces_to_persons([f], [a, b]) == [None]
        assert assign_faces_to_persons([f], [a, b], other_faces=[other]) == [0]

    def test_free_person_far_away_is_no_rival(self):
        p, elsewhere = person(100), person(900)
        assert assign_faces_to_persons([face_of(p)], [p, elsewhere]) == [0]


class TestExtraFaces:
    def test_duplicate_of_a_buffalo_face_is_dropped(self):
        own = [[100, 100, 160, 170]]
        detector = [[98, 95, 162, 168], [400, 100, 460, 170]]
        assert extra_faces(own, detector) == [1]

    def test_no_buffalo_faces_keeps_all(self):
        assert extra_faces([], [[0, 0, 10, 10], [20, 20, 30, 30]]) == [0, 1]

    def test_no_detector_faces(self):
        assert extra_faces([[0, 0, 10, 10]], []) == []

    def test_small_box_inside_a_large_one_is_a_duplicate(self):
        """Detectors disagree on tightness; a box inside the other is the same face."""
        own = [[100, 100, 200, 220]]
        assert extra_faces(own, [[120, 130, 180, 200]]) == []


def image(h: int = 400, w: int = 400, value: int = 200) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


class TestCrops:
    def test_face_crop_is_the_box(self):
        img = image()
        img[50:100, 60:120] = 10
        crop = face_crop(img, [60.7, 50.2, 120.9, 100.5])
        assert crop.shape == (50, 60, 3)
        assert (crop == 10).all()

    def test_face_crop_is_clamped_and_empty_is_none(self):
        img = image()
        assert face_crop(img, [-30, -30, 20, 20]).shape == (20, 20, 3)
        assert face_crop(img, [500, 500, 600, 600]) is None

    def test_person_crop_blacks_out_faces(self):
        img = image(500, 500)
        crop = person_crop(img, [0, 0, 200, 400], [], [[50, 20, 110, 80]])
        assert crop.shape == (400, 200, 3)
        assert (crop[20:80, 50:110] == 0).all()
        assert (crop[100:, :] == 200).all()

    def test_person_crop_blacks_out_other_persons_to_the_edge(self):
        """An overlapping person near the crop edge is cut to that edge (CROP_ROUND_RATE)."""
        img = image(400, 600)
        crop = person_crop(img, [0, 0, 300, 400], [[250, 10, 550, 400]], [])
        # MiVOLO clamps boxes to the last pixel, so the 400 px tall image gives 399 rows
        assert crop.shape == (399, 300, 3)
        assert (crop[:, 250:] == 0).all()  # y rounded to 0 and to the bottom
        assert (crop[:, :250] == 200).all()

    def test_small_person_is_none(self):
        img = image()
        assert person_crop(img, [0, 0, MIN_PERSON_SIZE - 1, 300], [], []) is None

    def test_mostly_covered_person_is_none(self):
        img = image(400, 600)
        others = [[0, 0, 140, 400], [160, 0, 300, 400]]
        assert person_crop(img, [0, 0, 300, 400], others, []) is None
