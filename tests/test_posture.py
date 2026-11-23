import numpy as np
import pytest

from posture import fall, movement
from posture.keypoints import (L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST,
                               R_ANKLE, R_HIP, R_KNEE, R_SHOULDER)
from posture.synthetic import hide, rotate, standing_pose
from posture.visibility import missing_joints, visibility_score


# ---------- fall ----------
def test_upright_is_steady():
    k, c = standing_pose()
    assert fall.classify_posture(k, c).status == fall.STEADY


@pytest.mark.parametrize("deg,expected", [(0, "steady"), (30, "steady"), (44, "steady"),
                                          (50, "fallen"), (90, "fallen"), (180, "fallen")])
def test_tilt_boundary_is_45_degrees(deg, expected):
    k, c = standing_pose()
    assert fall.classify_posture(rotate(k, deg), c).status == expected


def test_tilt_symmetric_left_and_right():
    k, c = standing_pose()
    assert fall.classify_posture(rotate(k, 70), c).status == fall.classify_posture(rotate(k, -70), c).status


def test_hidden_hips_gives_unknown_not_fallen():
    k, c = standing_pose()
    p = fall.classify_posture(k, hide(c, [L_HIP, R_HIP]))
    assert p.status == fall.UNKNOWN


def test_knees_hidden_falls_back_to_hips():
    k, c = standing_pose()
    p = fall.classify_posture(rotate(k, 90), hide(c, [L_KNEE, R_KNEE]))
    assert p.status == fall.FALLEN and p.slope is not None


def test_sitting_with_raised_knees_is_not_inverted():
    k, c = standing_pose()
    k = k.copy()
    k[L_KNEE][1] = k[R_KNEE][1] = 70   # knees up near hip height, still below shoulders
    k[L_ANKLE][1] = k[R_ANKLE][1] = 120
    assert not fall.is_inverted(k, c)


def test_inverted_detected_when_legs_above_shoulders():
    k, c = standing_pose()
    assert fall.is_inverted(rotate(k, 180), c)


def test_low_confidence_everywhere_is_unknown():
    k, _ = standing_pose()
    assert fall.classify_posture(k, np.zeros(17)).status == fall.UNKNOWN


def test_threshold_is_configurable():
    k, c = standing_pose()
    k60 = rotate(k, 60)
    assert fall.classify_posture(k60, c).status == fall.FALLEN
    assert fall.classify_posture(k60, c, slope_threshold=0.5).status == fall.STEADY


# ---------- movement ----------
def test_joint_angle_right_angle_and_straight():
    assert movement.joint_angle((0, 1), (0, 0), (1, 0)) == pytest.approx(90, abs=0.1)
    assert movement.joint_angle((0, 1), (0, 0), (0, -1)) == pytest.approx(180, abs=0.1)


def _runner():
    k, c = standing_pose()
    k = k.copy()
    # sharply bend left elbow and left knee
    k[L_WRIST] = k[L_ELBOW] + np.array([20.0, -10.0])      # forearm folded up
    k[L_ANKLE] = k[L_KNEE] + np.array([30.0, -20.0])       # shin folded back
    return k, c


def test_bent_arm_and_leg_is_running():
    k, c = _runner()
    assert movement.classify_movement(k, c)[0] == movement.RUNNING


def test_standing_is_non_running():
    k, c = standing_pose()
    assert movement.classify_movement(k, c)[0] == movement.NON_RUNNING


def test_one_leg_hidden_gives_unknown():
    k, c = standing_pose()
    label, _ = movement.classify_movement(k, hide(c, [L_ANKLE]))
    assert label == movement.UNKNOWN


# ---------- visibility ----------
def test_visibility_score_and_missing_names():
    _, c = standing_pose()
    assert visibility_score(c) == 1.0 and missing_joints(c) == []
    c2 = hide(c, [L_KNEE, R_KNEE, L_ANKLE])
    assert visibility_score(c2) == pytest.approx(9 / 12)
    assert set(missing_joints(c2)) == {"left_knee", "right_knee", "left_ankle"}
