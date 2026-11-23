"""Fall detection from a single frame of 2D pose keypoints.

Rule: draw a line from the shoulder midpoint to the knee midpoint (hips if the
knees are hidden). A person who is upright has a steep line (|slope| >= 1,
i.e. more than 45 degrees from horizontal); a person lying down has a shallow
one. A person with the legs above the torso is flagged as inverted.

Coordinates are image coordinates: x to the right, y DOWN.
"""
from dataclasses import dataclass

import numpy as np

from .keypoints import (L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, R_ANKLE, R_HIP,
                        R_KNEE, R_SHOULDER, visible)

CONF = 0.7
SLOPE_THRESHOLD = 1.0

STEADY, FALLEN, UNKNOWN = "steady", "fallen", "unknown"


@dataclass(frozen=True)
class Posture:
    status: str            # steady | fallen | unknown
    slope: float | None    # torso-line slope (inf when exactly vertical), None if not computed
    reason: str


def is_inverted(kpts, conf, threshold=CONF):
    """Both knees (or both ankles) are higher in the image than the shoulders.

    The first version compared against the lowest torso joint (hips included),
    which also fired for a person sitting with raised knees. Using the
    shoulders as the reference keeps sitting steady.
    """
    shoulders = [j for j in (L_SHOULDER, R_SHOULDER) if conf[j] > threshold]
    if not shoulders:
        return False
    ref_y = np.mean([kpts[j][1] for j in shoulders])
    for pair in ((L_KNEE, R_KNEE), (L_ANKLE, R_ANKLE)):
        if visible(conf, pair, threshold) and all(kpts[j][1] < ref_y for j in pair):
            return True
    return False


def classify_posture(kpts, conf, threshold=CONF, slope_threshold=SLOPE_THRESHOLD):
    """Classify one person as steady, fallen or unknown."""
    kpts = np.asarray(kpts, dtype=float)
    conf = np.asarray(conf, dtype=float)

    if is_inverted(kpts, conf, threshold):
        return Posture(FALLEN, None, "inverted: legs above shoulders")

    if not (visible(conf, (L_SHOULDER, R_SHOULDER), threshold)
            and visible(conf, (L_HIP, R_HIP), threshold)):
        return Posture(UNKNOWN, None, "shoulders or hips not visible")

    top = (kpts[L_SHOULDER] + kpts[R_SHOULDER]) / 2
    if visible(conf, (L_KNEE, R_KNEE), threshold):
        bottom = (kpts[L_KNEE] + kpts[R_KNEE]) / 2
    else:
        bottom = (kpts[L_HIP] + kpts[R_HIP]) / 2

    dx, dy = bottom[0] - top[0], bottom[1] - top[1]
    if abs(dx) <= 1:
        return Posture(STEADY, float("inf"), "torso line vertical")
    slope = dy / dx
    if abs(slope) < slope_threshold:
        return Posture(FALLEN, float(slope), f"torso line shallow (|slope|={abs(slope):.2f})")
    return Posture(STEADY, float(slope), f"torso line steep (|slope|={abs(slope):.2f})")
