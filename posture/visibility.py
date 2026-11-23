"""Which body joints did the pose model fail to see?

This replaces the earlier "amputee / disabled person" detector. A low-confidence
joint almost always means occlusion, a crop at the frame edge or a bad frame,
not a missing limb, so labelling people from it is not valid. What the signal
*is* good for is a data-quality flag: how much of this person can the other
rules (fall, movement) trust?
"""
from .keypoints import BODY_JOINTS, NAMES

CONF = 0.7


def missing_joints(conf, threshold=CONF):
    """Names of body joints (shoulders down) below the confidence threshold."""
    return [NAMES[j] for j in BODY_JOINTS if conf[j] <= threshold]


def visibility_score(conf, threshold=CONF):
    """Fraction of the 12 body joints that are visible, in [0, 1]."""
    return 1 - len(missing_joints(conf, threshold)) / len(BODY_JOINTS)
