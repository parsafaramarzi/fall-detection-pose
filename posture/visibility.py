"""Which body joints did the pose model fail to see?

A low-confidence joint almost always means occlusion, a crop at the frame edge
or a bad frame, not a missing limb, so it must not be used to label people.
What the signal *is* good for is a data-quality flag: how much of this person
can the fall rule trust?
"""
from .keypoints import BODY_JOINTS, NAMES

CONF = 0.7


def missing_joints(conf, threshold=CONF):
    """Names of body joints (shoulders down) below the confidence threshold."""
    return [NAMES[j] for j in BODY_JOINTS if conf[j] <= threshold]


def visibility_score(conf, threshold=CONF):
    """Fraction of the 12 body joints that are visible, in [0, 1]."""
    return 1 - len(missing_joints(conf, threshold)) / len(BODY_JOINTS)
