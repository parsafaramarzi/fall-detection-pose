"""Sensitivity analysis of the rules on a synthetic skeleton -> results/.

There is no labelled video in this repo, so rather than claim an accuracy this
script measures how the rules behave as a perfect skeleton is tilted and as
joints are hidden. It shows where the fall rule switches and how it reacts to hidden joints.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from posture import fall
from posture.keypoints import L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, R_ANKLE, R_HIP, R_KNEE, R_SHOULDER
from posture.synthetic import hide, rotate, standing_pose

OUT = Path("results")
OUT.mkdir(exist_ok=True)


def tilt_sweep():
    k, c = standing_pose()
    rows = []
    for deg in range(0, 181, 5):
        p = fall.classify_posture(rotate(k, deg), c)
        rows.append({"tilt_deg": deg, "status": p.status, "slope": p.slope, "reason": p.reason})
    return pd.DataFrame(rows)


def occlusion_cases():
    k, c = standing_pose()
    k90 = rotate(k, 90)
    cases = {
        "upright, all joints visible": (k, c),
        "lying (90 deg), all visible": (k90, c),
        "lying, knees hidden": (k90, hide(c, [L_KNEE, R_KNEE, L_ANKLE, R_ANKLE])),
        "lying, one shoulder hidden": (k90, hide(c, [L_SHOULDER])),
        "lying, both hips hidden": (k90, hide(c, [L_HIP, R_HIP])),
        "upright, both hips hidden": (k, hide(c, [L_HIP, R_HIP])),
        "upside down (180 deg)": (rotate(k, 180), c),
    }
    rows = []
    for n, (kk, cc) in cases.items():
        p = fall.classify_posture(kk, cc)
        rows.append({"case": n, "status": p.status, "reason": p.reason})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    sweep = tilt_sweep()
    sweep.to_csv(OUT / "fall_vs_tilt.csv", index=False)
    occ = occlusion_cases()
    occ.to_csv(OUT / "fall_occlusion_cases.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 3))
    colors = sweep.status.map({"steady": "tab:green", "fallen": "tab:red", "unknown": "tab:grey"})
    ax.scatter(sweep.tilt_deg, [1] * len(sweep), c=colors, s=120, marker="s")
    ax.axvline(45, color="k", ls="--")
    ax.set_yticks([])
    ax.set_xlabel("body tilt from vertical (degrees)")
    ax.set_title("Fall rule on a perfect skeleton (green steady, red fallen)")
    fig.tight_layout()
    fig.savefig(OUT / "fall_vs_tilt.png", dpi=130)

    print(sweep[["tilt_deg", "status"]].T.to_string(header=False))
    print(occ.to_string(index=False))
