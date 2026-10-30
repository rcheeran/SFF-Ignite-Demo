#!/usr/bin/env python3
"""Jaw-angle calibration: how the commanded wrist yaw maps to the jaw axis on the table.

HELPERS ONLY. The measurement procedure that used to live here inferred the jaw angle from the
angle a PLACED CUBE was left at, and it is deleted rather than deprecated: it depends on the cube
never rotating in the jaws, which is exactly the thing wrist alignment exists to fix. Two runs of
it disagreed on the SIGN of the gain (-1 then +1) because it was measuring slip, not jaws.

`auto_data.jaw_direct` measures the jaw axis directly with an empty gripper (gain +1, offset
+2.6 deg, spread 2.4 deg) and writes the same jaw_yaw.json this module reads.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "jaw_yaw.json")


def load(path=None):
    """(gain, offset_deg) from the saved measurement, or None if it has not been run."""
    path = path or OUT_JSON
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    return d["gain"], d["offset_deg"]


def yaw_for_angle(cube_angle_deg, cal=None):
    """Wrist yaw that lines the jaws up with two opposite faces of a cube lying at this angle
    (both angles in the ROBOT frame, folded to +-45). Returns 0 when uncalibrated."""
    cal = cal or load()
    if cal is None:
        return 0.0
    gain, offset = cal
    return ((gain * (cube_angle_deg - offset) + 45.0) % 90.0) - 45.0


def fit_yaw(obs):
    """(gain, offset, spread) for placed = gain*yaw + offset, with gain constrained to +-1.

    Angles live modulo 90 deg (a square looks the same every quarter turn), so a plain line fit
    through the folded values is meaningless - it read a "gain" of 2.34 on data that is actually
    gain -1. Instead: assume the physics (commanding a yaw turns the tool by exactly that yaw, so
    |gain| = 1), try both signs, and average the residual offsets CIRCULARLY on the 90 deg period.
    """
    cmd = np.array([o[0] for o in obs], float)
    got = np.array([o[1] for o in obs], float)
    best = None
    for gain in (1.0, -1.0):
        d = np.radians((got - gain * cmd) * 4.0)          # 90 deg period -> full turn
        mean = np.arctan2(np.sin(d).mean(), np.cos(d).mean())
        offset = np.degrees(mean) / 4.0
        resid = ((got - (gain * cmd + offset) + 45.0) % 90.0) - 45.0
        spread = float(np.abs(resid).max())
        if best is None or spread < best[2]:
            best = (gain, ((offset + 45.0) % 90.0) - 45.0, spread)
    return best
