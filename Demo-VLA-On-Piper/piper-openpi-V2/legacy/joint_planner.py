#!/usr/bin/env python3
"""Deterministic joint-space planner - wrist held CAMERA-UP.

Why joint space, and why the wrist is frozen:
  * The firmware IK silently refuses many targets (it simply does not move), and
    it picks its own wrist roll, which flips the wrist-mounted D405 UNDERNEATH
    the gripper - risking the camera and the flange on every descent.
  * Solving orientation per-target also swings J6 (smallest motor, carries the
    gripper + camera) and it thermally trips: 66-77 C under Cartesian control
    versus a steady 37-38 C with the wrist frozen.

So: J1 aims, (J2, J3) set radius and height, and J4/J5/J6 are constants that
keep the camera up. Measured gripper pitch stays within -30.6..-35.3 deg across
the whole map, so grasp geometry is consistent everywhere. A can is a cylinder,
so the gripper's yaw about vertical does not matter.

    planner = JointPlanner()
    j = planner.joints_for(x, y, z)     # 6 joint angles, degrees
"""
import json
import os

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "joint_table_cameraup.json")

# J1 = bearing + offset; the gripper sits off the J1 axis. Measured for THIS
# wrist configuration by measure_j1_offset() below.
J1_OFFSET_DEG = 5.6

# Region actually covered by the sampled table AFTER dropping every sample that
# sat below the table floor. Z_MIN is the floor, not the lowest thing the arm
# can physically do - configurations below this pressed the gripper into the
# table.
R_MIN, R_MAX = 235.0, 365.0
Z_MIN, Z_MAX = 108.0, 300.0


class JointPlanner:
    def __init__(self, table_path=TABLE):
        d = json.load(open(table_path))
        self.j4, self.j5, self.j6 = d["j4"], d["j5"], d["j6"]
        rows = d["rows"]
        pts = np.array([[r["r"], r["z"]] for r in rows], dtype=np.float64)
        vals = np.array([[r["j2"], r["j3"]] for r in rows], dtype=np.float64)
        self._lin = LinearNDInterpolator(pts, vals)
        self._near = NearestNDInterpolator(pts, vals)
        self.pts = pts
        self.pitch = np.array([r["rpy"][1] for r in rows])

    def reachable(self, x, y, z, margin=0.0):
        r = float(np.hypot(x, y))
        return (R_MIN + margin <= r <= R_MAX - margin) and (Z_MIN + margin <= z <= Z_MAX - margin)

    def joints_for(self, x, y, z):
        """Clamped into the validated region - never returns a sub-floor config."""
        r = float(np.clip(np.hypot(x, y), R_MIN, R_MAX))
        z = float(np.clip(z, Z_MIN, Z_MAX))
        j1 = float(np.degrees(np.arctan2(y, x))) + J1_OFFSET_DEG
        v = self._lin(r, z)
        if v is None or np.any(np.isnan(v)):
            v = self._near(r, z)                 # just outside the convex hull
        j2, j3 = np.asarray(v, dtype=np.float64).reshape(-1)
        return [j1, float(j2), float(j3), self.j4, self.j5, self.j6]


if __name__ == "__main__":
    p = JointPlanner()
    print(f"wrist held at J4={p.j4} J5={p.j5} J6={p.j6}  (camera up)")
    print(f"table covers r {p.pts[:,0].min():.0f}..{p.pts[:,0].max():.0f} mm, "
          f"z {p.pts[:,1].min():.0f}..{p.pts[:,1].max():.0f} mm")
    print(f"gripper pitch across table: {p.pitch.min():.1f}..{p.pitch.max():.1f} deg")
    for (x, y, z) in [(280, 0, 105), (280, 120, 105), (250, -120, 215), (330, 60, 160)]:
        print(f"  ({x:3d},{y:4d},{z:3d}) -> {[round(v,1) for v in p.joints_for(x, y, z)]}")
