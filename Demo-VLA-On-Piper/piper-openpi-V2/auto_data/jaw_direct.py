#!/usr/bin/env python3
"""Measure the jaw axis DIRECTLY from the top camera, with nothing in the gripper.

Why not the cube method (auto_data/jaw_yaw.py): it infers the jaw angle from the angle a placed
cube was left at, which only works if the cube never rotates in the jaws. It does - the same
commanded yaw of 0 left the cube at -36.2 deg on one run and -19.8 deg on the next. Using a
slipping grasp to calibrate the fix for slipping is circular, and the two runs disagreed even on
the SIGN of the gain.

Here the arm holds nothing. It parks the open jaws over a clear patch of table at a series of
commanded yaws and photographs them; the jaws are the only dark object there, and the line through
the two finger tips is the axis we want. No grasp, nothing to slip.

    python -m auto_data.jaw_direct                # ~1 min, writes auto_data/jaw_yaw.json
    python -m auto_data.jaw_direct --show         # also save annotated frames to data/snapshots/
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

import piper.config as C
from auto_data import auto_collect as A
from auto_data.auto_collect import hold, joints_for, stream
from auto_data.calibrate import CALIB_JSON, image_angle_to_robot
from auto_data.jaw_yaw import OUT_JSON, fit_yaw
from auto_data.vision import white_balance
from piper.cameras import open_cameras
from piper.robot import connect_for_motion

# an empty patch of table, so the only dark thing under the camera is the gripper
TEST_XY = (285.0, 0.0)
JAW_Z = 175.0                  # tool tip ~60 mm: jaws clearly above the table, still sharp
YAWS = [0.0, 20.0, -20.0, 40.0]
DARK_MAX = 70                  # jaws are matt black; the lit table is far brighter
MIN_AREA = 300
SEARCH_RADIUS = 45             # px around the tool centre. Tuned on real frames: at 140 the search
                               # swallows the wrist motor, body and cables, whose principal axis is
                               # the ARM's (elongation 1.5, axis meaningless). At 45 only the two
                               # fingers are inside and elongation rises to ~2.6.
MIN_ELONGATION = 1.5


def jaw_angle(bgr, centre_px, radius=SEARCH_RADIUS):
    """Image angle of the line through the jaws, or None.

    Looks only within `radius` of where the gripper is expected, takes the dark pixels there, and
    fits the principal axis. Two fingers either side of the tool centre make a strongly elongated
    cloud, so the major axis IS the jaw axis.
    """
    x0 = max(0, int(centre_px[0] - radius)); x1 = min(bgr.shape[1], int(centre_px[0] + radius))
    y0 = max(0, int(centre_px[1] - radius)); y1 = min(bgr.shape[0], int(centre_px[1] + radius))
    patch = white_balance(bgr)[y0:y1, x0:x1]
    dark = (patch.max(axis=2) < DARK_MAX).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    pts = np.column_stack(np.where(dark > 0))[:, ::-1]
    if len(pts) < 30:
        return None, (x0, y0, x1, y1), dark
    mean = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - mean)
    if s[0] < MIN_ELONGATION * s[1]:       # not elongated enough to trust an axis
        return None, (x0, y0, x1, y1), dark
    ang = np.degrees(np.arctan2(vt[0][1], vt[0][0]))
    return ((ang + 45.0) % 90.0) - 45.0, (x0, y0, x1, y1), dark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if not os.path.exists(CALIB_JSON):
        raise SystemExit("ABORT: run `python -m auto_data.calibrate` first")
    cal = json.load(open(CALIB_JSON))
    Amat = cal["A"]
    from auto_data.calibrate import px_to_xy

    # where the tool centre lands in the image: invert the calibration for TEST_XY
    Ainv = cv2.invertAffineTransform(np.asarray(Amat, float))
    centre_px = cv2.transform(np.array([[[TEST_XY[0], TEST_XY[1]]]], float), Ainv)[0, 0]
    print(f"tool centre should appear near px ({centre_px[0]:.0f},{centre_px[1]:.0f})")

    cams = None
    robot = connect_for_motion()
    try:
        cams = open_cameras(names=("top",))
        if not cams["top"].wait(5.0):
            raise SystemExit("ABORT: no frames from the top camera")
        robot.home(grip_mm=C.GRIP_OPEN_MM)

        obs = []
        for yaw in YAWS:
            q = joints_for([TEST_XY[0], TEST_XY[1], JAW_Z], yaw_deg=yaw)
            stream(robot, q, 5.0, C.GRIP_OPEN_MM)
            hold(robot, q, 0.8, C.GRIP_OPEN_MM)
            time.sleep(0.4)
            frame = cams["top"].read()
            ang, box, dark = jaw_angle(frame, centre_px)
            if ang is None:
                print(f"  yaw {yaw:+6.1f}: jaws not resolved")
            else:
                rob = image_angle_to_robot(Amat, ang)
                obs.append((yaw, rob))
                print(f"  yaw {yaw:+6.1f}: jaw axis {ang:+6.1f} deg in image -> {rob:+6.1f} deg robot")
            if args.show:
                v = frame.copy()
                cv2.rectangle(v, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 1)
                if ang is not None:
                    c = (int(centre_px[0]), int(centre_px[1]))
                    d = (int(80 * np.cos(np.radians(ang))), int(80 * np.sin(np.radians(ang))))
                    cv2.line(v, (c[0] - d[0], c[1] - d[1]), (c[0] + d[0], c[1] + d[1]), (0, 0, 255), 2)
                cv2.imwrite(os.path.join(C.SNAP_DIR, f"jaw_yaw{int(yaw)}.jpg"), v)

        if len(obs) < 2:
            raise SystemExit(f"ABORT: only {len(obs)} usable measurements")
        gain, offset, spread = fit_yaw(obs)
        print(f"\njaw_axis = {gain:+.0f} * yaw {offset:+.1f} deg   (spread {spread:.1f} deg)")
        json.dump({"gain": float(gain), "offset_deg": float(offset), "spread_deg": float(spread),
                   "method": "direct jaw axis, empty gripper",
                   "obs": [{"yaw": float(a), "placed": float(b)} for a, b in obs]},
                  open(OUT_JSON, "w"), indent=1)
        print(f"saved -> {OUT_JSON}")
        if spread > 8:
            print("NOTE: >8 deg spread - check data/snapshots/jaw_yaw*.jpg that the red line lies "
                  "along the fingers")
    finally:
        if cams:
            for c in cams.values():
                c.close()
        robot.rest()


if __name__ == "__main__":
    main()
