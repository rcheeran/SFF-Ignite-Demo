#!/usr/bin/env python3
"""Measure how the jaws move in the overhead image when the arm moves - no grasping, no descent.

The first servo attempt walked the arm 67 mm PAST the cube while claiming its error was shrinking,
which means one of two things was wrong and the log could not tell them apart:
  * the sign/scale converting a pixel error into a millimetre step, or
  * `find_jaws` locking onto something that is not the jaws (the arm's shadow is dark too, and it
    moves with the arm, so a shadow-tracking servo still produces plausible-looking numbers).

So: park the jaws at known table positions, photograph each, and report where the detector thinks
they are. That gives the true Jacobian d(pixel)/d(mm) - and the annotated frames show whether it
is following fingers or a shadow.

    python -m auto_data.servo_check          # ~1 min, writes data/snapshots/servocheck_*.jpg
"""
import json
import os
import time

import cv2
import numpy as np

import piper.config as C
from auto_data.auto_collect import hold, joints_for_tip, stream
from auto_data.servo import find_jaws
from piper.cameras import open_cameras
from piper.robot import connect_for_motion

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB = os.path.join(HERE, "calib.json")
BASE = (320.0, 0.0)
TIP_Z = 60.0
PROBES = [(0.0, 0.0), (30.0, 0.0), (-30.0, 0.0), (0.0, 30.0), (0.0, -30.0)]


def main():
    A = np.asarray(json.load(open(CALIB))["A"], float)
    Ainv = cv2.invertAffineTransform(A)
    cams = None
    robot = connect_for_motion()
    try:
        cams = open_cameras(names=("top",))
        if not cams["top"].wait(5.0):
            raise SystemExit("ABORT: no frames from the top camera")
        robot.home(grip_mm=C.GRIP_OPEN_MM)

        obs = []
        for dx, dy in PROBES:
            x, y = BASE[0] + dx, BASE[1] + dy
            stream(robot, joints_for_tip([x, y, TIP_Z]), 4.0, C.GRIP_OPEN_MM)
            hold(robot, joints_for_tip([x, y, TIP_Z]), 0.6, C.GRIP_OPEN_MM)
            time.sleep(0.4)
            frame = cams["top"].read()
            # search where the CALIBRATION says the tool is - the same assumption the servo makes
            exp = cv2.transform(np.array([[[x, y]]], float), Ainv)[0, 0]
            j = find_jaws(frame, exp)
            v = frame.copy()
            cv2.circle(v, (int(exp[0]), int(exp[1])), 55, (0, 255, 255), 1)
            cv2.drawMarker(v, (int(exp[0]), int(exp[1])), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
            if j:
                cv2.drawMarker(v, (int(j[0]), int(j[1])), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 20, 2)
                obs.append((x, y, j[0], j[1]))
                print(f"  tip ({x:6.1f},{y:6.1f}) -> expected px ({exp[0]:6.1f},{exp[1]:6.1f}) "
                      f"| jaws found ({j[0]:6.1f},{j[1]:6.1f}) {j[2]:5d} px "
                      f"| off by {np.hypot(j[0]-exp[0], j[1]-exp[1]):5.1f} px")
            else:
                print(f"  tip ({x:6.1f},{y:6.1f}) -> jaws NOT FOUND near ({exp[0]:.0f},{exp[1]:.0f})")
            cv2.imwrite(os.path.join(C.SNAP_DIR, f"servocheck_{int(dx)}_{int(dy)}.jpg"), v)

        if len(obs) >= 3:
            P = np.array([[o[0], o[1], 1.0] for o in obs])
            D = np.array([[o[2], o[3]] for o in obs])
            J = np.linalg.lstsq(P, D, rcond=None)[0][:2]      # d(px)/d(mm), 2x2
            print(f"\nmeasured d(pixel)/d(mm):\n{np.round(J, 4)}")
            print(f"calibration's inverse (what the servo assumes):\n{np.round(Ainv[:, :2], 4)}")
            print(f"\nmm per pixel from the measurement: {np.round(np.linalg.inv(J), 3).tolist()}")
        print("\nframes -> data/snapshots/servocheck_*.jpg  "
              "(yellow = where the calibration expects the tool, red = what find_jaws locked onto)")
    finally:
        if cams:
            for c in cams.values():
                c.close()
        robot.rest()


if __name__ == "__main__":
    main()
