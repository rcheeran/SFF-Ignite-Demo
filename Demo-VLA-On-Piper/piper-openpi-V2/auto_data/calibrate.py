#!/usr/bin/env python3
"""Camera -> table calibration: a homography from top-camera pixels to robot XY.

The robot calibrates itself. You hand it a cube once; it then repeats, for a spread of known
spots: carry the cube there, put it down, RETREAT TO HOME, detect the cube, pick it up again.

Retreating before detecting is the whole point. The obvious version - hold the object and detect
it - was tried on the previous robot and gave 11 mm of residual, because the jaws holding the
object also hide part of it and the hidden fraction changes with wrist angle. An object standing
free on the table has no such bias.

    python -m auto_data.calibrate                 # 6 spots, ~4 min, writes auto_data/calib.json
    python -m auto_data.calibrate --check         # no motion: re-fit from the saved pairs, print residuals

Output: auto_data/calib.json {H, pairs, residual_mm}. `px_to_xy()` is what the collector uses.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

import piper.config as C
from auto_data import auto_collect as A
from auto_data.auto_collect import (GRIP_HOLD_MM, HELD_MAX_MM, HELD_MIN_MM, hold, joints_for_tip,
                                    stream, tip_z)
from auto_data.frames import (CALIB_JSON, ROI, conditioning, fit, image_angle_to_robot,
                               loo_error, px_to_xy)
from auto_data.jaw_yaw import load as load_jaw_cal, yaw_for_angle
from auto_data.vision import detect
from piper.cameras import open_cameras
from piper.robot import connect_for_motion

HERE = os.path.dirname(os.path.abspath(__file__))

# spread over the area the IK reaches and the camera sees; more spread = better conditioned fit
# Where the CUBE sits (tool-tip xy), not where the flange goes. Spread as a GRID over as much of
# the workspace as the arm reaches, never along a line:
#   * an affine needs 3 non-collinear points, and a near-collinear set fits its own points to ~1 mm
#     while inventing a shear and extrapolating badly (measured: 84 mm),
#   * the fit's accuracy decays with distance from the points it was given, so a calibration
#     covering only the middle forces the collector to work in only the middle. Measured reach at
#     grasp height with the tip landing within 5 mm at every yaw: x 150-560, y up to +-260.
# Extra spots also mean a few failed grasps still leave a well-conditioned set.
SPOTS = [(250.0, -150.0), (250.0, 0.0), (250.0, 150.0),
         (325.0, -180.0), (325.0, -60.0), (325.0, 60.0), (325.0, 180.0),
         (400.0, -150.0), (400.0, 0.0), (400.0, 150.0),
         (460.0, -80.0), (460.0, 80.0)]
TIP_Z_CARRY = 117.0            # tool tip clear of the table while carrying
TIP_Z_GRASP = 20.0             # tool tip when closing on a cube on the table
TIP_Z_SERVO = 60.0             # tool tip while visually aligning: just above the cube, so the jaws
                               # and the cube sit at similar heights and the oblique camera does
                               # not put them at different pixels for the same table position
LOAD_SECS = 25







def cmd_check():
    d = json.load(open(CALIB_JSON))
    pairs = [((p["px"], p["py"]), (p["x"], p["y"])) for p in d["pairs"]]
    M, res = fit(pairs)
    scale, aniso = conditioning(M)
    print(f"{len(pairs)} pairs, residual mean {res.mean():.1f} mm  max {res.max():.1f} mm  "
          f"scale {scale:.2f} mm/px  anisotropy {aniso:.2f}")
    print(f"  leave-one-out error {loo_error(pairs):.1f} mm  <- the honest accuracy")
    for p, r in zip(d["pairs"], res):
        print(f"  px ({p['px']:6.1f},{p['py']:6.1f}) -> ({p['x']:6.1f},{p['y']:6.1f})  err {r:5.1f} mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", choices=["red", "blue"], default="blue",
                    help="which cube to carry (blue detects larger on this rig)")
    ap.add_argument("--check", action="store_true", help="no motion: re-fit the saved pairs")
    ap.add_argument("--grasp-z", type=float, default=TIP_Z_GRASP,
                    help="TOOL TIP height above the table when closing on the cube "
                         f"(default {TIP_Z_GRASP:.0f} mm). Lower it if the jaws skim the cube's top.")
    args = ap.parse_args()
    grasp_z = args.grasp_z
    if args.check:
        return cmd_check()

    label = f"{args.cube}_cube"
    # A previous position calibration lets us express the cube's image angle in the robot frame;
    # the jaw calibration turns that into a wrist yaw. Without either we simply grasp at yaw 0,
    # which is exactly how this script behaved before and still bootstraps from nothing.
    prev_A = json.load(open(CALIB_JSON))["A"] if os.path.exists(CALIB_JSON) else None
    jaw_cal = load_jaw_cal()
    print(f"jaw alignment: {'on (gain %+.0f, offset %+.1f deg)' % jaw_cal if jaw_cal and prev_A else 'OFF - grasping at yaw 0'}")

    cams = None
    robot = connect_for_motion()
    try:
        cams = open_cameras(names=("top", "wrist"))
        for n in ("top", "wrist"):
            if not cams[n].wait(5.0):
                raise SystemExit(f"ABORT: no frames from the {n} camera")
        robot.home(grip_mm=C.GRIP_OPEN_MM)
        print(f"grasping with the tool tip {grasp_z:.0f} mm above the table")

        for t in range(LOAD_SECS, 0, -1):
            print(f"\rPUT THE {args.cube.upper()} CUBE between the open jaws -- closing in {t:2d}s "
                  f"(grip {robot.grip_mm():.0f} mm) ", end="", flush=True)
            hold(robot, robot.joints_deg(), 1.0, C.GRIP_OPEN_MM)
        print()
        hold(robot, robot.joints_deg(), 1.5, GRIP_HOLD_MM)
        if not (HELD_MIN_MM <= robot.grip_mm() <= HELD_MAX_MM):
            raise SystemExit(f"ABORT: gripper reads {robot.grip_mm():.0f} mm - no cube held")

        pairs = []
        for i, (x, y) in enumerate(SPOTS):
            q_up = joints_for_tip([x, y, TIP_Z_CARRY])
            q_dn = joints_for_tip([x, y, grasp_z], q_up)

            stream(robot, q_up, 5.0, GRIP_HOLD_MM)          # carry it there
            stream(robot, q_dn, 2.5, GRIP_HOLD_MM)
            hold(robot, q_dn, 0.8, C.GRIP_OPEN_MM)          # put it down
            stream(robot, q_up, 2.0, C.GRIP_OPEN_MM)
            robot.home(grip_mm=C.GRIP_OPEN_MM)              # get OUT of the picture before looking
            time.sleep(0.6)

            found = [o for o in detect(cams["top"].read(), roi=ROI) if o["label"] == label]

            # Line the jaws up with the cube before reaching for it. With a fixed wrist the jaws
            # closed along one world direction and caught a rotated cube on a corner - that is what
            # lost the cube twice in the first six spots. The POSITION stays the commanded one on
            # purpose: re-picking there is the proof that the cube is where we think it is, which
            # is what makes the pair trustworthy. Only the angle comes from vision.
            # Re-pick at the COMMANDED spot, never at the detected one. The arm physically put
            # the cube there, so that is ground truth; the detected position depends on the very
            # calibration being measured. Reaching for the detection was tried and it failed at
            # the edges of the workspace, where the old fit is extrapolating: at (250,0) - inside
            # the previously fitted region - detection and command agreed to 6 mm, while at
            # (250,-150) they differed by 74 mm. That gap is the fit's error, not the cube moving.
            # Only the ANGLE comes from vision; angle needs no positional accuracy.
            yaw, gx, gy, shift = 0.0, x, y, None
            if found and prev_A is not None:
                cube_ang = image_angle_to_robot(prev_A, found[0]["angle"])
                yaw = yaw_for_angle(cube_ang, jaw_cal)
                dx, dy = px_to_xy(prev_A, found[0]["px"], found[0]["py"])
                shift = float(np.hypot(dx - x, dy - y))   # reported only: fit error + real movement
            # NO visual servoing here. It was tried and it made things strictly worse, for a
            # reason worth recording: the overhead camera sits ~35 deg off vertical, so the jaws
            # at servo height appear ~30 px from the table point directly beneath them. That gap
            # is PARALLAX and should not be nulled - driving it to zero walks the arm off target
            # by the parallax difference between the jaws' height and the cube's (measured: the
            # arm ended 67 mm past a cube while the "error" appeared to shrink). Comparing pixels
            # of things at different heights under an oblique camera is not a valid measurement.
            stream(robot, joints_for_tip([gx, gy, TIP_Z_CARRY], yaw_deg=yaw), 5.0, C.GRIP_OPEN_MM)
            q_dn_y = joints_for_tip([gx, gy, grasp_z], yaw_deg=yaw)
            stream(robot, q_dn_y, 2.5, C.GRIP_OPEN_MM)
            hold(robot, q_dn_y, 0.5, C.GRIP_OPEN_MM)        # settle, jaws still OPEN
            cv2.imwrite(os.path.join(C.SNAP_DIR, f"grasp{i}_{int(x)}_{int(y)}_wrist.jpg"),
                        cams["wrist"].read())               # what are the jaws about to close on?
            hold(robot, q_dn_y, 1.2, GRIP_HOLD_MM)
            q_up_y = joints_for_tip([gx, gy, TIP_Z_CARRY], yaw_deg=yaw)
            stream(robot, q_up_y, 2.0, GRIP_HOLD_MM)
            regrasped = HELD_MIN_MM <= robot.grip_mm() <= HELD_MAX_MM

            # A pair is only trustworthy when the re-pick SUCCEEDED: that is the proof the cube
            # actually sat at the commanded spot. If the cube slipped in the jaws on the way, it
            # landed somewhere else and pairing that pixel with (x, y) would poison the homography.
            if found and regrasped:
                o = found[0]
                pairs.append(((o["px"], o["py"]), (x, y)))
                print(f"  spot {i} ({x:.0f},{y:.0f}): px ({o['px']:6.1f},{o['py']:6.1f}) "
                      f"seen at ({gx:.0f},{gy:.0f}) shift {shift:4.0f} mm, jaws yawed {yaw:+.0f}  "
                      f"re-picked OK")
            elif not found:
                print(f"  spot {i} ({x:.0f},{y:.0f}): {label} not seen - pair dropped")
            else:
                print(f"  spot {i} ({x:.0f},{y:.0f}): re-pick failed (grip {robot.grip_mm():.0f} mm) "
                      f"at ({gx:.0f},{gy:.0f})"
                      + (f", release shift {shift:.0f} mm" if shift is not None else "")
                      + " - pair dropped")

            if not regrasped:
                for t in range(LOAD_SECS, 0, -1):
                    print(f"\r  PUT THE CUBE BACK SQUARELY between the jaws -- closing in {t:2d}s "
                          f"(grip {robot.grip_mm():.0f} mm) ", end="", flush=True)
                    hold(robot, robot.joints_deg(), 1.0, C.GRIP_OPEN_MM)
                print()
                hold(robot, robot.joints_deg(), 1.5, GRIP_HOLD_MM)
                if not (HELD_MIN_MM <= robot.grip_mm() <= HELD_MAX_MM):
                    print("  still no cube - stopping")
                    break

        if len(pairs) < 3:
            raise SystemExit(f"ABORT: only {len(pairs)} good spots, need 3 for an affine fit")
        M, res = fit(pairs)
        scale, aniso = conditioning(M)
        print(f"\naffine from {len(pairs)} pairs: residual mean {res.mean():.1f} mm, max {res.max():.1f} mm")
        loo = loo_error(pairs)
        print(f"  scale {scale:.2f} mm/px, anisotropy {aniso:.2f}, leave-one-out {loo:.1f} mm")
        if loo > 8:
            print("  WARNING: leave-one-out error is large - this fit does not generalise even "
                  "though its own residual looks small. Get more spots, spread over a grid.")
        json.dump({"A": M.tolist(), "roi": ROI,
                   "pairs": [{"px": p[0][0], "py": p[0][1], "x": p[1][0], "y": p[1][1]} for p in pairs],
                   "residual_mean_mm": float(res.mean()), "residual_max_mm": float(res.max())},
                  open(CALIB_JSON, "w"), indent=1)
        print(f"saved -> {CALIB_JSON}")
        if res.mean() > 8:
            print("NOTE: >8 mm mean residual - re-seat the camera or re-run before collecting")
    finally:
        if cams:
            for c in cams.values():
                c.close()
        robot.rest()


if __name__ == "__main__":
    main()
