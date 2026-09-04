#!/usr/bin/env python3
"""One-time overhead-camera -> table-XY calibration (homography).

Procedure:
  1. Arm goes to the taught start pose (joint space), gripper opens.
  2. You place the soda can INTO the gripper; press Enter; gripper closes.
  3. The arm visits a grid of XY positions at fixed height/orientation.
     At each stop, the overhead camera detects the can's red centroid ->
     one (pixel, table-XY) correspondence.
  4. Homography is fit and saved to overhead_calib.json with residuals.
  5. Arm returns near start, gripper opens - take the can back.

The table is planar, so a homography fully maps overhead pixels to base-frame
XY at can height. If the overhead camera is ever moved or bumped: re-run this.

Run in ~/pi05-venv:  python calibrate_overhead.py
"""
import json
import os
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from piper_sdk import C_PiperInterface_V2

IFACE = "can_piper"
# Realtek 5MP static camera: widest coverage (~855x642mm), 30fps at full res,
# sees arm base + gripper + can. Mounted rotated 90deg (arm base at frame right).
STATIC_DEV = "/dev/video6"
CAP_W, CAP_H = 2592, 1944
SPEED = 20
HZ = 15
GRIP_OPEN_UM = 70000
GRIP_HOLD_UM = 52000       # ~52mm: closed on the 57mm can with grip force
START_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "start_pose.json")
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "overhead_calib.json")

# calibration grid (base-frame mm), fixed height & taught orientation
LOAD_SECS = 25
CAL_Z = 200.0
CAL_XY = [(150, -250), (150, 0), (150, 250),
          (275, -250), (275, -125), (275, 0), (275, 125), (275, 250),
          (400, -250), (400, 0), (400, 250),
          (200, -180), (350, 180)]


def detect_red(bgr, min_area=600):
    """Red-can centroid in pixels, or None.

    Union of two tests - neither alone survives this room's lighting range:
      * HSV hue+saturation: good in vivid light, but bright daylight washes the
        can down to S~20 where it vanishes.
      * red dominance (R clearly above G and B): survives washed-out light and,
        unlike a low HSV saturation floor, does not match warm-lit plywood.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = (cv2.inRange(hsv, (0, 110, 50), (12, 255, 255))
          | cv2.inRange(hsv, (166, 110, 50), (180, 255, 255)))
    f = bgr.astype(np.int16)
    B, G, R = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    m2 = (((R - np.maximum(G, B)) > 35) & (R > 70)).astype(np.uint8) * 255
    m = cv2.bitwise_or(m1, m2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
    if not cnts:
        return None
    M = cv2.moments(max(cnts, key=cv2.contourArea))
    return (M["m10"] / M["m00"], M["m01"] / M["m00"])


def open_overhead():
    cap = cv2.VideoCapture(STATIC_DEV, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    for _ in range(10):
        cap.read()
    return cap


def grab(cap):
    for _ in range(4):   # flush stale buffer
        cap.read()
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("overhead camera: no frame")
    return frame


def read_pose(p):
    ep = p.GetArmEndPoseMsgs().end_pose
    return np.array([ep.X_axis, ep.Y_axis, ep.Z_axis,
                     ep.RX_axis, ep.RY_axis, ep.RZ_axis]) / 1000.0


def set_grip(p, um, secs=2.5, effort=1000):
    """Command the gripper. MUST hold CAN-control mode (MotionCtrl_2) while
    streaming GripperCtrl - in STANDBY the arm silently ignores the command,
    and a single shot does not latch. Verified with test_gripper2.py.
    """
    t0 = time.time()
    while time.time() - t0 < secs:
        p.MotionCtrl_2(0x01, 0x01, SPEED, 0x00)
        p.GripperCtrl(int(um), effort, 0x01, 0)
        time.sleep(1.0 / HZ)


def stream_pose(p, target6, secs):
    t0 = time.time()
    while time.time() - t0 < secs:
        p.MotionCtrl_2(0x01, 0x00, SPEED, 0x00)
        p.EndPoseCtrl(*(int(v * 1000) for v in target6))
        time.sleep(1.0 / HZ)


def goto_joints(p, joints, secs=20, tol=2.0):
    cmd = [int(j * 1000) for j in joints]
    t0 = time.time()
    while time.time() - t0 < secs:
        p.MotionCtrl_2(0x01, 0x01, SPEED, 0x00)
        p.JointCtrl(*cmd)
        time.sleep(1.0 / HZ)
        js = p.GetArmJointMsgs().joint_state
        cur = [getattr(js, f"joint_{i+1}") / 1000.0 for i in range(6)]
        if all(abs(a - b) < tol for a, b in zip(cur, joints)):
            return True
    return False


def main():
    start = json.load(open(START_JSON))
    cap = open_overhead()

    p = C_PiperInterface_V2(IFACE)
    p.ConnectPort()
    time.sleep(1.0)
    deadline = time.time() + 5
    while time.time() < deadline:
        p.EnablePiper()
        en = p.GetArmEnableStatus()
        if en and all(en):
            break
        time.sleep(0.2)
    en = p.GetArmEnableStatus()
    if not en or not all(en):
        raise SystemExit(f"ABORT: arm did not enable {en} (teach button off?)")
    p.MotorAngleLimitMaxSpdSet(motor_num=6, max_angle_limit=-600, min_angle_limit=-1790, max_joint_spd=0x7FFF)
    set_grip(p, GRIP_OPEN_UM)

    print("-> going to taught start pose")
    goto_joints(p, start["joints"])

    for t in range(LOAD_SECS, 0, -1):
        print(f"\rPLACE THE CAN between the open jaws -- closing in {t:2d}s ", end="", flush=True)
        time.sleep(1.0)
    print("\rclosing on can...                                   ")
    set_grip(p, GRIP_HOLD_UM)
    print("gripper closed on can.")

    # The taught wrist orientation is "handed": commanding it verbatim at every
    # XY makes IK reject the entire +Y half of the table. Rotating it about the
    # base Z by the change in base angle keeps grasp geometry identical and
    # reaches everywhere (verified 7/7 targets, <=1mm error).
    R0 = Rot.from_euler("xyz", start["pose"][3:6], degrees=True)
    th0 = np.arctan2(start["pose"][1], start["pose"][0])

    def ori_for(x, y):
        dth = np.arctan2(y, x) - th0
        return (Rot.from_euler("z", dth) * R0).as_euler("xyz", degrees=True)

    pixels, xys = [], []
    for (x, y) in CAL_XY:
        tgt = [x, y, CAL_Z, *ori_for(x, y)]
        print(f"-> waypoint ({x}, {y}, {CAL_Z})")
        stream_pose(p, tgt, 6.0)
        cur = read_pose(p)
        if np.linalg.norm(cur[:2] - np.array([x, y])) > 25:
            print(f"   SKIP: arm did not reach (at {np.round(cur[:2],0)})")
            continue
        time.sleep(0.8)  # settle
        c = detect_red(grab(cap))
        if c is None:
            print("   SKIP: can not detected in overhead view")
            continue
        # use the arm's actual XY (more truthful than the commanded target)
        pixels.append(c)
        xys.append([float(cur[0]), float(cur[1])])
        print(f"   ok: pixel ({c[0]:.0f}, {c[1]:.0f}) <-> table ({cur[0]:.0f}, {cur[1]:.0f})")

    if len(pixels) < 4:
        raise SystemExit(f"ABORT: only {len(pixels)} good correspondences (need >=4)")

    H, _ = cv2.findHomography(np.array(pixels), np.array(xys))
    # residuals
    proj = cv2.perspectiveTransform(np.array(pixels, dtype=np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)
    res = np.linalg.norm(proj - np.array(xys), axis=1)
    print(f"\nhomography residuals (mm): mean {res.mean():.1f}, max {res.max():.1f}")

    json.dump({
        "H": H.tolist(),
        "capture_size": [CAP_W, CAP_H],
        "pixels": pixels, "xys": xys,
        "residual_mean_mm": float(res.mean()), "residual_max_mm": float(res.max()),
        "cal_z_mm": CAL_Z,
    }, open(OUT_JSON, "w"), indent=1)
    print(f"saved -> {OUT_JSON}")

    print("-> returning to start; gripper will open, catch the can")
    goto_joints(p, start["joints"])
    set_grip(p, GRIP_OPEN_UM)
    print("done.")


if __name__ == "__main__":
    main()
