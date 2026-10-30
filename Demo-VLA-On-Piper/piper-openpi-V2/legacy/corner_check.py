#!/usr/bin/env python3
"""Corner check: how does robot XY map into each static camera, and what is visible?

Drives the arm (holding the can) to a set of known base-frame XY points and
records, for each, the can's pixel position in both static cameras. That tells us
  * which image axis corresponds to robot X vs Y (and the sign),
  * which points are actually reachable,
  * which points are actually in frame.
Output: corner_check.json + an annotated overlay per camera.

Run in ~/pi05-venv:  python corner_check.py
"""
import json
import os
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from piper_sdk import C_PiperInterface_V2

IFACE = "can_piper"
CAMS = {                      # name -> (device, width, height)
    "realtek": ("/dev/video6", 2592, 1944),
    "hp":      ("/dev/video8", 1280, 720),
}
SPEED = 20
HZ = 15
GRIP_OPEN_UM = 70000
GRIP_HOLD_UM = 52000
HERE = os.path.dirname(os.path.abspath(__file__))
START_JSON = os.path.join(HERE, "start_pose.json")
OUT_JSON = os.path.join(HERE, "corner_check.json")
OUT_DIR = os.path.join(HERE, "logs", "corner_check")

LOAD_SECS = 25          # time to place the can in the jaws before it closes
PROBE_Z = 200.0
# spread across the plausible workspace; unreachable/invisible ones are just skipped
PROBE_XY = [(150, -250), (150, 0), (150, 250),
            (275, -250), (275, 0), (275, 250),
            (400, -250), (400, 0), (400, 250)]


def detect_red(bgr, min_area=600):
    """Red-can centroid in pixels, or None.

    Union of two tests, because neither alone survives this room's lighting range:
      * HSV hue+saturation - works when the can is vividly lit, but under bright
        daylight the can washes out to S~20 and disappears.
      * red dominance (R clearly above G and B) - survives washed-out light, and
        unlike a low HSV saturation floor it does NOT match warm-lit plywood.
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


def open_cams():
    caps = {}
    for name, (dev, w, h) in CAMS.items():
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)      # geometry must not shift
        for _ in range(8):
            cap.read()
        caps[name] = cap
    return caps


def grab(cap):
    for _ in range(4):
        cap.read()
    ok, f = cap.read()
    return f if ok else None


def read_pose(p):
    ep = p.GetArmEndPoseMsgs().end_pose
    return np.array([ep.X_axis, ep.Y_axis, ep.Z_axis,
                     ep.RX_axis, ep.RY_axis, ep.RZ_axis]) / 1000.0


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


def stream_pose(p, target6, secs):
    t0 = time.time()
    while time.time() - t0 < secs:
        p.MotionCtrl_2(0x01, 0x00, SPEED, 0x00)
        p.EndPoseCtrl(*(int(v * 1000) for v in target6))
        time.sleep(1.0 / HZ)


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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    start = json.load(open(START_JSON))
    caps = open_cams()

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
    if not all(p.GetArmEnableStatus()):
        raise SystemExit("ABORT: arm not enabled (teach button off?)")
    p.MotorAngleLimitMaxSpdSet(motor_num=6, max_angle_limit=-600, min_angle_limit=-1790, max_joint_spd=0x7FFF)
    set_grip(p, GRIP_OPEN_UM)

    print("-> start pose")
    goto_joints(p, start["joints"])
    # timed pause instead of input(): this is normally launched without a tty
    for s in range(LOAD_SECS, 0, -1):
        print(f"\rPLACE THE CAN between the gripper jaws -- closing in {s:2d}s ", end="", flush=True)
        time.sleep(1.0)
    print("\rclosing gripper on can...                        ")
    set_grip(p, GRIP_HOLD_UM)
    print(f"   gripper now {p.GetArmGripperMsgs().gripper_state.grippers_angle/1000.0:.1f} mm")

    # The taught wrist orientation is "handed": commanding it verbatim at every
    # XY makes IK reject the entire +Y half. Rotating it about the base Z by the
    # change in base angle keeps the grasp geometry identical and reaches
    # everywhere (verified: 7/7 targets, <=1mm error).
    R0 = Rot.from_euler("xyz", start["pose"][3:6], degrees=True)
    th0 = np.arctan2(start["pose"][1], start["pose"][0])

    def ori_for(x, y):
        dth = np.arctan2(y, x) - th0
        return (Rot.from_euler("z", dth) * R0).as_euler("xyz", degrees=True)

    results = []
    for (x, y) in PROBE_XY:
        print(f"-> ({x}, {y})", end="  ")
        stream_pose(p, [x, y, PROBE_Z, *ori_for(x, y)], 6.0)
        cur = read_pose(p)
        reached = float(np.linalg.norm(cur[:2] - np.array([x, y])))
        if reached > 30:
            print(f"UNREACHABLE (stopped at {np.round(cur[:2],0)})")
            results.append({"target": [x, y], "reached": False, "actual": cur[:3].tolist()})
            continue
        time.sleep(0.8)
        rec = {"target": [x, y], "reached": True, "actual": cur[:3].tolist(), "pixels": {}}
        seen = []
        for name, cap in caps.items():
            f = grab(cap)
            c = detect_red(f) if f is not None else None
            rec["pixels"][name] = None if c is None else [float(c[0]), float(c[1])]
            if c is not None:
                seen.append(name)
            if f is not None:
                cv2.imwrite(os.path.join(OUT_DIR, f"{name}_{x}_{y}.jpg"), f)
        results.append(rec)
        print(f"at {np.round(cur[:2],0)}  visible in: {seen or 'NONE'}")

    json.dump({"probe_z": PROBE_Z, "results": results}, open(OUT_JSON, "w"), indent=1)
    print(f"\nsaved -> {OUT_JSON}")

    print("-> back to start; gripper opens, catch the can")
    goto_joints(p, start["joints"])
    set_grip(p, GRIP_OPEN_UM)
    for cap in caps.values():
        cap.release()
    print("done.")


if __name__ == "__main__":
    main()
