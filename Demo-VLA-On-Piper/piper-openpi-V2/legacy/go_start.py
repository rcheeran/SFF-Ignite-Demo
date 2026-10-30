#!/usr/bin/env python3
"""Move the Piper to the taught episode-start pose (JOINT space, streamed).

The start configuration was captured by drag-teaching (start_pose.json):
nearly joint-zero but with the wrist at J5~14, J6~-165 - the wrist roll the
demos actually used (pose-space EndPoseCtrl rejects this region near the
wrist singularity, so joints are commanded directly, like go_zero).

PREREQ: teach button OFF. Usage: python go_start.py
"""
import json
import os
import time

from piper_sdk import C_PiperInterface_V2

IFACE = "can_piper"
SPEED = 20
HZ = 15
TIMEOUT = 25.0
TOL_DEG = 2.0
START_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "start_pose.json")
GRIP_OPEN_UM = 70000
# J6 firmware limit matching the task region (re-asserted; not power-cycle-persistent)
J6_MAX, J6_MIN = -600, -1790  # 0.1 deg


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


def joints_deg(p):
    js = p.GetArmJointMsgs().joint_state
    return [getattr(js, f"joint_{i+1}") / 1000.0 for i in range(6)]


def main():
    target = json.load(open(START_JSON))["joints"]
    print("target joints:", target)

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
        raise SystemExit(f"ABORT: arm did not enable {en} - is the TEACH button OFF?")

    p.MotorAngleLimitMaxSpdSet(motor_num=6, max_angle_limit=J6_MAX, min_angle_limit=J6_MIN, max_joint_spd=0x7FFF)
    set_grip(p, GRIP_OPEN_UM)

    print("from:", [round(j, 1) for j in joints_deg(p)])
    cmd = [int(j * 1000) for j in target]
    t0 = time.time()
    while time.time() - t0 < TIMEOUT:
        p.MotionCtrl_2(0x01, 0x01, SPEED, 0x00)
        p.JointCtrl(*cmd)
        time.sleep(1.0 / HZ)
        if all(abs(a - b) < TOL_DEG for a, b in zip(joints_deg(p), target)):
            break
    cur = joints_deg(p)
    print("arrived:", [round(j, 1) for j in cur])
    ok = all(abs(a - b) < TOL_DEG for a, b in zip(cur, target))
    print("AT START POSE" if ok else "WARNING: did not fully reach the start pose")


if __name__ == "__main__":
    main()
