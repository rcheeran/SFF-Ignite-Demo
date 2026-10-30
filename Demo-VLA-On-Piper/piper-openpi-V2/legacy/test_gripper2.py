#!/usr/bin/env python3
"""Gripper diagnosis, round 2.

Round 1 finding: ctrl_mode was STANDBY(0x0) - the arm ignores commands until
MotionCtrl_2 puts it in CAN control mode. Every working script we have calls
MotionCtrl_2 before JointCtrl/EndPoseCtrl; the gripper path never did.

Here we hold CAN control mode while streaming the gripper command, and escalate
effort. Verified by reading the angle back.

PREREQ: TEACH BUTTON OFF. Run in ~/pi05-venv.
"""
import time

from piper_sdk import C_PiperInterface_V2

IFACE = "can_piper"
OPEN_UM = 70000
CLOSE_UM = 20000
HZ = 15


def angle_mm(p):
    return p.GetArmGripperMsgs().gripper_state.grippers_angle / 1000.0


def mode(p):
    return str(p.GetArmStatus().arm_status.ctrl_mode)


def gstat(p):
    f = p.GetArmGripperMsgs().gripper_state.foc_status
    return f"enabled={f.driver_enable_status} err={f.driver_error_status} homing={f.homing_status}"


def stream(p, um, effort, secs=3.0, code=0x01):
    """Hold CAN-control mode AND the gripper command together, every tick."""
    t0 = time.time()
    while time.time() - t0 < secs:
        p.MotionCtrl_2(0x01, 0x01, 30, 0x00)   # 0x01 = CAN control mode
        p.GripperCtrl(um, effort, code, 0)
        time.sleep(1.0 / HZ)


def trial(p, label, um, effort, pre=None):
    before = angle_mm(p)
    if pre:
        pre()
    stream(p, um, effort)
    time.sleep(0.5)
    after = angle_mm(p)
    print(f"  {label:38s} {before:5.1f} -> {after:5.1f} mm  (d={after-before:+.1f})  mode={mode(p)}")
    return abs(after - before) > 2.0


def main():
    p = C_PiperInterface_V2(IFACE)
    p.ConnectPort()
    time.sleep(1.0)
    if "TEACH" in mode(p).upper():
        raise SystemExit("ABORT: teach mode is ON - switch it off first.")

    for _ in range(6):
        p.EnablePiper()
        time.sleep(0.4)
        if all(p.GetArmEnableStatus()):
            break
    print("arm enabled:", p.GetArmEnableStatus())
    print("start mode :", mode(p), "| gripper", gstat(p), f"angle={angle_mm(p):.1f}mm\n")

    moved = {}
    print("D: CAN-control mode held while streaming gripper (effort 1000)")
    moved["D close"] = trial(p, "close, effort 1000", CLOSE_UM, 1000)
    moved["D open"] = trial(p, "open,  effort 1000", OPEN_UM, 1000)

    print("\nE: same, higher effort 3000")
    moved["E close"] = trial(p, "close, effort 3000", CLOSE_UM, 3000)
    moved["E open"] = trial(p, "open,  effort 3000", OPEN_UM, 3000)

    print("\nF: enable+clear (0x03) first, then stream, effort 5000 (max)")
    def pre_f():
        p.MotionCtrl_2(0x01, 0x01, 30, 0x00)
        p.GripperCtrl(0, 5000, 0x03, 0)
        time.sleep(0.5)
    moved["F close"] = trial(p, "close, effort 5000", CLOSE_UM, 5000, pre_f)
    moved["F open"] = trial(p, "open,  effort 5000", OPEN_UM, 5000, pre_f)

    print("\n================ SUMMARY ================")
    for k, v in moved.items():
        print(f"  {'MOVED  ' if v else 'no move'}  {k}")
    if any(moved.values()):
        print("\n=> gripper IS commandable over CAN; the missing piece was CAN-control mode.")
    else:
        print("\n=> still nothing. Next suspect: gripper needs homing (homing_status=False),")
        print("   or the gripper is on a separate CAN id not driven in this firmware mode.")
    print(f"final: {angle_mm(p):.1f}mm  {gstat(p)}")


if __name__ == "__main__":
    main()
