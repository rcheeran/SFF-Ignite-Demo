#!/usr/bin/env python3
"""Move the Piper to its zero / home pose (all 6 joints -> 0 deg), reliably.

The Piper needs the motion command STREAMED (repeated) to latch - a single
JointCtrl often doesn't execute - so this streams it until the arm arrives.

PREREQ: Joint-5 TEACH button OFF (slave mode) so the arm accepts CAN control.
Clear space, E-stop in hand. Run in ~/venv.

Usage: python3 go_zero.py [can_iface]     (default: can_piper)
"""
import sys
import time

from piper_sdk import C_PiperInterface_V2

IFACE = sys.argv[1] if len(sys.argv) > 1 else "can_piper"
SPEED = 30


def joints_deg(p):
    s = p.GetArmJointMsgs().joint_state
    return [round(getattr(s, f"joint_{i+1}") / 1000.0, 1) for i in range(6)]


def main():
    p = C_PiperInterface_V2(IFACE)
    p.ConnectPort()
    time.sleep(1.0)

    # enable all joints
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

    print("before:", joints_deg(p))
    print("moving to zero (streaming command)...")
    for _ in range(50):  # up to ~8 s
        p.MotionCtrl_2(0x01, 0x01, SPEED, 0x00)  # CAN_CTRL, MOVE_J
        p.JointCtrl(0, 0, 0, 0, 0, 0)
        time.sleep(0.15)
        if max(abs(v) for v in joints_deg(p)) < 0.5:
            break

    print("after: ", joints_deg(p))
    print("done. (arm left enabled/holding at zero)")


if __name__ == "__main__":
    main()
