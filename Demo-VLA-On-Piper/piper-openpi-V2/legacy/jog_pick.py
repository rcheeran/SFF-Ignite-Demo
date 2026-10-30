#!/usr/bin/env python3
"""jog_pick.py - jog the Piper end-effector under CAN control to define the
pick-and-place waypoints, then save them to waypoints.json.

Teach mode freezes pose feedback, so instead we jog under CAN_CTRL (feedback is
live). Position the gripper over/at the can and capture named waypoints.

PREREQ: Joint-5 TEACH button OFF (slave mode) so the arm accepts CAN commands.
Clear workspace, E-stop in hand. Run in ~/venv.

Keys:
  translate: w/s = +/-X (fwd/back)   a/d = +/-Y (left/right)   r/f = +/-Z (up/down)
  rotate:    i/k = +/-RX   o/l = +/-RY   n/m = +/-RZ
  gripper:   [ = close   ] = open
  step:      + / -   (mm per press)
  capture:   1=observe  2=pregrasp  3=grasp  4=lift    v=list captured
  p=print pose   q=save & quit
"""
import json
import os
import sys
import termios
import threading
import time
import tty

from piper_sdk import C_PiperInterface_V2

CAN = "can_piper"
OUT = os.path.expanduser("~/Documents/piper-openpi/waypoints.json")
F = 1000.0            # mm/deg -> 0.001mm/0.001deg for the SDK
SPEED = 25            # percent
GRIP_MAX_MM = 70.0


class Jog:
    def __init__(self):
        self.p = C_PiperInterface_V2(CAN)
        self.p.ConnectPort()
        time.sleep(1.0)
        self._enable()
        e = self.p.GetArmEndPoseMsgs().end_pose
        self.t = [e.X_axis / F, e.Y_axis / F, e.Z_axis / F,
                  e.RX_axis / F, e.RY_axis / F, e.RZ_axis / F]  # mm / deg
        self.grip = GRIP_MAX_MM      # start open
        self.step = 5.0
        self.wp = {}
        self.run = True
        threading.Thread(target=self._stream, daemon=True).start()

    def _enable(self):
        d = time.time() + 5
        while time.time() < d:
            self.p.EnablePiper()
            en = self.p.GetArmEnableStatus()
            if en and all(en):
                return
            time.sleep(0.2)
        raise SystemExit("ABORT: arm did not enable (is the TEACH button OFF?).")

    def _stream(self):
        while self.run:
            self.p.MotionCtrl_2(0x01, 0x00, SPEED, 0x00)  # CAN_CTRL, MOVE_P
            x, y, z, rx, ry, rz = (round(v * F) for v in self.t)
            self.p.EndPoseCtrl(x, y, z, rx, ry, rz)
            self.p.GripperCtrl(round(self.grip * F), 1000, 0x01, 0)
            time.sleep(0.05)

    def fb(self):
        e = self.p.GetArmEndPoseMsgs().end_pose
        return [e.X_axis / F, e.Y_axis / F, e.Z_axis / F,
                e.RX_axis / F, e.RY_axis / F, e.RZ_axis / F]


def getkey():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    print(__doc__)
    print("PREREQ: TEACH button OFF. Enabling + starting stream...")
    j = Jog()
    trans = {"w": (0, 1), "s": (0, -1), "a": (1, -1), "d": (1, 1), "r": (2, 1), "f": (2, -1)}
    rot = {"i": (3, 1), "k": (3, -1), "o": (4, 1), "l": (4, -1), "n": (5, 1), "m": (5, -1)}
    caps = {"1": "observe", "2": "pregrasp", "3": "grasp", "4": "lift"}
    print("ready - jog the arm.\n")
    while True:
        k = getkey()
        if k == "q":
            break
        elif k in trans:
            ax, sgn = trans[k]; j.t[ax] += sgn * j.step
        elif k in rot:
            ax, sgn = rot[k]; j.t[ax] += sgn * 2.0
        elif k == "[":
            j.grip = max(0.0, j.grip - 3.0)
        elif k == "]":
            j.grip = min(GRIP_MAX_MM, j.grip + 3.0)
        elif k == "+" or k == "=":
            j.step = min(20.0, j.step + 1.0)
        elif k == "-":
            j.step = max(1.0, j.step - 1.0)
        elif k == "p":
            print(f"\n target={[round(v,1) for v in j.t]} grip={j.grip:.0f} | fb={[round(v,1) for v in j.fb()]}")
        elif k in caps:
            j.wp[caps[k]] = {"pose": [round(v, 2) for v in j.fb()], "grip": round(j.grip, 2)}
            print(f"\n captured {caps[k]}: {j.wp[caps[k]]}")
        elif k == "v":
            print("\n" + json.dumps(j.wp, indent=2))
        print(f"  X{j.t[0]:6.0f} Y{j.t[1]:6.0f} Z{j.t[2]:6.0f} "
              f"RX{j.t[3]:5.0f} RY{j.t[4]:5.0f} RZ{j.t[5]:5.0f} grip{j.grip:3.0f} step{j.step:.0f}   ",
              end="\r", flush=True)
    j.run = False
    time.sleep(0.2)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(j.wp, open(OUT, "w"), indent=2)
    print(f"\n\nsaved waypoints {list(j.wp)} -> {OUT}")


if __name__ == "__main__":
    main()
