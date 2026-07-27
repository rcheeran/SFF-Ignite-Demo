#!/usr/bin/env python3
"""Read the arm's current end-pose - use this to capture real-world
coordinates for pick_can.py (or any other script) by physically
positioning the gripper and reading back where the arm thinks it is.

No motion commands are sent by this script itself. To physically move the
arm by hand: enter drag-teach mode first (grag_teach_ctrl=0x01), move the
gripper to the desired position, then run this script (or press Ctrl+C to
stop the loop below) and note the printed X/Y/Z. Exit teach mode afterward
(grag_teach_ctrl=0x02) before sending any scripted motion commands.

Usage: python3 scripts/read_current_pose.py [can_iface]
"""
import sys
import time
from piper_sdk import *

CAN_IFACE = sys.argv[1] if len(sys.argv) > 1 else "can0"


def main():
    piper = C_PiperInterface_V2(CAN_IFACE)
    piper.ConnectPort()
    time.sleep(1.0)

    print("Reading end pose (Ctrl+C to stop)...")
    print("Position the gripper where you want, then note the X/Y/Z printed below.\n")
    try:
        while True:
            pose = piper.GetArmEndPoseMsgs()
            if pose.Hz > 0:
                p = pose.end_pose
                print(f"\rX={p.X_axis/1000:8.1f}mm  Y={p.Y_axis/1000:8.1f}mm  Z={p.Z_axis/1000:8.1f}mm   "
                      f"RX={p.RX_axis/1000:7.2f}deg  RY={p.RY_axis/1000:7.2f}deg  RZ={p.RZ_axis/1000:7.2f}deg",
                      end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopped.")


if __name__ == "__main__":
    main()
