#!/usr/bin/env python3
"""Read-only connectivity check for the Piper arm over CAN, using piper_sdk.

No motion commands are sent. Connects, enables the arm, and reports CAN fps,
arm status, joint angles, and enable state.

Before running this, the arm must be in Slave mode - see SETUP.md. If it's
still in Master mode (or was switched without a restart), ArmStatus.Hz will
read 0.0 and Control Mode will show TEACHING_MODE instead of STANDBY/CAN_CTRL,
even though the CAN link itself looks healthy.

Usage: python3 scripts/check_connection.py [can_iface]
"""
import sys
import time
from piper_sdk import *

CAN_IFACE = sys.argv[1] if len(sys.argv) > 1 else "can0"


def main():
    piper = C_PiperInterface_V2(CAN_IFACE)
    piper.ConnectPort()
    time.sleep(1.0)

    print("fps:", piper.GetCanFps())

    status = piper.GetArmStatus()
    print("\nArmStatus.Hz:", getattr(status, "Hz", "n/a"),
          "(0.0 means no real status frame received - check Slave mode / restart, see SETUP.md)")
    print(status)

    angles = piper.GetArmJointMsgs()
    print("JointMsgs.Hz:", getattr(angles, "Hz", "n/a"))
    print(angles)

    print("EnableStatus:", piper.GetArmEnableStatus())


if __name__ == "__main__":
    main()
