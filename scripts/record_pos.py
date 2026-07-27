#!/usr/bin/env python3
"""Record fixed joint positions while the arm is in drag-teach mode.

Ported from AgileX's own recordPos_new_en.py (agilexrobotics/Agilex-College,
piper/recordAndPlayPos/), which already uses C_PiperInterface_V2 directly -
no changes needed to the core logic, just repo conventions.

Workflow (see AgileX's Agilex-College README for the full walkthrough with
photos):
1. Run this script.
2. Short-press the physical teach button on the arm to enter teach mode.
3. Physically move the gripper to each position you want to record; press
   Enter to record it, repeat as needed.
4. Type 'q' to stop recording.
5. Short-press the teach button again to exit teach mode.

Positions are stored in pos.csv (next to this script) as one line per
position: 6 joint angles in radians, plus gripper opening in meters.
scripts/play_pos.py reads this same file/format back.

No motion commands are sent by this script - it only reads state.

Usage: python3 scripts/record_pos.py
"""
import os
import time
from piper_sdk import *

CAN_IFACE = "can0"
HAVE_GRIPPER = True
TEACH_MODE_TIMEOUT_S = 10.0
CSV_PATH = os.path.join(os.path.dirname(__file__), "pos.csv")


def get_pos(piper):
    """Return (j1..j6 in rad[, gripper opening in meters])."""
    joint_state = piper.GetArmJointMsgs().joint_state
    joints = tuple(getattr(joint_state, f"joint_{i+1}") / 1e3 * 0.0174533 for i in range(6))
    if HAVE_GRIPPER:
        return joints + (piper.GetArmGripperMsgs().gripper_state.grippers_angle / 1e6,)
    return joints


def main():
    piper = C_PiperInterface_V2(CAN_IFACE)
    piper.ConnectPort()
    time.sleep(0.1)

    print("INFO: Please click the teach button to enter the teaching mode")
    deadline = time.time() + TEACH_MODE_TIMEOUT_S
    while piper.GetArmStatus().arm_status.ctrl_mode != 2:  # 2 = TEACHING_MODE
        if time.time() > deadline:
            print("ERROR: Teaching mode detection timeout, please check whether teaching mode is enabled")
            return
        time.sleep(0.01)

    count = 1
    with open(CSV_PATH, "w") as csv_file:
        while input("INPUT: Enter q to exit, press Enter directly to record: ") != "q":
            current_pos = get_pos(piper)
            print(f"INFO: {count}th position, recorded position: {current_pos}")
            csv_file.write(",".join(map(str, current_pos)) + "\n")
            count += 1

    print(f"INFO: Recording ends ({count - 1} position(s) saved to {CSV_PATH}).")
    print("INFO: Click the teach button again to exit the teaching mode.")


if __name__ == "__main__":
    main()
