#!/usr/bin/env python3
"""Replay joint positions recorded by useful-piper-scripts/record_pos.py.

Ported from AgileX's own playPos_new_en.py (agilexrobotics/Agilex-College,
piper/recordAndPlayPos/) - the core logic, including the joint 2/3/5
gravity-fall safety check in stop(), is kept exactly as AgileX wrote it.
Do not skip or shortcut that check: exiting teach mode with joints 2/3/5
away from a safe folded position risks the arm falling under gravity when
motor holding torque is released.

*** THIS SENDS REAL MOTION COMMANDS TO THE ARM. ***

Prerequisites: the arm must have exited teach mode (short-press the teach
button after recording) before running this.

Usage: python3 useful-piper-scripts/play_pos.py
"""
import os
import time
import csv
from piper_sdk import *

CAN_IFACE = "can_piper"
HAVE_GRIPPER = True
PLAY_TIMES = 1          # 0 = infinite loop
PLAY_INTERVAL_S = 0     # negative = manual (press Enter) advance between positions
SPEED_PERCENT = 30      # conservative speed - AgileX's demo defaults to 100
CAN_CTRL_TIMEOUT_S = 5.0
CSV_PATH = os.path.join(os.path.dirname(__file__), "pos.csv")


def get_pos(piper):
    joint_state = piper.GetArmJointMsgs().joint_state
    joints = tuple(getattr(joint_state, f"joint_{i+1}") / 1e3 * 0.0174533 for i in range(6))
    if HAVE_GRIPPER:
        return joints + (piper.GetArmGripperMsgs().gripper_state.grippers_angle / 1e6,)
    return joints


def stop(piper):
    """Must be called before the first CAN-mode switch after teach mode.

    Waits until joints 2, 3, 5 are within a safe range before releasing
    the emergency-stop hold, to prevent the arm falling under gravity.
    """
    piper.EmergencyStop(0x01)
    time.sleep(1.0)
    limit_angle = [0.1745, 0.7854, 0.2094]  # rad
    pos = get_pos(piper)
    while not (abs(pos[1]) < limit_angle[0] and abs(pos[2]) < limit_angle[0]
               and limit_angle[2] < pos[4] < limit_angle[1]):
        time.sleep(0.01)
        pos = get_pos(piper)
    piper.EmergencyStop(0x02)
    time.sleep(1.0)


def enable(piper):
    while not piper.EnablePiper():
        time.sleep(0.01)
    if HAVE_GRIPPER:
        time.sleep(0.01)
        piper.GripperCtrl(0, 1000, 0x01, 0x00)
    piper.ModeCtrl(0x01, 0x01, SPEED_PERCENT, 0x00)
    print("INFO: Enable successful")


def main():
    try:
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            track = list(csv.reader(f))
            if not track:
                print("ERROR: Position file is empty")
                return
            track = [[float(v) for v in row] for row in track]
    except FileNotFoundError:
        print("ERROR: Position file does not exist - run useful-piper-scripts/record_pos.py first")
        return

    piper = C_PiperInterface_V2(CAN_IFACE)
    piper.ConnectPort()
    time.sleep(0.1)

    print("step 1: Please ensure the robotic arm has exited the teaching mode before playback")
    if piper.GetArmStatus().arm_status.ctrl_mode != 1:
        stop(piper)  # required the first time switching out of teach mode into CAN mode
    deadline = time.time() + CAN_CTRL_TIMEOUT_S
    while piper.GetArmStatus().arm_status.ctrl_mode != 1:  # 1 = CAN_CTRL
        if time.time() > deadline:
            print("ERROR: Failed to switch to CAN mode, please check if teach mode is exited")
            return
        piper.ModeCtrl(0x01, 0x01, SPEED_PERCENT, 0x00)
        time.sleep(0.01)

    enable(piper)
    count = 0
    input("step 2: Press Enter to start playing positions")
    while PLAY_TIMES == 0 or abs(PLAY_TIMES) != count:
        for n, pos in enumerate(track):
            while True:
                joints = [round(v / 0.0174533 * 1e3) for v in pos[:6]]
                piper.MotionCtrl_2(0x01, 0x01, SPEED_PERCENT, 0x00)
                piper.JointCtrl(*joints)
                time.sleep(0.01)
                current_pos = get_pos(piper)
                print(f"INFO: {count + 1}th playback, {n + 1}th position, "
                      f"current position: {current_pos}, target position: {pos}")
                if all(abs(current_pos[i] - pos[i]) < 0.0698 for i in range(6)):
                    break
            if HAVE_GRIPPER and len(pos) == 7:
                piper.GripperCtrl(round(pos[-1] * 1e6), 1000, 0x01, 0x00)
                time.sleep(0.5)
            if PLAY_INTERVAL_S < 0:
                if n != len(track) - 1 and input("INPUT: Enter 'q' to exit, press Enter directly to play: ") == 'q':
                    return
            else:
                time.sleep(PLAY_INTERVAL_S)
        count += 1


if __name__ == "__main__":
    main()
