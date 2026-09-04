#!/usr/bin/env python3
"""Find out WHICH command takes this arm out of teach mode. Diagnostic only - collects no data.

    python -m tools.teach_probe          # ~1 min, hold the arm throughout

Three guesses at this have now been wrong (MotionCtrl_1's grag_teach_ctrl byte, a second press of the teach
button, streaming ModeCtrl, and the EmergencyStop pair), each inferred from documentation rather than measured
on this arm. So this tries every candidate one at a time and prints ctrl_mode after each, until one works.

ctrl_mode values (piper_sdk arm_feedback_status.ArmMsgStatus.CtrlMode):
    0x00 standby   0x01 CAN_CTRL   0x02 TEACHING_MODE   0x03 ethernet   0x04 wifi
    0x05 offline trajectory        0x06 LINKAGE_TEACHING_INPUT_MODE

The last step is the one that matters: ctrl_mode reading 1 is not proof of anything if JointCtrl is still being
ignored, so the probe finishes by commanding a 3 deg J1 move and reporting whether the arm actually moved.
"""
import time

import numpy as np

from piper import config as C
from piper.robot import connect_for_motion

SETTLE_S = 2.0        # after each candidate: how long to watch ctrl_mode before calling it a failure
STREAM_S = 2.0        # how long to stream a repeated command
PROBE_DEG = 3.0       # size of the "did JointCtrl actually take?" test move


def mode(robot) -> tuple:
    raw = int(robot._p.GetArmStatus().arm_status.ctrl_mode)
    return raw, robot.ctrl_mode()


def watch(robot, secs=SETTLE_S) -> int:
    """Poll ctrl_mode for `secs`, return the last value seen (and print it if it changes)."""
    t0, last = time.time(), mode(robot)[0]
    while time.time() - t0 < secs:
        raw = mode(robot)[0]
        if raw != last:
            print(f"      ctrl_mode {last} -> {raw}")
            last = raw
        time.sleep(0.05)
    return last


def stream(robot, fn, secs=STREAM_S):
    t0 = time.time()
    while time.time() - t0 < secs and int(robot._p.GetArmStatus().arm_status.ctrl_mode) != 1:
        fn()
        time.sleep(0.01)


def candidates(robot):
    """(name, callable) in increasing order of bluntness."""
    p = robot._p
    return [
        ("MotionCtrl_1(0,0,0x02)  'end teach record'", lambda: p.MotionCtrl_1(0x00, 0x00, 0x02)),
        ("ModeCtrl(0x01,0x01) streamed 2 s", lambda: stream(robot, lambda: p.ModeCtrl(0x01, 0x01, C.SPEED_HOME, 0x00))),
        ("MotionCtrl_2(0x01,0x01) streamed 2 s",
         lambda: stream(robot, lambda: p.MotionCtrl_2(0x01, 0x01, C.SPEED_HOME, 0x00))),
        ("EmergencyStop(0x01) + EmergencyStop(0x02)",
         lambda: (p.EmergencyStop(0x01), time.sleep(1.0), p.EmergencyStop(0x02), time.sleep(1.0))),
        ("ModeCtrl streamed again (after e-stop)",
         lambda: stream(robot, lambda: p.ModeCtrl(0x01, 0x01, C.SPEED_HOME, 0x00))),
        ("EnablePiper() x10", lambda: [(p.EnablePiper(), time.sleep(0.2)) for _ in range(10)]),
        ("MasterSlaveConfig(0xFC,0,0,0)  'motion output arm' (SLAVE)",
         lambda: p.MasterSlaveConfig(0xFC, 0, 0, 0)),
        ("ModeCtrl streamed again (after slave config)",
         lambda: stream(robot, lambda: p.ModeCtrl(0x01, 0x01, C.SPEED_HOME, 0x00))),
    ]


def main():
    robot = connect_for_motion(countdown=0)
    try:
        print(f"\nstart: ctrl_mode {mode(robot)}  joints {np.round(robot.joints_deg(), 1).tolist()}")
        print("\nHOLD THE ARM, then press the TEACH BUTTON (the joints go limp).")
        t0 = time.time()
        while not robot.in_teach() and time.time() - t0 < 120.0:
            time.sleep(0.2)
        if not robot.in_teach():
            raise SystemExit("ABORT: never entered teach mode - is the arm powered? is that the right button?")
        print(f"in teach mode: ctrl_mode {mode(robot)}")
        print("move the arm by hand a little, so we can tell it is really limp, then keep holding it.")
        time.sleep(3.0)
        print(f"joints now {np.round(robot.joints_deg(), 1).tolist()}\n")

        print("trying each exit candidate - KEEP HOLDING THE ARM")
        worked = None
        for name, fn in candidates(robot):
            if not robot.in_teach():
                break
            print(f"  -> {name}")
            try:
                fn()
            except Exception as e:
                print(f"      raised {type(e).__name__}: {e}")
            raw = watch(robot)
            print(f"      ctrl_mode after: {raw} ({robot.ctrl_mode()})")
            if raw == 1:
                worked = name
                break

        print()
        if worked:
            print(f"LEFT TEACH MODE after: {worked}")
        elif not robot.in_teach():
            print(f"left teach mode, but ctrl_mode is {mode(robot)} (not CAN_CTRL)")
        else:
            print(f"STILL IN TEACH MODE after every candidate: ctrl_mode {mode(robot)}")
            print("Nothing here works on this firmware - the arm needs a power cycle.")
            return

        # ctrl_mode reading 1 proves nothing on its own: check the arm actually obeys JointCtrl now
        print(f"\nenable: {robot._enable_motors()}  enabled {robot.enabled()}")
        j0 = robot.joints_deg()
        target = j0.copy()
        target[0] += PROBE_DEG
        print(f"commanding J1 {j0[0]:.1f} -> {target[0]:.1f} deg for 3 s...")
        t0 = time.time()
        while time.time() - t0 < 3.0:
            robot.send_joints(target, None, C.SPEED_REST)
            time.sleep(1.0 / C.CONTROL_HZ)
        moved = robot.joints_deg()[0] - j0[0]
        print(f"J1 moved {moved:+.1f} deg -> JointCtrl is {'OBEYED' if abs(moved) > 0.5 else 'STILL IGNORED'}")
    finally:
        robot.rest()


if __name__ == "__main__":
    main()
