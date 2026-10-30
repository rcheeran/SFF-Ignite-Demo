#!/usr/bin/env python3
"""Connect to the Piper arm, go to zero position, then trace a 10 inch
square with the gripper endpoint using linear (MOVE_L) Cartesian motion.

*** THIS SENDS REAL MOTION COMMANDS TO THE ARM. ***

Prerequisites (see prepare-jetson/SETUP.md): arm in Slave mode, CAN interface up, arm
enabled and reachable via piper_sdk. Verify with prepare-piper/check_connection.py
first.

The square is traced in the X-Z plane (forward distance x height), at a
fixed lateral (Y) position, gripper pointing down/forward (RY=85 deg,
matching AgileX's own demos). All 4 corners below were individually
verified reachable on this arm before being wired into the script - if you
change CENTER_X_MM / CENTER_Z_MM / SIDE_LENGTH_MM, re-verify each corner
individually first (see the corner-mapping approach in the project history)
since this arm's usable workspace at RY=85 deg is much smaller than its
~626mm max reach, and out-of-range targets fail silently at the CAN level
(arm_status flips to TARGET_POS_EXCEEDS_LIMIT, but motion_status does not
report failure - both must be checked).

Usage: python3 useful-piper-scripts/draw_square.py [can_iface]   (default: can_piper)
"""
import sys
import time
from piper_sdk import *

CAN_IFACE = sys.argv[1] if len(sys.argv) > 1 else "can_piper"
MM_PER_INCH = 25.4

SIDE_LENGTH_MM = 10 * MM_PER_INCH   # 254mm, per the ask
CENTER_X_MM = 300.0                 # forward distance to the square's center
CENTER_Z_MM = 380.0                 # height of the square's center
FIXED_Y_MM = 0.0                    # lateral position (fixed - square is in the X-Z plane)
SPEED_PERCENT = 30                  # conservative speed for a first run

# Gripper orientation held constant through the square (deg), matching
# AgileX's own end-pose demos (gripper pointing down/forward).
RX_DEG, RY_DEG, RZ_DEG = 0.0, 85.0, 0.0

FACTOR = 1000  # SDK units: mm * 1000, deg * 1000 (per piper_sdk EndPoseCtrl convention)

half = SIDE_LENGTH_MM / 2
CORNERS = [
    (CENTER_X_MM - half, FIXED_Y_MM, CENTER_Z_MM - half),
    (CENTER_X_MM + half, FIXED_Y_MM, CENTER_Z_MM - half),
    (CENTER_X_MM + half, FIXED_Y_MM, CENTER_Z_MM + half),
    (CENTER_X_MM - half, FIXED_Y_MM, CENTER_Z_MM + half),
    (CENTER_X_MM - half, FIXED_Y_MM, CENTER_Z_MM - half),  # close the loop
]


class UnreachableTarget(RuntimeError):
    pass


def send_end_pose(piper, x_mm, y_mm, z_mm, move_mode=0x02):
    """move_mode: 0x00 = MOVE_P, 0x02 = MOVE_L (linear)."""
    piper.MotionCtrl_2(0x01, move_mode, SPEED_PERCENT, 0x00)
    piper.EndPoseCtrl(
        round(x_mm * FACTOR),
        round(y_mm * FACTOR),
        round(z_mm * FACTOR),
        round(RX_DEG * FACTOR),
        round(RY_DEG * FACTOR),
        round(RZ_DEG * FACTOR),
    )


def wait_motion_done(piper, timeout=8.0, poll_interval=0.05):
    """Poll GetArmStatus() for motion_status == 0 (reach target successfully).

    Also checks arm_status - a target that exceeds the arm's reachable
    workspace flips arm_status to TARGET_POS_EXCEEDS_LIMIT (or similar)
    without ever setting motion_status to a failure value, so motion_status
    alone is not sufficient to detect a rejected target.
    """
    time.sleep(0.2)
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        status = piper.GetArmStatus()
        if status.Hz > 0:
            if status.arm_status.arm_status != 0:
                raise UnreachableTarget(
                    f"arm_status={status.arm_status.arm_status} - target rejected "
                    "(likely exceeds reachable workspace at this orientation)"
                )
            if status.arm_status.motion_status == 0:
                return True
        time.sleep(poll_interval)
    print(f"    timed out after {timeout}s waiting for motion complete")
    return False


def go_to_zero(piper):
    print("Going to zero position...")
    piper.ModeCtrl(0x01, 0x01, SPEED_PERCENT, 0x00)
    piper.JointCtrl(0, 0, 0, 0, 0, 0)
    wait_motion_done(piper)
    angles = piper.GetArmJointMsgs()
    print(f"  joints: {angles.joint_state.joint_1}, {angles.joint_state.joint_2}, "
          f"{angles.joint_state.joint_3}, {angles.joint_state.joint_4}, "
          f"{angles.joint_state.joint_5}, {angles.joint_state.joint_6}")


def main():
    piper = C_PiperInterface_V2(CAN_IFACE)
    piper.ConnectPort()
    time.sleep(1.0)

    print("Enabling arm...")
    deadline = time.time() + 5
    while time.time() < deadline and not piper.EnablePiper():
        time.sleep(0.05)
    if not piper.GetArmEnableStatus() or not all(piper.GetArmEnableStatus()):
        raise RuntimeError("Arm did not enable - aborting before sending any motion.")

    go_to_zero(piper)

    print(f"\nMoving to start corner {CORNERS[0]} (MOVE_P)...")
    send_end_pose(piper, *CORNERS[0], move_mode=0x00)
    wait_motion_done(piper)

    for i, corner in enumerate(CORNERS[1:], start=1):
        print(f"Moving to corner {i}: {corner} (MOVE_L)...")
        send_end_pose(piper, *corner, move_mode=0x02)
        wait_motion_done(piper)

    print("\nSquare complete. Returning to zero position...")
    go_to_zero(piper)


if __name__ == "__main__":
    main()
