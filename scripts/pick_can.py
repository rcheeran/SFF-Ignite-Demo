#!/usr/bin/env python3
"""Pick up a soda can with the gripper: hover above it, descend, grasp,
lift, move to a drop-off point, release.

*** THIS SENDS REAL MOTION COMMANDS TO THE ARM. ***

Before running:
1. Fill in CAN_POSITION_MM below - the grasp point (center of the can, at
   the height/depth you want the gripper to close around it). Get this
   with scripts/read_current_pose.py: enter drag-teach mode, physically
   move the gripper to that position, and read the printed X/Y/Z. (Or
   measure it by hand relative to the arm base - X=forward, Y=left/right,
   Z=height, in mm.)
2. Fill in DROPOFF_POSITION_MM the same way.
3. Adjust CAN_DIAMETER_MM / GRIP_WIDTH_MM to your actual can.
4. Verify both positions are reachable BEFORE running the full sequence -
   see the corner-mapping approach used for draw_square.py. A target that
   exceeds the arm's reachable workspace at this orientation fails
   silently at the CAN level (arm_status flips, motion_status does not
   always reflect it) - do not skip this check.

Usage: python3 scripts/pick_can.py
"""
import time
from piper_sdk import *

CAN_IFACE = "can0"
SPEED_PERCENT = 30  # conservative speed

# --- TODO: fill these in before running (see instructions above) ---
CAN_POSITION_MM = None       # (x, y, z) - grasp point, gripper closes here
DROPOFF_POSITION_MM = None   # (x, y, z) - where to release the can
# ---------------------------------------------------------------------

HOVER_HEIGHT_MM = 80.0   # how far above the grasp/dropoff point to pre-position
CAN_DIAMETER_MM = 66.0   # standard 12oz soda can - adjust to your actual can
GRIP_WIDTH_MM = 55.0     # gripper closes to this width to grasp (< can diameter)
OPEN_WIDTH_MM = 80.0     # gripper opens to this width before approaching
GRIP_EFFORT = 1000       # 0.001 N*m units - 1000 = 1 N*m, matches AgileX demos

# Gripper orientation held constant (deg) - top-down grasp, matching
# AgileX's own end-pose demos (gripper pointing down/forward).
RX_DEG, RY_DEG, RZ_DEG = 0.0, 85.0, 0.0

FACTOR = 1000  # SDK units: mm * 1000, deg * 1000


class UnreachableTarget(RuntimeError):
    pass


def send_end_pose(piper, x_mm, y_mm, z_mm, move_mode=0x02):
    """move_mode: 0x00 = MOVE_P, 0x02 = MOVE_L (linear)."""
    piper.MotionCtrl_2(0x01, move_mode, SPEED_PERCENT, 0x00)
    piper.EndPoseCtrl(
        round(x_mm * FACTOR), round(y_mm * FACTOR), round(z_mm * FACTOR),
        round(RX_DEG * FACTOR), round(RY_DEG * FACTOR), round(RZ_DEG * FACTOR),
    )


def wait_motion_done(piper, timeout=8.0, poll_interval=0.05):
    """Poll GetArmStatus() for motion_status == 0 (reach target successfully).

    Checks arm_status too - a target that exceeds the reachable workspace
    flips arm_status without motion_status ever reflecting success/failure
    reliably by itself.
    """
    time.sleep(0.2)
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        status = piper.GetArmStatus()
        if status.Hz > 0:
            if status.arm_status.arm_status != 0:
                raise UnreachableTarget(
                    f"arm_status={status.arm_status.arm_status} - target rejected"
                )
            if status.arm_status.motion_status == 0:
                return True
            elif status.arm_status.motion_status == 1:
                print("    motion_status=REACH_TARGET_POS_FAILED - stopping")
                return False
        time.sleep(poll_interval)
    print(f"    timed out after {timeout}s waiting for motion complete")
    return False


def set_gripper(piper, width_mm, effort=GRIP_EFFORT):
    piper.GripperCtrl(round(width_mm * FACTOR), effort, 0x01, 0)
    time.sleep(1.0)


def go_to_zero(piper):
    print("Going to zero position...")
    piper.ModeCtrl(0x01, 0x01, SPEED_PERCENT, 0x00)
    piper.JointCtrl(0, 0, 0, 0, 0, 0)
    wait_motion_done(piper)


def move_to(piper, position_mm, label):
    x, y, z = position_mm
    hover = (x, y, z + HOVER_HEIGHT_MM)
    print(f"  hovering above {label} {hover}...")
    send_end_pose(piper, *hover)
    if not wait_motion_done(piper):
        raise UnreachableTarget(f"could not reach hover point above {label}")
    print(f"  descending to {label} {position_mm}...")
    send_end_pose(piper, *position_mm)
    if not wait_motion_done(piper):
        raise UnreachableTarget(f"could not reach {label}")


def lift_from(piper, position_mm, label):
    x, y, z = position_mm
    hover = (x, y, z + HOVER_HEIGHT_MM)
    print(f"  lifting from {label} to {hover}...")
    send_end_pose(piper, *hover)
    if not wait_motion_done(piper):
        raise UnreachableTarget(f"could not lift from {label}")


def main():
    if CAN_POSITION_MM is None or DROPOFF_POSITION_MM is None:
        raise RuntimeError(
            "CAN_POSITION_MM / DROPOFF_POSITION_MM not set - see the instructions "
            "at the top of this file. Use scripts/read_current_pose.py to capture "
            "real coordinates first, and verify they're reachable before running."
        )

    piper = C_PiperInterface_V2(CAN_IFACE)
    piper.ConnectPort()
    time.sleep(1.0)

    print("Enabling arm...")
    deadline = time.time() + 5
    while time.time() < deadline and not piper.EnablePiper():
        time.sleep(0.05)
    if not all(piper.GetArmEnableStatus()):
        raise RuntimeError("Arm did not fully enable - aborting before sending any motion.")

    print("Enabling and opening gripper...")
    set_gripper(piper, OPEN_WIDTH_MM)

    go_to_zero(piper)

    print(f"\nMoving to can at {CAN_POSITION_MM}...")
    move_to(piper, CAN_POSITION_MM, "can")

    print(f"Closing gripper to {GRIP_WIDTH_MM}mm...")
    set_gripper(piper, GRIP_WIDTH_MM)

    lift_from(piper, CAN_POSITION_MM, "can")

    print(f"\nMoving to drop-off at {DROPOFF_POSITION_MM}...")
    move_to(piper, DROPOFF_POSITION_MM, "drop-off")

    print(f"Opening gripper to release...")
    set_gripper(piper, OPEN_WIDTH_MM)

    lift_from(piper, DROPOFF_POSITION_MM, "drop-off")

    print("\nDone. Returning to zero position...")
    go_to_zero(piper)


if __name__ == "__main__":
    main()
