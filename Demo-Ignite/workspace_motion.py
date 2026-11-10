#!/usr/bin/env python3
"""Validate or replay originally taught candy-workspace corners."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import threading
import time

import numpy as np

from demo import (
    CALIBRATION_DIR, WORKSPACE_MIN_MM, WORKSPACE_MAX_MM,
    DEMO_SETTINGS, DemoError, MARGINED_LIMITS, PIPER_C, finite_vector,
    euler_to_rot, fk_flange, flange_for_tip, load_json,
    move_robot, require_healthy, solve_waypoint,
    return_to_zero, tip_position, validate_joint_path_safety, validate_target_workspace,
)
from piper.robot import Piper, connect_for_teach

DEFAULT_RECORD = CALIBRATION_DIR / "calibration/workspace_teaching.json"
PREVIEW_CORNER_OFFSETS_MM = {
    "2": [0.0, 70.0, -8.20008306013348],
    # C4's original tip plus this offset is [144, -170, 18] mm.
    "4": [-17.5716867000212, -44.36457140168528, 0.576945121752715],
    # C6 and C8 target [440, -170, 18] and [440, 206, 18] mm.
    "6": [-0.77995125443726, -60.99932052008623, -2.388909600809612],
    "8": [5.3861374392412, 27.79580766696915, -1.167384231547487],
}
PREVIEW_MINIMUM_XY_MM = [144.0, -170.0]
PREVIEW_MAXIMUM_Y_MM = 206.3166357672802


def hold_until_power_off(piper: Piper, hold_joints: np.ndarray) -> None:
    """Hold the supplied joints without changing grip until the operator confirms power-off.

    Confirmation is manual; this does not electronically detect controller power
    or automatically disable motors while the arm may be unsupported.
    """
    confirmed = threading.Event()

    def wait_for_confirmation() -> None:
        """Wait for the operator's power-off confirmation and signal the hold loop."""
        input(
            "Turn the Piper controller OFF while supporting the arm. "
            "Once it is OFF, press Enter: "
        )
        confirmed.set()

    threading.Thread(target=wait_for_confirmation, daemon=True).start()
    while not confirmed.is_set():
        piper.send_joints(hold_joints, None, PIPER_C.SPEED_REST)
        confirmed.wait(timeout=0.05)


def parse_args() -> argparse.Namespace:
    """Require explicit single/all-corner selection and permissions for motion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--corner", type=int, help="one-based recorded corner number")
    selection.add_argument("--corners", type=int, nargs="+", help="recorded corner numbers in the requested order")
    selection.add_argument("--all-corners", action="store_true", help="visit all recorded candy corners in order")
    parser.add_argument("--can-interface", default=PIPER_C.CAN_IFACE)
    parser.add_argument("--check-only", action="store_true", help="validate/list only; never connect to CAN")
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument("--setup-reviewed", action="store_true")
    return parser.parse_args()


def load_corners(record_path: Path) -> tuple[list[dict], int, float]:
    """Validate original measurements and apply script-local experimental replay settings."""
    record = load_json(record_path)
    tool_length = float(record["tool_length_mm"])
    if not math.isfinite(tool_length) or not math.isclose(tool_length, PIPER_C.TOOL_LEN_MM, abs_tol=1e-6):
        raise DemoError("recorded tool length differs from the current Piper tool length")
    settings = DEMO_SETTINGS
    values = record["workspaces"]["candy"]
    if not isinstance(values, list) or not values:
        raise DemoError("no originally taught candy corners are available")
    measured = np.stack([
        finite_vector(point["tool_tip_base_point_mm"], 3, "recorded candy corner") for point in values
    ])
    minimum, maximum = measured.min(axis=0), measured.max(axis=0)
    preview_maximum = np.maximum(maximum, WORKSPACE_MAX_MM)
    preview_minimum = np.minimum(minimum, WORKSPACE_MIN_MM)
    preview_minimum[:2] = np.minimum(
        preview_minimum[:2], finite_vector(PREVIEW_MINIMUM_XY_MM, 2, "preview minimum XY")
    )
    if not math.isfinite(PREVIEW_MAXIMUM_Y_MM):
        raise DemoError("preview maximum Y must be finite")
    preview_maximum[1] = max(preview_maximum[1], PREVIEW_MAXIMUM_Y_MM)
    speed_value = float(settings["motion_speed_percent"])
    timeout = float(settings["move_timeout_seconds"])
    if not math.isfinite(speed_value) or not speed_value.is_integer() or not 1 <= speed_value <= 100:
        raise DemoError("motion speed must be an integer from 1 to 100")
    if not math.isfinite(timeout) or timeout <= 0:
        raise DemoError("move timeout must be positive and finite")
    offsets = PREVIEW_CORNER_OFFSETS_MM
    if not isinstance(offsets, dict) or any(key not in {str(i) for i in range(1, len(values) + 1)} for key in offsets):
        raise DemoError("candy corner replay offsets must map existing corner numbers to XYZ offsets")
    corners = []
    for index, point in enumerate(values, start=1):
        joints = finite_vector(point["joints_deg"], 6, f"corner {index} joints")
        recorded_tip = finite_vector(point["tool_tip_base_point_mm"], 3, f"corner {index} tip")
        computed_tip = finite_vector(tip_position(joints), 3, f"corner {index} FK tip")
        if np.linalg.norm(recorded_tip - computed_tip) > 1.0:
            raise DemoError(f"corner {index} recorded tip disagrees with current FK by more than 1 mm")
        validate_target_workspace(computed_tip, minimum, maximum)
        validate_joint_path_safety(f"corner {index} endpoint", [joints, joints])
        offset = finite_vector(offsets.get(str(index), [0, 0, 0]), 3, f"corner {index} replay offset")
        requested_tip = computed_tip + offset
        if np.any(offset):
            validate_target_workspace(requested_tip, preview_minimum, preview_maximum)
            orientation = euler_to_rot(fk_flange(joints)[3:])
            joints = solve_waypoint(
                f"adjusted candy corner {index}", flange_for_tip(requested_tip, orientation), orientation, joints
            )
            computed_tip = finite_vector(tip_position(joints), 3, f"adjusted corner {index} FK tip")
            rotation_error = (orientation.inv() * euler_to_rot(fk_flange(joints)[3:])).magnitude()
            if np.linalg.norm(computed_tip - requested_tip) > 1.0 or np.degrees(rotation_error) > 0.5:
                raise DemoError(f"adjusted corner {index} IK exceeds 1 mm / 0.5 degree target tolerance")
            validate_joint_path_safety(f"adjusted corner {index} endpoint", [joints, joints])
        corners.append({
            "index": index, "joints_deg": joints, "tool_tip_base_point_mm": computed_tip,
            "replay_offset_mm": offset,
        })
    return corners, min(int(speed_value), 10), timeout


def replay_corner(
    arm, corner: dict, speed: int, timeout: float, *,
    joint_limits: np.ndarray = MARGINED_LIMITS,
) -> None:
    """Check the actual current-to-target path, move without grip commands, and verify joints."""
    require_healthy(arm, "candy corner replay")
    current = finite_vector(arm.joints_deg(), 6, "current joints")
    target = corner["joints_deg"]
    validate_joint_path_safety(
        f"candy corner {corner['index']} replay", [current, target], joint_limits=joint_limits
    )
    tip = finite_vector(tip_position(target), 3, "target tool-tip coordinates")
    name = (
        f"candy corner {corner['index']} "
        f"(tool-tip base XYZ mm: X={tip[0]:.2f}, Y={tip[1]:.2f}, Z={tip[2]:.2f})"
    )
    move_robot(arm, name, target, None, timeout, speed)
    measured = finite_vector(arm.joints_deg(), 6, "measured arrival joints")
    if np.max(np.abs(measured - target)) > 2.0:
        raise DemoError("measured joints are outside the 2-degree arrival tolerance")
    print(f"Measured tool-tip XYZ: {np.round(tip_position(measured), 2).tolist()} mm")
    print("Review this physical boundary; no camera-derived point was used.")


def replay_sequence(arm, corners: list[dict], speed: int, timeout: float) -> None:
    """Check current-to-zero and the selected route, home first, then replay each segment."""
    if not corners:
        raise DemoError("select at least one corner")
    require_healthy(arm, "candy corner sequence")
    start = finite_vector(arm.joints_deg(), 6, "current joints")
    zero = np.zeros(6)
    factory_limits = np.asarray(PIPER_C.JOINT_LIMITS_DEG, dtype=np.float64)
    validate_joint_path_safety("pre-replay zero", [start, zero], joint_limits=factory_limits)
    validate_joint_path_safety(
        "zero to first candy corner", [zero, corners[0]["joints_deg"]], joint_limits=factory_limits
    )
    if len(corners) > 1:
        validate_joint_path_safety("candy corner sequence", [corner["joints_deg"] for corner in corners])
    return_to_zero(arm, context="pre-replay zero")
    for index, corner in enumerate(corners):
        replay_corner(
            arm, corner, speed, timeout,
            joint_limits=factory_limits if index == 0 else MARGINED_LIMITS,
        )
        time.sleep(1.0)
    print(f"[OK] Completed {len(corners)} recorded candy corner(s).", flush=True)


def main() -> int:
    """Default to offline listing; replay the approved sequence and safely end holding."""
    args = parse_args()
    arm = None
    try:
        corners, speed, timeout = load_corners(args.record)
        print(
            f"Script-local preview: minimum XY={PREVIEW_MINIMUM_XY_MM}, maximum Y={PREVIEW_MAXIMUM_Y_MM:.2f} mm; "
            "live demo configuration and teaching record are unchanged."
        )
        for corner in corners:
            print(f"Corner {corner['index']}: tip XYZ={np.round(corner['tool_tip_base_point_mm'], 2).tolist()} mm")
            if np.any(corner.get("replay_offset_mm", [0, 0, 0])):
                print(f"  Replay adjustment from taught point: {np.round(corner['replay_offset_mm'], 3).tolist()} mm")
        indices = args.corners if args.corners is not None else ([args.corner] if args.corner is not None else None)
        if indices is not None:
            if any(not 1 <= index <= len(corners) for index in indices):
                raise DemoError(f"corner numbers must be between 1 and {len(corners)}")
            if len(set(indices)) != len(indices):
                raise DemoError("corner numbers must not be repeated")
        if args.check_only or not args.allow_motion:
            print("Validation complete; no CAN connection or robot motion.")
            return 0
        if not args.setup_reviewed or (indices is None and not args.all_corners):
            raise DemoError("replay requires --corner N, --corners N ..., or --all-corners, plus --allow-motion --setup-reviewed")
        if not sys.stdin.isatty():
            raise DemoError("motion requires an interactive terminal for shutdown confirmation")
        selected = corners if args.all_corners else [corners[index - 1] for index in indices]
        print(
            f"Will replay corner(s) {[corner['index'] for corner in selected]} at {speed}% speed, "
            "with no gripper commands and no power-off between corners.\n"
            f"FIRST move to SDK zero (all six joints 0) at {PIPER_C.SPEED_REST}% speed.\n"
            "Restart the controller after teaching. Remove candy and people; clear the entire "
            "current-to-zero, zero-to-first-corner, and corner-to-corner route. "
            "Joint checks do not model collisions.\n"
            "Keep emergency stop accessible. Ctrl+C stops the test without automatic zero return.",
            flush=True,
        )
        if input("Type MOVE to confirm the entire selected route is reviewed and clear: ").strip() != "MOVE":
            print("Replay cancelled; no robot connection.")
            return 0
        print("Connecting and enabling motors in 5 seconds.", flush=True)
        time.sleep(5)
        arm = connect_for_teach(iface=args.can_interface)
        replay_sequence(arm, selected, speed, timeout)
        return 0
    except KeyboardInterrupt:
        print("\nReplay interrupted; no automatic return or gripper release.", file=sys.stderr)
        return 130
    except (RuntimeError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: candy workspace replay stopped: {exc}", file=sys.stderr)
        return 2
    finally:
        if arm is not None:
            print("SUPPORT THE ARM. Holding current pose until controller power-off confirmation.", flush=True)
            try:
                hold_until_power_off(arm, arm.joints_deg())
            finally:
                arm.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
