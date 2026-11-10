#!/usr/bin/env python3
"""Measure D435 demo workspaces and delivery orientation using read-only CAN."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "prepare-piper/homography"))
sys.path.insert(0, str(ROOT / "Demo-Ignite"))

from common import write_json_atomic  # noqa: E402
from demo import DEMO_SETTINGS, DemoError, finite_vector, load_json, RELEASE_CLEARANCE_MM  # noqa: E402
from piper import config as C  # noqa: E402
from piper.robot import connect_readonly  # noqa: E402
from piper.transforms import euler_to_rot, fk_flange, tip_position  # noqa: E402
from teach_scan_poses import collect_stable_pose, validate_replayable_pose  # noqa: E402

CAMERA_DIR = ROOT / "prepare-piper/hand-eye-realsense435"
DEFAULT_CONFIG = CAMERA_DIR / "config.json"
DEFAULT_OUTPUT = CAMERA_DIR / "calibration/workspace_teaching.json"
CLEARANCE_KEYS = ("approach_height_mm", "pregrasp_height_mm", "hand_approach_height_mm")


def parse_args() -> argparse.Namespace:
    """Parse measurement paths and stability thresholds; no motion options exist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--can-interface", default="can_piper")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument("--maximum-joint-stddev-deg", type=float, default=0.25)
    parser.add_argument("--maximum-position-stddev-mm", type=float, default=0.75)
    return parser.parse_args()


def require_active_teach(arm) -> None:
    """Require physical teach recording, not its stopped-but-still-teach state."""
    feedback_hz = arm.feedback_hz()
    if not math.isfinite(feedback_hz) or feedback_hz <= 0:
        raise DemoError("joint feedback is unavailable; no measurement can be accepted")
    if not arm.in_teach() or arm.teach_status_code() == arm.TEACH_STATUS_STOPPED:
        raise DemoError("active physical teach mode is required; keep supporting the arm")


def capture_point(arm, args: argparse.Namespace) -> dict:
    """Derive tip and RPY from mean joints, avoiding wrapped Euler-angle averages."""
    require_active_teach(arm)
    pose = collect_stable_pose(
        arm, args.samples, args.sample_interval,
        args.maximum_joint_stddev_deg, args.maximum_position_stddev_mm,
    )
    require_active_teach(arm)
    joints = finite_vector(pose["joints_deg"], 6, "taught joints")
    finite_vector(pose["end_pose_mm_deg"], 6, "taught end pose")
    finite_vector(pose["joint_stddev_deg"], 6, "joint spread")
    finite_vector(pose["position_stddev_mm"], 3, "position spread")
    validate_replayable_pose(pose, C.IK_LIMIT_MARGIN_DEG)
    point = finite_vector(tip_position(joints), 3, "taught tool tip")
    if point[2] < C.TIP_Z_FLOOR_MM:
        raise DemoError(f"tool tip is below the {C.TIP_Z_FLOOR_MM:g} mm safety floor")
    return {
        **pose,
        "tool_tip_base_point_mm": point.tolist(),
        "fk_end_pose_mm_deg": finite_vector(fk_flange(joints), 6, "FK pose").tolist(),
    }


def workspace_bounds(points: list[dict]) -> tuple[list[float], list[float]]:
    """Return the measured XYZ envelope without expanding beyond sampled points."""
    if len(points) < 2:
        raise DemoError("capture at least two opposite 3D corners")
    coordinates = np.stack([
        finite_vector(point["tool_tip_base_point_mm"], 3, "workspace point")
        for point in points
    ])
    minimum, maximum = coordinates.min(axis=0), coordinates.max(axis=0)
    if np.any(minimum >= maximum):
        raise DemoError("workspace points must span X, Y, and Z; include both height limits")
    return minimum.tolist(), maximum.tolist()


def capture_workspace(arm, args: argparse.Namespace, label: str) -> list[dict]:
    """Collect operator-selected corners for candy centers or palm centers."""
    print(
        f"\n{label.upper()} WORKSPACE: hand-guide the gripper TIP to the permitted "
        "center-position boundaries, in robot-base coordinates.\n"
        "Capture the corners at BOTH lower and upper heights; preferably all eight. "
        "Use an empty marked region, never a person's hand. These are center "
        f"bounds, not release points {RELEASE_CLEARANCE_MM:g} mm above a palm.",
        flush=True,
    )
    points = []
    while True:
        require_active_teach(arm)
        action = input(f"{label}: Enter=capture corner, u=undo, q=finish: ").strip().lower()
        if action == "u":
            if points:
                points.pop()
                print("Last corner removed.")
            else:
                print("No corners to remove.")
            continue
        if action == "q":
            try:
                minimum, maximum = workspace_bounds(points)
            except DemoError as exc:
                print(f"INCOMPLETE: {exc}")
                continue
            print(f"{label} bounds: min={minimum}, max={maximum} mm")
            return points
        if action:
            print("Use Enter, u, or q.")
            continue
        try:
            point = capture_point(arm, args)
        except (RuntimeError, ValueError) as exc:
            require_active_teach(arm)
            print(f"REJECTED corner: {exc}")
            continue
        points.append(point)
        print(f"Recorded corner {len(points)}: {point['tool_tip_base_point_mm']} mm")


def capture_delivery_orientation(arm, args: argparse.Namespace) -> dict:
    """Record a downward delivery orientation using a dummy palm position."""
    while True:
        input(
            "\nWith NO PERSON underneath, hand-guide the gripper to the intended "
            f"release pose {RELEASE_CLEARANCE_MM:g} mm above a marked dummy palm "
            "center, with the gripper facing downward. Hold still and press Enter: "
        )
        try:
            pose = capture_point(arm, args)
            rpy = pose["fk_end_pose_mm_deg"][3:]
            if euler_to_rot(rpy).as_matrix()[2, 2] >= 0:
                raise DemoError("the gripper tool axis must face downward")
        except (RuntimeError, ValueError) as exc:
            require_active_teach(arm)
            print(f"REJECTED orientation: {exc}")
            continue
        print(f"Delivery RPY: {rpy} degrees")
        return pose


def confirm_power_off() -> None:
    """Require manual power-off confirmation; this read-only utility cannot hold the arm."""
    print(
        "\nKEEP SUPPORTING THE ARM. This utility sends NO holding or motor commands. "
        "Stop teach recording with the physical button, then turn controller power OFF.",
        flush=True,
    )
    while True:
        try:
            if input("Type OFF once controller power is OFF: ").strip().upper() == "OFF":
                return
            print("Power-off is not confirmed; keep supporting the arm.")
        except KeyboardInterrupt:
            print("\nKeep supporting the arm and confirm power-off before continuing.")


def prompt_clearances(settings: dict) -> dict:
    """Collect physically reviewed positive clearances without inventing defaults."""
    labels = {
        "approach_height_mm": "Pickup approach height ABOVE the grasp tip",
        "pregrasp_height_mm": "Pre-grasp height ABOVE the grasp tip",
        "hand_approach_height_mm": "Hand approach height ABOVE the release point",
    }
    print("\nController is off. Enter clearances reviewed using the empty setup, in mm.")
    while True:
        values = {}
        for key in CLEARANCE_KEYS:
            while True:
                saved = settings.get(key)
                text = input(f"{labels[key]} (saved={saved}; no new default): ").strip()
                try:
                    value = float(text if text else saved)
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError("must be positive and finite")
                except (TypeError, ValueError) as exc:
                    print(f"INVALID clearance: {exc}")
                    continue
                values[key] = value
                break
        if values["pregrasp_height_mm"] < values["approach_height_mm"]:
            return values
        print("INVALID clearances: pre-grasp must be below pickup approach; enter again.")


def measured_settings(record: dict) -> dict:
    """Validate measurements and return only the settings this utility owns."""
    settings = {}
    for label, prefix in (("candy", "target"), ("hand", "hand")):
        minimum, maximum = workspace_bounds(record["workspaces"][label])
        settings[f"{prefix}_workspace_min_mm"] = minimum
        settings[f"{prefix}_workspace_max_mm"] = maximum
    rpy = finite_vector(record["delivery_pose"]["fk_end_pose_mm_deg"], 6, "delivery pose")[3:]
    if euler_to_rot(rpy).as_matrix()[2, 2] >= 0:
        raise DemoError("delivery tool axis must face downward")
    settings["hand_delivery_rpy_deg"] = rpy.tolist()
    for key in CLEARANCE_KEYS:
        value = float(record["clearances"][key])
        if not math.isfinite(value) or value <= 0:
            raise DemoError(f"{key} must be positive and finite")
        settings[key] = value
    if settings["pregrasp_height_mm"] >= settings["approach_height_mm"]:
        raise DemoError("pre-grasp height must be below pickup approach height")
    return settings


def validate_output_path(config_path: Path, output_path: Path) -> None:
    """Prevent accidentally replacing configuration or another teaching/calibration artifact."""
    if config_path.resolve() == output_path.resolve():
        raise DemoError("configuration and teaching record paths must differ")
    protected_names = {
        "config.json", "candy_grip.json", "scan_poses.json", "hand_eye.json",
        "validation.json", "depth_correction.json", "depth_verification.json",
    }
    if output_path.resolve().name in protected_names:
        raise DemoError(f"refusing to overwrite protected artifact: {output_path}")


def save_measurements(config_path: Path, output_path: Path, record: dict) -> None:
    """Save confirmed measurements separately, leaving calibration configuration unchanged."""
    validate_output_path(config_path, output_path)
    settings = measured_settings(record)
    existing = DEMO_SETTINGS
    if "minimum_pregrasp_height_mm" in existing:
        minimum = float(existing["minimum_pregrasp_height_mm"])
        if not math.isfinite(minimum) or minimum <= 0 or minimum > settings["pregrasp_height_mm"]:
            raise DemoError("saved minimum pre-grasp height conflicts with the taught clearance")
        if "pregrasp_height_step_mm" in existing:
            step = float(existing["pregrasp_height_step_mm"])
            if not math.isfinite(step) or step <= 0 or step > settings["pregrasp_height_mm"] - minimum:
                raise DemoError("saved pre-grasp step conflicts with the taught clearance")
    write_json_atomic(output_path, {**record, "settings": settings})


def main() -> int:
    """Measure under physical teach control, disconnect safely, and save only on confirmation."""
    args = parse_args()
    arm = None
    try:
        thresholds = (
            args.sample_interval, args.maximum_joint_stddev_deg,
            args.maximum_position_stddev_mm,
        )
        if args.samples < 3 or any(not math.isfinite(value) or value <= 0 for value in thresholds):
            raise DemoError("use at least three samples and positive finite stability settings")
        validate_output_path(args.config, args.output)
        config = load_json(args.config)
        settings = DEMO_SETTINGS
        if not sys.stdin.isatty():
            raise DemoError("run in an interactive terminal for physical teach and power-off prompts")
        print(
            "READ-ONLY CAN: no enable, zero, replay, gripper, or hold commands.\n"
            "Clear the workspace, keep emergency stop accessible, and support the arm "
            "before using the physical Teach button. No other arm controller may run.",
            flush=True,
        )
        record = {
            "schema_version": 1, "frame": "Piper base; positions mm, rotations degrees",
            "config_path": str(args.config.resolve()), "tool_length_mm": C.TOOL_LEN_MM,
            "release_clearance_mm": RELEASE_CLEARANCE_MM,
            "sampling": {
                "sample_count": args.samples, "interval_seconds": args.sample_interval,
                "maximum_joint_stddev_deg": args.maximum_joint_stddev_deg,
                "maximum_position_stddev_mm": args.maximum_position_stddev_mm,
            },
        }
        if isinstance(config.get("camera"), dict) and "serial" in config["camera"]:
            record["camera_serial"] = config["camera"]["serial"]
        try:
            arm = connect_readonly(iface=args.can_interface)
            input("Support the arm, enter PHYSICAL TEACH mode, then press Enter: ")
            require_active_teach(arm)
            record["workspaces"] = {
                label: capture_workspace(arm, args, label) for label in ("candy", "hand")
            }
            record["delivery_pose"] = capture_delivery_orientation(arm, args)
        finally:
            if arm is not None:
                try:
                    confirm_power_off()
                finally:
                    arm.disconnect()
        record["clearances"] = prompt_clearances(settings)
        record["captured_at"] = datetime.now(timezone.utc).isoformat()
        print("\nProposed demo settings:", measured_settings(record))
        if input("Save these measurements? [y/N]: ").strip().lower() not in ("y", "yes"):
            print("Measurements not saved; existing configuration is unchanged.")
            return 0
        save_measurements(args.config, args.output, record)
        print(f"Teaching record: {args.output}\nCalibration configuration unchanged.")
        print("Review measurements before updating the workspace variables or DEMO_SETTINGS in Demo-Ignite/demo.py.")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted; no unconfirmed measurements saved.", file=sys.stderr)
        return 130
    except (RuntimeError, OSError, EOFError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: workspace teaching failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
