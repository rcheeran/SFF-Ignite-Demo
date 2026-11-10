#!/usr/bin/env python3
"""Teach the Piper candy grasp height and interactive gripper widths."""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
HOMOGRAPHY_DIR = PROJECT_DIR / "prepare-piper" / "homography"
HAND_EYE_DIR = PROJECT_DIR / "prepare-piper" / "hand-eye-realsense405"
PIPER_DIR = PROJECT_DIR / "Demo-VLA-On-Piper" / "piper-openpi-V2"
sys.path.insert(0, str(HOMOGRAPHY_DIR))
sys.path.insert(0, str(HAND_EYE_DIR))
sys.path.insert(0, str(PIPER_DIR))

from common import CalibrationError, load_json, write_json_atomic  # noqa: E402
from piper import config as PIPER_C  # noqa: E402
from piper.robot import Piper, connect_for_teach  # noqa: E402
from piper.transforms import tip_position  # noqa: E402
from teach_scan_poses import (  # noqa: E402
    collect_stable_pose,
    move_to_sdk_zero,
    validate_replayable_pose,
)
from verify_depth_coordinates import hold_until_power_off  # noqa: E402

CAMERA_DIR = PROJECT_DIR / "prepare-piper" / "hand-eye-realsense435"
DEFAULT_CONFIG = CAMERA_DIR / "config.json"
DEFAULT_OUTPUT = CAMERA_DIR / "calibration" / "candy_grip.json"


class DemoError(RuntimeError):
    """Invalid candy-grip teaching settings or measurements."""


def parse_args() -> argparse.Namespace:
    """Parse teaching, stability, and CAN interface settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--can-interface", default="can_piper")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument("--maximum-joint-stddev-deg", type=float, default=0.25)
    parser.add_argument("--maximum-position-stddev-mm", type=float, default=0.75)
    parser.add_argument("--zero-timeout", type=float, default=12.0)
    parser.add_argument("--gripper-settle-seconds", type=float, default=1.5)
    return parser.parse_args()


def validate_gripper_widths(open_mm: float, close_mm: float) -> None:
    """Require commandable widths with meaningful clearance between them."""
    values = np.asarray([open_mm, close_mm], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise DemoError("gripper widths must be finite")
    if np.any(values < PIPER_C.GRIP_CLOSED_MM) or np.any(
        values > PIPER_C.GRIP_OPEN_MM
    ):
        raise DemoError(
            f"gripper widths must stay within "
            f"{PIPER_C.GRIP_CLOSED_MM:.1f}..{PIPER_C.GRIP_OPEN_MM:.1f}mm"
        )
    if open_mm - close_mm < 5.0:
        raise DemoError("open width must be at least 5mm greater than close width")


def initial_grip_settings(record: dict) -> dict:
    """Use the last teaching record for jaw prompts, not calibration configuration."""
    open_mm = float(record.get("grip_open_command_mm", PIPER_C.GRIP_OPEN_MM))
    close_mm = float(record.get("grip_close_command_mm", PIPER_C.GRIP_CLOSED_MM))
    validate_gripper_widths(open_mm, close_mm)
    return {"grip_open_mm": open_mm, "grip_close_mm": close_mm}


def save_taught_grip(
    config_path: Path,
    output_path: Path,
    record: dict,
) -> None:
    """Save confirmed grip measurements without modifying calibration configuration."""
    if config_path.resolve() == output_path.resolve():
        raise DemoError("configuration and teaching record paths must differ")
    grasp_z = float(record["grasp_tip_z_mm"])
    open_mm = float(record["grip_open_command_mm"])
    close_mm = float(record["grip_close_command_mm"])
    if not math.isfinite(grasp_z):
        raise DemoError("grasp tool-tip Z must be finite")
    if grasp_z < PIPER_C.TIP_Z_FLOOR_MM:
        raise DemoError("grasp tool-tip Z must not be below the safety floor")
    validate_gripper_widths(open_mm, close_mm)
    write_json_atomic(output_path, record)


def command_gripper_at_hold(
    piper: Piper,
    hold_joints: np.ndarray,
    width_mm: float,
    settle_seconds: float,
) -> float:
    """Hold the taught arm pose while commanding and measuring a gripper width."""
    deadline = time.monotonic() + settle_seconds
    next_tick = time.monotonic()
    while time.monotonic() < deadline:
        faults = piper.faults()
        if faults:
            raise RuntimeError(f"Piper faults while testing gripper: {faults}")
        if not all(piper.enabled()):
            raise RuntimeError("one or more Piper joints became disabled")
        piper.send_joints(hold_joints, width_mm, speed=PIPER_C.SPEED_REST)
        next_tick += 1.0 / PIPER_C.CONTROL_HZ
        time.sleep(max(0.0, next_tick - time.monotonic()))
    return float(piper.grip_mm())


def prompt_width(
    piper: Piper,
    hold_joints: np.ndarray,
    label: str,
    default_mm: float,
    settle_seconds: float,
) -> tuple[float, float]:
    """Interactively command candidate widths until the operator accepts one."""
    while True:
        raw = input(
            f"\nEnter {label} gripper command in mm "
            f"[{default_mm:.1f}], or q to abort: "
        ).strip()
        if raw.lower() == "q":
            raise KeyboardInterrupt
        try:
            command_mm = default_mm if not raw else float(raw)
            if not math.isfinite(command_mm):
                raise ValueError("width must be finite")
            if not PIPER_C.GRIP_CLOSED_MM <= command_mm <= PIPER_C.GRIP_OPEN_MM:
                raise ValueError(
                    f"width must be within {PIPER_C.GRIP_CLOSED_MM:.1f}.."
                    f"{PIPER_C.GRIP_OPEN_MM:.1f}mm"
                )
        except ValueError as exc:
            print(f"REJECTED {label} width: {exc}")
            continue
        input(
            f"Keep clear of the jaws. Press Enter to command "
            f"{command_mm:.1f}mm: "
        )
        measured_mm = command_gripper_at_hold(
            piper,
            hold_joints,
            command_mm,
            settle_seconds,
        )
        print(
            f"{label.capitalize()} command={command_mm:.1f}mm, "
            f"measured={measured_mm:.1f}mm"
        )
        accepted = input("Accept this width? [y/N]: ").strip().lower()
        if accepted in ("y", "yes"):
            return command_mm, measured_mm
        default_mm = command_mm


def main() -> int:
    """Run physical pose teaching, interactive jaw tuning, and atomic saving."""
    args = parse_args()
    if (
        args.samples < 3
        or args.sample_interval <= 0
        or args.maximum_joint_stddev_deg <= 0
        or args.maximum_position_stddev_mm <= 0
        or args.zero_timeout <= 0
        or args.gripper_settle_seconds <= 0
    ):
        print(
            "ERROR: sample count must be >= 3 and limits must be positive",
            file=sys.stderr,
        )
        return 2
    try:
        load_json(args.config)
        grip_settings = initial_grip_settings(load_json(args.output) if args.output.exists() else {})
    except (CalibrationError, DemoError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: cannot start candy-grip teaching: {exc}", file=sys.stderr)
        return 2

    piper: Piper | None = None
    hold_joints: np.ndarray | None = None
    record: dict | None = None
    return_code = 0
    try:
        piper = connect_for_teach(iface=args.can_interface)
        print(
            "\nClear the path to the folded SDK zero pose and keep the "
            "emergency stop accessible."
        )
        hold_joints = move_to_sdk_zero(
            piper,
            args.zero_timeout,
            "PRE-TEACH ZERO",
        )
        print("\nSupport the arm and press the PHYSICAL TEACH BUTTON.")
        if not piper.wait_teach(True):
            raise RuntimeError("Piper did not enter teach mode")
        hold_joints = None
        print("TEACH MODE detected.")

        pose = None
        while pose is None:
            input(
                "\nPlace a Starburst at the pickup location. Move the OPEN "
                "gripper so the candy is centered between the jaws at the "
                "desired closing height and orientation. Hold still, then "
                "press Enter to capture: "
            )
            try:
                candidate = collect_stable_pose(
                    piper,
                    args.samples,
                    args.sample_interval,
                    args.maximum_joint_stddev_deg,
                    args.maximum_position_stddev_mm,
                )
                validate_replayable_pose(
                    candidate,
                    PIPER_C.IK_LIMIT_MARGIN_DEG,
                )
                joints = np.asarray(candidate["joints_deg"], dtype=np.float64)
                tool_tip = tip_position(joints)
                if float(tool_tip[2]) < PIPER_C.TIP_Z_FLOOR_MM:
                    raise RuntimeError(
                        f"tool-tip Z {tool_tip[2]:.1f}mm is below the "
                        f"{PIPER_C.TIP_Z_FLOOR_MM:.1f}mm safety floor"
                    )
                pose = candidate
                print(
                    "\nACCEPTED candy grasp pose:"
                    f"\n  tool tip: {np.round(tool_tip, 1).tolist()}mm"
                    f"\n  joints: {np.round(joints, 1).tolist()}deg"
                )
            except (RuntimeError, ValueError) as exc:
                print(f"REJECTED candy grasp pose: {exc}")

        print("\nKEEP HOLDING THE ARM.")
        print("Press the PHYSICAL TEACH BUTTON again; the arm should stiffen.")
        if not piper.wait_teach_stopped():
            raise RuntimeError("teach recording did not stop")
        hold_joints = np.asarray(pose["joints_deg"], dtype=np.float64)
        input("Arm stiffened. Press Enter to restore stable CAN hold: ")
        if not piper.teach_stop(hold_joints=hold_joints):
            raise RuntimeError("Piper did not return to stable CAN control")

        print(
            "\nThe arm will hold the taught pose while you test jaw widths. "
            "Keep fingers clear and use the emergency stop if motion is "
            "unexpected."
        )
        open_command, open_measured = prompt_width(
            piper,
            hold_joints,
            "open",
            float(grip_settings["grip_open_mm"]),
            args.gripper_settle_seconds,
        )
        close_command, close_measured = prompt_width(
            piper,
            hold_joints,
            "close",
            float(grip_settings["grip_close_mm"]),
            args.gripper_settle_seconds,
        )
        validate_gripper_widths(open_command, close_command)

        tool_tip = tip_position(hold_joints)
        record = {
            "schema_version": 1,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "frame": "Piper base; positions and distances in millimetres",
            "tool_tip_base_point_mm": tool_tip.tolist(),
            "grasp_tip_z_mm": float(tool_tip[2]),
            "grip_open_command_mm": open_command,
            "grip_open_measured_mm": open_measured,
            "grip_close_command_mm": close_command,
            "grip_close_measured_mm": close_measured,
            "joints_deg": hold_joints.tolist(),
            "end_pose_mm_deg": pose["end_pose_mm_deg"],
            "joint_stddev_deg": pose["joint_stddev_deg"],
            "position_stddev_mm": pose["position_stddev_mm"],
            "sample_count": pose["sample_count"],
            "config_path": str(args.config),
        }
        print(
            "\nTeaching result:"
            f"\n  grasp tool-tip Z: {record['grasp_tip_z_mm']:.1f}mm"
            f"\n  open command/measured: "
            f"{open_command:.1f}/{open_measured:.1f}mm"
            f"\n  close command/measured: "
            f"{close_command:.1f}/{close_measured:.1f}mm"
        )
        if input("Save these values? [y/N]: ").strip().lower() not in ("y", "yes"):
            raise KeyboardInterrupt
        reopened_mm = command_gripper_at_hold(
            piper,
            hold_joints,
            open_command,
            args.gripper_settle_seconds,
        )
        print(
            f"Gripper reopened to {reopened_mm:.1f}mm. Remove the candy while "
            "the arm remains held."
        )
        save_taught_grip(args.config, args.output, record)
        print("Calibration configuration unchanged. Review the saved values and update DEMO_SETTINGS in Demo-Ignite/demo.py.")
        print(f"Teaching record written to {args.output}")
    except KeyboardInterrupt:
        print("\nInterrupted; no unconfirmed values were saved.", file=sys.stderr)
        return_code = 130
    except (CalibrationError, DemoError, OSError, RuntimeError) as exc:
        print(f"ERROR: candy-grip teaching failed: {exc}", file=sys.stderr)
        return_code = 1
    finally:
        if piper is not None and hold_joints is None and piper.in_teach():
            print(
                "\nKEEP SUPPORTING THE ARM. Press the PHYSICAL TEACH BUTTON "
                "again; the arm should stiffen.",
                file=sys.stderr,
            )
            if piper.wait_teach_stopped():
                hold_joints = piper.joints_deg()
                if not piper.teach_stop(hold_joints=hold_joints):
                    print(
                        "SAFETY ERROR: CAN hold could not be restored. Keep "
                        "supporting the arm and turn controller power off.",
                        file=sys.stderr,
                    )
            else:
                print(
                    "SAFETY ERROR: teach recording did not stop. Keep supporting "
                    "the arm and turn controller power off.",
                    file=sys.stderr,
                )
        if piper is not None and hold_joints is not None and not piper.in_teach():
            hold_joints = piper.joints_deg()
            print(
                "\nKEEP SUPPORTING THE ARM. The current pose will remain "
                "streamed until controller power is OFF."
            )
            hold_until_power_off(piper, hold_joints)
        if piper is not None:
            piper.disconnect()
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
