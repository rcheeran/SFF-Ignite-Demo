#!/usr/bin/env python3
"""Teach an absolute Piper-base tool-tip Z for safe object release."""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
HOMOGRAPHY_DIR = PROJECT_DIR / "prepare-piper" / "homography"
HAND_EYE_DIR = PROJECT_DIR / "prepare-piper" / "hand-eye-realsense405"
PIPER_DIR = PROJECT_DIR / "Demo-Ignite"
sys.path.insert(0, str(HOMOGRAPHY_DIR))
sys.path.insert(0, str(HAND_EYE_DIR))
sys.path.insert(0, str(PIPER_DIR))
sys.path.insert(0, str(PROJECT_DIR / "Demo-Product-Day"))

from common import CalibrationError, load_json, write_json_atomic  # noqa: E402
from demo import (  # noqa: E402
    DEFAULT_CONFIG,
    DemoError,
    validate_demo_config,
)
from piper.robot import Piper, connect_for_teach  # noqa: E402
from piper.transforms import tip_position  # noqa: E402
from teach_scan_poses import collect_stable_pose, move_to_sdk_zero  # noqa: E402
from verify_depth_coordinates import hold_until_power_off  # noqa: E402

DEFAULT_OUTPUT = HAND_EYE_DIR / "calibration" / "release_height.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--can-interface", default="can_piper")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument("--maximum-joint-stddev-deg", type=float, default=0.25)
    parser.add_argument("--maximum-position-stddev-mm", type=float, default=0.75)
    parser.add_argument("--zero-timeout", type=float, default=12.0)
    return parser.parse_args()


def save_taught_height(
    config_path: Path,
    output_path: Path,
    config: dict,
    record: dict,
) -> None:
    updated = dict(config)
    updated["demo"] = dict(config["demo"])
    updated["demo"]["hand_release_tip_z_mm"] = float(
        record["saved_release_tip_z_mm"]
    )
    validate_demo_config(updated)
    write_json_atomic(config_path, updated)
    write_json_atomic(output_path, record)


def main() -> int:
    args = parse_args()
    if (
        args.samples < 3
        or args.sample_interval <= 0
        or args.maximum_joint_stddev_deg <= 0
        or args.maximum_position_stddev_mm <= 0
        or args.zero_timeout <= 0
    ):
        print(
            "ERROR: sample count must be >= 3 and limits must be positive",
            file=sys.stderr,
        )
        return 2

    try:
        config = load_json(args.config)
        validate_demo_config(config)
    except (CalibrationError, DemoError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: cannot start release-height teaching: {exc}", file=sys.stderr)
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

        while record is None:
            input(
                "\nMove the empty gripper tool tip to the desired SAFE absolute "
                "release height, hold the arm still, then press Enter to "
                "capture: "
            )
            try:
                pose = collect_stable_pose(
                    piper,
                    args.samples,
                    args.sample_interval,
                    args.maximum_joint_stddev_deg,
                    args.maximum_position_stddev_mm,
                )
                joints = np.asarray(pose["joints_deg"], dtype=np.float64)
                tool_tip = tip_position(joints)
                measured_tip_z = float(tool_tip[2])
                saved_tip_z = float(math.ceil(measured_tip_z))
                record = {
                    "schema_version": 3,
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "frame": "Piper base; positions and distances in millimetres",
                    "tool_tip_base_point_mm": tool_tip.tolist(),
                    "measured_release_tip_z_mm": measured_tip_z,
                    "saved_release_tip_z_mm": saved_tip_z,
                    "joints_deg": joints.tolist(),
                    "end_pose_mm_deg": pose["end_pose_mm_deg"],
                    "joint_stddev_deg": pose["joint_stddev_deg"],
                    "position_stddev_mm": pose["position_stddev_mm"],
                    "sample_count": pose["sample_count"],
                    "config_path": str(args.config),
                }
                print(
                    "\nACCEPTED release height:"
                    f"\n  tool tip: {np.round(tool_tip, 1).tolist()}mm"
                    f"\n  measured base Z: {measured_tip_z:.1f}mm"
                    f"\n  saved base Z: {saved_tip_z:.1f}mm"
                )
            except (DemoError, RuntimeError, ValueError) as exc:
                print(f"REJECTED release-height capture: {exc}")

        print("\nKEEP HOLDING THE ARM.")
        print("Press the PHYSICAL TEACH BUTTON again; the arm should stiffen.")
        if not piper.wait_teach_stopped():
            raise RuntimeError("teach recording did not stop")
        hold_joints = piper.joints_deg()
        input("Arm stiffened. Press Enter to restore stable CAN hold and save: ")
        if not piper.teach_stop(hold_joints=hold_joints):
            raise RuntimeError("Piper did not return to stable CAN control")

        save_taught_height(args.config, args.output, config, record)
        print(
            f"\nSaved release tool-tip base Z "
            f"{record['saved_release_tip_z_mm']:.1f}mm to {args.config}"
        )
        print(f"Teaching record written to {args.output}")
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return_code = 130
    except (CalibrationError, DemoError, OSError, RuntimeError) as exc:
        print(f"ERROR: release-height teaching failed: {exc}", file=sys.stderr)
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
