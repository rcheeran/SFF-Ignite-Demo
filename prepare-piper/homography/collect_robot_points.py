#!/usr/bin/env python3
"""Collect Piper base coordinates at each physical ArUco marker center.

Purpose:
    Guide the Piper by hand in teach/drag mode so the chosen tool contact point
    touches each marker center, then press Enter to record repeated Cartesian
    telemetry samples. This script only reads Piper telemetry. It does not
    enable the arm, enter/exit teach mode, or send any motion command.

Expected output:
    calibration/robot_points.json
        For every marker found by capture_markers.py: averaged X/Y/Z and
        RX/RY/RZ, per-axis standard deviations, and all raw samples. The solver
        pairs these X/Y values with marker pixel centers.

Example:
    python3 prepare-piper/homography/collect_robot_points.py
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from piper_sdk import C_PiperInterface_V2

from common import (
    CalibrationError,
    load_json,
    marker_map,
    require_number,
    write_json_atomic,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.json"
DEFAULT_PIXELS = SCRIPT_DIR / "calibration" / "marker_pixels.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "calibration" / "robot_points.json"
POSE_KEYS = ("x", "y", "z", "rx", "ry", "rz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pixels", type=Path, default=DEFAULT_PIXELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--can-interface", default="can_piper")
    return parser.parse_args()


def read_end_pose(piper: C_PiperInterface_V2) -> np.ndarray:
    message = piper.GetArmEndPoseMsgs()
    if getattr(message, "Hz", 0) <= 0:
        raise RuntimeError("Piper end-pose telemetry is unavailable")
    pose = message.end_pose
    return np.array(
        [
            pose.X_axis / 1000.0,
            pose.Y_axis / 1000.0,
            pose.Z_axis / 1000.0,
            pose.RX_axis / 1000.0,
            pose.RY_axis / 1000.0,
            pose.RZ_axis / 1000.0,
        ],
        dtype=np.float64,
    )


def collect_samples(
    piper: C_PiperInterface_V2, count: int, interval: float
) -> np.ndarray:
    samples = []
    for _ in range(count):
        samples.append(read_end_pose(piper))
        time.sleep(interval)
    return np.stack(samples)


def main() -> int:
    args = parse_args()
    try:
        config = load_json(args.config)
        pixels = load_json(args.pixels)
        pixel_markers = marker_map(pixels, str(args.pixels))
        required_ids = [int(value) for value in pixels["required_marker_ids"]]
        collection = config["collection"]
        sample_count = int(collection["pose_samples_per_marker"])
        interval = float(collection["sample_interval_seconds"])
        maximum_stddev = float(collection["maximum_position_stddev_mm"])
        if sample_count < 2 or interval <= 0 or maximum_stddev <= 0:
            raise CalibrationError("collection settings must be positive")
    except (KeyError, TypeError, ValueError, CalibrationError) as exc:
        print(f"ERROR: invalid input: {exc}", file=sys.stderr)
        return 2

    missing = sorted(set(required_ids) - set(pixel_markers))
    if missing:
        print(f"ERROR: pixel file is missing markers {missing}", file=sys.stderr)
        return 2

    piper = C_PiperInterface_V2(args.can_interface)
    piper.ConnectPort()
    time.sleep(1.0)
    print("READ-ONLY: this script sends no robot or mode commands.")
    print("Put the Piper in teach/drag mode using the approved physical procedure.")
    print("Use the same precise point between the gripper tips for every contact.\n")

    marker_results: dict[str, dict] = {}
    for marker_id in required_ids:
        center = pixel_markers[marker_id]["center"]
        center_u = require_number(center, "u", f"marker {marker_id} center")
        center_v = require_number(center, "v", f"marker {marker_id} center")
        while True:
            input(
                f"Touch the tool point to marker ID {marker_id} center "
                f"(pixel {center_u:.2f}, {center_v:.2f}); then press Enter: "
            )
            try:
                samples = collect_samples(piper, sample_count, interval)
            except RuntimeError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1
            mean = samples.mean(axis=0)
            stddev = samples.std(axis=0)
            position_stddev = float(np.max(stddev[:3]))
            print(
                f"  Mean X={mean[0]:.2f} Y={mean[1]:.2f} Z={mean[2]:.2f} mm; "
                f"max position stddev={position_stddev:.3f} mm"
            )
            if position_stddev <= maximum_stddev:
                break
            print(
                f"  Unstable measurement: required <= {maximum_stddev:.3f} mm. "
                "Hold the arm still and retry."
            )

        marker_results[str(marker_id)] = {
            "pixel_center": {
                "u": center_u,
                "v": center_v,
            },
            "mean_mm_deg": {
                key: float(mean[index]) for index, key in enumerate(POSE_KEYS)
            },
            "stddev_mm_deg": {
                key: float(stddev[index]) for index, key in enumerate(POSE_KEYS)
            },
            "samples_mm_deg": [
                {key: float(sample[index]) for index, key in enumerate(POSE_KEYS)}
                for sample in samples
            ],
        }

    result = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "can_interface": args.can_interface,
        "tool_contact_point": "physical point centered between the closed gripper tips",
        "sample_count_per_marker": sample_count,
        "source_marker_pixels": str(args.pixels),
        "markers": marker_results,
    }
    write_json_atomic(args.output, result)
    print(f"\nRobot marker points written to {args.output}")
    print("Exit teach/drag mode using the approved procedure before any commanded motion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
