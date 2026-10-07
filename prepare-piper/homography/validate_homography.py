#!/usr/bin/env python3
"""Validate a homography with independent board points not used for fitting.

Purpose:
    Transform independently measured image pixels into Piper base X/Y and
    compare those predictions with independently touched robot coordinates.
    This script uses saved JSON only and has no camera or robot access.

Expected output:
    calibration/validation_report.json
        Per-point predicted/measured coordinates and errors, aggregate mean,
        RMSE and maximum error, configured limits, and an overall pass/fail.
    The process exits 0 when all criteria pass and 1 when calibration quality
    is inadequate.

Example:
    python3 prepare-piper/homography/validate_homography.py \
        --points prepare-piper/homography/calibration/validation_points.json
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from common import (
    CalibrationError,
    error_metrics,
    load_json,
    require_number,
    transform_pixels,
    write_json_atomic,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.json"
DEFAULT_HOMOGRAPHY = SCRIPT_DIR / "calibration" / "homography.json"
DEFAULT_POINTS = SCRIPT_DIR / "calibration" / "validation_points.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "calibration" / "validation_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--homography", type=Path, default=DEFAULT_HOMOGRAPHY)
    parser.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_json(args.config)
        calibration = load_json(args.homography)
        validation = load_json(args.points)
        homography = np.asarray(calibration["homography"], dtype=np.float64)
        if homography.shape != (3, 3) or not np.all(np.isfinite(homography)):
            raise CalibrationError("homography must be a finite 3x3 matrix")
        raw_points = validation.get("points")
        if not isinstance(raw_points, list) or len(raw_points) < 3:
            raise CalibrationError("at least three independent validation points are required")
        pixel_points = np.array(
            [
                [
                    require_number(point["pixel"], "u", f"point {index} pixel"),
                    require_number(point["pixel"], "v", f"point {index} pixel"),
                ]
                for index, point in enumerate(raw_points)
            ]
        )
        robot_points = np.array(
            [
                [
                    require_number(point["robot_mm"], "x", f"point {index} robot_mm"),
                    require_number(point["robot_mm"], "y", f"point {index} robot_mm"),
                ]
                for index, point in enumerate(raw_points)
            ]
        )
        acceptance = config["acceptance"]
        maximum_rmse = float(acceptance["maximum_validation_rmse_mm"])
        maximum_error = float(acceptance["maximum_validation_error_mm"])
    except (KeyError, TypeError, ValueError, CalibrationError) as exc:
        print(f"ERROR: invalid validation input: {exc}", file=sys.stderr)
        return 2

    predicted = transform_pixels(homography, pixel_points)
    metrics = error_metrics(predicted, robot_points)
    quality_passed = (
        metrics["rmse_mm"] <= maximum_rmse
        and metrics["max_error_mm"] <= maximum_error
    )
    rows = []
    for index, point in enumerate(raw_points):
        error = float(np.linalg.norm(predicted[index] - robot_points[index]))
        row = {
            "name": str(point.get("name", f"point-{index + 1}")),
            "pixel": {"u": float(pixel_points[index, 0]), "v": float(pixel_points[index, 1])},
            "measured_robot_mm": {
                "x": float(robot_points[index, 0]),
                "y": float(robot_points[index, 1]),
            },
            "projected_robot_mm": {
                "x": float(predicted[index, 0]),
                "y": float(predicted[index, 1]),
            },
            "error_mm": error,
        }
        rows.append(row)
        print(f"{row['name']}: error={error:.3f} mm")

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_homography": str(args.homography),
        "metrics": metrics,
        "acceptance": {
            "maximum_validation_rmse_mm": maximum_rmse,
            "maximum_validation_error_mm": maximum_error,
            "quality_passed": quality_passed,
        },
        "points": rows,
    }
    write_json_atomic(args.output, report)
    print(f"Validation RMSE: {metrics['rmse_mm']:.3f} mm")
    print(f"Maximum error: {metrics['max_error_mm']:.3f} mm")
    print(f"Result: {'PASS' if quality_passed else 'FAIL'}")
    print(f"Validation report written to {args.output}")
    return 0 if quality_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
