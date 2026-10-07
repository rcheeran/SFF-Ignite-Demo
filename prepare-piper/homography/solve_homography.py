#!/usr/bin/env python3
"""Solve and quality-check the fixed-pose image-to-Piper homography.

Purpose:
    Pair each averaged ArUco pixel center (u, v) with its manually measured
    Piper base coordinate (X, Y), solve a planar 3x3 homography, and calculate
    per-marker fit errors. This script uses saved JSON only and has no camera
    or robot access.

Expected output:
    calibration/homography.json
        The 3x3 pixel-to-Piper-XY matrix, mean table Z, camera/scan metadata,
        inlier flags, per-marker errors, RMSE, maximum error, and pass/fail
        acceptance thresholds. The file is written only when all fit criteria
        pass unless --allow-failed-quality is supplied.

Example:
    python3 prepare-piper/homography/solve_homography.py
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
    marker_map,
    require_number,
    solve_homography,
    transform_pixels,
    write_json_atomic,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.json"
DEFAULT_PIXELS = SCRIPT_DIR / "calibration" / "marker_pixels.json"
DEFAULT_ROBOT = SCRIPT_DIR / "calibration" / "robot_points.json"
DEFAULT_SCAN_POSE = SCRIPT_DIR / "calibration" / "scan_pose.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "calibration" / "homography.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pixels", type=Path, default=DEFAULT_PIXELS)
    parser.add_argument("--robot-points", type=Path, default=DEFAULT_ROBOT)
    parser.add_argument("--scan-pose", type=Path, default=DEFAULT_SCAN_POSE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-failed-quality", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_json(args.config)
        pixels_document = load_json(args.pixels)
        robot_document = load_json(args.robot_points)
        scan_pose = load_json(args.scan_pose)
        pixel_markers = marker_map(pixels_document, str(args.pixels))
        robot_markers = marker_map(robot_document, str(args.robot_points))
        marker_ids = sorted(set(pixel_markers) & set(robot_markers))
        if len(marker_ids) < 4:
            raise CalibrationError("fewer than four marker IDs exist in both input files")
        missing_robot = sorted(set(pixel_markers) - set(robot_markers))
        if missing_robot:
            raise CalibrationError(f"robot points are missing marker IDs {missing_robot}")

        pixel_points = np.array(
            [
                [
                    require_number(pixel_markers[marker_id]["center"], "u", f"marker {marker_id}"),
                    require_number(pixel_markers[marker_id]["center"], "v", f"marker {marker_id}"),
                ]
                for marker_id in marker_ids
            ]
        )
        robot_points = np.array(
            [
                [
                    require_number(
                        robot_markers[marker_id]["mean_mm_deg"], "x", f"marker {marker_id}"
                    ),
                    require_number(
                        robot_markers[marker_id]["mean_mm_deg"], "y", f"marker {marker_id}"
                    ),
                ]
                for marker_id in marker_ids
            ]
        )
        table_z_values = np.array(
            [
                require_number(
                    robot_markers[marker_id]["mean_mm_deg"], "z", f"marker {marker_id}"
                )
                for marker_id in marker_ids
            ]
        )
        acceptance = config["acceptance"]
        ransac_threshold = float(acceptance["ransac_threshold_mm"])
        maximum_table_z_range = float(acceptance["maximum_table_z_range_mm"])
        maximum_rmse = float(acceptance["maximum_fit_rmse_mm"])
        maximum_error = float(acceptance["maximum_fit_error_mm"])
        homography, inliers = solve_homography(
            pixel_points, robot_points, ransac_threshold
        )
    except (KeyError, TypeError, ValueError, CalibrationError) as exc:
        print(f"ERROR: invalid calibration input: {exc}", file=sys.stderr)
        return 2

    predicted = transform_pixels(homography, pixel_points)
    metrics = error_metrics(predicted, robot_points)
    per_marker = []
    for index, marker_id in enumerate(marker_ids):
        error = float(np.linalg.norm(predicted[index] - robot_points[index]))
        per_marker.append(
            {
                "marker_id": marker_id,
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
                "ransac_inlier": bool(inliers[index]),
            }
        )

    quality_passed = (
        metrics["rmse_mm"] <= maximum_rmse
        and metrics["max_error_mm"] <= maximum_error
        and float(np.ptp(table_z_values)) <= maximum_table_z_range
        and bool(np.all(inliers))
    )
    print(f"Fit RMSE: {metrics['rmse_mm']:.3f} mm (limit {maximum_rmse:.3f})")
    print(
        f"Maximum error: {metrics['max_error_mm']:.3f} mm "
        f"(limit {maximum_error:.3f})"
    )
    print(
        f"Table Z range: {np.ptp(table_z_values):.3f} mm "
        f"(limit {maximum_table_z_range:.3f})"
    )
    for row in per_marker:
        print(
            f"  Marker {row['marker_id']}: error={row['error_mm']:.3f} mm, "
            f"inlier={row['ransac_inlier']}"
        )
    if not quality_passed and not args.allow_failed_quality:
        print(
            "ERROR: fit failed acceptance criteria; homography was not written",
            file=sys.stderr,
        )
        return 1

    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mapping": "[u, v, 1] image pixels to [X, Y, 1] Piper base millimetres",
        "homography": homography.tolist(),
        "table_z_mm": float(np.mean(table_z_values)),
        "table_z_range_mm": float(np.ptp(table_z_values)),
        "camera": pixels_document.get("camera"),
        "image": pixels_document.get("image"),
        "aruco_dictionary": pixels_document.get("aruco_dictionary"),
        "scan_pose": scan_pose,
        "fit_metrics": metrics,
        "acceptance": {
            "maximum_fit_rmse_mm": maximum_rmse,
            "maximum_fit_error_mm": maximum_error,
            "maximum_table_z_range_mm": maximum_table_z_range,
            "quality_passed": quality_passed,
        },
        "points": per_marker,
    }
    write_json_atomic(args.output, result)
    print(f"Homography written to {args.output}")
    if not quality_passed:
        print("WARNING: output was written with failed quality by explicit request")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
