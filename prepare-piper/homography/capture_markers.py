#!/usr/bin/env python3
"""Capture stable ArUco marker pixels from the wrist-mounted RealSense D405.

Purpose:
    With the Piper held at its fixed calibration/scanning pose, collect many
    color frames, refine each ArUco corner to subpixel precision, and average
    the observed corners and centers. This script uses only the camera; it does
    not connect to or command the robot.

Expected output:
    calibration/marker_pixels.json
        Camera identity, intrinsics, image settings, marker centers/corners,
        observation counts, and pixel stability statistics.
    calibration/marker_capture.jpg
        The final color frame annotated with detected marker IDs and averaged
        center positions for visual inspection.

Example:
    python3 prepare-piper/homography/capture_markers.py
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from common import CalibrationError, load_json, parse_marker_ids, write_json_atomic

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "calibration" / "marker_pixels.json"
DEFAULT_IMAGE = SCRIPT_DIR / "calibration" / "marker_capture.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    return parser.parse_args()


def camera_info(profile: object) -> dict:
    import pyrealsense2 as rs

    device = profile.get_device()
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_profile.get_intrinsics()
    return {
        "name": device.get_info(rs.camera_info.name),
        "serial": device.get_info(rs.camera_info.serial_number),
        "firmware": device.get_info(rs.camera_info.firmware_version),
        "intrinsics": {
            "width": intrinsics.width,
            "height": intrinsics.height,
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "ppx": intrinsics.ppx,
            "ppy": intrinsics.ppy,
            "distortion_model": str(intrinsics.model),
            "distortion_coefficients": list(intrinsics.coeffs),
        },
    }


def main() -> int:
    args = parse_args()
    try:
        config = load_json(args.config)
        camera_config = config["camera"]
        aruco_config = config["aruco"]
        required_ids = parse_marker_ids(aruco_config["required_marker_ids"])
        dictionary_name = str(aruco_config["dictionary"])
        dictionary_id = getattr(cv2.aruco, dictionary_name, None)
        if dictionary_id is None:
            raise CalibrationError(f"unknown OpenCV ArUco dictionary: {dictionary_name}")
        width = int(camera_config["width"])
        height = int(camera_config["height"])
        fps = int(camera_config["fps"])
        warmup_frames = int(camera_config["warmup_frames"])
        sample_frames = int(camera_config["sample_frames"])
        minimum_observations = int(aruco_config["minimum_observations_per_marker"])
        maximum_stddev = float(aruco_config["maximum_center_stddev_px"])
    except (KeyError, TypeError, ValueError, CalibrationError) as exc:
        print(f"ERROR: invalid configuration: {exc}", file=sys.stderr)
        return 2

    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    rs_config = rs.config()
    configured_serial = camera_config.get("serial")
    if configured_serial:
        rs_config.enable_device(str(configured_serial))
    rs_config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    observations: dict[int, list[np.ndarray]] = defaultdict(list)
    last_frame: np.ndarray | None = None
    last_corners: list[np.ndarray] = []
    last_ids: np.ndarray | None = None
    pipeline_started = False

    print("Starting RealSense color stream. Keep the Piper at the fixed scan pose.")
    try:
        profile = pipeline.start(rs_config)
        pipeline_started = True
        detected_camera = camera_info(profile)
        if configured_serial and detected_camera["serial"] != str(configured_serial):
            raise CalibrationError(
                f"expected camera {configured_serial}, got {detected_camera['serial']}"
            )

        for _ in range(warmup_frames):
            pipeline.wait_for_frames(timeout_ms=5000)

        for frame_index in range(sample_frames):
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data()).copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            last_frame = frame
            last_corners = corners
            last_ids = ids
            if ids is not None:
                for marker_corners, raw_id in zip(corners, ids.reshape(-1), strict=True):
                    marker_id = int(raw_id)
                    if marker_id not in required_ids:
                        continue
                    refined = (
                        marker_corners.reshape(4, 2).astype(np.float32).reshape(-1, 1, 2)
                    )
                    cv2.cornerSubPix(
                        gray,
                        refined,
                        (5, 5),
                        (-1, -1),
                        (
                            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                            30,
                            0.01,
                        ),
                    )
                    observations[marker_id].append(
                        refined.reshape(4, 2).astype(np.float64)
                    )
            counts = ", ".join(
                f"{marker_id}:{len(observations[marker_id])}" for marker_id in required_ids
            )
            print(f"\rFrame {frame_index + 1}/{sample_frames}; observations {counts}", end="")
        print()
    except (RuntimeError, CalibrationError) as exc:
        print(f"\nERROR: camera capture failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if pipeline_started:
            pipeline.stop()

    if last_frame is None:
        print("ERROR: no color frame was received", file=sys.stderr)
        return 1

    failures: list[str] = []
    marker_results: dict[str, dict] = {}
    averaged_centers: dict[int, np.ndarray] = {}
    for marker_id in required_ids:
        samples = observations[marker_id]
        if len(samples) < minimum_observations:
            failures.append(
                f"marker {marker_id}: {len(samples)} observations, need {minimum_observations}"
            )
            continue
        corner_samples = np.stack(samples)
        center_samples = corner_samples.mean(axis=1)
        mean_corners = corner_samples.mean(axis=0)
        mean_center = center_samples.mean(axis=0)
        center_stddev = center_samples.std(axis=0)
        maximum_axis_stddev = float(np.max(center_stddev))
        if maximum_axis_stddev > maximum_stddev:
            failures.append(
                f"marker {marker_id}: center stddev {maximum_axis_stddev:.3f}px "
                f"exceeds {maximum_stddev:.3f}px"
            )
        averaged_centers[marker_id] = mean_center
        marker_results[str(marker_id)] = {
            "center": {"u": float(mean_center[0]), "v": float(mean_center[1])},
            "corners": [
                {"u": float(corner[0]), "v": float(corner[1])}
                for corner in mean_corners
            ],
            "observation_count": len(samples),
            "center_stddev_px": {
                "u": float(center_stddev[0]),
                "v": float(center_stddev[1]),
            },
        }

    if failures:
        print("ERROR: marker capture did not meet acceptance criteria:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    if last_ids is not None:
        cv2.aruco.drawDetectedMarkers(last_frame, last_corners, last_ids)
    for marker_id, center in averaged_centers.items():
        point = tuple(np.rint(center).astype(int))
        cv2.circle(last_frame, point, 5, (0, 0, 255), -1)
        cv2.putText(
            last_frame,
            f"ID {marker_id} ({center[0]:.1f}, {center[1]:.1f})",
            (point[0] + 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    args.image.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.image), last_frame):
        print(f"ERROR: could not write annotated image: {args.image}", file=sys.stderr)
        return 1
    result = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "capture_duration_seconds": sample_frames / fps,
        "image": {"width": width, "height": height, "fps": fps},
        "camera": detected_camera,
        "aruco_dictionary": dictionary_name,
        "required_marker_ids": required_ids,
        "markers": marker_results,
        "annotated_image": str(args.image),
    }
    write_json_atomic(args.output, result)
    print(f"Marker pixels written to {args.output}")
    print(f"Annotated capture written to {args.image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
