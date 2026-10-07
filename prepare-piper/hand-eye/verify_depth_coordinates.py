#!/usr/bin/env python3
"""Verify D405 depth coordinates against the calibrated fixed ArUco marker.

Purpose:
    Optionally align the wrist in physical teach mode using a browser preview,
    then capture aligned D405 color/depth frames while Marker ID 1 and the Piper
    arm remain stationary. Deproject the marker-center pixel into a metric 3D
    camera point, transform it into the Piper base frame with the saved
    hand-eye matrix and live Piper pose, and compare it with the marker position
    saved by solve_hand_eye.py. Without alignment, verification uses read-only
    Piper telemetry. The optional alignment stage takes control back at the
    measured current joints and continuously streams that exact hold pose
    through measurement and safe controller power-off; it never commands a new
    target pose.

Expected output:
    calibration/depth_verification.json
        Camera/depth metadata, every accepted observation, stability metrics,
        base-coordinate error metrics, configured limits, and pass/fail.
    calibration/depth_verification.jpg
        The last accepted color frame annotated with the sampled pixel, depth,
        calculated base coordinate, and coordinate error.

Example:
    python3 prepare-piper/hand-eye/verify_depth_coordinates.py
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

HAND_EYE_DIR = Path(__file__).resolve().parent
HOMOGRAPHY_DIR = HAND_EYE_DIR.parent / "homography"
PIPER_APP_DIR = HAND_EYE_DIR.parents[1] / "piper-openpi-V2"
sys.path.insert(0, str(HOMOGRAPHY_DIR))
sys.path.insert(0, str(HAND_EYE_DIR))
sys.path.insert(0, str(PIPER_APP_DIR))

from common import CalibrationError, load_json, write_json_atomic  # noqa: E402
from hand_eye_common import (  # noqa: E402
    HandEyeError,
    average_transforms,
    piper_pose_to_transform,
    rotation_error_degrees,
)
from piper import config as PIPER_C  # noqa: E402
from piper.robot import Piper, connect_for_teach, connect_readonly  # noqa: E402
from record_scan_pose import FfmpegRealSensePreview  # noqa: E402

DEFAULT_CONFIG = HAND_EYE_DIR / "config.json"
DEFAULT_CALIBRATION = HAND_EYE_DIR / "calibration" / "hand_eye.json"
DEFAULT_DEPTH_CORRECTION = HAND_EYE_DIR / "calibration" / "depth_correction.json"
DEFAULT_OUTPUT = HAND_EYE_DIR / "calibration" / "depth_verification.json"
DEFAULT_IMAGE = HAND_EYE_DIR / "calibration" / "depth_verification.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument(
        "--depth-correction",
        type=Path,
        default=DEFAULT_DEPTH_CORRECTION,
    )
    parser.add_argument(
        "--no-depth-correction",
        action="store_true",
        help="ignore an existing depth correction for a raw-depth diagnostic",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--can-interface", default="can_piper")
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="seconds to wait for enough valid Marker-1 depth observations",
    )
    parser.add_argument(
        "--align-in-teach",
        action="store_true",
        help="show the browser preview and hand-guide the wrist before verification",
    )
    return parser.parse_args()


def rigid_transform(document: dict, key: str) -> np.ndarray:
    matrix = np.asarray(document.get(key), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise HandEyeError(f"{key} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8):
        raise HandEyeError(f"{key} is not a rigid transform")
    return matrix


def robust_center_depth_m(
    depth_raw: np.ndarray,
    center_uv: np.ndarray,
    depth_scale: float,
    radius_px: int,
    minimum_pixels: int,
    minimum_depth_m: float,
    maximum_depth_m: float,
) -> tuple[float, int, float]:
    """Return median center depth, valid count and median absolute deviation."""
    if depth_raw.ndim != 2:
        raise HandEyeError("depth image must be a 2D array")
    u, v = np.rint(np.asarray(center_uv, dtype=np.float64)).astype(int)
    height, width = depth_raw.shape
    left, right = max(0, u - radius_px), min(width, u + radius_px + 1)
    top, bottom = max(0, v - radius_px), min(height, v + radius_px + 1)
    values = depth_raw[top:bottom, left:right].astype(np.float64) * depth_scale
    valid = values[
        np.isfinite(values)
        & (values >= minimum_depth_m)
        & (values <= maximum_depth_m)
    ]
    if valid.size < minimum_pixels:
        raise HandEyeError(
            f"only {valid.size} valid depth pixels near marker center; "
            f"need at least {minimum_pixels}"
        )
    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    return median, int(valid.size), mad


def transform_camera_point_to_base(
    camera_point_mm: np.ndarray,
    gripper_to_base: np.ndarray,
    camera_to_gripper: np.ndarray,
) -> np.ndarray:
    point = np.asarray(camera_point_mm, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise HandEyeError("camera point must contain three finite values")
    homogeneous = np.append(point, 1.0)
    return (gripper_to_base @ camera_to_gripper @ homogeneous)[:3]


def marker_center(corners: np.ndarray) -> np.ndarray:
    points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    return np.mean(points, axis=0)


def camera_serial_from_calibration(calibration: dict) -> str | None:
    camera = calibration.get("camera")
    if not isinstance(camera, dict):
        return None
    serial = camera.get("serial")
    return None if serial is None else str(serial)


def load_depth_correction(
    path: Path, disabled: bool
) -> tuple[float, float, dict | None]:
    if disabled or not path.exists():
        return 1.0, 0.0, None
    document = load_json(path)
    acceptance = document.get("acceptance")
    if (
        not isinstance(acceptance, dict)
        or acceptance.get("quality_passed") is not True
    ):
        raise HandEyeError(
            f"depth correction has not passed quality checks: {path}"
        )
    scale = float(document["scale"])
    offset = float(document["offset_mm"])
    if not math.isfinite(scale) or not math.isfinite(offset) or scale <= 0:
        raise HandEyeError(f"invalid depth correction values in {path}")
    return scale, offset, document


def align_marker_in_teach(
    camera_config: dict,
    aruco_config: dict,
    marker_id: int,
    can_interface: str,
) -> tuple[FfmpegRealSensePreview, Piper, np.ndarray]:
    preview_config = {
        "dictionary": aruco_config["dictionary"],
        "required_marker_ids": [marker_id],
    }
    preview = FfmpegRealSensePreview(
        camera_config, preview_config, enable_depth=True
    )
    piper: Piper | None = None
    completed = False
    try:
        preview.start()
        piper = connect_for_teach(iface=can_interface)
        print(f"\nLive Marker-{marker_id} alignment view: {preview.url}")
        print("Hold the arm and press the PHYSICAL TEACH BUTTON.")
        if not piper.wait_teach(True):
            raise RuntimeError("Piper did not enter teach mode")
        preview.check()
        print("TEACH MODE detected.")
        input(
            f"Move the wrist until Marker {marker_id} is fully visible and "
            "the overlay says READY, let it settle, then press Enter: "
        )
        preview.wait_for_marker(marker_id, timeout=3.0)
        if not piper.in_teach():
            raise RuntimeError("Piper left teach mode before alignment was confirmed")
        print("\nKEEP HOLDING THE ARM.")
        print("Press the PHYSICAL TEACH BUTTON again; the arm should stiffen.")
        if not piper.wait_teach_stopped():
            raise RuntimeError(
                "teach recording did not stop; keep holding the arm and power-cycle safely"
            )
        hold_joints = piper.joints_deg()
        input("Arm stiffened. Press Enter to take CAN control back: ")
        if not piper.teach_stop(hold_joints=hold_joints):
            raise RuntimeError(
                "Piper did not leave teach mode and enable its motors; "
                "keep holding it and power-cycle safely"
            )
        print("Control taken back; holding torque has returned.")
        completed = True
        return preview, piper, hold_joints
    finally:
        if not completed and piper is not None:
            if piper.in_teach():
                print(
                    "\nKEEP HOLDING THE ARM and press the PHYSICAL TEACH BUTTON "
                    "again to stiffen it.",
                    file=sys.stderr,
                )
                if not piper.wait_teach_stopped() or not piper.teach_stop():
                    print(
                        "SAFETY ERROR: recovery failed; keep holding the arm and "
                        "power-cycle safely",
                        file=sys.stderr,
                    )
            piper.disconnect()
        if not completed:
            preview.close()


def finalize_verification(
    args: argparse.Namespace,
    observations: list[dict],
    robot_transforms: list[np.ndarray],
    last_annotated: np.ndarray,
    detected_serial: str,
    width: int,
    height: int,
    fps: int,
    depth_scale: float,
    correction_document: dict | None,
    depth_scale_correction: float,
    depth_offset_mm: float,
    dictionary_name: str,
    marker_id: int,
    reference_target_to_base: np.ndarray,
    maximum_coordinate_rmse: float,
    maximum_position_deviation: float,
    maximum_rotation_deviation: float,
    maximum_consensus_error: float,
    maximum_temporal_depth_mad: float,
    minimum_temporal_inlier_fraction: float,
    temporal_mad_multiplier: float,
    temporal_outlier_floor: float,
) -> int:
    robot_mean = average_transforms(robot_transforms)
    position_errors = np.asarray(
        [value["coordinate_error_mm"] for value in observations],
        dtype=np.float64,
    )
    robot_position_errors = np.asarray(
        [
            np.linalg.norm(value[:3, 3] - robot_mean[:3, 3])
            for value in robot_transforms
        ]
    )
    robot_rotation_errors = np.asarray(
        [rotation_error_degrees(robot_mean, value) for value in robot_transforms]
    )
    coordinate_rmse = float(math.sqrt(float(np.mean(np.square(position_errors)))))
    raw_depths_mm = np.asarray(
        [value["raw_depth_m"] * 1000.0 for value in observations],
        dtype=np.float64,
    )
    temporal_median_depth = float(np.median(raw_depths_mm))
    temporal_depth_mad = float(
        np.median(np.abs(raw_depths_mm - temporal_median_depth))
    )
    temporal_threshold = max(
        temporal_mad_multiplier * temporal_depth_mad,
        temporal_outlier_floor,
    )
    temporal_inliers = (
        np.abs(raw_depths_mm - temporal_median_depth) <= temporal_threshold
    )
    temporal_inlier_fraction = float(np.mean(temporal_inliers))
    base_points = np.asarray(
        [value["base_point_mm"] for value in observations],
        dtype=np.float64,
    )
    consensus_base_point = np.median(base_points[temporal_inliers], axis=0)
    consensus_error = float(
        np.linalg.norm(
            consensus_base_point - reference_target_to_base[:3, 3]
        )
    )
    inlier_errors = position_errors[temporal_inliers]
    inlier_coordinate_rmse = float(
        math.sqrt(float(np.mean(np.square(inlier_errors))))
    )
    for observation, is_inlier in zip(
        observations, temporal_inliers, strict=True
    ):
        observation["temporal_depth_inlier"] = bool(is_inlier)
    maximum_robot_position_deviation = float(np.max(robot_position_errors))
    maximum_robot_rotation_deviation = float(np.max(robot_rotation_errors))
    quality_passed = (
        coordinate_rmse <= maximum_coordinate_rmse
        and consensus_error <= maximum_consensus_error
        and temporal_depth_mad <= maximum_temporal_depth_mad
        and temporal_inlier_fraction >= minimum_temporal_inlier_fraction
        and maximum_robot_position_deviation <= maximum_position_deviation
        and maximum_robot_rotation_deviation <= maximum_rotation_deviation
    )
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "calibration": str(args.calibration),
        "camera": {
            "serial": detected_serial,
            "width": width,
            "height": height,
            "fps": fps,
            "depth_scale_m": depth_scale,
        },
        "depth_correction": {
            "applied": correction_document is not None,
            "source": (
                None if correction_document is None else str(args.depth_correction)
            ),
            "scale": depth_scale_correction,
            "offset_mm": depth_offset_mm,
        },
        "marker": {
            "dictionary": dictionary_name,
            "id": marker_id,
            "reference_target_to_base_mm": reference_target_to_base[:3, 3].tolist(),
        },
        "sample_count": len(observations),
        "coordinate_rmse_mm": coordinate_rmse,
        "inlier_coordinate_rmse_mm": inlier_coordinate_rmse,
        "mean_coordinate_error_mm": float(np.mean(position_errors)),
        "maximum_coordinate_error_mm": float(np.max(position_errors)),
        "temporal_consensus": {
            "raw_depth_median_mm": temporal_median_depth,
            "raw_depth_mad_mm": temporal_depth_mad,
            "outlier_threshold_mm": temporal_threshold,
            "inlier_count": int(np.sum(temporal_inliers)),
            "inlier_fraction": temporal_inlier_fraction,
            "base_point_mm": consensus_base_point.tolist(),
            "coordinate_error_mm": consensus_error,
        },
        "maximum_robot_position_deviation_mm": maximum_robot_position_deviation,
        "maximum_robot_rotation_deviation_deg": maximum_robot_rotation_deviation,
        "acceptance": {
            "maximum_coordinate_rmse_mm": maximum_coordinate_rmse,
            "maximum_consensus_error_mm": maximum_consensus_error,
            "maximum_temporal_depth_mad_mm": maximum_temporal_depth_mad,
            "minimum_temporal_inlier_fraction": minimum_temporal_inlier_fraction,
            "maximum_robot_position_deviation_mm": maximum_position_deviation,
            "maximum_robot_rotation_deviation_deg": maximum_rotation_deviation,
            "quality_passed": quality_passed,
        },
        "observations": observations,
    }
    write_json_atomic(args.output, result)
    args.image.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.image), last_annotated):
        raise RuntimeError(f"could not write annotated image {args.image}")
    print(
        f"\nDepth coordinate verification {'PASSED' if quality_passed else 'FAILED'}: "
        f"consensus={consensus_error:.2f}mm, "
        f"inliers={np.sum(temporal_inliers)}/{len(observations)}, "
        f"depth MAD={temporal_depth_mad:.2f}mm "
        f"(all-frame RMSE={coordinate_rmse:.2f}mm, "
        f"max={np.max(position_errors):.2f}mm)"
    )
    print(f"Report: {args.output}")
    print(f"Image:  {args.image}")
    return 0 if quality_passed else 1


def run_held_verification(
    args: argparse.Namespace,
    preview: FfmpegRealSensePreview,
    piper: Piper,
    hold_joints: np.ndarray,
    marker_id: int,
    sample_count: int,
    radius_px: int,
    minimum_depth_pixels: int,
    minimum_depth_m: float,
    maximum_depth_m: float,
    minimum_marker_side_px: float,
    depth_scale_correction: float,
    depth_offset_mm: float,
    camera_to_gripper: np.ndarray,
    reference_target_to_base: np.ndarray,
) -> tuple[list[dict], list[np.ndarray], np.ndarray]:
    import pyrealsense2 as rs

    observations: list[dict] = []
    robot_transforms: list[np.ndarray] = []
    last_annotated = None
    sequence = -1
    deadline = time.monotonic() + args.timeout
    last_reason = "marker not detected"
    print(
        f"Collecting {sample_count} corrected depth observations while "
        "continuously holding the taught joints."
    )
    while len(observations) < sample_count and time.monotonic() < deadline:
        piper.send_joints(hold_joints, None, PIPER_C.SPEED_REST)
        try:
            frame, depth_raw, corners, sequence = preview.wait_for_marker_with_depth(
                marker_id, after_sequence=sequence, timeout=0.2
            )
        except RuntimeError as exc:
            last_reason = str(exc)
            continue
        side_px = float(
            np.mean(
                [
                    np.linalg.norm(corners[(index + 1) % 4] - corners[index])
                    for index in range(4)
                ]
            )
        )
        if side_px < minimum_marker_side_px:
            last_reason = (
                f"marker side {side_px:.1f}px is below "
                f"{minimum_marker_side_px:.1f}px"
            )
            continue
        center_uv = marker_center(corners)
        try:
            raw_depth_m, valid_pixels, depth_mad = robust_center_depth_m(
                depth_raw,
                center_uv,
                float(preview.depth_scale),
                radius_px,
                minimum_depth_pixels,
                minimum_depth_m,
                maximum_depth_m,
            )
        except HandEyeError as exc:
            last_reason = str(exc)
            continue
        corrected_depth_m = (
            depth_scale_correction * raw_depth_m * 1000.0 + depth_offset_mm
        ) / 1000.0
        camera_point_mm = (
            np.asarray(
                rs.rs2_deproject_pixel_to_point(
                    preview.color_intrinsics,
                    [float(center_uv[0]), float(center_uv[1])],
                    corrected_depth_m,
                )
            )
            * 1000.0
        )
        piper_pose = piper.pose()
        gripper_to_base = piper_pose_to_transform(piper_pose)
        base_point_mm = transform_camera_point_to_base(
            camera_point_mm, gripper_to_base, camera_to_gripper
        )
        error_mm = float(
            np.linalg.norm(base_point_mm - reference_target_to_base[:3, 3])
        )
        observations.append(
            {
                "index": len(observations) + 1,
                "pixel_uv": center_uv.tolist(),
                "marker_side_px": side_px,
                "raw_depth_m": raw_depth_m,
                "corrected_depth_m": corrected_depth_m,
                "depth_valid_pixel_count": valid_pixels,
                "depth_mad_mm": depth_mad * 1000.0,
                "camera_point_mm": camera_point_mm.tolist(),
                "piper_pose_mm_deg": piper_pose.tolist(),
                "base_point_mm": base_point_mm.tolist(),
                "coordinate_error_mm": error_mm,
            }
        )
        robot_transforms.append(gripper_to_base)
        annotated = frame.copy()
        cv2.polylines(
            annotated, [np.rint(corners).astype(np.int32)], True, (0, 255, 0), 2
        )
        cv2.circle(
            annotated,
            tuple(np.rint(center_uv).astype(int)),
            radius_px + 3,
            (0, 255, 0),
            2,
        )
        for line_index, text in enumerate(
            [
                (
                    f"depth={raw_depth_m * 1000.0:.1f}"
                    f"->{corrected_depth_m * 1000.0:.1f}mm"
                ),
                (
                    f"base=({base_point_mm[0]:.1f},"
                    f"{base_point_mm[1]:.1f},{base_point_mm[2]:.1f})mm"
                ),
                f"error={error_mm:.1f}mm",
            ]
        ):
            cv2.putText(
                annotated,
                text,
                (8, 24 + line_index * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        last_annotated = annotated
        piper.send_joints(hold_joints, None, PIPER_C.SPEED_REST)
        print(
            f"  {len(observations):02d}/{sample_count}: "
            f"depth={raw_depth_m * 1000.0:.1f}"
            f"->{corrected_depth_m * 1000.0:.1f}mm, error={error_mm:.2f}mm"
        )
    if len(observations) < sample_count or last_annotated is None:
        raise RuntimeError(
            f"collected {len(observations)}/{sample_count} observations; "
            f"last rejection: {last_reason}"
        )
    return observations, robot_transforms, last_annotated


def hold_until_power_off(piper: Piper, hold_joints: np.ndarray) -> None:
    confirmed = threading.Event()

    def wait_for_confirmation() -> None:
        input(
            "Turn the Piper controller OFF while supporting the arm. "
            "Once it is OFF, press Enter: "
        )
        confirmed.set()

    threading.Thread(target=wait_for_confirmation, daemon=True).start()
    while not confirmed.is_set():
        piper.send_joints(hold_joints, None, PIPER_C.SPEED_REST)
        confirmed.wait(timeout=0.05)


def main() -> int:
    args = parse_args()
    try:
        config = load_json(args.config)
        calibration = load_json(args.calibration)
        camera_config = config["camera"]
        aruco_config = config["aruco"]
        depth_config = config["depth_verification"]
        serial = str(camera_config.get("serial") or "")
        width = int(camera_config["width"])
        height = int(camera_config["height"])
        fps = int(camera_config["fps"])
        marker_id = int(aruco_config["target_marker_id"])
        minimum_marker_side_px = float(aruco_config["minimum_marker_side_px"])
        sample_count = int(depth_config["sample_count"])
        warmup_frames = int(depth_config["warmup_frames"])
        radius_px = int(depth_config["center_window_radius_px"])
        minimum_depth_pixels = int(depth_config["minimum_valid_depth_pixels"])
        minimum_depth_m = float(depth_config["minimum_depth_m"])
        maximum_depth_m = float(depth_config["maximum_depth_m"])
        maximum_position_deviation = float(
            depth_config["maximum_robot_position_deviation_mm"]
        )
        maximum_rotation_deviation = float(
            depth_config["maximum_robot_rotation_deviation_deg"]
        )
        maximum_coordinate_rmse = float(
            depth_config["maximum_coordinate_rmse_mm"]
        )
        maximum_consensus_error = float(
            depth_config["maximum_consensus_error_mm"]
        )
        maximum_temporal_depth_mad = float(
            depth_config["maximum_temporal_depth_mad_mm"]
        )
        minimum_temporal_inlier_fraction = float(
            depth_config["minimum_temporal_inlier_fraction"]
        )
        temporal_mad_multiplier = float(
            depth_config["temporal_mad_multiplier"]
        )
        temporal_outlier_floor = float(
            depth_config["temporal_outlier_floor_mm"]
        )
        camera_to_gripper = rigid_transform(calibration, "camera_to_gripper")
        reference_target_to_base = rigid_transform(
            calibration, "mean_target_to_base"
        )
        depth_scale_correction, depth_offset_mm, correction_document = (
            load_depth_correction(
                args.depth_correction,
                args.no_depth_correction,
            )
        )
        expected_serial = camera_serial_from_calibration(calibration)
        if sample_count < 3 or warmup_frames < 0 or radius_px < 1:
            raise HandEyeError("invalid depth verification sampling configuration")
        if args.timeout <= 0:
            raise HandEyeError("--timeout must be positive")
    except (KeyError, TypeError, ValueError, CalibrationError, HandEyeError) as exc:
        print(f"ERROR: cannot start depth verification: {exc}", file=sys.stderr)
        return 2

    dictionary_name = str(aruco_config["dictionary"])
    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None:
        print(f"ERROR: unknown ArUco dictionary {dictionary_name}", file=sys.stderr)
        return 2
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(dictionary_id),
        cv2.aruco.DetectorParameters(),
    )

    if args.align_in_teach:
        preview = None
        held_piper = None
        try:
            preview, held_piper, hold_joints = align_marker_in_teach(
                camera_config,
                aruco_config,
                marker_id,
                args.can_interface,
            )
            detected_serial = str(preview.camera["serial"])
            if serial and detected_serial != serial:
                raise RuntimeError(
                    f"expected D405 serial {serial}, detected {detected_serial}"
                )
            if expected_serial and detected_serial != expected_serial:
                raise RuntimeError(
                    f"calibration belongs to D405 {expected_serial}, "
                    f"but detected {detected_serial}"
                )
            observations, robot_transforms, last_annotated = (
                run_held_verification(
                    args,
                    preview,
                    held_piper,
                    hold_joints,
                    marker_id,
                    sample_count,
                    radius_px,
                    minimum_depth_pixels,
                    minimum_depth_m,
                    maximum_depth_m,
                    minimum_marker_side_px,
                    depth_scale_correction,
                    depth_offset_mm,
                    camera_to_gripper,
                    reference_target_to_base,
                )
            )
            result = finalize_verification(
                args,
                observations,
                robot_transforms,
                last_annotated,
                detected_serial,
                width,
                height,
                fps,
                float(preview.depth_scale),
                correction_document,
                depth_scale_correction,
                depth_offset_mm,
                dictionary_name,
                marker_id,
                reference_target_to_base,
                maximum_coordinate_rmse,
                maximum_position_deviation,
                maximum_rotation_deviation,
                maximum_consensus_error,
                maximum_temporal_depth_mad,
                minimum_temporal_inlier_fraction,
                temporal_mad_multiplier,
                temporal_outlier_floor,
            )
        except KeyboardInterrupt:
            print("\nHeld verification cancelled.", file=sys.stderr)
            return 130
        except RuntimeError as exc:
            print(f"ERROR: held verification failed: {exc}", file=sys.stderr)
            return 1
        finally:
            if held_piper is not None:
                print(
                    "\nKEEP SUPPORTING THE ARM. The saved taught pose will remain "
                    "streamed until you confirm controller power is OFF."
                )
                hold_until_power_off(held_piper, hold_joints)
                held_piper.disconnect()
            if preview is not None:
                preview.close()
        return result

    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    rs_config = rs.config()
    if serial:
        rs_config.enable_device(serial)
    rs_config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    rs_config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    align = rs.align(rs.stream.color)
    piper: Piper | None = None
    started = False
    observations: list[dict] = []
    robot_transforms: list[np.ndarray] = []
    last_annotated: np.ndarray | None = None
    try:
        profile = pipeline.start(rs_config)
        started = True
        device = profile.get_device()
        detected_serial = device.get_info(rs.camera_info.serial_number)
        if serial and detected_serial != serial:
            raise RuntimeError(
                f"expected D405 serial {serial}, detected {detected_serial}"
            )
        if expected_serial and detected_serial != expected_serial:
            raise RuntimeError(
                f"calibration belongs to D405 {expected_serial}, "
                f"but detected {detected_serial}"
            )
        depth_scale = float(device.first_depth_sensor().get_depth_scale())
        color_profile = profile.get_stream(
            rs.stream.color
        ).as_video_stream_profile()
        color_intrinsics = color_profile.get_intrinsics()
        for _ in range(warmup_frames):
            pipeline.wait_for_frames(timeout_ms=5000)

        piper = connect_readonly(iface=args.can_interface)
        if piper.in_teach():
            raise RuntimeError(
                "Piper reports teach mode; use --align-in-teach so one connection "
                "can continuously hold the taught pose through verification"
            )
        print(
            f"Collecting {sample_count} aligned depth observations of Marker "
            f"{marker_id}; keep the arm and marker stationary."
        )
        deadline = time.monotonic() + args.timeout
        last_reason = "marker not detected"
        while len(observations) < sample_count and time.monotonic() < deadline:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                last_reason = "aligned color/depth frame unavailable"
                continue
            color = np.asanyarray(color_frame.get_data()).copy()
            depth_raw = np.asanyarray(depth_frame.get_data())
            corners, ids, _ = detector.detectMarkers(color)
            if ids is None or marker_id not in ids.reshape(-1):
                last_reason = f"Marker ID {marker_id} is not visible"
                continue
            marker_index = list(ids.reshape(-1)).index(marker_id)
            marker_corners = corners[marker_index].reshape(4, 2)
            side_px = float(
                np.mean(
                    [
                        np.linalg.norm(
                            marker_corners[(index + 1) % 4]
                            - marker_corners[index]
                        )
                        for index in range(4)
                    ]
                )
            )
            if side_px < minimum_marker_side_px:
                last_reason = (
                    f"marker side {side_px:.1f}px is below "
                    f"{minimum_marker_side_px:.1f}px"
                )
                continue
            center_uv = marker_center(marker_corners)
            try:
                raw_depth_m, valid_pixels, depth_mad = robust_center_depth_m(
                    depth_raw,
                    center_uv,
                    depth_scale,
                    radius_px,
                    minimum_depth_pixels,
                    minimum_depth_m,
                    maximum_depth_m,
                )
            except HandEyeError as exc:
                last_reason = str(exc)
                continue
            corrected_depth_m = (
                depth_scale_correction * raw_depth_m * 1000.0
                + depth_offset_mm
            ) / 1000.0
            camera_point_m = np.asarray(
                rs.rs2_deproject_pixel_to_point(
                    color_intrinsics,
                    [float(center_uv[0]), float(center_uv[1])],
                    corrected_depth_m,
                ),
                dtype=np.float64,
            )
            camera_point_mm = camera_point_m * 1000.0
            piper_pose = piper.pose()
            gripper_to_base = piper_pose_to_transform(piper_pose)
            base_point_mm = transform_camera_point_to_base(
                camera_point_mm, gripper_to_base, camera_to_gripper
            )
            error_mm = float(
                np.linalg.norm(
                    base_point_mm - reference_target_to_base[:3, 3]
                )
            )
            observation = {
                "index": len(observations) + 1,
                "pixel_uv": [float(center_uv[0]), float(center_uv[1])],
                "marker_side_px": side_px,
                "raw_depth_m": raw_depth_m,
                "corrected_depth_m": corrected_depth_m,
                "depth_valid_pixel_count": valid_pixels,
                "depth_mad_mm": depth_mad * 1000.0,
                "camera_point_mm": camera_point_mm.tolist(),
                "piper_pose_mm_deg": piper_pose.tolist(),
                "base_point_mm": base_point_mm.tolist(),
                "coordinate_error_mm": error_mm,
            }
            observations.append(observation)
            robot_transforms.append(gripper_to_base)

            annotated = color.copy()
            cv2.aruco.drawDetectedMarkers(
                annotated,
                [corners[marker_index]],
                np.asarray([[marker_id]], dtype=np.int32),
            )
            pixel = tuple(np.rint(center_uv).astype(int))
            cv2.circle(annotated, pixel, radius_px + 3, (0, 255, 0), 2)
            lines = [
                (
                    f"depth={raw_depth_m * 1000.0:.1f}"
                    f"->{corrected_depth_m * 1000.0:.1f}mm"
                ),
                (
                    f"base=({base_point_mm[0]:.1f},"
                    f"{base_point_mm[1]:.1f},{base_point_mm[2]:.1f})mm"
                ),
                f"error={error_mm:.1f}mm",
            ]
            for line_index, text in enumerate(lines):
                cv2.putText(
                    annotated,
                    text,
                    (8, 24 + line_index * 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            last_annotated = annotated
            print(
                f"  {len(observations):02d}/{sample_count}: "
                f"depth={raw_depth_m * 1000.0:.1f}"
                f"->{corrected_depth_m * 1000.0:.1f}mm, "
                f"error={error_mm:.2f}mm"
            )

        if len(observations) < sample_count:
            raise RuntimeError(
                f"collected {len(observations)}/{sample_count} observations "
                f"within {args.timeout:.1f}s; last rejection: {last_reason}"
            )
    except KeyboardInterrupt:
        print("\nCancelled; no verification report was written.", file=sys.stderr)
        return 130
    except (RuntimeError, HandEyeError) as exc:
        print(f"ERROR: depth verification failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if piper is not None:
            piper.disconnect()
        if started:
            pipeline.stop()

    if last_annotated is None:
        print("ERROR: verification produced no annotated frame", file=sys.stderr)
        return 1
    return finalize_verification(
        args,
        observations,
        robot_transforms,
        last_annotated,
        detected_serial,
        width,
        height,
        fps,
        depth_scale,
        correction_document,
        depth_scale_correction,
        depth_offset_mm,
        dictionary_name,
        marker_id,
        reference_target_to_base,
        maximum_coordinate_rmse,
        maximum_position_deviation,
        maximum_rotation_deviation,
        maximum_consensus_error,
        maximum_temporal_depth_mad,
        minimum_temporal_inlier_fraction,
        temporal_mad_multiplier,
        temporal_outlier_floor,
    )


if __name__ == "__main__":
    raise SystemExit(main())
