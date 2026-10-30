#!/usr/bin/env python3
"""Calibrate systematic D405 Z-depth scale and offset using fixed Marker ID 1.

Purpose:
    In one physical teach session, collect the same stationary marker from at
    least seven varied camera distances. For each pose, compare robust aligned
    D405 depth with the marker depth predicted by the validated hand-eye
    transform and fixed marker base position. Fit:

        corrected_z_mm = scale * raw_z_mm + offset_mm

    The script enters teach mode only through the physical button. On exit, it
    waits for a second physical-button press to restore stiffness, snapshots
    the taught joints, and takes CAN control back while holding that pose.

Expected output:
    calibration/depth_samples.json
        Saved after every accepted pose so observations survive interruption.
    calibration/depth_correction.json
        Fitted scale/offset, distance span, residual metrics and pass/fail.
    calibration/depth_sample_NNN.jpg
        Annotated image for every accepted distance.

Example:
    python3 prepare-piper/hand-eye/calibrate_depth.py
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

HAND_EYE_DIR = Path(__file__).resolve().parent
HOMOGRAPHY_DIR = HAND_EYE_DIR.parent / "homography"
PIPER_APP_DIR = HAND_EYE_DIR.parents[1] / "Demo-VLA-On-Piper" / "piper-openpi-V2"
sys.path.insert(0, str(HOMOGRAPHY_DIR))
sys.path.insert(0, str(HAND_EYE_DIR))
sys.path.insert(0, str(PIPER_APP_DIR))

from common import CalibrationError, load_json, write_json_atomic  # noqa: E402
from hand_eye_common import (  # noqa: E402
    HandEyeError,
    average_transforms,
    piper_pose_to_transform,
)
from piper.robot import Piper, connect_for_teach  # noqa: E402
from record_scan_pose import FfmpegRealSensePreview  # noqa: E402
from verify_depth_coordinates import (  # noqa: E402
    marker_center,
    rigid_transform,
    robust_center_depth_m,
)

DEFAULT_CONFIG = HAND_EYE_DIR / "config.json"
DEFAULT_HAND_EYE = HAND_EYE_DIR / "calibration" / "hand_eye.json"
DEFAULT_SAMPLES = HAND_EYE_DIR / "calibration" / "depth_samples.json"
DEFAULT_OUTPUT = HAND_EYE_DIR / "calibration" / "depth_correction.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--hand-eye", type=Path, default=DEFAULT_HAND_EYE)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--can-interface", default="can_piper")
    return parser.parse_args()


def fit_depth_correction(
    raw_depth_mm: np.ndarray, expected_depth_mm: np.ndarray
) -> tuple[float, float, np.ndarray]:
    raw = np.asarray(raw_depth_mm, dtype=np.float64)
    expected = np.asarray(expected_depth_mm, dtype=np.float64)
    if raw.shape != expected.shape or raw.ndim != 1 or len(raw) < 3:
        raise HandEyeError("matching one-dimensional depth arrays are required")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(expected)):
        raise HandEyeError("depth arrays must contain finite values")
    design = np.column_stack([raw, np.ones_like(raw)])
    scale, offset = np.linalg.lstsq(design, expected, rcond=None)[0]
    residuals = scale * raw + offset - expected
    return float(scale), float(offset), residuals


def leave_one_out_errors(
    raw_depth_mm: np.ndarray, expected_depth_mm: np.ndarray
) -> np.ndarray:
    raw = np.asarray(raw_depth_mm, dtype=np.float64)
    expected = np.asarray(expected_depth_mm, dtype=np.float64)
    if len(raw) < 4:
        raise HandEyeError("at least four poses are needed for cross-validation")
    errors = []
    for index in range(len(raw)):
        keep = np.arange(len(raw)) != index
        scale, offset, _ = fit_depth_correction(raw[keep], expected[keep])
        errors.append(scale * raw[index] + offset - expected[index])
    return np.asarray(errors, dtype=np.float64)


def expected_marker_in_camera(
    gripper_to_base: np.ndarray,
    camera_to_gripper: np.ndarray,
    target_to_base: np.ndarray,
) -> np.ndarray:
    base_to_camera = gripper_to_base @ camera_to_gripper
    return (np.linalg.inv(base_to_camera) @ target_to_base)[:3, 3]


def collect_pose(
    preview: FfmpegRealSensePreview,
    piper: Piper,
    marker_id: int,
    frame_count: int,
    interval: float,
    radius_px: int,
    minimum_pixels: int,
    minimum_depth_m: float,
    maximum_depth_m: float,
    camera_to_gripper: np.ndarray,
    target_to_base: np.ndarray,
) -> tuple[dict, np.ndarray]:
    raw_depths = []
    expected_points = []
    centers = []
    piper_poses = []
    marker_sides = []
    depth_mads = []
    last_frame = None
    last_corners = None
    sequence = -1
    for _ in range(frame_count):
        frame, depth, corners, sequence = preview.wait_for_marker_with_depth(
            marker_id, after_sequence=sequence, timeout=3.0
        )
        center = marker_center(corners)
        raw_depth_m, _, depth_mad = robust_center_depth_m(
            depth,
            center,
            float(preview.depth_scale),
            radius_px,
            minimum_pixels,
            minimum_depth_m,
            maximum_depth_m,
        )
        piper_pose = piper.pose()
        gripper_to_base = piper_pose_to_transform(piper_pose)
        expected_point = expected_marker_in_camera(
            gripper_to_base, camera_to_gripper, target_to_base
        )
        raw_depths.append(raw_depth_m * 1000.0)
        expected_points.append(expected_point)
        centers.append(center)
        piper_poses.append(piper_pose)
        marker_sides.append(
            np.mean(
                [
                    np.linalg.norm(corners[(index + 1) % 4] - corners[index])
                    for index in range(4)
                ]
            )
        )
        depth_mads.append(depth_mad * 1000.0)
        last_frame, last_corners = frame, corners
        time.sleep(interval)

    transforms = [piper_pose_to_transform(value) for value in piper_poses]
    mean_transform = average_transforms(transforms)
    mean_expected = expected_marker_in_camera(
        mean_transform, camera_to_gripper, target_to_base
    )
    translation_deviation = max(
        np.linalg.norm(value[:3, 3] - mean_transform[:3, 3])
        for value in transforms
    )
    center_array = np.stack(centers)
    mean_center = np.mean(center_array, axis=0)
    result = {
        "raw_depth_mm": float(np.median(raw_depths)),
        "raw_depth_range_mm": [
            float(np.min(raw_depths)),
            float(np.max(raw_depths)),
        ],
        "expected_camera_point_mm": mean_expected.tolist(),
        "expected_depth_mm": float(mean_expected[2]),
        "raw_minus_expected_mm": float(np.median(raw_depths) - mean_expected[2]),
        "mean_pixel_uv": mean_center.tolist(),
        "marker_center_stddev_px": np.std(center_array, axis=0).tolist(),
        "mean_marker_side_px": float(np.mean(marker_sides)),
        "median_depth_mad_mm": float(np.median(depth_mads)),
        "maximum_robot_translation_deviation_mm": float(translation_deviation),
        "mean_piper_pose_mm_deg": np.mean(piper_poses, axis=0).tolist(),
    }
    annotated = last_frame.copy()
    cv2.polylines(
        annotated,
        [np.rint(last_corners).astype(np.int32)],
        True,
        (0, 255, 0),
        2,
    )
    cv2.circle(
        annotated,
        tuple(np.rint(mean_center).astype(int)),
        radius_px + 3,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        annotated,
        (
            f"raw={result['raw_depth_mm']:.1f}mm "
            f"expected={result['expected_depth_mm']:.1f}mm"
        ),
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return result, annotated


def main() -> int:
    args = parse_args()
    try:
        config = load_json(args.config)
        hand_eye = load_json(args.hand_eye)
        camera_config = config["camera"]
        aruco = config["aruco"]
        verify = config["depth_verification"]
        depth_config = config["depth_calibration"]
        marker_id = int(aruco["target_marker_id"])
        minimum_marker_side = float(aruco["minimum_marker_side_px"])
        minimum_poses = int(depth_config["minimum_poses"])
        frames_per_pose = int(depth_config["frames_per_pose"])
        interval = float(depth_config["sample_interval_seconds"])
        minimum_span = float(depth_config["minimum_expected_depth_span_mm"])
        minimum_scale = float(depth_config["minimum_scale"])
        maximum_scale = float(depth_config["maximum_scale"])
        maximum_fit_rmse = float(depth_config["maximum_fit_rmse_mm"])
        maximum_cv_rmse = float(
            depth_config["maximum_cross_validation_rmse_mm"]
        )
        maximum_cv_error = float(
            depth_config["maximum_cross_validation_error_mm"]
        )
        radius_px = int(verify["center_window_radius_px"])
        minimum_pixels = int(verify["minimum_valid_depth_pixels"])
        minimum_depth_m = float(verify["minimum_depth_m"])
        maximum_depth_m = float(verify["maximum_depth_m"])
        maximum_robot_deviation = float(
            verify["maximum_robot_position_deviation_mm"]
        )
        camera_to_gripper = rigid_transform(hand_eye, "camera_to_gripper")
        target_to_base = rigid_transform(hand_eye, "mean_target_to_base")
    except (KeyError, TypeError, ValueError, CalibrationError, HandEyeError) as exc:
        print(f"ERROR: cannot start depth calibration: {exc}", file=sys.stderr)
        return 2

    preview_config = {
        "dictionary": aruco["dictionary"],
        "required_marker_ids": [marker_id],
    }
    preview = FfmpegRealSensePreview(
        camera_config, preview_config, enable_depth=True
    )
    piper: Piper | None = None
    samples: list[dict] = []
    try:
        preview.start()
        piper = connect_for_teach(iface=args.can_interface)
        print(f"\nLive Marker-{marker_id} view: {preview.url}")
        print("The marker must remain fixed throughout this calibration.")
        print("Hold the arm and press the PHYSICAL TEACH BUTTON.")
        if not piper.wait_teach(True):
            raise RuntimeError("Piper did not enter teach mode")
        print("TEACH MODE detected.")
        print(
            "Capture near, middle and far distances with varied tilt. Keep the "
            "entire marker visible and hold the arm still for each capture."
        )
        while True:
            choice = input(
                f"\nPress Enter to capture distance {len(samples) + 1}, or q to "
                f"finish ({len(samples)}/{minimum_poses} minimum): "
            ).strip().lower()
            if choice == "q":
                if len(samples) < minimum_poses:
                    print(f"Need {minimum_poses - len(samples)} more poses.")
                    continue
                break
            if not piper.in_teach():
                raise RuntimeError("Piper left teach mode during depth calibration")
            try:
                sample, annotated = collect_pose(
                    preview,
                    piper,
                    marker_id,
                    frames_per_pose,
                    interval,
                    radius_px,
                    minimum_pixels,
                    minimum_depth_m,
                    maximum_depth_m,
                    camera_to_gripper,
                    target_to_base,
                )
            except (RuntimeError, HandEyeError) as exc:
                print(f"REJECTED: {exc}")
                continue
            reasons = []
            if sample["mean_marker_side_px"] < minimum_marker_side:
                reasons.append(
                    f"marker side {sample['mean_marker_side_px']:.1f}px "
                    f"< {minimum_marker_side:.1f}px"
                )
            if sample["maximum_robot_translation_deviation_mm"] > maximum_robot_deviation:
                reasons.append(
                    f"arm moved {sample['maximum_robot_translation_deviation_mm']:.2f}mm "
                    f"> {maximum_robot_deviation:.2f}mm"
                )
            if reasons:
                print("REJECTED: " + "; ".join(reasons))
                continue
            index = len(samples) + 1
            image_path = args.samples.parent / f"depth_sample_{index:03d}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(image_path), annotated):
                raise RuntimeError(f"could not write {image_path}")
            sample.update(
                {
                    "index": index,
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "image": str(image_path),
                }
            )
            samples.append(sample)
            write_json_atomic(
                args.samples,
                {
                    "schema_version": 1,
                    "camera": preview.camera,
                    "marker_id": marker_id,
                    "samples": samples,
                },
            )
            depths = [value["expected_depth_mm"] for value in samples]
            span = max(depths) - min(depths)
            print(
                f"ACCEPTED {index}: raw={sample['raw_depth_mm']:.1f}mm, "
                f"expected={sample['expected_depth_mm']:.1f}mm, "
                f"difference={sample['raw_minus_expected_mm']:+.1f}mm, "
                f"span={span:.1f}mm"
            )

        print("\nKEEP HOLDING THE ARM.")
        print("Press the PHYSICAL TEACH BUTTON again; the arm should stiffen.")
        if not piper.wait_teach_stopped():
            raise RuntimeError("teach recording did not stop")
        hold_joints = piper.joints_deg()
        input("Arm stiffened. Press Enter to take CAN control back: ")
        if not piper.teach_stop(hold_joints=hold_joints):
            raise RuntimeError("Piper did not return to stable CAN control")
        print("Control taken back; holding the taught pose.")
    except KeyboardInterrupt:
        print("\nInterrupted; accepted samples remain saved.", file=sys.stderr)
        return 130
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if piper is not None:
            if piper.in_teach():
                print(
                    "SAFETY: keep holding the arm and press the physical Teach "
                    "button again to stiffen it.",
                    file=sys.stderr,
                )
            piper.disconnect()
        preview.close()

    expected = np.asarray([value["expected_depth_mm"] for value in samples])
    raw = np.asarray([value["raw_depth_mm"] for value in samples])
    span = float(np.ptp(expected))
    if span < minimum_span:
        print(
            f"ERROR: expected-depth span {span:.1f}mm is below {minimum_span:.1f}mm; "
            "collect a wider near-to-far range",
            file=sys.stderr,
        )
        return 1
    scale, offset, residuals = fit_depth_correction(raw, expected)
    rmse = float(math.sqrt(float(np.mean(np.square(residuals)))))
    cv_errors = leave_one_out_errors(raw, expected)
    cv_rmse = float(math.sqrt(float(np.mean(np.square(cv_errors)))))
    cv_max_error = float(np.max(np.abs(cv_errors)))
    quality_passed = (
        minimum_scale <= scale <= maximum_scale
        and rmse <= maximum_fit_rmse
        and cv_rmse <= maximum_cv_rmse
        and cv_max_error <= maximum_cv_error
    )
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "corrected_z_mm = scale * raw_z_mm + offset_mm",
        "scale": scale,
        "offset_mm": offset,
        "pose_count": len(samples),
        "expected_depth_span_mm": span,
        "fit_rmse_mm": rmse,
        "fit_max_error_mm": float(np.max(np.abs(residuals))),
        "cross_validation_rmse_mm": cv_rmse,
        "cross_validation_mean_absolute_error_mm": float(
            np.mean(np.abs(cv_errors))
        ),
        "cross_validation_max_error_mm": cv_max_error,
        "acceptance": {
            "minimum_scale": minimum_scale,
            "maximum_scale": maximum_scale,
            "maximum_fit_rmse_mm": maximum_fit_rmse,
            "maximum_cross_validation_rmse_mm": maximum_cv_rmse,
            "maximum_cross_validation_error_mm": maximum_cv_error,
            "quality_passed": quality_passed,
        },
        "samples": [
            {
                "index": sample["index"],
                "raw_depth_mm": sample["raw_depth_mm"],
                "expected_depth_mm": sample["expected_depth_mm"],
                "corrected_depth_mm": float(
                    scale * sample["raw_depth_mm"] + offset
                ),
                "residual_mm": float(residual),
                "cross_validation_error_mm": float(cv_error),
            }
            for sample, residual, cv_error in zip(
                samples, residuals, cv_errors, strict=True
            )
        ],
    }
    write_json_atomic(args.output, result)
    print(
        f"\nDepth correction {'PASSED' if quality_passed else 'FAILED'}: "
        f"scale={scale:.8f}, offset={offset:+.3f}mm, RMSE={rmse:.3f}mm"
    )
    print(f"Correction written to {args.output}")
    return 0 if quality_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
