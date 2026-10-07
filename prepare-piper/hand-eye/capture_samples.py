#!/usr/bin/env python3
"""Collect eye-in-hand calibration samples using stationary ArUco marker ID 1.

Purpose:
    Start the D405 browser preview, wait for physical teach mode, and let the
    operator move the wrist through at least 15 varied poses while the same
    fixed 101.6 mm marker remains visible. At each accepted pose, average
    synchronized marker corners, marker-to-camera PnP, and Piper end-pose
    telemetry. On exit, use the firmware-proven control sequence to enable the
    motors and hold the measured current joints; no new target pose is commanded.

Expected output:
    calibration/samples.json
        Camera intrinsics plus each Piper gripper-to-base transform and ArUco
        target-to-camera transform, with stability/reprojection diagnostics.
    calibration/sample_NNN.jpg
        An annotated image for every accepted pose.

Example:
    python3 prepare-piper/hand-eye/capture_samples.py
"""

from __future__ import annotations

import argparse
import sys
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

from hand_eye_common import (  # noqa: E402
    HandEyeError,
    average_transforms,
    piper_pose_to_transform,
)
from record_scan_pose import (  # noqa: E402
    FfmpegRealSensePreview,
)
from common import CalibrationError, load_json, write_json_atomic  # noqa: E402
from piper.robot import Piper, connect_for_teach  # noqa: E402

DEFAULT_CONFIG = HAND_EYE_DIR / "config.json"
DEFAULT_OUTPUT = HAND_EYE_DIR / "calibration" / "samples.json"
POSE_KEYS = ("x", "y", "z", "rx", "ry", "rz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--can-interface", default="can_piper")
    return parser.parse_args()


def read_piper_pose(piper: Piper) -> np.ndarray:
    return piper.pose()


def marker_object_points(size_mm: float) -> np.ndarray:
    half = size_mm / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def estimate_target_to_camera(
    corners: np.ndarray,
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> tuple[np.ndarray, float]:
    ok, rotation_vector, translation = cv2.solvePnP(
        object_points,
        corners.astype(np.float64),
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        raise RuntimeError("OpenCV could not estimate the marker pose")
    rotation, _ = cv2.Rodrigues(rotation_vector)
    projected, _ = cv2.projectPoints(
        object_points, rotation_vector, translation, camera_matrix, distortion
    )
    reprojection_error = float(
        np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - corners) ** 2, axis=1)))
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation.reshape(3)
    return matrix, reprojection_error


def collect_one_sample(
    preview: FfmpegRealSensePreview,
    piper: Piper,
    marker_id: int,
    marker_size_mm: float,
    frame_count: int,
    interval: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    object_points = marker_object_points(marker_size_mm)
    marker_corners = []
    robot_poses = []
    target_transforms = []
    reprojection_errors = []
    last_frame = None
    sequence = -1
    for _ in range(frame_count):
        frame, corners, sequence = preview.wait_for_marker(
            marker_id, after_sequence=sequence, timeout=3
        )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        refined = corners.astype(np.float32).reshape(-1, 1, 2)
        cv2.cornerSubPix(
            gray,
            refined,
            (5, 5),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
        )
        refined = refined.reshape(4, 2).astype(np.float64)
        target_to_camera, reprojection = estimate_target_to_camera(
            refined,
            object_points,
            preview.camera_matrix,
            preview.distortion_coefficients,
        )
        marker_corners.append(refined)
        robot_poses.append(read_piper_pose(piper))
        target_transforms.append(target_to_camera)
        reprojection_errors.append(reprojection)
        last_frame = frame
        time.sleep(interval)

    corner_array = np.stack(marker_corners)
    pose_array = np.stack(robot_poses)
    mean_gripper_to_base = average_transforms(
        [piper_pose_to_transform(pose) for pose in pose_array]
    )
    mean_target_to_camera = average_transforms(target_transforms)
    mean_corners = np.mean(corner_array, axis=0)
    center_stddev = np.std(np.mean(corner_array, axis=1), axis=0)
    position_stddev = np.std(pose_array[:, :3], axis=0)
    diagnostics = {
        "marker_center_stddev_px": {
            "u": float(center_stddev[0]),
            "v": float(center_stddev[1]),
        },
        "robot_position_stddev_mm": {
            "x": float(position_stddev[0]),
            "y": float(position_stddev[1]),
            "z": float(position_stddev[2]),
        },
        "mean_reprojection_error_px": float(np.mean(reprojection_errors)),
        "max_reprojection_error_px": float(np.max(reprojection_errors)),
        "mean_marker_side_px": float(
            np.mean(
                [
                    np.linalg.norm(mean_corners[(index + 1) % 4] - mean_corners[index])
                    for index in range(4)
                ]
            )
        ),
    }
    return (
        mean_gripper_to_base,
        mean_target_to_camera,
        last_frame,
        {"corners": mean_corners, **diagnostics},
    )


def ensure_exit_teach(
    piper: Piper,
    preview: FfmpegRealSensePreview,
    prompt_operator: bool = False,
) -> None:
    if not piper.in_teach():
        return
    print("\nKEEP HOLDING THE ARM.")
    print("Press the PHYSICAL TEACH BUTTON again; the arm should stiffen.")
    if not piper.wait_teach_stopped():
        raise RuntimeError(
            "teach recording did not stop; keep holding the arm and power-cycle safely"
        )
    hold_joints = piper.joints_deg()
    if prompt_operator:
        input("Arm stiffened. Press Enter to take CAN control back: ")
    else:
        print("Arm stiffened. Taking CAN control back...")
    if not piper.teach_stop(hold_joints=hold_joints):
        raise RuntimeError(
            "Piper did not leave teach mode and enable its motors; "
            "keep holding it and power-cycle safely"
        )
    print("Control taken back; holding torque has returned.")


def main() -> int:
    args = parse_args()
    try:
        config = load_json(args.config)
        camera_config = config["camera"]
        aruco = config["aruco"]
        collection = config["collection"]
        marker_id = int(aruco["target_marker_id"])
        marker_size_mm = float(aruco["marker_size_mm"])
        minimum_side_px = float(aruco["minimum_marker_side_px"])
        minimum_samples = int(collection["minimum_samples"])
        frame_count = int(collection["frames_per_sample"])
        interval = float(collection["sample_interval_seconds"])
        maximum_center_stddev = float(collection["maximum_marker_center_stddev_px"])
        maximum_position_stddev = float(collection["maximum_robot_position_stddev_mm"])
        maximum_reprojection = float(collection["maximum_reprojection_error_px"])
        if marker_size_mm <= 0 or minimum_samples < 3 or frame_count < 2:
            raise HandEyeError("invalid marker size or sampling limits")
    except (KeyError, TypeError, ValueError, CalibrationError, HandEyeError) as exc:
        print(f"ERROR: invalid configuration: {exc}", file=sys.stderr)
        return 2

    preview_config = {"dictionary": aruco["dictionary"], "required_marker_ids": [marker_id]}
    preview = FfmpegRealSensePreview(camera_config, preview_config)
    piper: Piper | None = None
    samples: list[dict] = []
    try:
        preview.start()
        piper = connect_for_teach(iface=args.can_interface)
        print(f"\nLive Marker-{marker_id} view: {preview.url}")
        print("The marker must remain fixed for the entire calibration.")
        print("Hold the arm and press the PHYSICAL TEACH BUTTON.")
        if not piper.wait_teach(True):
            raise RuntimeError("Piper did not enter teach mode")
        preview.check()
        print("TEACH MODE detected.")
        print(
            "Collect poses at different positions, distances, tilts and wrist rotations. "
            "Keep the entire marker visible."
        )

        while True:
            choice = input(
                f"\nPress Enter to capture pose {len(samples) + 1}, "
                f"or q to finish ({len(samples)}/{minimum_samples} minimum): "
            ).strip().lower()
            if choice == "q":
                if len(samples) < minimum_samples:
                    print(f"Need {minimum_samples - len(samples)} more samples.")
                    continue
                break
            if not piper.in_teach():
                raise RuntimeError("Piper left teach mode during sample collection")
            try:
                gripper_to_base, target_to_camera, frame, diagnostics = collect_one_sample(
                    preview,
                    piper,
                    marker_id,
                    marker_size_mm,
                    frame_count,
                    interval,
                )
            except RuntimeError as exc:
                print(f"REJECTED: {exc}")
                continue
            reasons = []
            center_stddev = max(diagnostics["marker_center_stddev_px"].values())
            position_stddev = max(diagnostics["robot_position_stddev_mm"].values())
            if center_stddev > maximum_center_stddev:
                reasons.append(
                    f"marker moved {center_stddev:.2f}px > {maximum_center_stddev:.2f}px"
                )
            if position_stddev > maximum_position_stddev:
                reasons.append(
                    f"robot moved {position_stddev:.2f}mm > {maximum_position_stddev:.2f}mm"
                )
            if diagnostics["max_reprojection_error_px"] > maximum_reprojection:
                reasons.append(
                    f"reprojection {diagnostics['max_reprojection_error_px']:.2f}px "
                    f"> {maximum_reprojection:.2f}px"
                )
            if diagnostics["mean_marker_side_px"] < minimum_side_px:
                reasons.append(
                    f"marker side {diagnostics['mean_marker_side_px']:.1f}px "
                    f"< {minimum_side_px:.1f}px"
                )
            if reasons:
                print("REJECTED: " + "; ".join(reasons))
                continue

            index = len(samples) + 1
            corners = diagnostics.pop("corners")
            image_path = args.output.parent / f"sample_{index:03d}.jpg"
            annotated = frame.copy()
            cv2.polylines(
                annotated,
                [np.rint(corners).astype(np.int32)],
                True,
                (0, 255, 0),
                2,
            )
            rotation_vector, _ = cv2.Rodrigues(target_to_camera[:3, :3])
            cv2.drawFrameAxes(
                annotated,
                preview.camera_matrix,
                preview.distortion_coefficients,
                rotation_vector,
                target_to_camera[:3, 3],
                marker_size_mm / 2,
            )
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(image_path), annotated):
                raise RuntimeError(f"could not write {image_path}")
            sample = {
                "index": index,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "gripper_to_base": gripper_to_base.tolist(),
                "target_to_camera": target_to_camera.tolist(),
                "diagnostics": diagnostics,
                "image": str(image_path),
            }
            samples.append(sample)
            write_json_atomic(
                args.output,
                {
                    "schema_version": 1,
                    "camera": preview.camera,
                    "marker": {
                        "dictionary": aruco["dictionary"],
                        "id": marker_id,
                        "black_square_size_mm": marker_size_mm,
                    },
                    "piper_pose_convention": "Rz(RZ) * Ry(RY) * Rx(RX), translation mm",
                    "samples": samples,
                },
            )
            print(
                f"ACCEPTED sample {index}: marker={diagnostics['mean_marker_side_px']:.1f}px, "
                f"reprojection={diagnostics['mean_reprojection_error_px']:.3f}px"
            )

        ensure_exit_teach(piper, preview, prompt_operator=True)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        if piper is not None:
            try:
                ensure_exit_teach(piper, preview)
            except RuntimeError as exc:
                print(f"SAFETY ERROR: {exc}", file=sys.stderr)
        return 130
    except (RuntimeError, HandEyeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if piper is not None:
            try:
                ensure_exit_teach(piper, preview)
            except RuntimeError as safety_exc:
                print(f"SAFETY ERROR: {safety_exc}", file=sys.stderr)
        return 1
    finally:
        if piper is not None:
            piper.disconnect()
        preview.close()

    print(f"\nCollected {len(samples)} samples in {args.output}")
    print("Next: python3 prepare-piper/hand-eye/solve_hand_eye.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
