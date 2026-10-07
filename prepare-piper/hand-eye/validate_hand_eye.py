#!/usr/bin/env python3
"""Validate the solved wrist D405 hand-eye transform from fresh robot poses.

Purpose:
    Capture at least five new observations of the same stationary Marker ID 1
    without recalculating the hand-eye transform. For every observation,
    project the marker pose into the Piper base frame and compare it with the
    fixed-marker pose saved by solve_hand_eye.py. On exit, use the firmware-
    proven control sequence to enable the motors and hold the measured current
    joints; no new target pose is commanded.

Expected output:
    calibration/validation.json
        Per-pose base-frame marker estimates and errors, aggregate translation
        and rotation RMSE, and an overall quality pass/fail result.
    calibration/validation_NNN.jpg
        An annotated D405 image for each accepted validation pose.

Example:
    python3 prepare-piper/hand-eye/validate_hand_eye.py
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
PIPER_APP_DIR = HAND_EYE_DIR.parents[1] / "piper-openpi-V2"
sys.path.insert(0, str(HOMOGRAPHY_DIR))
sys.path.insert(0, str(HAND_EYE_DIR))
sys.path.insert(0, str(PIPER_APP_DIR))

from common import CalibrationError, load_json, write_json_atomic  # noqa: E402
from capture_samples import collect_one_sample  # noqa: E402
from hand_eye_common import HandEyeError, rotation_error_degrees  # noqa: E402
from record_scan_pose import (  # noqa: E402
    FfmpegRealSensePreview,
)
from piper.robot import Piper, connect_for_teach  # noqa: E402

DEFAULT_CONFIG = HAND_EYE_DIR / "config.json"
DEFAULT_CALIBRATION = HAND_EYE_DIR / "calibration" / "hand_eye.json"
DEFAULT_OUTPUT = HAND_EYE_DIR / "calibration" / "validation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--can-interface", default="can_piper")
    parser.add_argument("--poses", type=int, default=5)
    return parser.parse_args()


def rigid_transform(document: dict, key: str) -> np.ndarray:
    value = np.asarray(document.get(key), dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise HandEyeError(f"{key} must be a finite 4x4 matrix")
    if not np.allclose(value[3], [0, 0, 0, 1], atol=1e-8):
        raise HandEyeError(f"{key} is not a rigid transform")
    return value


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


def validation_result(
    args: argparse.Namespace,
    calibration: dict,
    reference_target_to_base: np.ndarray,
    observations: list[dict],
    maximum_translation_rmse: float,
    maximum_rotation_rmse: float,
    safety_exit_completed: bool,
) -> dict:
    translation_errors = np.asarray(
        [value["translation_error_mm"] for value in observations]
    )
    rotation_errors = np.asarray(
        [value["rotation_error_deg"] for value in observations]
    )
    translation_rmse = float(
        math.sqrt(float(np.mean(np.square(translation_errors))))
    )
    rotation_rmse = float(math.sqrt(float(np.mean(np.square(rotation_errors)))))
    quality_passed = (
        translation_rmse <= maximum_translation_rmse
        and rotation_rmse <= maximum_rotation_rmse
    )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "calibration": str(args.calibration),
        "selected_method": calibration.get("selected_method"),
        "pose_count": len(observations),
        "reference_target_to_base": reference_target_to_base.tolist(),
        "translation_rmse_mm": translation_rmse,
        "rotation_rmse_deg": rotation_rmse,
        "maximum_translation_error_mm": float(np.max(translation_errors)),
        "maximum_rotation_error_deg": float(np.max(rotation_errors)),
        "acceptance": {
            "maximum_translation_rmse_mm": maximum_translation_rmse,
            "maximum_rotation_rmse_deg": maximum_rotation_rmse,
            "quality_passed": quality_passed,
        },
        "safety_exit_completed": safety_exit_completed,
        "observations": observations,
    }


def main() -> int:
    args = parse_args()
    try:
        config = load_json(args.config)
        calibration = load_json(args.calibration)
        if args.poses < 3:
            raise HandEyeError("--poses must be at least 3")
        camera_config = config["camera"]
        aruco = config["aruco"]
        collection = config["collection"]
        acceptance = config["acceptance"]
        marker_id = int(aruco["target_marker_id"])
        marker_size_mm = float(aruco["marker_size_mm"])
        minimum_side_px = float(aruco["minimum_marker_side_px"])
        frame_count = int(collection["frames_per_sample"])
        interval = float(collection["sample_interval_seconds"])
        maximum_center_stddev = float(
            collection["maximum_marker_center_stddev_px"]
        )
        maximum_position_stddev = float(
            collection["maximum_robot_position_stddev_mm"]
        )
        maximum_reprojection = float(collection["maximum_reprojection_error_px"])
        maximum_translation_rmse = float(
            acceptance["maximum_target_translation_rmse_mm"]
        )
        maximum_rotation_rmse = float(
            acceptance["maximum_target_rotation_rmse_deg"]
        )
        camera_to_gripper = rigid_transform(calibration, "camera_to_gripper")
        reference_target_to_base = rigid_transform(
            calibration, "mean_target_to_base"
        )
    except (KeyError, TypeError, ValueError, CalibrationError, HandEyeError) as exc:
        print(f"ERROR: cannot start validation: {exc}", file=sys.stderr)
        return 2

    preview_config = {
        "dictionary": aruco["dictionary"],
        "required_marker_ids": [marker_id],
    }
    preview = FfmpegRealSensePreview(camera_config, preview_config)
    piper: Piper | None = None
    observations: list[dict] = []
    result: dict | None = None
    try:
        preview.start()
        piper = connect_for_teach(iface=args.can_interface)
        print(f"\nLive Marker-{marker_id} view: {preview.url}")
        print("Do not move the marker or camera mount.")
        print("Use NEW poses that were not used during calibration.")
        print("Hold the arm and press the PHYSICAL TEACH BUTTON.")
        if not piper.wait_teach(True):
            raise RuntimeError("Piper did not enter teach mode")
        preview.check()
        print("TEACH MODE detected.")

        while len(observations) < args.poses:
            input(
                f"\nMove to new validation pose {len(observations) + 1}/{args.poses}, "
                "let the arm and image settle, then press Enter: "
            )
            if not piper.in_teach():
                raise RuntimeError("Piper left teach mode during validation")
            try:
                gripper_to_base, target_to_camera, frame, diagnostics = (
                    collect_one_sample(
                        preview,
                        piper,
                        marker_id,
                        marker_size_mm,
                        frame_count,
                        interval,
                    )
                )
            except RuntimeError as exc:
                print(f"REJECTED: {exc}")
                continue

            center_stddev = max(diagnostics["marker_center_stddev_px"].values())
            position_stddev = max(diagnostics["robot_position_stddev_mm"].values())
            reasons = []
            if center_stddev > maximum_center_stddev:
                reasons.append(
                    f"marker moved {center_stddev:.2f}px > "
                    f"{maximum_center_stddev:.2f}px"
                )
            if position_stddev > maximum_position_stddev:
                reasons.append(
                    f"robot moved {position_stddev:.2f}mm > "
                    f"{maximum_position_stddev:.2f}mm"
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

            target_to_base = (
                gripper_to_base @ camera_to_gripper @ target_to_camera
            )
            translation_error = float(
                np.linalg.norm(
                    target_to_base[:3, 3] - reference_target_to_base[:3, 3]
                )
            )
            rotation_error = rotation_error_degrees(
                reference_target_to_base, target_to_base
            )
            index = len(observations) + 1
            corners = diagnostics.pop("corners")
            image_path = args.output.parent / f"validation_{index:03d}.jpg"
            annotated = frame.copy()
            cv2.polylines(
                annotated,
                [np.rint(corners).astype(np.int32)],
                True,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                annotated,
                f"base error: {translation_error:.2f} mm / {rotation_error:.2f} deg",
                (8, 460),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(image_path), annotated):
                raise RuntimeError(f"could not write {image_path}")
            observations.append(
                {
                    "index": index,
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "target_to_base": target_to_base.tolist(),
                    "translation_error_mm": translation_error,
                    "rotation_error_deg": rotation_error,
                    "diagnostics": diagnostics,
                    "image": str(image_path),
                }
            )
            print(
                f"ACCEPTED pose {index}: base error={translation_error:.3f}mm, "
                f"{rotation_error:.3f}deg"
            )

        result = validation_result(
            args,
            calibration,
            reference_target_to_base,
            observations,
            maximum_translation_rmse,
            maximum_rotation_rmse,
            safety_exit_completed=False,
        )
        write_json_atomic(args.output, result)
        print(f"\nValidation measurements saved to {args.output}")
        ensure_exit_teach(piper, preview, prompt_operator=True)
        result["safety_exit_completed"] = True
        write_json_atomic(args.output, result)
    except KeyboardInterrupt:
        print("\nInterrupted; validation result was not written.", file=sys.stderr)
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

    if result is None:
        print("ERROR: validation completed without a result", file=sys.stderr)
        return 1
    translation_rmse = float(result["translation_rmse_mm"])
    rotation_rmse = float(result["rotation_rmse_deg"])
    quality_passed = bool(result["acceptance"]["quality_passed"])
    print(
        f"\nValidation {'PASSED' if quality_passed else 'FAILED'}: "
        f"translation RMSE={translation_rmse:.3f}mm, "
        f"rotation RMSE={rotation_rmse:.3f}deg"
    )
    print(f"Validation report written to {args.output}")
    return 0 if quality_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
