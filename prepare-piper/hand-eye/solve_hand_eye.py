#!/usr/bin/env python3
"""Solve the wrist D405-to-Piper hand-eye transform and score consistency.

Purpose:
    Run OpenCV's Tsai, Park, Horaud, Andreff and Daniilidis eye-in-hand
    solvers on the collected Marker-1 observations. Select the transform that
    makes the fixed marker most stationary in Piper base coordinates. This
    script uses saved JSON only and sends no camera or robot commands.

Expected output:
    calibration/hand_eye.json
        The selected camera-to-gripper 4x4 transform used as:
        base_T_camera = base_T_gripper @ gripper_T_camera,
        all candidate scores, pose-diversity diagnostics, stationary-target
        residuals, and quality pass/fail.

Example:
    python3 prepare-piper/hand-eye/solve_hand_eye.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HAND_EYE_DIR = Path(__file__).resolve().parent
HOMOGRAPHY_DIR = HAND_EYE_DIR.parent / "homography"
sys.path.insert(0, str(HOMOGRAPHY_DIR))
sys.path.insert(0, str(HAND_EYE_DIR))

from common import CalibrationError, load_json, write_json_atomic  # noqa: E402
from hand_eye_common import (  # noqa: E402
    HandEyeError,
    average_transforms,
    motion_diversity,
    solve_hand_eye_candidates,
)

DEFAULT_CONFIG = HAND_EYE_DIR / "config.json"
DEFAULT_SAMPLES = HAND_EYE_DIR / "calibration" / "samples.json"
DEFAULT_OUTPUT = HAND_EYE_DIR / "calibration" / "hand_eye.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-failed-quality", action="store_true")
    return parser.parse_args()


def matrix_from_sample(sample: dict, key: str, index: int) -> np.ndarray:
    matrix = np.asarray(sample.get(key), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise HandEyeError(f"sample {index} has invalid {key}")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8):
        raise HandEyeError(f"sample {index} {key} is not a rigid transform")
    return matrix


def main() -> int:
    args = parse_args()
    try:
        config = load_json(args.config)
        document = load_json(args.samples)
        samples = document.get("samples")
        if not isinstance(samples, list):
            raise HandEyeError("samples.json must contain a samples list")
        minimum_samples = int(config["collection"]["minimum_samples"])
        if len(samples) < minimum_samples:
            raise HandEyeError(
                f"need at least {minimum_samples} samples, found {len(samples)}"
            )
        gripper_to_base = [
            matrix_from_sample(sample, "gripper_to_base", index)
            for index, sample in enumerate(samples, 1)
        ]
        target_to_camera = [
            matrix_from_sample(sample, "target_to_camera", index)
            for index, sample in enumerate(samples, 1)
        ]
        diversity = motion_diversity(gripper_to_base)
        acceptance = config["acceptance"]
        minimum_translation = float(acceptance["minimum_translation_span_mm"])
        minimum_rotation = float(acceptance["minimum_rotation_from_first_deg"])
        minimum_rank = int(acceptance["minimum_rotation_axis_rank"])
        maximum_translation_rmse = float(
            acceptance["maximum_target_translation_rmse_mm"]
        )
        maximum_rotation_rmse = float(
            acceptance["maximum_target_rotation_rmse_deg"]
        )
        if diversity["translation_span_mm"] < minimum_translation:
            raise HandEyeError(
                f"translation span {diversity['translation_span_mm']:.1f}mm "
                f"is below {minimum_translation:.1f}mm"
            )
        if diversity["maximum_rotation_from_first_deg"] < minimum_rotation:
            raise HandEyeError(
                f"rotation span {diversity['maximum_rotation_from_first_deg']:.1f}deg "
                f"is below {minimum_rotation:.1f}deg"
            )
        if diversity["rotation_axis_rank"] < minimum_rank:
            raise HandEyeError(
                f"rotation-axis rank {diversity['rotation_axis_rank']} "
                f"is below {minimum_rank}; vary tilt and roll around multiple axes"
            )
        candidates = solve_hand_eye_candidates(gripper_to_base, target_to_camera)
    except (KeyError, TypeError, ValueError, CalibrationError, HandEyeError) as exc:
        print(f"ERROR: cannot solve hand-eye calibration: {exc}", file=sys.stderr)
        return 2

    best = candidates[0]
    metrics = best["metrics"]
    quality_passed = (
        metrics["target_translation_rmse_mm"] <= maximum_translation_rmse
        and metrics["target_rotation_rmse_deg"] <= maximum_rotation_rmse
    )
    print(
        f"Selected {best['method']}: translation RMSE "
        f"{metrics['target_translation_rmse_mm']:.3f}mm, rotation RMSE "
        f"{metrics['target_rotation_rmse_deg']:.3f}deg"
    )
    for candidate in candidates:
        candidate_metrics = candidate["metrics"]
        print(
            f"  {candidate['method']}: "
            f"{candidate_metrics['target_translation_rmse_mm']:.3f}mm, "
            f"{candidate_metrics['target_rotation_rmse_deg']:.3f}deg"
        )
    if not quality_passed and not args.allow_failed_quality:
        print(
            "ERROR: calibration failed acceptance criteria; output was not written",
            file=sys.stderr,
        )
        return 1

    mean_base_target = average_transforms(best["base_targets"])
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(samples),
        "selected_method": best["method"],
        "frame_convention": {
            "camera_to_gripper": "maps D405 optical-frame points into the Piper reported end frame",
            "runtime_composition": "base_T_camera = base_T_gripper @ gripper_T_camera",
            "piper_euler": "Rz(RZ) * Ry(RY) * Rx(RX), translation millimetres",
        },
        "camera_to_gripper": best["camera_to_gripper"].tolist(),
        "gripper_to_camera": np.linalg.inv(best["camera_to_gripper"]).tolist(),
        "mean_target_to_base": mean_base_target.tolist(),
        "motion_diversity": diversity,
        "metrics": metrics,
        "acceptance": {
            "maximum_target_translation_rmse_mm": maximum_translation_rmse,
            "maximum_target_rotation_rmse_deg": maximum_rotation_rmse,
            "quality_passed": quality_passed,
        },
        "candidates": [
            {
                "method": candidate["method"],
                "camera_to_gripper": candidate["camera_to_gripper"].tolist(),
                "metrics": candidate["metrics"],
                "score": float(candidate["score"]),
            }
            for candidate in candidates
        ],
        "camera": document.get("camera"),
        "marker": document.get("marker"),
    }
    write_json_atomic(args.output, result)
    print(f"Hand-eye calibration written to {args.output}")
    if not quality_passed:
        print("WARNING: failed-quality output was written by explicit request")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
