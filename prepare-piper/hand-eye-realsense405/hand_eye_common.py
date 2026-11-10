#!/usr/bin/env python3
"""Shared rigid-transform and scoring helpers for wrist-camera hand-eye calibration."""

from __future__ import annotations

import math
from typing import Iterable

import cv2
import numpy as np


class HandEyeError(ValueError):
    """Raised when hand-eye inputs are malformed or geometrically inadequate."""


def transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return matrix


def euler_xyz_degrees_to_rotation(angles: Iterable[float]) -> np.ndarray:
    """Piper RX/RY/RZ convention: R = Rz(RZ) * Ry(RY) * Rx(RX)."""
    rx, ry, rz = np.radians(np.asarray(list(angles), dtype=np.float64))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return np.array(
        [
            [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
            [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
            [-sy, sx * cy, cx * cy],
        ],
        dtype=np.float64,
    )


def piper_pose_to_transform(pose_mm_deg: Iterable[float]) -> np.ndarray:
    pose = np.asarray(list(pose_mm_deg), dtype=np.float64)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise HandEyeError("Piper pose must contain six finite values")
    return transform(euler_xyz_degrees_to_rotation(pose[3:]), pose[:3])


def average_rotations(rotations: Iterable[np.ndarray]) -> np.ndarray:
    matrices = np.stack([np.asarray(rotation).reshape(3, 3) for rotation in rotations])
    u, _, vt = np.linalg.svd(np.sum(matrices, axis=0))
    mean = u @ vt
    if np.linalg.det(mean) < 0:
        u[:, -1] *= -1
        mean = u @ vt
    return mean


def rotation_error_degrees(left: np.ndarray, right: np.ndarray) -> float:
    relative = np.asarray(left)[:3, :3].T @ np.asarray(right)[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def average_transforms(transforms: Iterable[np.ndarray]) -> np.ndarray:
    matrices = np.stack([np.asarray(value).reshape(4, 4) for value in transforms])
    return transform(
        average_rotations(matrices[:, :3, :3]),
        np.mean(matrices[:, :3, 3], axis=0),
    )


def stationary_target_metrics(
    gripper_to_base: list[np.ndarray],
    target_to_camera: list[np.ndarray],
    camera_to_gripper: np.ndarray,
) -> tuple[dict[str, float], list[np.ndarray]]:
    base_to_target = [
        base_gripper @ camera_to_gripper @ camera_target
        for base_gripper, camera_target in zip(
            gripper_to_base, target_to_camera, strict=True
        )
    ]
    mean = average_transforms(base_to_target)
    translation_errors = np.array(
        [
            np.linalg.norm(value[:3, 3] - mean[:3, 3])
            for value in base_to_target
        ],
        dtype=np.float64,
    )
    rotation_errors = np.array(
        [rotation_error_degrees(mean, value) for value in base_to_target],
        dtype=np.float64,
    )
    metrics = {
        "target_translation_mean_error_mm": float(np.mean(translation_errors)),
        "target_translation_rmse_mm": float(
            np.sqrt(np.mean(np.square(translation_errors)))
        ),
        "target_translation_max_error_mm": float(np.max(translation_errors)),
        "target_rotation_mean_error_deg": float(np.mean(rotation_errors)),
        "target_rotation_rmse_deg": float(
            np.sqrt(np.mean(np.square(rotation_errors)))
        ),
        "target_rotation_max_error_deg": float(np.max(rotation_errors)),
    }
    return metrics, base_to_target


def motion_diversity(gripper_to_base: list[np.ndarray]) -> dict[str, float | int]:
    if len(gripper_to_base) < 2:
        raise HandEyeError("at least two poses are needed to measure diversity")
    translations = np.stack([value[:3, 3] for value in gripper_to_base])
    translation_span = float(
        np.linalg.norm(np.max(translations, axis=0) - np.min(translations, axis=0))
    )
    relative_rotation_vectors = []
    maximum_rotation = 0.0
    reference = gripper_to_base[0][:3, :3]
    for value in gripper_to_base[1:]:
        relative = reference.T @ value[:3, :3]
        vector, _ = cv2.Rodrigues(relative)
        vector = vector.reshape(3)
        relative_rotation_vectors.append(vector)
        maximum_rotation = max(maximum_rotation, math.degrees(np.linalg.norm(vector)))
    rotation_rank = int(
        np.linalg.matrix_rank(np.stack(relative_rotation_vectors), tol=1e-3)
    )
    return {
        "translation_span_mm": translation_span,
        "maximum_rotation_from_first_deg": maximum_rotation,
        "rotation_axis_rank": rotation_rank,
    }


def solve_hand_eye_candidates(
    gripper_to_base: list[np.ndarray],
    target_to_camera: list[np.ndarray],
) -> list[dict]:
    if len(gripper_to_base) != len(target_to_camera) or len(gripper_to_base) < 3:
        raise HandEyeError("matching lists of at least three poses are required")
    methods = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    rotations_gripper_to_base = [value[:3, :3] for value in gripper_to_base]
    translations_gripper_to_base = [value[:3, 3] for value in gripper_to_base]
    rotations_target_to_camera = [value[:3, :3] for value in target_to_camera]
    translations_target_to_camera = [value[:3, 3] for value in target_to_camera]
    candidates = []
    for name, method in methods.items():
        try:
            rotation, translation = cv2.calibrateHandEye(
                rotations_gripper_to_base,
                translations_gripper_to_base,
                rotations_target_to_camera,
                translations_target_to_camera,
                method=method,
            )
            camera_to_gripper = transform(rotation, translation)
            if not np.all(np.isfinite(camera_to_gripper)):
                continue
            metrics, base_targets = stationary_target_metrics(
                gripper_to_base, target_to_camera, camera_to_gripper
            )
            candidates.append(
                {
                    "method": name,
                    "camera_to_gripper": camera_to_gripper,
                    "metrics": metrics,
                    "base_targets": base_targets,
                    "score": metrics["target_translation_rmse_mm"]
                    + 5.0 * metrics["target_rotation_rmse_deg"],
                }
            )
        except cv2.error:
            continue
    if not candidates:
        raise HandEyeError("all OpenCV hand-eye methods failed")
    return sorted(candidates, key=lambda candidate: candidate["score"])
