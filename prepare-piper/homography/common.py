#!/usr/bin/env python3
"""Shared data validation and geometry helpers for homography calibration."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import cv2
import numpy as np


class CalibrationError(ValueError):
    """Raised when calibration input is missing, malformed, or degenerate."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CalibrationError(f"file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CalibrationError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CalibrationError(f"expected a JSON object in {path}")
    return data


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def parse_marker_ids(values: list[Any]) -> list[int]:
    if len(values) < 4:
        raise CalibrationError("at least four marker IDs are required")
    try:
        marker_ids = [int(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise CalibrationError("marker IDs must be integers") from exc
    if len(set(marker_ids)) != len(marker_ids):
        raise CalibrationError("marker IDs must be unique")
    return marker_ids


def marker_map(document: dict[str, Any], source_name: str) -> dict[int, dict[str, Any]]:
    markers = document.get("markers")
    if not isinstance(markers, dict):
        raise CalibrationError(f"{source_name} must contain a 'markers' object")
    parsed: dict[int, dict[str, Any]] = {}
    for raw_id, value in markers.items():
        try:
            marker_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise CalibrationError(f"invalid marker ID {raw_id!r} in {source_name}") from exc
        if not isinstance(value, dict):
            raise CalibrationError(f"marker {marker_id} in {source_name} must be an object")
        parsed[marker_id] = value
    return parsed


def solve_homography(
    pixel_points: np.ndarray,
    robot_points: np.ndarray,
    ransac_threshold_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    pixel_points = np.asarray(pixel_points, dtype=np.float64)
    robot_points = np.asarray(robot_points, dtype=np.float64)
    if pixel_points.shape != robot_points.shape or pixel_points.ndim != 2:
        raise CalibrationError("pixel and robot points must have matching Nx2 shapes")
    if pixel_points.shape[0] < 4 or pixel_points.shape[1] != 2:
        raise CalibrationError("at least four 2D point pairs are required")
    for name, points in (("pixel", pixel_points), ("robot", robot_points)):
        if not np.all(np.isfinite(points)):
            raise CalibrationError(f"{name} points contain non-finite values")
        hull_area = abs(float(cv2.contourArea(cv2.convexHull(points.astype(np.float32)))))
        if hull_area < 1e-6:
            raise CalibrationError(f"{name} points are collinear or degenerate")

    method = cv2.RANSAC if len(pixel_points) > 4 else 0
    homography, mask = cv2.findHomography(
        pixel_points,
        robot_points,
        method=method,
        ransacReprojThreshold=ransac_threshold_mm,
    )
    if homography is None or mask is None:
        raise CalibrationError("OpenCV could not solve the homography")
    if not np.all(np.isfinite(homography)) or np.linalg.matrix_rank(homography) < 3:
        raise CalibrationError("the solved homography is invalid or singular")
    if abs(float(homography[2, 2])) < 1e-12:
        raise CalibrationError("the solved homography cannot be normalized")
    homography /= homography[2, 2]
    return homography, mask.reshape(-1).astype(bool)


def transform_pixels(homography: np.ndarray, pixel_points: np.ndarray) -> np.ndarray:
    points = np.asarray(pixel_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise CalibrationError("pixel points must have shape Nx2")
    transformed = cv2.perspectiveTransform(points.reshape(-1, 1, 2), homography)
    return transformed.reshape(-1, 2)


def error_metrics(predicted: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    errors = np.linalg.norm(
        np.asarray(predicted, dtype=np.float64) - np.asarray(expected, dtype=np.float64),
        axis=1,
    )
    if len(errors) == 0:
        raise CalibrationError("cannot calculate metrics without points")
    return {
        "mean_error_mm": float(np.mean(errors)),
        "rmse_mm": float(math.sqrt(float(np.mean(np.square(errors))))),
        "max_error_mm": float(np.max(errors)),
    }


def require_number(container: dict[str, Any], key: str, context: str) -> float:
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationError(f"{context}.{key} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise CalibrationError(f"{context}.{key} must be finite")
    return number
