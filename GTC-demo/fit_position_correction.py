#!/usr/bin/env python3
"""Fit the GTC camera-to-robot XY correction from --measure-offsets readings (no robot).

Each reading in logs/grasp_offsets.jsonl holds the camera's uncorrected candy
position, the jaw midpoint the arm actually reached (from its joints), and the
operator's measured candy offset from that midpoint, so the candy's true
position is jaw point + offset. The fit maps camera XY to true XY:

    true XY = matrix @ camera XY + offset

with a full 2x2 matrix from 4 or more readings, or an offset only from fewer.
Paste the printed settings into DEMO_SETTINGS in demo.py.

    python GTC-demo/fit_position_correction.py --since 2026-11-23T15:40
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

DEFAULT_LOG = Path(__file__).resolve().parent / "logs" / "grasp_offsets.jsonl"
MINIMUM_AFFINE_READINGS = 4


def load_readings(path: Path, since: datetime | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (camera XY, true XY) arrays from the readings recorded at or after `since`."""
    camera, true = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        recorded = datetime.fromisoformat(record["recorded_at"])
        if since is not None and recorded < since:
            continue
        camera.append(np.asarray(record["uncorrected_base_point_mm"], dtype=np.float64)[:2])
        true.append(np.asarray(record["jaw_point_mm"], dtype=np.float64)[:2] + np.asarray(record["offset_mm"], dtype=np.float64))
    return np.asarray(camera).reshape(-1, 2), np.asarray(true).reshape(-1, 2)


def fit_correction(camera_xy: np.ndarray, true_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares (matrix, offset) with true = matrix @ camera + offset."""
    if len(camera_xy) == 0:
        raise ValueError("no readings to fit")
    if len(camera_xy) < MINIMUM_AFFINE_READINGS:
        return np.eye(2), np.mean(true_xy - camera_xy, axis=0)
    design = np.column_stack([camera_xy, np.ones(len(camera_xy))])
    solution, *_ = np.linalg.lstsq(design, true_xy, rcond=None)
    return solution[:2].T, solution[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path, nargs="?", default=DEFAULT_LOG, help="grasp offset JSON-lines file")
    parser.add_argument("--since", help="only readings recorded at or after this ISO time (local time unless it has an offset)")
    args = parser.parse_args()
    since = None
    if args.since:
        since = datetime.fromisoformat(args.since)
        if since.tzinfo is None:
            since = since.astimezone()
    try:
        camera_xy, true_xy = load_readings(args.log, since)
        matrix, offset = fit_correction(camera_xy, true_xy)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    corrected = camera_xy @ matrix.T + offset
    print(f"{len(camera_xy)} reading(s); {'matrix and offset' if len(camera_xy) >= MINIMUM_AFFINE_READINGS else 'offset only'}")
    for before_xy, after_xy, actual in zip(camera_xy, corrected, true_xy):
        print(
            f"  camera {np.round(before_xy, 1).tolist()} true {np.round(actual, 1).tolist()}: "
            f"error {np.round(before_xy - actual, 1).tolist()} -> {np.round(after_xy - actual, 1).tolist()} mm"
        )
    residual = np.linalg.norm(corrected - true_xy, axis=1)
    print(f"Residual after the fit: mean {residual.mean():.1f} mm, max {residual.max():.1f} mm")
    print("Settings for DEMO_SETTINGS in demo.py:")
    print(f'    "candy_xy_correction_mm": {np.round(offset, 3).tolist()},')
    print(f'    "candy_xy_correction_matrix": {np.round(matrix, 4).tolist()},')
    if np.any(np.abs(matrix - np.eye(2)) > 0.3):
        print("WARNING: the matrix is far from the identity; demo.py will reject it. Check the readings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
