#!/usr/bin/env python3
"""Run wrist-mounted D435 calibration with isolated settings and output files."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

WORKFLOW_DIR = Path(__file__).resolve().parent
ENGINE_DIR = WORKFLOW_DIR.parent / "hand-eye-realsense405"
CONFIG = WORKFLOW_DIR / "config.json"
OUTPUT_DIR = WORKFLOW_DIR / "calibration"
STAGES = {
    "capture": "capture_samples.py",
    "solve": "solve_hand_eye.py",
    "validate": "validate_hand_eye.py",
    "depth": "calibrate_depth.py",
    "verify": "verify_depth_coordinates.py",
    "click": "click_to_coordinate.py",
}
PREREQUISITES = {
    "validate": ("hand_eye.json",),
    "depth": ("hand_eye.json", "validation.json"),
    "verify": ("hand_eye.json", "validation.json", "depth_correction.json"),
    "click": (
        "hand_eye.json", "validation.json",
        "depth_correction.json", "depth_verification.json",
    ),
}


def build_command(stage: str) -> list[str]:
    script = ENGINE_DIR / STAGES[stage]
    if not script.is_file():
        raise FileNotFoundError(f"shared calibration script not found: {script}")
    hand_eye = str(OUTPUT_DIR / "hand_eye.json")
    correction = str(OUTPUT_DIR / "depth_correction.json")
    options = {
        "capture": ["--output", str(OUTPUT_DIR / "samples.json")],
        "solve": [
            "--samples", str(OUTPUT_DIR / "samples.json"),
            "--output", hand_eye,
        ],
        "validate": [
            "--calibration", hand_eye,
            "--output", str(OUTPUT_DIR / "validation.json"),
        ],
        "depth": [
            "--hand-eye", hand_eye,
            "--samples", str(OUTPUT_DIR / "depth_samples.json"),
            "--output", correction,
        ],
        "verify": [
            "--calibration", hand_eye,
            "--depth-correction", correction,
            "--output", str(OUTPUT_DIR / "depth_verification.json"),
            "--image", str(OUTPUT_DIR / "depth_verification.jpg"),
            "--align-in-teach",
        ],
        "click": [
            "--hand-eye", hand_eye,
            "--depth-correction", correction,
            "--output", str(OUTPUT_DIR / "click_coordinates.json"),
        ],
    }
    return [
        sys.executable, str(script), "--config", str(CONFIG), *options[stage]
    ]


def check_dependencies() -> None:
    cv2 = importlib.import_module("cv2")
    required = (
        "aruco", "solvePnP", "Rodrigues", "projectPoints", "calibrateHandEye"
    )
    missing = [name for name in required if not hasattr(cv2, name)]
    if missing:
        raise RuntimeError(
            f"OpenCV {cv2.__version__} is missing {', '.join(missing)}. "
            "Activate ~/venv-ignite-demo and install this workflow's requirements.txt."
        )
    for name in ("numpy", "scipy", "can", "piper_sdk", "pyrealsense2"):
        importlib.import_module(name)


def require_prerequisites(stage: str) -> None:
    for filename in PREREQUISITES.get(stage, ()):
        path = OUTPUT_DIR / filename
        document = json.loads(path.read_text(encoding="utf-8"))
        acceptance = document.get("acceptance") if isinstance(document, dict) else None
        if (
            not isinstance(acceptance, dict)
            or acceptance.get("quality_passed") is not True
        ):
            raise RuntimeError(f"calibration has not passed quality checks: {path}")


def check_camera(config: dict) -> None:
    rs = importlib.import_module("pyrealsense2")
    serial = str(config["camera"]["serial"])
    context = rs.context()
    for device in context.query_devices():
        if device.get_info(rs.camera_info.serial_number) == serial:
            print(
                f"Camera detected: {device.get_info(rs.camera_info.name)}, "
                f"serial {serial}. No stream or robot control was started."
            )
            return
    raise RuntimeError(
        f"configured D435 serial {serial} is not accessible; "
        "check its USB connection and permissions"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("check", *STAGES),
        help="'check' only verifies dependencies and camera enumeration; "
        "other stages may enter physical teach/hold mode except offline 'solve'",
    )
    args = parser.parse_args()
    try:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        check_dependencies()
        if args.stage == "check":
            check_camera(config)
            print(f"Configuration: {CONFIG}")
            print(f"New calibration results will be saved under {OUTPUT_DIR}")
            return 0
        require_prerequisites(args.stage)
        command = build_command(args.stage)
        # Replace the launcher so the calibration tool owns Ctrl+C and cleanup.
        os.execv(command[0], command)
    except (ImportError, OSError, RuntimeError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: cannot start D435 calibration: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
