#!/usr/bin/env python3
"""Capture lossless RGB frames from a taught scan pose for candy-detector tuning.

By default this only validates calibration, the scan pose, and the camera; it
never connects to CAN. With --allow-motion --setup-reviewed and a typed MOVE
confirmation it moves current -> SDK zero -> scan pose without gripper commands,
captures one or more rounds of frames while holding the scan pose, then makes a
checked SDK-zero return and disconnects. Saved images are for offline tuning;
they are never used as robot targets.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np

from demo import (
    CALIBRATION_DIR, DEFAULT_SCAN_POSES, PIPER_C, PROMPT,
    WORKSPACE_CORNERS_MM, WORKSPACE_MAX_Z_MM, WORKSPACE_MIN_Z_MM,
    DemoError, average_transforms, finite_vector, load_hand_eye, load_setup,
    piper_pose_to_transform, prepare_robot_for_scan, require_healthy,
    return_to_zero, rotation_error_degrees, validate_joint_path_safety,
)
from detection_client import DetectionError
from piper.robot import connect_robot
from vision_client import DEFAULT_DETECTION_URL, DEFAULT_VISION_URL, HttpClient, VisionClient

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "captures"
LABEL_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,40}")
MAXIMUM_FRAMES = 20


def parse_args() -> argparse.Namespace:
    """Parse capture options; motion requires explicit permission flags."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--calibration-dir", type=Path, default=CALIBRATION_DIR / "calibration")
    parser.add_argument("--scan-poses", type=Path, default=DEFAULT_SCAN_POSES)
    parser.add_argument("--scan", type=int, default=1, help="one-based taught scan pose (default 1)")
    parser.add_argument("--frames", type=int, default=3, help=f"frames per round, 1-{MAXIMUM_FRAMES} (default 3)")
    parser.add_argument("--frame-interval", type=float, default=0.5, help="seconds between frames (default 0.5)")
    parser.add_argument("--label", default="", help="filename label for every round, e.g. evening-light")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vision-url", default=DEFAULT_VISION_URL)
    parser.add_argument("--detection-url", default=DEFAULT_DETECTION_URL)
    parser.add_argument("--confidence", type=float, default=0.05, help="YOLO reference threshold")
    parser.add_argument("--no-yolo", action="store_true", help="do not save YOLO reference detections")
    parser.add_argument("--can-interface", default=PIPER_C.CAN_IFACE)
    parser.add_argument("--check-only", action="store_true", help="validate only; never connect to CAN")
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument("--setup-reviewed", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="print robot connection and zero-motion diagnostics")
    return parser.parse_args()


def validate_options(args: argparse.Namespace) -> None:
    """Reject unusable frame counts, intervals, thresholds, and labels before any I/O."""
    if not 1 <= args.frames <= MAXIMUM_FRAMES:
        raise DemoError(f"--frames must be between 1 and {MAXIMUM_FRAMES}")
    if not math.isfinite(args.frame_interval) or args.frame_interval < 0:
        raise DemoError("--frame-interval must be finite and nonnegative")
    if not math.isfinite(args.confidence) or not 0 < args.confidence < 1:
        raise DemoError("--confidence must be between zero and one")
    if args.label and not LABEL_PATTERN.fullmatch(args.label):
        raise DemoError("--label may contain only letters, digits, '-' and '_' (max 40)")


def check_camera(camera: HttpClient, camera_serial: str) -> None:
    """Require the camera service for the calibrated D435 with RGB-D capture."""
    health = camera.health()
    if (
        health.get("service") != "camera"
        or health.get("camera_serial") != camera_serial
        or "rgbd-capture" not in (health.get("capabilities") or [])
    ):
        raise DemoError("camera service must be the calibrated D435 with rgbd-capture")
    print(f"[OK] Camera {camera_serial} at {camera.base_url}", flush=True)


def reference_detector(objects: VisionClient, enabled: bool):
    """Return the YOLO detector for reference labels, or None when skipped/unavailable."""
    if not enabled:
        print("YOLO reference detections disabled (--no-yolo).", flush=True)
        return None
    try:
        objects.detector.check()
    except DetectionError as exc:
        print(f"WARNING: YOLO reference unavailable; saving images without it: {exc}", flush=True)
        return None
    print(f"[OK] YOLO reference detector {objects.detector.model_name}", flush=True)
    return objects.detector


def capture_frame(
    arm: Any, objects: VisionClient, detector: Any, confidence: float,
    camera_to_gripper: np.ndarray, settings: dict, hand_eye: dict, scan: dict, stem: Path,
) -> tuple[Path, dict[str, Any]]:
    """Capture one frame while stationary and save its PNG plus pose/calibration metadata."""
    image_path, metadata_path = stem.with_suffix(".png"), stem.with_suffix(".json")
    if image_path.exists() or metadata_path.exists():
        raise DemoError(f"refusing to overwrite existing capture {stem}")
    require_healthy(arm, "tuning capture")
    joints = finite_vector(arm.joints_deg(), 6, "capture joints")
    before = finite_vector(arm.pose(), 6, "pose before capture")
    captured, image = objects.capture_png()
    after = finite_vector(arm.pose(), 6, "pose after capture")
    before_transform, after_transform = piper_pose_to_transform(before), piper_pose_to_transform(after)
    moved = float(np.linalg.norm(after[:3] - before[:3]))
    rotated = rotation_error_degrees(before_transform, after_transform)
    if (
        moved > float(settings["maximum_scan_position_deviation_mm"])
        or rotated > float(settings["maximum_scan_rotation_deviation_deg"])
    ):
        raise DemoError(f"arm moved {moved:.2f}mm / {rotated:.2f}deg during capture")
    gripper_to_base = average_transforms([before_transform, after_transform])
    reference: dict[str, Any] = {"enabled": detector is not None}
    if detector is not None:
        reference.update(model=detector.model_name, prompt=PROMPT, confidence=confidence)
        try:
            reference["detections"] = detector.detect_png(image, PROMPT, confidence)
        except DetectionError as exc:
            reference["error"] = str(exc)
    metadata = {
        "schema_version": 1,
        "purpose": "offline candy-detector tuning; not a robot target",
        "frame_convention": "X_to_Y matrices map points from frame X into frame Y; millimetres",
        "image": image_path.name,
        "token": captured["token"],
        "frame_sequence": captured["frame_sequence"],
        "captured_at": captured.get("captured_at"),
        "width": captured.get("width"),
        "height": captured.get("height"),
        "camera_serial": hand_eye["camera"]["serial"],
        "camera_intrinsics": hand_eye["camera"].get("intrinsics"),
        "scan_index": int(scan["index"]),
        "scan_joints_deg": np.asarray(scan["joints_deg"], dtype=float).tolist(),
        "measured_joints_deg": joints.tolist(),
        "robot_pose_before_mm_deg": before.tolist(),
        "robot_pose_after_mm_deg": after.tolist(),
        "robot_position_deviation_mm": moved,
        "robot_rotation_deviation_deg": rotated,
        "gripper_to_base": gripper_to_base.tolist(),
        "camera_to_gripper": np.asarray(camera_to_gripper, dtype=float).tolist(),
        "camera_to_base": (gripper_to_base @ camera_to_gripper).tolist(),
        "workspace": {
            "corners_mm": [list(corner) for corner in WORKSPACE_CORNERS_MM],
            "minimum_z_mm": WORKSPACE_MIN_Z_MM,
            "maximum_z_mm": WORKSPACE_MAX_Z_MM,
            "grasp_tip_z_mm": float(settings["grasp_tip_z_mm"]),
        },
        "yolo_reference": reference,
    }
    stem.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(image)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return image_path, reference


def capture_rounds(
    arm: Any, objects: VisionClient, detector: Any, camera_to_gripper: np.ndarray,
    settings: dict, hand_eye: dict, scan: dict, args: argparse.Namespace,
) -> int:
    """Hold the scan pose and capture operator-triggered rounds until q/EOF."""
    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    saved = 0
    round_index = 0
    print(
        "Arm is enabled and holding the scan pose. Keep clear of the arm joints; "
        "keep emergency stop accessible.", flush=True,
    )
    while True:
        try:
            response = input(
                "Arrange the scene. Enter = capture, a label (letters/digits/-/_) = capture "
                "with that label, q = finish: "
            ).strip()
        except EOFError:
            return saved
        if response.lower() == "q":
            return saved
        if response and not LABEL_PATTERN.fullmatch(response):
            print("Labels may contain only letters, digits, '-' and '_' (max 40).", flush=True)
            continue
        round_index += 1
        label = "_".join(part for part in (args.label, response) if part)
        prefix = f"scan{int(scan['index'])}_{session}_r{round_index:02d}" + (f"_{label}" if label else "")
        for frame in range(1, args.frames + 1):
            if frame > 1:
                time.sleep(args.frame_interval)
            image_path, reference = capture_frame(
                arm, objects, detector, args.confidence, camera_to_gripper,
                settings, hand_eye, scan, args.output_dir / f"{prefix}_f{frame}",
            )
            saved += 1
            if not reference["enabled"]:
                summary = "no YOLO reference"
            elif "error" in reference:
                summary = f"YOLO reference failed: {reference['error']}"
            else:
                summary = f"YOLO reference: {len(reference['detections'])} detection(s)"
            print(f"Saved {image_path} ({summary})", flush=True)


def main() -> int:
    """Validate by default; with explicit permission, capture at the scan pose and return to zero."""
    args = parse_args()
    arm = None
    exit_code = 0
    try:
        validate_options(args)
        hand_eye = load_hand_eye(args.calibration_dir)
        settings, camera_to_gripper, scans = load_setup(args, hand_eye)
        if not 1 <= args.scan <= len(scans):
            raise DemoError(f"--scan must be between 1 and {len(scans)}")
        scan = scans[args.scan - 1]
        validate_joint_path_safety(
            "zero to selected scan pose", [np.zeros(6), scan["joints_deg"]],
            joint_limits=np.asarray(PIPER_C.JOINT_LIMITS_DEG, dtype=np.float64),
        )
        check_camera(HttpClient(args.vision_url), hand_eye["camera"]["serial"])
        objects = VisionClient(args.vision_url, detection_url=args.detection_url)
        detector = reference_detector(objects, not args.no_yolo)
        print(f"[OK] Accepted calibration and scan pose {args.scan}; output: {args.output_dir}", flush=True)
        if args.check_only or not args.allow_motion:
            print("Validation complete; no CAN connection or robot motion.", flush=True)
            return 0
        if not args.setup_reviewed:
            raise DemoError("capture motion requires --allow-motion --setup-reviewed")
        if not sys.stdin.isatty():
            raise DemoError("motion requires an interactive terminal for confirmation")
        print(
            f"Will move current -> SDK zero -> scan pose {args.scan} at {settings['motion_speed_percent']}% "
            "speed with no gripper commands, hold there while you capture, then return to SDK zero "
            "and disconnect.\nClear the current-to-zero and zero-to-scan paths. Joint checks do not "
            "model collisions. Keep emergency stop accessible.",
            flush=True,
        )
        if input("Type MOVE to confirm the route is reviewed and clear: ").strip() != "MOVE":
            print("Capture cancelled; no robot connection.", flush=True)
            return 0
        print("Connecting in 5 seconds.", flush=True)
        for _ in range(5):
            time.sleep(1)
        arm = connect_robot(iface=args.can_interface, verbose=args.verbose)
        prepare_robot_for_scan(arm, scan, settings)
        time.sleep(float(settings["scan_settle_seconds"]))
        saved = capture_rounds(arm, objects, detector, camera_to_gripper, settings, hand_eye, scan, args)
        print(f"[OK] Saved {saved} frame(s) to {args.output_dir}", flush=True)
    except KeyboardInterrupt:
        if arm is None:
            print("Interrupted before arm connection; no zero-return motion.", file=sys.stderr)
        exit_code = 130
    except (RuntimeError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: scan capture stopped: {exc}", file=sys.stderr)
        exit_code = 2
    finally:
        if arm is not None:
            try:
                return_to_zero(arm, context="capture zero return", verbose=args.verbose)
            except KeyboardInterrupt:
                print("Zero return interrupted; disconnecting without further motion.", file=sys.stderr)
                exit_code = 130
            except (RuntimeError, OSError, KeyError, TypeError, ValueError) as exc:
                print(f"ERROR: Cannot return to zero; disconnecting without further motion: {exc}", file=sys.stderr)
                if exit_code == 0:
                    exit_code = 2
            finally:
                arm.disconnect()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
