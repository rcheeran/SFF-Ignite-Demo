#!/usr/bin/env python3
"""Compare a surveyed candy-surface point with token-bound camera coordinates."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import math
from pathlib import Path
import sys

import cv2
import numpy as np

import teach_workspaces as teaching
from demo import (
    DemoError, average_transforms, finite_vector, load_hand_eye, load_json,
    piper_pose_to_transform, require_quality, rigid_transform,
    transform_camera_point_to_base,
)
from vision_client import HttpClient

DEFAULT_OUTPUT = teaching.CAMERA_DIR / "calibration/candy_point_verification.json"


def parse_args() -> argparse.Namespace:
    """Parse reference, camera, sampling, and explicit comparison thresholds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-url", default="http://127.0.0.1:8094")
    parser.add_argument("--calibration-dir", type=Path, default=teaching.DEFAULT_OUTPUT.parent)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reference-point-mm", type=float, nargs=3, metavar=("X", "Y", "Z"),
        help="independently surveyed base XYZ; otherwise record a hand-guided tool-tip reference",
    )
    parser.add_argument("--can-interface", default="can_piper")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument("--maximum-joint-stddev-deg", type=float, default=0.25)
    parser.add_argument("--maximum-position-stddev-mm", type=float, default=0.75)
    parser.add_argument("--maximum-capture-position-change-mm", type=float, default=1.0)
    parser.add_argument("--maximum-capture-rotation-change-deg", type=float, default=0.5)
    parser.add_argument("--maximum-error-mm", type=float, default=10.0)
    return parser.parse_args()


def load_calibration(directory: Path, health: dict) -> np.ndarray:
    """Require accepted D435 artifacts and calibrated depth from the same camera."""
    hand_eye = load_hand_eye(directory)
    serial = hand_eye["camera"]["serial"]
    for name in ("validation.json", "depth_correction.json", "depth_verification.json"):
        document = load_json(directory / name)
        require_quality(document, name, directory / name)
        if name == "depth_verification.json" and document.get("camera", {}).get("serial") != serial:
            raise DemoError("depth verification camera does not match hand-eye calibration")
    if (
        health.get("camera_serial") != serial or health.get("preview_only") is not False
        or health.get("depth_calibrated") is not True
        or "deproject" not in health.get("capabilities", [])
    ):
        raise DemoError("matching calibrated camera with deprojection capability is required")
    return rigid_transform(hand_eye, "camera_to_gripper")


def freeze_capture(arm, camera: HttpClient, args: argparse.Namespace) -> tuple[dict, bytes, np.ndarray]:
    """Bind RGB/depth to a token and measured wrist transform, rejecting capture motion."""
    teaching.require_active_teach(arm)
    before = piper_pose_to_transform(finite_vector(arm.pose(), 6, "before capture pose"))
    capture = camera._json_request("/api/rgbd-capture", {})
    after = piper_pose_to_transform(finite_vector(arm.pose(), 6, "after capture pose"))
    teaching.require_active_teach(arm)
    translation = float(np.linalg.norm(after[:3, 3] - before[:3, 3]))
    relative_rotation = before[:3, :3].T @ after[:3, :3]
    rotation = math.degrees(math.acos(float(np.clip((np.trace(relative_rotation) - 1) / 2, -1, 1))))
    if translation > args.maximum_capture_position_change_mm or rotation > args.maximum_capture_rotation_change_deg:
        raise DemoError(f"arm moved during capture: {translation:.2f} mm / {rotation:.2f} degrees")
    token = capture.get("token")
    if (
        capture.get("status") != "ok" or not isinstance(token, str) or not token.isalnum()
        or type(capture.get("frame_sequence")) is not int
        or capture.get("image_png_url") != f"/rgbd/{token}.png"
    ):
        raise DemoError("invalid token-bound RGB-D capture response")
    image = camera._request_bytes(capture["image_png_url"])
    return capture, image, average_transforms([before, after])


def write_pixel_viewer(path: Path, image: bytes) -> tuple[int, int]:
    """Save a local frozen-image click viewer reporting exact original pixel coordinates."""
    decoded = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise DemoError("camera capture is not a decodable image")
    height, width = decoded.shape[:2]
    encoded = base64.b64encode(image).decode("ascii")
    html = """<!doctype html><meta charset="utf-8"><title>Candy point verification</title>
<p>Click the SAME marked reference point. Copy u v to the terminal.
Use the flat surface away from edges; deprojection samples a 17-by-17 depth window.</p>
<img id="frame" style="max-width:100%%" src="data:image/png;base64,%s">
<pre id="pixel">No pixel selected</pre>
<script>
const frame = document.getElementById('frame');
frame.onclick = event => {
  const box = frame.getBoundingClientRect();
  const u = Math.min(frame.naturalWidth - 1, Math.max(0,
    Math.floor((event.clientX - box.left) * frame.naturalWidth / box.width)));
  const v = Math.min(frame.naturalHeight - 1, Math.max(0,
    Math.floor((event.clientY - box.top) * frame.naturalHeight / box.height)));
  document.getElementById('pixel').textContent = u + ' ' + v;
};
</script>""" % encoded
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return width, height


def compare_point(
    reference: np.ndarray, response: dict, capture: dict, wrist: np.ndarray,
    camera_to_gripper: np.ndarray, pixel: list[int], maximum_error: float,
) -> dict:
    """Report camera-minus-reference XYZ and Euclidean error without changing bounds."""
    if (
        response.get("status") != "ok" or response.get("token") != capture["token"]
        or response.get("pixel_uv") != pixel
    ):
        raise DemoError("deprojection does not match the selected pixel and frozen frame")
    camera_point = finite_vector(response.get("camera_point_mm"), 3, "camera point")
    known = finite_vector(reference, 3, "independent reference point")
    measured = transform_camera_point_to_base(camera_point, wrist, camera_to_gripper)
    error = measured - known
    distance = float(np.linalg.norm(error))
    return {
        "reference_base_point_mm": known.tolist(),
        "camera_base_point_mm": measured.tolist(),
        "camera_minus_reference_mm": error.tolist(),
        "euclidean_error_mm": distance,
        "acceptance": {"maximum_error_mm": maximum_error, "quality_passed": distance <= maximum_error},
        "deprojection": response,
    }


def main() -> int:
    """Collect reference and camera data under manual teach control; always disconnect safely."""
    args = parse_args()
    arm = None
    try:
        thresholds = (
            args.sample_interval, args.maximum_joint_stddev_deg, args.maximum_position_stddev_mm,
            args.maximum_capture_position_change_mm, args.maximum_capture_rotation_change_deg,
            args.maximum_error_mm,
        )
        if args.samples < 3 or any(not math.isfinite(value) or value <= 0 for value in thresholds):
            raise DemoError("use at least three samples and positive finite thresholds")
        if args.output.name != "candy_point_verification.json":
            raise DemoError("output filename must be candy_point_verification.json to protect other artifacts")
        if not sys.stdin.isatty():
            raise DemoError("physical measurement requires an interactive terminal")
        camera = HttpClient(args.camera_url)
        health = camera.health()
        calibration = load_calibration(args.calibration_dir, health)
        reference_pose = None
        print(
            "READ-ONLY: no motor enable, joint, gripper, or hold commands.\n"
            "Stop all arm controllers. Keep supporting the arm in physical teach mode.\n"
            "Use a stationary marked dummy surface at candy-top height, with NO PERSON underneath.\n"
            "Do not reach below the tool-tip safety floor or move the marked surface.",
            flush=True,
        )
        try:
            arm = teaching.connect_readonly(iface=args.can_interface)
            input("Support the arm, enter PHYSICAL TEACH recording, then press Enter: ")
            teaching.require_active_teach(arm)
            if args.reference_point_mm is None:
                input("Hand-guide the tool TIP to the marked point without compressing it; hold still and press Enter: ")
                reference_pose = teaching.capture_point(arm, args)
                reference = np.array(reference_pose["tool_tip_base_point_mm"])
            else:
                reference = finite_vector(args.reference_point_mm, 3, "surveyed reference")
            print(f"Reference base XYZ: {reference.tolist()} mm")
            input(
                "Hand-guide to a viewing pose; move the tip away so the mark is unobstructed. "
                "Keep the D435 at least 300 mm from the surface; hold still and press Enter: "
            )
            capture, image, wrist = freeze_capture(arm, camera, args)
            viewer = args.output.with_suffix(".html")
            width, height = write_pixel_viewer(viewer, image)
            print(f"Open this frozen-image viewer in a browser: {viewer}")
        finally:
            if arm is not None:
                try:
                    teaching.confirm_power_off()
                finally:
                    arm.disconnect()
        while True:
            text = input("With controller OFF, click the marked point in the viewer; enter its u v coordinates: ")
            try:
                pixel = [int(value) for value in text.split()]
                if len(pixel) != 2 or not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
                    raise ValueError("pixel must be inside the frozen image")
            except ValueError as exc:
                print(f"INVALID pixel: {exc}")
                continue
            break
        response = camera._json_request("/api/deproject", {"token": capture["token"], "pixel_uv": pixel})
        result = compare_point(reference, response, capture, wrist, calibration, pixel, args.maximum_error_mm)
        print(f"Camera base XYZ: {result['camera_base_point_mm']} mm")
        print(f"Camera minus reference XYZ: {result['camera_minus_reference_mm']} mm")
        print(f"3D error: {result['euclidean_error_mm']:.2f} mm")
        passed = result["acceptance"]["quality_passed"]
        print(f"{'PASS' if passed else 'FAIL'} at {args.maximum_error_mm:g} mm comparison threshold.")
        print("This compares one point, not a certified calibration or verified workspace. Bounds are unchanged.")
        if input("Save verification record (including failures)? [y/N]: ").strip().lower() in ("y", "yes"):
            teaching.write_json_atomic(args.output, {
                **result, "captured_at": datetime.now(timezone.utc).isoformat(),
                "reference_pose": reference_pose, "capture": capture,
                "wrist_to_base": wrist.tolist(), "camera_to_gripper": calibration.tolist(),
                "camera_health": health, "tool_length_mm": teaching.C.TOOL_LEN_MM,
                "calibration_dir": str(args.calibration_dir.resolve()),
                "reference_method": "hand-guided tool tip" if reference_pose is not None else "operator-surveyed XYZ",
            })
            print(f"Saved {args.output}")
        else:
            print("Verification record not saved.")
        return 0 if passed else 2
    except KeyboardInterrupt:
        print("\nVerification interrupted; no automatic motion or bounds update.", file=sys.stderr)
        return 130
    except (RuntimeError, OSError, EOFError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: point verification failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
