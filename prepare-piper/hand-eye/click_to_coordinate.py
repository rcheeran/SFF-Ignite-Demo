#!/usr/bin/env python3
"""Convert browser-selected D405 pixels into Piper base-frame coordinates.

Purpose:
    Let the operator hand-guide the wrist to a useful view, then click any
    visible object or surface in the browser. For each click, collect 30 aligned
    depth frames, reject temporal outliers, apply the calibrated D405 depth
    correction, and transform the robust 3D point into Piper base coordinates.
    The exact taught joint pose is continuously streamed throughout the test.
    No new robot target or Cartesian motion is commanded.

Expected output:
    Browser results showing pixel, raw/corrected depth, camera XYZ, Piper base
    XYZ, temporal inliers and depth MAD.
    calibration/click_coordinates.json
        All successful click measurements from the session.
    calibration/click_coordinate_NNN.jpg
        The selected pixel and resulting coordinate for each measurement.

Example:
    python3 prepare-piper/hand-eye/click_to_coordinate.py
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
from hand_eye_common import (  # noqa: E402
    HandEyeError,
    average_transforms,
    piper_pose_to_transform,
    rotation_error_degrees,
)
from piper import config as PIPER_C  # noqa: E402
from piper.robot import Piper, connect_for_teach  # noqa: E402
from record_scan_pose import FfmpegRealSensePreview  # noqa: E402
from verify_depth_coordinates import (  # noqa: E402
    hold_until_power_off,
    load_depth_correction,
    rigid_transform,
    robust_center_depth_m,
    transform_camera_point_to_base,
)

DEFAULT_CONFIG = HAND_EYE_DIR / "config.json"
DEFAULT_HAND_EYE = HAND_EYE_DIR / "calibration" / "hand_eye.json"
DEFAULT_DEPTH_CORRECTION = HAND_EYE_DIR / "calibration" / "depth_correction.json"
DEFAULT_OUTPUT = HAND_EYE_DIR / "calibration" / "click_coordinates.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--hand-eye", type=Path, default=DEFAULT_HAND_EYE)
    parser.add_argument(
        "--depth-correction", type=Path, default=DEFAULT_DEPTH_CORRECTION
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--can-interface", default="can_piper")
    return parser.parse_args()


def validate_pixel(u: object, v: object, width: int, height: int) -> tuple[int, int]:
    if isinstance(u, bool) or isinstance(v, bool):
        raise ValueError("pixel coordinates must be numbers")
    try:
        pixel_u = int(round(float(u)))
        pixel_v = int(round(float(v)))
    except (TypeError, ValueError) as exc:
        raise ValueError("pixel coordinates must be numbers") from exc
    if not (0 <= pixel_u < width and 0 <= pixel_v < height):
        raise ValueError(
            f"pixel ({pixel_u}, {pixel_v}) is outside {width}x{height}"
        )
    return pixel_u, pixel_v


def temporal_inlier_mask(
    raw_depths_mm: np.ndarray,
    mad_multiplier: float,
    outlier_floor_mm: float,
) -> tuple[np.ndarray, float, float, float]:
    depths = np.asarray(raw_depths_mm, dtype=np.float64)
    if depths.ndim != 1 or len(depths) < 3 or not np.all(np.isfinite(depths)):
        raise HandEyeError("at least three finite temporal depths are required")
    median = float(np.median(depths))
    mad = float(np.median(np.abs(depths - median)))
    threshold = max(mad_multiplier * mad, outlier_floor_mm)
    return np.abs(depths - median) <= threshold, median, mad, threshold


class ClickCoordinatePreview(FfmpegRealSensePreview):
    def __init__(self, camera_config: dict, aruco_config: dict) -> None:
        super().__init__(camera_config, aruco_config, enable_depth=True)
        self.clicks: queue.Queue[tuple[int, int]] = queue.Queue(maxsize=1)
        self.finish_event = threading.Event()
        self.ready_event = threading.Event()
        self.busy_event = threading.Event()
        self.result_lock = threading.Lock()
        self.result: dict = {
            "status": "aligning",
            "message": "Align the camera in teach mode.",
        }

    def set_result(self, result: dict) -> None:
        with self.result_lock:
            self.result = result

    def _create_server(self) -> ThreadingHTTPServer:
        preview = self
        page = f"""<!doctype html>
<meta charset="utf-8">
<title>D405 click-to-coordinate</title>
<style>
body{{margin:0;background:#111;color:#eee;font:16px sans-serif}}
main{{max-width:900px;margin:auto;padding:12px}}
#frame{{display:block;width:100%;height:auto;cursor:crosshair;border:1px solid #555}}
#result{{white-space:pre-wrap;background:#222;padding:12px;min-height:9em}}
button{{font-size:16px;padding:8px 16px}}
</style>
<main>
<h2>D405 click-to-coordinate</h2>
<p>After alignment completes, click a stable surface away from edges.</p>
<img id="frame" src="/stream.mjpg" width="{self.width}" height="{self.height}">
<p><button id="finish">Finish session</button></p>
<pre id="result">Waiting for alignment...</pre>
</main>
<script>
const image = document.getElementById('frame');
const result = document.getElementById('result');
image.addEventListener('click', async event => {{
  const rect = image.getBoundingClientRect();
  const u = (event.clientX - rect.left) * {self.width} / rect.width;
  const v = (event.clientY - rect.top) * {self.height} / rect.height;
  const response = await fetch('/click', {{
    method:'POST', headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{u, v}})
  }});
  const body = await response.json();
  if (!response.ok) result.textContent = body.error;
}});
document.getElementById('finish').addEventListener('click', async () => {{
  await fetch('/finish', {{method:'POST'}});
  result.textContent = 'Finish requested. Follow the terminal shutdown prompt.';
}});
async function update() {{
  try {{
    const response = await fetch('/result', {{cache:'no-store'}});
    result.textContent = JSON.stringify(await response.json(), null, 2);
  }} catch (error) {{ result.textContent = String(error); }}
}}
setInterval(update, 500); update();
</script>""".encode()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: object) -> None:
                pass

            def send_json(self, status: int, value: dict) -> None:
                body = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(page)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(page)
                    return
                if self.path == "/result":
                    with preview.result_lock:
                        self.send_json(200, preview.result)
                    return
                if self.path != "/stream.mjpg":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                )
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    while not preview.stop_event.is_set():
                        with preview.frame_ready:
                            preview.frame_ready.wait(timeout=1)
                            jpeg = preview.latest_jpeg
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                        )
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_POST(self) -> None:
                if self.path == "/finish":
                    preview.finish_event.set()
                    self.send_json(200, {"status": "finishing"})
                    return
                if self.path != "/click":
                    self.send_error(404)
                    return
                try:
                    if not preview.ready_event.is_set():
                        self.send_json(
                            409, {"error": "finish teach alignment before clicking"}
                        )
                        return
                    if preview.busy_event.is_set():
                        self.send_json(
                            409, {"error": "a click is already being processed"}
                        )
                        return
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 1024:
                        raise ValueError("invalid request size")
                    body = json.loads(self.rfile.read(length))
                    if not isinstance(body, dict):
                        raise ValueError("request must be a JSON object")
                    pixel = validate_pixel(
                        body.get("u"), body.get("v"), preview.width, preview.height
                    )
                    preview.busy_event.set()
                    preview.clicks.put_nowait(pixel)
                    preview.set_result(
                        {
                            "status": "sampling",
                            "pixel_uv": list(pixel),
                            "message": "Collecting temporal depth samples...",
                        }
                    )
                    self.send_json(202, {"status": "sampling", "pixel_uv": pixel})
                except queue.Full:
                    preview.busy_event.clear()
                    self.send_json(409, {"error": "a click is already being processed"})
                except (ValueError, json.JSONDecodeError) as exc:
                    self.send_json(400, {"error": str(exc)})

        class ClickServer(ThreadingHTTPServer):
            allow_reuse_address = True
            daemon_threads = True

        return ClickServer(("0.0.0.0", self.port), Handler)


def measure_click(
    preview: ClickCoordinatePreview,
    piper: Piper,
    hold_joints: np.ndarray,
    pixel: tuple[int, int],
    sample_count: int,
    radius_px: int,
    minimum_pixels: int,
    minimum_depth_m: float,
    maximum_depth_m: float,
    correction_scale: float,
    correction_offset_mm: float,
    camera_to_gripper: np.ndarray,
    maximum_position_deviation: float,
    maximum_rotation_deviation: float,
    maximum_depth_mad: float,
    minimum_inlier_fraction: float,
    mad_multiplier: float,
    outlier_floor_mm: float,
) -> tuple[dict, np.ndarray]:
    import pyrealsense2 as rs

    raw_depths = []
    camera_points = []
    base_points = []
    robot_transforms = []
    frame = None
    sequence = -1
    for _ in range(sample_count):
        piper.send_joints(hold_joints, None, PIPER_C.SPEED_REST)
        deadline = time.monotonic() + 3.0
        with preview.frame_ready:
            while preview.frame_sequence <= sequence and time.monotonic() < deadline:
                preview.frame_ready.wait(timeout=0.05)
            preview.check()
            if (
                preview.frame_sequence <= sequence
                or preview.latest_frame is None
                or preview.latest_depth is None
            ):
                raise RuntimeError("timed out waiting for aligned depth")
            frame = preview.latest_frame.copy()
            depth = preview.latest_depth.copy()
            sequence = preview.frame_sequence
        raw_depth_m, _, _ = robust_center_depth_m(
            depth,
            np.asarray(pixel),
            float(preview.depth_scale),
            radius_px,
            minimum_pixels,
            minimum_depth_m,
            maximum_depth_m,
        )
        corrected_depth_m = (
            correction_scale * raw_depth_m * 1000.0 + correction_offset_mm
        ) / 1000.0
        camera_point_mm = (
            np.asarray(
                rs.rs2_deproject_pixel_to_point(
                    preview.color_intrinsics,
                    [float(pixel[0]), float(pixel[1])],
                    corrected_depth_m,
                )
            )
            * 1000.0
        )
        robot_transform = piper_pose_to_transform(piper.pose())
        base_point = transform_camera_point_to_base(
            camera_point_mm, robot_transform, camera_to_gripper
        )
        raw_depths.append(raw_depth_m * 1000.0)
        camera_points.append(camera_point_mm)
        base_points.append(base_point)
        robot_transforms.append(robot_transform)

    raw_depths_array = np.asarray(raw_depths)
    inliers, raw_median, depth_mad, threshold = temporal_inlier_mask(
        raw_depths_array, mad_multiplier, outlier_floor_mm
    )
    inlier_fraction = float(np.mean(inliers))
    robot_mean = average_transforms(robot_transforms)
    position_deviation = max(
        np.linalg.norm(value[:3, 3] - robot_mean[:3, 3])
        for value in robot_transforms
    )
    rotation_deviation = max(
        rotation_error_degrees(robot_mean, value) for value in robot_transforms
    )
    reasons = []
    if inlier_fraction < minimum_inlier_fraction:
        reasons.append(
            f"depth inlier fraction {inlier_fraction:.2f} < "
            f"{minimum_inlier_fraction:.2f}"
        )
    if depth_mad > maximum_depth_mad:
        reasons.append(
            f"depth MAD {depth_mad:.2f}mm > {maximum_depth_mad:.2f}mm"
        )
    if position_deviation > maximum_position_deviation:
        reasons.append(
            f"robot moved {position_deviation:.2f}mm > "
            f"{maximum_position_deviation:.2f}mm"
        )
    if rotation_deviation > maximum_rotation_deviation:
        reasons.append(
            f"robot rotated {rotation_deviation:.2f}deg > "
            f"{maximum_rotation_deviation:.2f}deg"
        )
    if reasons:
        raise RuntimeError("; ".join(reasons))
    camera_consensus = np.median(np.asarray(camera_points)[inliers], axis=0)
    base_consensus = np.median(np.asarray(base_points)[inliers], axis=0)
    corrected_median = correction_scale * raw_median + correction_offset_mm
    result = {
        "status": "success",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "pixel_uv": list(pixel),
        "raw_depth_median_mm": raw_median,
        "corrected_depth_median_mm": corrected_median,
        "depth_mad_mm": depth_mad,
        "temporal_outlier_threshold_mm": threshold,
        "inlier_count": int(np.sum(inliers)),
        "sample_count": sample_count,
        "inlier_fraction": inlier_fraction,
        "camera_point_mm": camera_consensus.tolist(),
        "base_point_mm": base_consensus.tolist(),
        "maximum_robot_position_deviation_mm": float(position_deviation),
        "maximum_robot_rotation_deviation_deg": float(rotation_deviation),
    }
    annotated = frame.copy()
    cv2.drawMarker(
        annotated, pixel, (0, 255, 0), cv2.MARKER_CROSS, 28, 2
    )
    cv2.putText(
        annotated,
        (
            f"base=({base_consensus[0]:.1f},"
            f"{base_consensus[1]:.1f},{base_consensus[2]:.1f})mm"
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
        click_config = config["click_coordinate"]
        correction_scale, correction_offset, correction = load_depth_correction(
            args.depth_correction, disabled=False
        )
        if correction is None:
            raise HandEyeError(
                f"depth correction does not exist: {args.depth_correction}"
            )
        camera_to_gripper = rigid_transform(hand_eye, "camera_to_gripper")
        sample_count = int(click_config["sample_count"])
        radius_px = int(click_config["center_window_radius_px"])
        minimum_pixels = int(click_config["minimum_valid_depth_pixels"])
        minimum_depth_m = float(click_config["minimum_depth_m"])
        maximum_depth_m = float(click_config["maximum_depth_m"])
        maximum_position_deviation = float(
            click_config["maximum_robot_position_deviation_mm"]
        )
        maximum_rotation_deviation = float(
            click_config["maximum_robot_rotation_deviation_deg"]
        )
        maximum_depth_mad = float(
            click_config["maximum_temporal_depth_mad_mm"]
        )
        minimum_inlier_fraction = float(
            click_config["minimum_temporal_inlier_fraction"]
        )
        mad_multiplier = float(click_config["temporal_mad_multiplier"])
        outlier_floor_mm = float(click_config["temporal_outlier_floor_mm"])
    except (KeyError, TypeError, ValueError, CalibrationError, HandEyeError) as exc:
        print(f"ERROR: cannot start click tool: {exc}", file=sys.stderr)
        return 2

    preview_config = {
        "dictionary": config["aruco"]["dictionary"],
        "required_marker_ids": [],
    }
    preview = ClickCoordinatePreview(camera_config, preview_config)
    piper: Piper | None = None
    hold_joints: np.ndarray | None = None
    results: list[dict] = []
    try:
        preview.start()
        piper = connect_for_teach(iface=args.can_interface)
        print(f"\nClick-to-coordinate view: {preview.url}")
        print("Hold the arm and press the PHYSICAL TEACH BUTTON.")
        if not piper.wait_teach(True):
            raise RuntimeError("Piper did not enter teach mode")
        print("TEACH MODE detected.")
        input(
            "Move the wrist until the test object is clearly visible, let it "
            "settle, then press Enter: "
        )
        print("\nKEEP HOLDING THE ARM.")
        print("Press the PHYSICAL TEACH BUTTON again; the arm should stiffen.")
        if not piper.wait_teach_stopped():
            raise RuntimeError("teach recording did not stop")
        hold_joints = piper.joints_deg()
        input("Arm stiffened. Press Enter to take CAN control back: ")
        if not piper.teach_stop(hold_joints=hold_joints):
            raise RuntimeError("Piper did not return to stable CAN control")
        preview.set_result(
            {
                "status": "ready",
                "message": "Click a stable surface away from edges.",
            }
        )
        preview.ready_event.set()
        print("READY: click a stable surface in the browser. Use Finish when done.")

        while not preview.finish_event.is_set():
            piper.send_joints(hold_joints, None, PIPER_C.SPEED_REST)
            try:
                pixel = preview.clicks.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                result, annotated = measure_click(
                    preview,
                    piper,
                    hold_joints,
                    pixel,
                    sample_count,
                    radius_px,
                    minimum_pixels,
                    minimum_depth_m,
                    maximum_depth_m,
                    correction_scale,
                    correction_offset,
                    camera_to_gripper,
                    maximum_position_deviation,
                    maximum_rotation_deviation,
                    maximum_depth_mad,
                    minimum_inlier_fraction,
                    mad_multiplier,
                    outlier_floor_mm,
                )
                index = len(results) + 1
                image_path = args.output.parent / f"click_coordinate_{index:03d}.jpg"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(image_path), annotated):
                    raise RuntimeError(f"could not write {image_path}")
                result["index"] = index
                result["image"] = str(image_path)
                results.append(result)
                write_json_atomic(
                    args.output,
                    {
                        "schema_version": 1,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "hand_eye": str(args.hand_eye),
                        "depth_correction": str(args.depth_correction),
                        "measurements": results,
                    },
                )
                preview.set_result(result)
                base = result["base_point_mm"]
                print(
                    f"CLICK {index}: base=({base[0]:.1f}, {base[1]:.1f}, "
                    f"{base[2]:.1f})mm, inliers="
                    f"{result['inlier_count']}/{result['sample_count']}, "
                    f"depth MAD={result['depth_mad_mm']:.2f}mm"
                )
            except (RuntimeError, HandEyeError) as exc:
                preview.set_result(
                    {
                        "status": "rejected",
                        "pixel_uv": list(pixel),
                        "error": str(exc),
                    }
                )
                print(f"CLICK REJECTED: {exc}")
            finally:
                preview.busy_event.clear()
    except KeyboardInterrupt:
        print("\nCancelled.")
    except RuntimeError as exc:
        print(f"ERROR: click tool failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if piper is not None and hold_joints is None and piper.in_teach():
            print(
                "\nKEEP SUPPORTING THE ARM. Press the PHYSICAL TEACH BUTTON "
                "again; the arm should stiffen.",
                file=sys.stderr,
            )
            if piper.wait_teach_stopped():
                hold_joints = piper.joints_deg()
                if not piper.teach_stop(hold_joints=hold_joints):
                    print(
                        "SAFETY ERROR: CAN hold could not be restored. Keep "
                        "supporting the arm and turn controller power off.",
                        file=sys.stderr,
                    )
            else:
                print(
                    "SAFETY ERROR: teach recording did not stop. Keep supporting "
                    "the arm and turn controller power off.",
                    file=sys.stderr,
                )
        if piper is not None and hold_joints is not None:
            print(
                "\nKEEP SUPPORTING THE ARM. The saved pose will remain streamed "
                "until controller power is OFF."
            )
            hold_until_power_off(piper, hold_joints)
        if piper is not None:
            piper.disconnect()
        preview.close()

    print(f"Saved {len(results)} click measurement(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
