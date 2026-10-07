#!/usr/bin/env python3
"""Align the wrist camera in teach mode and record the fixed scan pose.

Purpose:
    1. Start the configured RealSense D405 color stream.
    2. Pipe ArUco-annotated frames through FFmpeg for MJPEG encoding and serve
       the result over HTTP so the operator can align all required tags.
    3. Wait for the operator to enter teach mode with the physical teach button.
    4. Wait while the operator hand-guides the arm to the desired scan pose.
    5. Use the firmware-proven CAN sequence to exit teach mode, re-enable the
       motors, and hold the arm at its current joints before recording telemetry.

Safety:
    This script never commands a new joint or Cartesian target. Teach mode is
    entered with the physical button. To exit, it streams the Piper firmware's
    position-control command, enables the motors, and holds the measured current
    joints. Hold the arm until motor holding torque returns.

Expected output:
    A live browser page at http://<jetson-ip>:8094/ by default.
    The stream labels visible ArUco IDs and whether all required IDs are seen.

    calibration/scan_pose.json
        The averaged six-joint pose, Cartesian pose, stability statistics,
        camera identity/settings, required marker IDs, and preview URL.

    The FFmpeg server and D405 stream stop automatically when the script exits.

Example:
    python3 prepare-piper/homography/record_scan_pose.py
"""

from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
from piper_sdk import C_PiperInterface_V2

from common import CalibrationError, load_json, parse_marker_ids, write_json_atomic

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "calibration" / "scan_pose.json"
TEACH_CTRL_MODE = 0x02
TEACH_TIMEOUT_SECONDS = 120.0
SETTLE_SECONDS = 1.0
TEACH_EXIT_TIMEOUT_SECONDS = 8.0
TEACH_EXIT_SPEED_PERCENT = 15
TEACH_HOLD_SPEED_PERCENT = 10
GRIPPER_EFFORT = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--can-interface", default="can_piper")
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="do not open the D405 or start FFmpeg (diagnostic fallback only)",
    )
    return parser.parse_args()


def lan_ip() -> str:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 1))
        return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


class FfmpegRealSensePreview:
    """Own the D405, encode annotated frames with FFmpeg, and serve MJPEG."""

    def __init__(
        self,
        camera_config: dict,
        aruco_config: dict,
        enable_depth: bool = False,
    ) -> None:
        self.serial = str(camera_config.get("serial") or "")
        self.width = int(camera_config["width"])
        self.height = int(camera_config["height"])
        self.fps = int(camera_config["fps"])
        self.port = int(camera_config["preview_port"])
        self.jpeg_quality = int(camera_config["preview_jpeg_quality"])
        try:
            self.required_ids = [
                int(value) for value in aruco_config["required_marker_ids"]
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationError("required marker IDs must be a list of integers") from exc
        if len(set(self.required_ids)) != len(self.required_ids):
            raise CalibrationError("required marker IDs must be unique")
        dictionary_name = str(aruco_config["dictionary"])
        dictionary_id = getattr(cv2.aruco, dictionary_name, None)
        if dictionary_id is None:
            raise CalibrationError(f"unknown OpenCV ArUco dictionary: {dictionary_name}")
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(dictionary_id),
            cv2.aruco.DetectorParameters(),
        )
        self.enable_depth = enable_depth
        self.align = None
        self.depth_scale: float | None = None
        self.color_intrinsics = None
        self.pipeline = None
        self.process: subprocess.Popen | None = None
        self.capture_thread: threading.Thread | None = None
        self.encoder_thread: threading.Thread | None = None
        self.server_thread: threading.Thread | None = None
        self.server: ThreadingHTTPServer | None = None
        self.stop_event = threading.Event()
        self.frame_ready = threading.Condition()
        self.latest_jpeg: bytes | None = None
        self.latest_frame: np.ndarray | None = None
        self.latest_depth: np.ndarray | None = None
        self.latest_markers: dict[int, np.ndarray] = {}
        self.frame_sequence = 0
        self.error: str | None = None
        self.camera: dict = {}
        self.camera_matrix: np.ndarray | None = None
        self.distortion_coefficients: np.ndarray | None = None
        self.url = f"http://{lan_ip()}:{self.port}/"

    def start(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                "ffmpeg is not installed; rerun bash prepare-jetson/setup.sh"
            )

        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
        )
        if self.enable_depth:
            config.enable_stream(
                rs.stream.depth,
                self.width,
                self.height,
                rs.format.z16,
                self.fps,
            )
        profile = pipeline.start(config)
        device = profile.get_device()
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_profile.get_intrinsics()
        self.color_intrinsics = intrinsics
        if self.enable_depth:
            self.align = rs.align(rs.stream.color)
            self.depth_scale = float(
                device.first_depth_sensor().get_depth_scale()
            )
        detected_serial = device.get_info(rs.camera_info.serial_number)
        if self.serial and detected_serial != self.serial:
            pipeline.stop()
            raise RuntimeError(
                f"expected RealSense {self.serial}, detected {detected_serial}"
            )
        self.camera = {
            "name": device.get_info(rs.camera_info.name),
            "serial": detected_serial,
            "firmware": device.get_info(rs.camera_info.firmware_version),
            "intrinsics": {
                "width": intrinsics.width,
                "height": intrinsics.height,
                "fx": intrinsics.fx,
                "fy": intrinsics.fy,
                "ppx": intrinsics.ppx,
                "ppy": intrinsics.ppy,
                "distortion_model": str(intrinsics.model),
                "distortion_coefficients": list(intrinsics.coeffs),
            },
        }
        self.camera_matrix = np.array(
            [
                [intrinsics.fx, 0.0, intrinsics.ppx],
                [0.0, intrinsics.fy, intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.distortion_coefficients = np.asarray(
            intrinsics.coeffs, dtype=np.float64
        )
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            str(self.fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "mjpeg",
            "-q:v",
            str(self.jpeg_quality),
            "-f",
            "image2pipe",
            "pipe:1",
        ]
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception:
            pipeline.stop()
            raise
        self.pipeline = pipeline
        self.process = process
        self.capture_thread = threading.Thread(
            target=self._stream_loop, name="d405-ffmpeg-preview", daemon=True
        )
        self.encoder_thread = threading.Thread(
            target=self._read_encoded_frames,
            name="ffmpeg-mjpeg-reader",
            daemon=True,
        )
        self.server = self._create_server()
        self.server_thread = threading.Thread(
            target=self.server.serve_forever,
            name="d405-preview-http",
            daemon=True,
        )
        self.encoder_thread.start()
        self.capture_thread.start()
        self.server_thread.start()
        with self.frame_ready:
            self.frame_ready.wait_for(
                lambda: self.latest_jpeg is not None or self.error is not None,
                timeout=8,
            )
        self.check()
        if self.latest_jpeg is None:
            raise RuntimeError("D405/FFmpeg preview did not produce a frame within 8 seconds")

    def _create_server(self) -> ThreadingHTTPServer:
        preview = self
        page = (
            b"<!doctype html><meta charset='utf-8'><title>D405 scan alignment</title>"
            b"<style>body{margin:0;background:#111;color:#ddd;font:16px sans-serif}"
            b"img{display:block;max-width:100vw;max-height:100vh;margin:auto}</style>"
            b"<img src='/stream.mjpg' alt='D405 alignment stream'>"
        )

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: object) -> None:
                pass

            def do_GET(self) -> None:
                if self.path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(page)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(page)
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
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

        class PreviewServer(ThreadingHTTPServer):
            allow_reuse_address = True
            daemon_threads = True

        return PreviewServer(("0.0.0.0", self.port), Handler)

    def _stream_loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                frames = self.pipeline.wait_for_frames(timeout_ms=2000)
                if self.align is not None:
                    frames = self.align.process(frames)
                color = frames.get_color_frame()
                if not color:
                    continue
                frame = np.asanyarray(color.get_data()).copy()
                depth = None
                if self.enable_depth:
                    depth_frame = frames.get_depth_frame()
                    if not depth_frame:
                        continue
                    depth = np.asanyarray(depth_frame.get_data()).copy()
                corners, ids, _ = self.detector.detectMarkers(frame)
                visible_ids = [] if ids is None else [int(value) for value in ids.reshape(-1)]
                markers = {}
                if ids is not None:
                    markers = {
                        int(marker_id): marker_corners.reshape(4, 2).copy()
                        for marker_id, marker_corners in zip(
                            ids.reshape(-1), corners, strict=True
                        )
                    }
                with self.frame_ready:
                    self.latest_frame = frame.copy()
                    self.latest_depth = depth
                    self.latest_markers = markers
                    self.frame_sequence += 1
                    self.frame_ready.notify_all()
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                all_visible = set(self.required_ids).issubset(visible_ids)
                status = (
                    f"Visible IDs: {sorted(visible_ids)} | "
                    f"Required: {self.required_ids} | "
                    f"{'READY' if all_visible else 'ALIGN CAMERA'}"
                )
                color_bgr = (0, 180, 0) if all_visible else (0, 0, 255)
                cv2.rectangle(frame, (0, 0), (self.width, 34), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    status,
                    (8, 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    color_bgr,
                    1,
                    cv2.LINE_AA,
                )
                if self.process.poll() is not None:
                    stderr = self.process.stderr.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"FFmpeg exited unexpectedly: {stderr.strip()}")
                try:
                    self.process.stdin.write(frame.tobytes())
                except (BrokenPipeError, OSError) as exc:
                    if not self.stop_event.is_set():
                        stderr = self.process.stderr.read().decode(
                            "utf-8", errors="replace"
                        )
                        raise RuntimeError(
                            f"FFmpeg preview connection failed: {stderr.strip()}"
                        ) from exc
        except Exception as exc:
            if not self.stop_event.is_set():
                self.error = str(exc)
                with self.frame_ready:
                    self.frame_ready.notify_all()

    def wait_for_marker(
        self,
        marker_id: int,
        after_sequence: int = -1,
        timeout: float = 3.0,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        deadline = time.monotonic() + timeout
        with self.frame_ready:
            while time.monotonic() < deadline:
                if self.error:
                    raise RuntimeError(self.error)
                if (
                    self.frame_sequence > after_sequence
                    and self.latest_frame is not None
                    and marker_id in self.latest_markers
                ):
                    return (
                        self.latest_frame.copy(),
                        self.latest_markers[marker_id].copy(),
                        self.frame_sequence,
                    )
                self.frame_ready.wait(timeout=min(0.2, deadline - time.monotonic()))
        raise RuntimeError(f"marker ID {marker_id} was not visible within {timeout:.1f}s")

    def wait_for_marker_with_depth(
        self,
        marker_id: int,
        after_sequence: int = -1,
        timeout: float = 3.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        if not self.enable_depth:
            raise RuntimeError("depth was not enabled for this preview")
        deadline = time.monotonic() + timeout
        with self.frame_ready:
            while time.monotonic() < deadline:
                if self.error:
                    raise RuntimeError(self.error)
                if (
                    self.frame_sequence > after_sequence
                    and self.latest_frame is not None
                    and self.latest_depth is not None
                    and marker_id in self.latest_markers
                ):
                    return (
                        self.latest_frame.copy(),
                        self.latest_depth.copy(),
                        self.latest_markers[marker_id].copy(),
                        self.frame_sequence,
                    )
                self.frame_ready.wait(timeout=min(0.2, deadline - time.monotonic()))
        raise RuntimeError(
            f"marker ID {marker_id} with aligned depth was not visible "
            f"within {timeout:.1f}s"
        )

    def _read_encoded_frames(self) -> None:
        buffer = bytearray()
        try:
            while not self.stop_event.is_set():
                chunk = self.process.stdout.read(4096)
                if not chunk:
                    if self.process.poll() is not None and not self.stop_event.is_set():
                        stderr = self.process.stderr.read().decode(
                            "utf-8", errors="replace"
                        )
                        raise RuntimeError(
                            f"FFmpeg encoder stopped unexpectedly: {stderr.strip()}"
                        )
                    continue
                buffer.extend(chunk)
                while True:
                    start = buffer.find(b"\xff\xd8")
                    if start < 0:
                        buffer.clear()
                        break
                    end = buffer.find(b"\xff\xd9", start + 2)
                    if end < 0:
                        if start:
                            del buffer[:start]
                        break
                    jpeg = bytes(buffer[start : end + 2])
                    del buffer[: end + 2]
                    with self.frame_ready:
                        self.latest_jpeg = jpeg
                        self.frame_ready.notify_all()
        except Exception as exc:
            if not self.stop_event.is_set():
                self.error = str(exc)
                with self.frame_ready:
                    self.frame_ready.notify_all()

    def check(self) -> None:
        if self.error:
            raise RuntimeError(self.error)
        if self.process is not None and self.process.poll() is not None:
            stderr = self.process.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"FFmpeg exited unexpectedly: {stderr.strip()}")

    def close(self) -> None:
        self.stop_event.set()
        with self.frame_ready:
            self.frame_ready.notify_all()
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        for thread in (
            self.capture_thread,
            self.encoder_thread,
            self.server_thread,
        ):
            if thread is not None:
                thread.join(timeout=3)
        if self.pipeline is not None:
            self.pipeline.stop()


def control_mode(piper: C_PiperInterface_V2) -> int:
    status = piper.GetArmStatus()
    if getattr(status, "Hz", 0) <= 0:
        raise RuntimeError("Piper arm-status telemetry is unavailable")
    return int(status.arm_status.ctrl_mode)


def wait_for_teach_mode(
    piper: C_PiperInterface_V2,
    expected: bool,
    preview: FfmpegRealSensePreview | None,
    timeout: float = TEACH_TIMEOUT_SECONDS,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if preview is not None:
            preview.check()
        if (control_mode(piper) == TEACH_CTRL_MODE) == expected:
            return True
        time.sleep(0.2)
    return (control_mode(piper) == TEACH_CTRL_MODE) == expected


def take_control_from_teach(
    piper: C_PiperInterface_V2,
    preview: FfmpegRealSensePreview | None,
    timeout: float = TEACH_EXIT_TIMEOUT_SECONDS,
) -> bool:
    """Exit teach mode and hold the measured pose using this arm's proven sequence."""
    deadline = time.monotonic() + timeout
    while control_mode(piper) == TEACH_CTRL_MODE and time.monotonic() < deadline:
        if preview is not None:
            preview.check()
        piper.MotionCtrl_2(0x01, 0x01, TEACH_EXIT_SPEED_PERCENT, 0x00)
        time.sleep(0.01)
    if control_mode(piper) == TEACH_CTRL_MODE:
        return False

    piper.MotionCtrl_2(0x01, 0x01, TEACH_EXIT_SPEED_PERCENT, 0x00)
    time.sleep(0.1)
    deadline = time.monotonic() + timeout
    while not all(piper.GetArmEnableStatus()) and time.monotonic() < deadline:
        if preview is not None:
            preview.check()
        piper.EnablePiper()
        time.sleep(0.3)
    if not all(piper.GetArmEnableStatus()):
        return False

    joints = piper.GetArmJointMsgs().joint_state
    gripper = piper.GetArmGripperMsgs().gripper_state
    piper.GripperCtrl(
        int(gripper.grippers_angle),
        GRIPPER_EFFORT,
        0x03,
        0,
    )
    piper.MotionCtrl_2(0x01, 0x01, TEACH_HOLD_SPEED_PERCENT, 0x00)
    piper.JointCtrl(
        *[int(getattr(joints, f"joint_{index}")) for index in range(1, 7)]
    )
    return True


def read_pose_sample(piper: C_PiperInterface_V2) -> tuple[np.ndarray, np.ndarray]:
    joints_message = piper.GetArmJointMsgs()
    pose_message = piper.GetArmEndPoseMsgs()
    if getattr(joints_message, "Hz", 0) <= 0 or getattr(pose_message, "Hz", 0) <= 0:
        raise RuntimeError("Piper joint/end-pose telemetry is unavailable")
    if control_mode(piper) == TEACH_CTRL_MODE:
        raise RuntimeError("Piper re-entered teach mode while recording the scan pose")
    joints = joints_message.joint_state
    pose = pose_message.end_pose
    joint_degrees = np.array(
        [getattr(joints, f"joint_{index}") / 1000.0 for index in range(1, 7)],
        dtype=np.float64,
    )
    cartesian = np.array(
        [
            pose.X_axis / 1000.0,
            pose.Y_axis / 1000.0,
            pose.Z_axis / 1000.0,
            pose.RX_axis / 1000.0,
            pose.RY_axis / 1000.0,
            pose.RZ_axis / 1000.0,
        ],
        dtype=np.float64,
    )
    return joint_degrees, cartesian


def main() -> int:
    args = parse_args()
    try:
        config = load_json(args.config)
        camera_config = config["camera"]
        aruco_config = config["aruco"]
        collection = config["collection"]
        sample_count = int(collection["pose_samples_per_marker"])
        interval = float(collection["sample_interval_seconds"])
        maximum_stddev_mm = float(collection["maximum_position_stddev_mm"])
        if sample_count < 2 or interval <= 0 or maximum_stddev_mm <= 0:
            raise CalibrationError("collection settings must be positive")
    except (KeyError, TypeError, ValueError, CalibrationError) as exc:
        print(f"ERROR: invalid configuration: {exc}", file=sys.stderr)
        return 2

    preview: FfmpegRealSensePreview | None = None
    piper: C_PiperInterface_V2 | None = None
    try:
        if not args.no_preview:
            preview = FfmpegRealSensePreview(camera_config, aruco_config)
            preview.start()
            print(f"\nLive D405 alignment page: {preview.url}")
            print("Open that URL in a browser. The overlay turns green when all tags are visible.\n")

        piper = C_PiperInterface_V2(args.can_interface)
        piper.ConnectPort()
        time.sleep(1.0)
        control_mode(piper)

        print("Hold the arm, then press the PHYSICAL TEACH BUTTON to make it movable.")
        if not wait_for_teach_mode(piper, True, preview):
            raise RuntimeError(
                f"Piper did not enter teach mode within {TEACH_TIMEOUT_SECONDS:.0f} seconds"
            )
        print("TEACH MODE detected.")
        print("Guide the wrist until all marker IDs are visible and the overlay says READY.")
        print("Keep the board centered, avoid a grazing camera angle, and let the image settle.")
        input(
            "\nWhen alignment is correct, keep holding the arm and press Enter here: "
        )
        if control_mode(piper) != TEACH_CTRL_MODE:
            raise RuntimeError("Piper left teach mode before alignment was confirmed")

        print("\nKEEP HOLDING THE ARM.")
        input("Press Enter to take control back and stiffen the arm at this pose: ")
        if not take_control_from_teach(piper, preview):
            raise RuntimeError(
                "Piper did not leave teach mode and enable its motors; keep holding it"
            )
        print(f"Teach mode exited. Holding scan pose; settling for {SETTLE_SECONDS:.1f}s...")
        time.sleep(SETTLE_SECONDS)

        joint_samples: list[np.ndarray] = []
        cartesian_samples: list[np.ndarray] = []
        for _ in range(sample_count):
            if preview is not None:
                preview.check()
            joints, cartesian = read_pose_sample(piper)
            joint_samples.append(joints)
            cartesian_samples.append(cartesian)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nCancelled; no scan-pose file was written.", file=sys.stderr)
        return 130
    except (RuntimeError, CalibrationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if piper is not None:
            piper.DisconnectPort()
        if preview is not None:
            preview.close()

    joints_array = np.stack(joint_samples)
    cartesian_array = np.stack(cartesian_samples)
    joints_mean = joints_array.mean(axis=0)
    joints_stddev = joints_array.std(axis=0)
    cartesian_mean = cartesian_array.mean(axis=0)
    cartesian_stddev = cartesian_array.std(axis=0)
    position_stddev = float(np.max(cartesian_stddev[:3]))
    if position_stddev > maximum_stddev_mm:
        print(
            f"ERROR: scan pose position stddev {position_stddev:.3f}mm exceeds "
            f"{maximum_stddev_mm:.3f}mm; let the arm settle and retry",
            file=sys.stderr,
        )
        return 1

    result = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "can_interface": args.can_interface,
        "sample_count": sample_count,
        "camera": None if preview is None else preview.camera,
        "image": {
            "width": int(camera_config["width"]),
            "height": int(camera_config["height"]),
            "fps": int(camera_config["fps"]),
        },
        "required_marker_ids": parse_marker_ids(aruco_config["required_marker_ids"]),
        "preview_url": None if preview is None else preview.url,
        "joints_deg": {
            f"joint_{index + 1}": float(joints_mean[index]) for index in range(6)
        },
        "joint_stddev_deg": {
            f"joint_{index + 1}": float(joints_stddev[index]) for index in range(6)
        },
        "end_pose_mm_deg": {
            key: float(cartesian_mean[index])
            for index, key in enumerate(("x", "y", "z", "rx", "ry", "rz"))
        },
        "end_pose_stddev_mm_deg": {
            key: float(cartesian_stddev[index])
            for index, key in enumerate(("x", "y", "z", "rx", "ry", "rz"))
        },
    }
    write_json_atomic(args.output, result)
    print(f"Scan pose written to {args.output}")
    print(
        "Joint target (deg): "
        + ", ".join(f"J{i + 1}={value:.3f}" for i, value in enumerate(joints_mean))
    )
    print(
        "End pose: "
        f"X={cartesian_mean[0]:.2f} Y={cartesian_mean[1]:.2f} "
        f"Z={cartesian_mean[2]:.2f} mm"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
