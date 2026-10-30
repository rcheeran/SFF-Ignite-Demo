#!/usr/bin/env python3
"""D405 + OWL-ViT service with an FFmpeg MJPEG stream and three scan captures."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision import (
    OwlVitDetector,
    VisionError,
    annotate_capture,
    camera_point_from_pixel,
    robust_box_depth,
    robust_pixel_depth,
)


def env_float(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if not np.isfinite(value):
        raise VisionError(f"{name} must be finite")
    return value


def env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise VisionError(f"{name} must be positive")
    return value


def load_depth_correction(path: Path) -> tuple[float, float]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise VisionError(f"depth correction does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise VisionError(f"invalid depth correction JSON: {exc}") from exc
    acceptance = document.get("acceptance")
    if not isinstance(acceptance, dict) or acceptance.get("quality_passed") is not True:
        raise VisionError(f"depth correction has not passed quality checks: {path}")
    scale = float(document["scale"])
    offset = float(document["offset_mm"])
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(offset):
        raise VisionError("invalid depth correction scale or offset")
    return scale, offset


class D405FfmpegStream:
    def __init__(self) -> None:
        self.serial = os.environ.get("CAMERA_SERIAL", "")
        self.width = env_int("CAMERA_WIDTH", 640)
        self.height = env_int("CAMERA_HEIGHT", 480)
        self.fps = env_int("CAMERA_FPS", 30)
        self.jpeg_quality = env_int("FFMPEG_JPEG_QUALITY", 5)
        self.stop_event = threading.Event()
        self.frame_ready = threading.Condition()
        self.latest_frame: np.ndarray | None = None
        self.latest_depth: np.ndarray | None = None
        self.latest_jpeg: bytes | None = None
        self.sequence = 0
        self.error: str | None = None
        self.pipeline = None
        self.process: subprocess.Popen | None = None
        self.depth_scale: float | None = None
        self.intrinsics = None
        self.threads: list[threading.Thread] = []
        self.frame_annotator = None

    def start(self) -> None:
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
        )
        config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
        )
        profile = pipeline.start(config)
        device = profile.get_device()
        detected_serial = device.get_info(rs.camera_info.serial_number)
        if self.serial and detected_serial != self.serial:
            pipeline.stop()
            raise VisionError(
                f"expected D405 {self.serial}, detected {detected_serial}"
            )
        self.serial = detected_serial
        self.depth_scale = float(device.first_depth_sensor().get_depth_scale())
        self.intrinsics = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self.align = rs.align(rs.stream.color)
        command = [
            "ffmpeg",
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
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.pipeline = pipeline
        self.process = process
        self.threads = [
            threading.Thread(target=self._capture_loop, daemon=True),
            threading.Thread(target=self._encoder_loop, daemon=True),
        ]
        for thread in self.threads:
            thread.start()
        with self.frame_ready:
            self.frame_ready.wait_for(
                lambda: self.latest_jpeg is not None or self.error is not None,
                timeout=12,
            )
        self.check()
        if self.latest_jpeg is None:
            raise VisionError("D405/FFmpeg did not produce a frame within 12 seconds")

    def check(self) -> None:
        if self.error is not None:
            raise VisionError(self.error)

    def snapshot(
        self, after_sequence: int = -1, timeout: float = 3.0
    ) -> tuple[np.ndarray, np.ndarray, int]:
        deadline = time.monotonic() + timeout
        with self.frame_ready:
            while self.sequence <= after_sequence and time.monotonic() < deadline:
                self.frame_ready.wait(timeout=0.05)
            self.check()
            if (
                self.sequence <= after_sequence
                or self.latest_frame is None
                or self.latest_depth is None
            ):
                raise VisionError("timed out waiting for an aligned D405 frame")
            return (
                self.latest_frame.copy(),
                self.latest_depth.copy(),
                self.sequence,
            )

    def _capture_loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                frames = self.pipeline.wait_for_frames(timeout_ms=2000)
                aligned = self.align.process(frames)
                color = aligned.get_color_frame()
                depth = aligned.get_depth_frame()
                if not color or not depth:
                    continue
                frame = np.asanyarray(color.get_data()).copy()
                depth_raw = np.asanyarray(depth.get_data()).copy()
                with self.frame_ready:
                    self.latest_frame = frame.copy()
                    self.latest_depth = depth_raw
                    self.sequence += 1
                    self.frame_ready.notify_all()
                if self.frame_annotator is not None:
                    frame = self.frame_annotator(frame)
                cv2.rectangle(frame, (0, 0), (self.width, 34), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    "D405 LIVE - OWL-ViT service ready",
                    (8, 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 190, 0),
                    1,
                    cv2.LINE_AA,
                )
                if self.process.poll() is not None:
                    stderr = self.process.stderr.read().decode(errors="replace")
                    raise VisionError(f"FFmpeg exited: {stderr.strip()}")
                self.process.stdin.write(frame.tobytes())
        except Exception as exc:
            if not self.stop_event.is_set():
                self.error = str(exc)
                with self.frame_ready:
                    self.frame_ready.notify_all()

    def _encoder_loop(self) -> None:
        buffer = bytearray()
        try:
            while not self.stop_event.is_set():
                chunk = self.process.stdout.read(4096)
                if not chunk:
                    if self.process.poll() is not None:
                        stderr = self.process.stderr.read().decode(errors="replace")
                        raise VisionError(f"FFmpeg encoder stopped: {stderr.strip()}")
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

    def close(self) -> None:
        self.stop_event.set()
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
        if self.process is not None:
            if self.process.stdin:
                self.process.stdin.close()
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)


class VisionApplication:
    def __init__(self) -> None:
        self.model_name = os.environ.get(
            "OWL_MODEL", "google/owlvit-base-patch32"
        )
        self.default_confidence = env_float("DETECTION_CONFIDENCE", 0.2)
        if not 0 < self.default_confidence < 1:
            raise VisionError("DETECTION_CONFIDENCE must be between zero and one")
        self.depth_scale_correction, self.depth_offset_mm = load_depth_correction(
            Path(
                os.environ.get(
                    "DEPTH_CORRECTION",
                    "/calibration/depth_correction.json",
                )
            )
        )
        self.depth_settings = {
            "inset_fraction": env_float("BOX_INSET_FRACTION", 0.2),
            "minimum_pixels": env_int("MINIMUM_BOX_DEPTH_PIXELS", 100),
            "minimum_depth_m": env_float("MINIMUM_DEPTH_M", 0.05),
            "maximum_depth_m": env_float("MAXIMUM_DEPTH_M", 1.0),
            "cluster_bin_mm": env_float("DEPTH_CLUSTER_BIN_MM", 10.0),
            "cluster_tolerance_mm": env_float(
                "DEPTH_CLUSTER_TOLERANCE_MM", 15.0
            ),
            "minimum_cluster_fraction": env_float(
                "MINIMUM_DEPTH_CLUSTER_FRACTION", 0.35
            ),
        }
        print(f"Loading {self.model_name} on CUDA", flush=True)
        self.detector = OwlVitDetector(self.model_name, "cuda")
        self.camera = D405FfmpegStream()
        self.camera.frame_annotator = self.annotate_workspace
        self.capture_lock = threading.Lock()
        self.snapshots: dict[int, dict[str, Any]] = {}
        self.snapshot_jpegs: dict[int, bytes] = {}
        self.rgbd_frames: dict[str, dict[str, Any]] = {}
        self.workspace_overlay: dict[str, Any] | None = None
        self.workspace_overlay_lock = threading.Lock()
        dataset_dir = os.environ.get("DATASET_COLLECTION_DIR", "").strip()
        self.dataset_dir = Path(dataset_dir) if dataset_dir else None
        self.dataset_class_id = env_int("DATASET_CLASS_ID", 8)
        self.dataset_prompts = {
            " ".join(value.lower().replace("_", " ").replace("-", " ").split())
            for value in os.environ.get(
                "DATASET_COLLECTION_PROMPTS",
                "starburst,starburst candy",
            ).split(",")
            if value.strip()
        }
        if self.dataset_dir is not None:
            (self.dataset_dir / "images").mkdir(parents=True, exist_ok=True)
            (self.dataset_dir / "labels").mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        self.camera.start()
        print(
            f"Vision service ready: D405 {self.camera.serial}, "
            f"{self.camera.width}x{self.camera.height}@{self.camera.fps}",
            flush=True,
        )

    def localize_detection(
        self,
        detection: dict[str, Any],
        depth: np.ndarray,
    ) -> dict[str, Any]:
        """Add depth, camera XYZ, and optional long-axis geometry to one box."""
        candidate: dict[str, Any] = {
            "status": "rejected",
            "score": float(detection["score"]),
            "box_xyxy": [float(value) for value in detection["box_xyxy"]],
        }
        try:
            if detection.get("orientation_error"):
                raise VisionError(str(detection["orientation_error"]))
            pixel, raw_depth_mm, diagnostics = robust_box_depth(
                depth,
                detection["box_xyxy"],
                float(self.camera.depth_scale),
                **self.depth_settings,
            )
            corrected_depth_mm = (
                self.depth_scale_correction * raw_depth_mm
                + self.depth_offset_mm
            )
            camera_point = camera_point_from_pixel(
                self.camera.intrinsics,
                pixel,
                corrected_depth_mm,
            )
            orientation: dict[str, Any] = {}
            if "long_axis_pixels" in detection:
                axis_pixels = np.asarray(
                    detection["long_axis_pixels"],
                    dtype=np.float64,
                )
                if axis_pixels.shape != (2, 2):
                    raise VisionError(
                        "long-axis pixels must contain two image points"
                    )
                axis_camera_points = np.asarray(
                    [
                        camera_point_from_pixel(
                            self.camera.intrinsics,
                            tuple(endpoint),
                            corrected_depth_mm,
                        )
                        for endpoint in axis_pixels
                    ]
                )
                axis_vector = axis_camera_points[1] - axis_camera_points[0]
                axis_norm = float(np.linalg.norm(axis_vector))
                if axis_norm <= 1e-6:
                    raise VisionError(
                        "Starburst camera-frame orientation is degenerate"
                    )
                orientation = {
                    "long_axis_pixels": axis_pixels.tolist(),
                    "long_axis_camera_unit": (
                        axis_vector / axis_norm
                    ).tolist(),
                    "orientation_aspect_ratio": float(
                        detection["orientation_aspect_ratio"]
                    ),
                    "orientation_mask_fraction": float(
                        detection["orientation_mask_fraction"]
                    ),
                }
            candidate.update(
                {
                    "status": "detected",
                    "pixel_uv": list(pixel),
                    "raw_depth_mm": raw_depth_mm,
                    "corrected_depth_mm": corrected_depth_mm,
                    "camera_point_mm": camera_point.tolist(),
                    **orientation,
                    **diagnostics,
                }
            )
        except VisionError as exc:
            candidate["error"] = str(exc)
        return candidate

    def save_dataset_capture(
        self,
        frame: np.ndarray,
        prompt: str,
        detections: list[dict[str, Any]],
    ) -> dict[str, str]:
        """Persist an unannotated image and its provisional YOLO labels."""
        if self.dataset_dir is None or not detections:
            return {}
        normalized_prompt = " ".join(
            prompt.lower().replace("_", " ").replace("-", " ").split()
        )
        if normalized_prompt not in self.dataset_prompts:
            return {}

        height, width = frame.shape[:2]
        labels = []
        for detection in detections:
            left, top, right, bottom = [
                float(value) for value in detection["box_xyxy"]
            ]
            center_x = ((left + right) / 2.0) / width
            center_y = ((top + bottom) / 2.0) / height
            box_width = (right - left) / width
            box_height = (bottom - top) / height
            labels.append(
                f"{self.dataset_class_id} {center_x:.6f} {center_y:.6f} "
                f"{box_width:.6f} {box_height:.6f}"
            )

        captured_at = datetime.now(timezone.utc)
        stem = (
            f"{captured_at:%Y%m%dT%H%M%S}_{captured_at.microsecond:06d}Z_"
            f"{uuid.uuid4().hex[:8]}"
        )
        image_relative = Path("images") / f"{stem}.jpg"
        label_relative = Path("labels") / f"{stem}.txt"
        image_path = self.dataset_dir / image_relative
        label_path = self.dataset_dir / label_relative
        image_temporary = image_path.with_suffix(".jpg.tmp")
        label_temporary = label_path.with_suffix(".txt.tmp")

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        if not ok:
            raise VisionError("could not encode raw dataset JPEG")
        try:
            image_temporary.write_bytes(encoded.tobytes())
            label_temporary.write_text("\n".join(labels) + "\n", encoding="utf-8")
            os.replace(image_temporary, image_path)
            os.replace(label_temporary, label_path)
        except OSError as exc:
            for path in (
                image_temporary,
                label_temporary,
                image_path,
                label_path,
            ):
                path.unlink(missing_ok=True)
            raise VisionError(f"could not persist dataset capture: {exc}") from exc
        return {
            "dataset_image": image_relative.as_posix(),
            "dataset_label": label_relative.as_posix(),
        }

    def capture(self, slot: int, prompt: str, confidence: float) -> dict[str, Any]:
        if slot not in (1, 2, 3):
            raise VisionError("slot must be 1, 2, or 3")
        if not prompt.strip():
            raise VisionError("prompt cannot be empty")
        if not 0 < confidence < 1:
            raise VisionError("confidence must be between zero and one")
        if not self.capture_lock.acquire(blocking=False):
            raise VisionError("another inference capture is already running")
        try:
            frame, depth, sequence = self.camera.snapshot(
                after_sequence=self.camera.sequence
            )
            detections = self.detector.detect(frame, prompt.strip(), confidence)
            result: dict[str, Any] = {
                "status": "not_found",
                "slot": slot,
                "prompt": prompt.strip(),
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "frame_sequence": sequence,
                "model": self.model_name,
            }
            result.update(
                self.save_dataset_capture(frame, prompt.strip(), detections)
            )
            candidates = [
                self.localize_detection(detection, depth)
                for detection in detections
            ]
            valid_candidates = [
                candidate
                for candidate in candidates
                if candidate["status"] == "detected"
            ]
            result["detection_count"] = len(detections)
            result["candidate_count"] = len(valid_candidates)
            result["candidates"] = candidates
            if valid_candidates:
                result.update(valid_candidates[0])
            elif candidates:
                result.update(candidates[0])
            annotated = annotate_capture(frame, slot, prompt.strip(), result)
            annotated = self.annotate_workspace(annotated)
            ok, encoded = cv2.imencode(".jpg", annotated)
            if not ok:
                raise VisionError("could not encode capture JPEG")
            result["snapshot_url"] = f"/snapshots/{slot}.jpg"
            self.snapshots[slot] = result
            self.snapshot_jpegs[slot] = encoded.tobytes()
            return result
        finally:
            self.capture_lock.release()

    def set_workspace_overlay(
        self,
        base_to_camera: list[list[float]],
        minimum: list[float],
        maximum: list[float],
    ) -> dict[str, Any]:
        transform = np.asarray(base_to_camera, dtype=np.float64)
        lower = np.asarray(minimum, dtype=np.float64)
        upper = np.asarray(maximum, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise VisionError("base_to_camera must be a finite 4x4 matrix")
        if (
            lower.shape != (3,)
            or upper.shape != (3,)
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower >= upper)
        ):
            raise VisionError("workspace bounds must be finite increasing XYZ")
        with self.workspace_overlay_lock:
            self.workspace_overlay = {
                "base_to_camera": transform,
                "minimum": lower,
                "maximum": upper,
                "updated_monotonic": time.monotonic(),
            }
        return {"status": "ok", "expires_after_seconds": 5.0}

    def annotate_workspace(self, frame: np.ndarray) -> np.ndarray:
        with self.workspace_overlay_lock:
            overlay = self.workspace_overlay
            if (
                overlay is None
                or time.monotonic() - overlay["updated_monotonic"] > 5.0
            ):
                return frame
            transform = overlay["base_to_camera"].copy()
            lower = overlay["minimum"].copy()
            upper = overlay["maximum"].copy()
        camera_from_base = np.linalg.inv(transform)
        corners_base = np.asarray(
            [
                [x, y, z, 1.0]
                for x in (lower[0], upper[0])
                for y in (lower[1], upper[1])
                for z in (lower[2], upper[2])
            ],
            dtype=np.float64,
        )
        corners_camera = (camera_from_base @ corners_base.T).T[:, :3]
        intrinsics = self.camera.intrinsics
        pixels: list[tuple[int, int] | None] = []
        for x, y, z in corners_camera:
            if z <= 10.0:
                pixels.append(None)
                continue
            u = int(round(intrinsics.fx * x / z + intrinsics.ppx))
            v = int(round(intrinsics.fy * y / z + intrinsics.ppy))
            limit_x, limit_y = frame.shape[1] * 3, frame.shape[0] * 3
            pixels.append(
                (
                    int(np.clip(u, -limit_x, limit_x)),
                    int(np.clip(v, -limit_y, limit_y)),
                )
            )
        edges = (
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6),
            (5, 7), (6, 7),
        )
        annotated = frame.copy()
        for start, end in edges:
            if pixels[start] is not None and pixels[end] is not None:
                cv2.line(
                    annotated,
                    pixels[start],
                    pixels[end],
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
        cv2.putText(
            annotated,
            "GREEN: object safety workspace",
            (8, frame.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        return annotated

    def capture_rgbd(self) -> dict[str, Any]:
        frame, depth, sequence = self.camera.snapshot(
            after_sequence=self.camera.sequence
        )
        token = uuid.uuid4().hex
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise VisionError("could not encode synchronized RGB frame")
        captured_at = datetime.now(timezone.utc).isoformat()
        self.rgbd_frames[token] = {
            "depth": depth,
            "jpeg": encoded.tobytes(),
            "frame_sequence": sequence,
            "captured_at": captured_at,
        }
        while len(self.rgbd_frames) > 30:
            self.rgbd_frames.pop(next(iter(self.rgbd_frames)))
        return {
            "status": "ok",
            "token": token,
            "captured_at": captured_at,
            "frame_sequence": sequence,
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "image_url": f"/rgbd/{token}.jpg",
        }

    def deproject(self, token: str, pixel: list[int]) -> dict[str, Any]:
        stored = self.rgbd_frames.get(token)
        if stored is None:
            raise VisionError("unknown or expired RGB-D frame token")
        raw_depth_mm, diagnostics = robust_pixel_depth(
            stored["depth"],
            pixel,
            float(self.camera.depth_scale),
            radius=8,
            minimum_pixels=40,
            minimum_depth_m=float(self.depth_settings["minimum_depth_m"]),
            maximum_depth_m=float(self.depth_settings["maximum_depth_m"]),
        )
        corrected_depth_mm = (
            self.depth_scale_correction * raw_depth_mm + self.depth_offset_mm
        )
        camera_point = camera_point_from_pixel(
            self.camera.intrinsics,
            (int(pixel[0]), int(pixel[1])),
            corrected_depth_mm,
        )
        return {
            "status": "ok",
            "token": token,
            "pixel_uv": [int(pixel[0]), int(pixel[1])],
            "raw_depth_mm": raw_depth_mm,
            "corrected_depth_mm": corrected_depth_mm,
            "camera_point_mm": camera_point.tolist(),
            **diagnostics,
        }


def page_html() -> bytes:
    object_cards = "".join(
        f"""
        <section><h3>Scan {slot}</h3>
        <img id="shot{slot}" src="/snapshots/{slot}.jpg" alt="Scan {slot}">
        <pre id="meta{slot}">Not captured</pre></section>
        """
        for slot in (1, 2, 3)
    )
    hand_cards = "".join(
        f"""
        <section><h3>Hand scan {slot}</h3>
        <img id="handshot{slot}" alt="Hand scan {slot}">
        <pre id="handmeta{slot}">Not captured</pre></section>
        """
        for slot in (4, 5, 6)
    )
    return f"""<!doctype html>
<meta charset="utf-8">
<title>Piper OWL-ViT vision</title>
<style>
body{{margin:0;background:#111;color:#eee;font:15px sans-serif}}
main{{max-width:1280px;margin:auto;padding:12px}}
.live{{width:100%;max-width:960px;border:1px solid #555}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
section{{background:#222;padding:8px}} section img{{width:100%;min-height:120px}}
pre{{white-space:pre-wrap;font-size:12px}} @media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
</style>
<main><h2>Piper D405 / OWL-ViT service</h2>
<img class="live" src="/stream.mjpg" alt="Live D405 stream">
<p style="color:#00ff00">Green wireframe: projected object safety workspace
X=100..450 mm, Y=-350..350 mm, Z=-50..100 mm. The overlay appears only while
fresh Piper pose telemetry is being supplied by demo.py.</p>
<h2>Latest hand scan</h2><div class="grid">{hand_cards}</div>
<h2>Latest object scan</h2><div class="grid">{object_cards}</div></main>
<script>
async function refresh(){{
  const r=await fetch('/api/snapshots',{{cache:'no-store'}});
  const data=await r.json();
  for(const slot of [1,2,3]){{
    const value=data.snapshots[String(slot)];
    if(value){{
      document.getElementById('shot'+slot).src='/snapshots/'+slot+'.jpg?t='+Date.now();
      document.getElementById('meta'+slot).textContent=JSON.stringify(value,null,2);
    }}
  }}
  try{{
    const hr=await fetch('/api/hand-snapshots',{{cache:'no-store'}});
    const handData=await hr.json();
    for(const slot of [4,5,6]){{
      const value=handData.snapshots[String(slot)];
      if(value){{
        document.getElementById('handshot'+slot).src='/hand-snapshots/'+slot+'.jpg?t='+Date.now();
        document.getElementById('handmeta'+slot).textContent=JSON.stringify(value,null,2);
      }}
    }}
  }}catch(error){{
    for(const slot of [4,5,6]){{
      document.getElementById('handmeta'+slot).textContent='Hand service unavailable: '+error;
    }}
  }}
}}
setInterval(refresh,1000); refresh();
</script>""".encode()


def create_server(app: VisionApplication, port: int) -> ThreadingHTTPServer:
    page = page_html()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            print(f"http: {format % args}", flush=True)

        def send_bytes(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status: int, value: dict[str, Any]) -> None:
            self.send_bytes(
                status,
                "application/json",
                json.dumps(value).encode("utf-8"),
            )

        def do_GET(self) -> None:
            path = urllib.parse.urlsplit(self.path).path
            if path in ("/", "/index.html"):
                self.send_bytes(200, "text/html; charset=utf-8", page)
                return
            if path == "/healthz":
                try:
                    app.camera.check()
                    self.send_json(
                        200,
                        {
                            "status": "ok",
                            "model": app.model_name,
                            "camera_serial": app.camera.serial,
                            "cuda": True,
                        },
                    )
                except VisionError as exc:
                    self.send_json(503, {"status": "error", "error": str(exc)})
                return
            if path == "/api/snapshots":
                self.send_json(
                    200,
                    {
                        "snapshots": {
                            str(slot): value
                            for slot, value in app.snapshots.items()
                        }
                    },
                )
                return
            if path == "/api/hand-snapshots":
                try:
                    with urllib.request.urlopen(
                        "http://127.0.0.1:8095/api/snapshots",
                        timeout=3,
                    ) as response:
                        body = response.read()
                    self.send_bytes(200, "application/json", body)
                except Exception as exc:
                    self.send_json(
                        503,
                        {"status": "error", "error": str(exc), "snapshots": {}},
                    )
                return
            if path.startswith("/hand-snapshots/") and path.endswith(".jpg"):
                slot_name = path.removeprefix("/hand-snapshots/")
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:8095/snapshots/{slot_name}",
                        timeout=3,
                    ) as response:
                        body = response.read()
                    self.send_bytes(200, "image/jpeg", body)
                except Exception:
                    self.send_error(404)
                return
            if path.startswith("/snapshots/") and path.endswith(".jpg"):
                try:
                    slot = int(path.removeprefix("/snapshots/").removesuffix(".jpg"))
                    jpeg = app.snapshot_jpegs[slot]
                except (ValueError, KeyError):
                    self.send_error(404)
                    return
                self.send_bytes(200, "image/jpeg", jpeg)
                return
            if path.startswith("/rgbd/") and path.endswith(".jpg"):
                token = path.removeprefix("/rgbd/").removesuffix(".jpg")
                stored = app.rgbd_frames.get(token)
                if stored is None:
                    self.send_error(404)
                else:
                    self.send_bytes(200, "image/jpeg", stored["jpeg"])
                return
            if path != "/stream.mjpg":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while not app.camera.stop_event.is_set():
                    with app.camera.frame_ready:
                        app.camera.frame_ready.wait(timeout=1)
                        jpeg = app.camera.latest_jpeg
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
            path = urllib.parse.urlsplit(self.path).path
            if path == "/api/rgbd-capture":
                try:
                    self.send_json(200, app.capture_rgbd())
                except VisionError as exc:
                    self.send_json(400, {"status": "rejected", "error": str(exc)})
                return
            if path not in (
                "/api/capture",
                "/api/deproject",
                "/api/workspace-overlay",
            ):
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 4096:
                    raise VisionError("request body must be 1..4096 bytes")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise VisionError("request must be a JSON object")
                if path == "/api/workspace-overlay":
                    result = app.set_workspace_overlay(
                        body.get("base_to_camera"),
                        body.get("minimum_mm"),
                        body.get("maximum_mm"),
                    )
                elif path == "/api/deproject":
                    pixel = body.get("pixel_uv")
                    if (
                        not isinstance(pixel, list)
                        or len(pixel) != 2
                        or not all(isinstance(value, int) for value in pixel)
                    ):
                        raise VisionError("pixel_uv must contain two integers")
                    result = app.deproject(str(body.get("token", "")), pixel)
                else:
                    result = app.capture(
                        int(body.get("slot")),
                        str(body.get("prompt", "")),
                        float(body.get("confidence", app.default_confidence)),
                    )
                self.send_json(200, result)
            except (VisionError, ValueError, TypeError, json.JSONDecodeError) as exc:
                status = 409 if "already running" in str(exc) else 400
                self.send_json(status, {"status": "rejected", "error": str(exc)})

    class Server(ThreadingHTTPServer):
        allow_reuse_address = True
        daemon_threads = True

    return Server(("0.0.0.0", port), Handler)


def main() -> int:
    port = env_int("PORT", 8094)
    app = VisionApplication()
    try:
        app.start()
        server = create_server(app, port)
        print(f"HTTP/FFmpeg UI listening on 0.0.0.0:{port}", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping vision service", flush=True)
    finally:
        app.camera.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
