"""Intel RealSense D405 wrist camera for the Piper toolset.

Wraps a single RealSense color+depth pipeline behind a small thread-safe API:
grab a color snapshot (saved to disk), read depth at the image centre or a
region, and report device info. pyrealsense2/opencv are imported lazily so the
toolset runs without them (mock mode / no camera).
"""
from __future__ import annotations

import os
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_DIR = os.path.join(_HERE, "captures")


class CameraUnavailable(RuntimeError):
    pass


class PiperCamera:
    def __init__(self, width=640, height=480, fps=30, warmup=15):
        self._w, self._h, self._fps, self._warmup = width, height, fps, warmup
        self._lock = threading.Lock()
        self._pipeline = None
        self._info = {}
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    def start(self):
        try:
            import pyrealsense2 as rs
        except Exception as e:  # noqa: BLE001
            raise CameraUnavailable(f"pyrealsense2 not available: {e}")
        ctx = rs.context()
        devs = ctx.query_devices()
        if len(devs) == 0:
            raise CameraUnavailable("no RealSense device detected — check USB")
        dev = devs[0]
        self._info = {
            "name": dev.get_info(rs.camera_info.name),
            "serial": dev.get_info(rs.camera_info.serial_number),
            "firmware": dev.get_info(rs.camera_info.firmware_version),
        }
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self._w, self._h, rs.format.bgr8, self._fps)
        config.enable_stream(rs.stream.depth, self._w, self._h, rs.format.z16, self._fps)
        pipeline.start(config)
        # warm up so auto-exposure settles
        for _ in range(self._warmup):
            pipeline.wait_for_frames(timeout_ms=5000)
        self._pipeline = pipeline
        return self._info

    @property
    def info(self):
        return dict(self._info)

    def _frames(self):
        if self._pipeline is None:
            raise CameraUnavailable("camera not started")
        return self._pipeline.wait_for_frames(timeout_ms=5000)

    def capture(self, label: str = "snapshot") -> dict:
        """Grab a color frame + centre depth, save a JPG, return metadata."""
        import cv2
        import numpy as np
        with self._lock:
            frames = self._frames()
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                raise CameraUnavailable("incomplete frame (no color/depth)")
            color_img = np.asanyarray(color.get_data())
            depth_img = np.asanyarray(depth.get_data())
            path = os.path.join(SNAPSHOT_DIR, f"{label}_{int(time.time())}.jpg")
            cv2.imwrite(path, color_img)
        h, w = depth_img.shape
        return {
            "path": path,
            "width": int(w), "height": int(h),
            "center_depth_mm": int(depth_img[h // 2, w // 2]),
            "device": self._info.get("name", "RealSense"),
        }

    def depth_at(self, u_frac: float = 0.5, v_frac: float = 0.5, patch: int = 5) -> dict:
        """Median depth (mm) in a small patch around a normalized (u,v) point."""
        import numpy as np
        with self._lock:
            frames = self._frames()
            depth = frames.get_depth_frame()
            if not depth:
                raise CameraUnavailable("no depth frame")
            depth_img = np.asanyarray(depth.get_data())
        h, w = depth_img.shape
        u = int(np.clip(u_frac, 0, 1) * (w - 1))
        v = int(np.clip(v_frac, 0, 1) * (h - 1))
        r = max(1, patch)
        region = depth_img[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1]
        valid = region[region > 0]
        return {
            "u_frac": round(u_frac, 3), "v_frac": round(v_frac, 3),
            "pixel": [u, v],
            "depth_mm": int(np.median(valid)) if valid.size else 0,
            "valid": bool(valid.size),
        }

    def stop(self):
        with self._lock:
            if self._pipeline is not None:
                try:
                    self._pipeline.stop()
                except Exception:
                    pass
                self._pipeline = None
