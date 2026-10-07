"""Hardware-free HTTP routing tests for the vision service."""

from __future__ import annotations

import threading
import tempfile
import unittest
import urllib.request
import json
import time
from pathlib import Path
from types import SimpleNamespace

from app import create_server
from app import VisionApplication
import cv2
import numpy as np
from unittest import mock


class VisionHttpTests(unittest.TestCase):
    def test_capture_returns_all_localized_candidates(self) -> None:
        app = VisionApplication.__new__(VisionApplication)
        app.capture_lock = threading.Lock()
        app.model_name = "candy-yolo11s"
        app.depth_scale_correction = 1.0
        app.depth_offset_mm = 0.0
        app.depth_settings = {}
        app.snapshots = {}
        app.snapshot_jpegs = {}
        app.dataset_dir = None
        app.annotate_workspace = lambda frame: frame
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        depth = np.full((120, 160), 300, dtype=np.uint16)
        app.camera = SimpleNamespace(
            sequence=10,
            depth_scale=0.001,
            intrinsics=object(),
            snapshot=lambda after_sequence: (frame, depth, after_sequence + 1),
        )
        app.detector = SimpleNamespace(
            detect=lambda *_args: [
                {"score": 0.9, "box_xyxy": [10.0, 20.0, 50.0, 70.0]},
                {"score": 0.8, "box_xyxy": [90.0, 25.0, 140.0, 75.0]},
            ]
        )
        localized = [
            {
                "status": "detected",
                "score": score,
                "box_xyxy": box,
                "pixel_uv": pixel,
                "corrected_depth_mm": 300.0,
                "camera_point_mm": [float(pixel[0]), float(pixel[1]), 300.0],
            }
            for score, box, pixel in (
                (0.9, [10.0, 20.0, 50.0, 70.0], [30, 45]),
                (0.8, [90.0, 25.0, 140.0, 75.0], [115, 50]),
            )
        ]

        with mock.patch.object(
            app,
            "localize_detection",
            side_effect=localized,
        ):
            result = app.capture(1, "starburst candy", 0.25)

        self.assertEqual(result["status"], "detected")
        self.assertEqual(result["detection_count"], 2)
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(result["score"], 0.9)

    def test_dataset_capture_saves_raw_image_and_yolo_labels(self) -> None:
        app = VisionApplication.__new__(VisionApplication)
        app.dataset_class_id = 8
        app.dataset_prompts = {"starburst", "starburst candy"}
        frame = np.full((100, 200, 3), 127, dtype=np.uint8)
        detections = [
            {"score": 0.9, "box_xyxy": [20.0, 10.0, 100.0, 50.0]},
            {"score": 0.8, "box_xyxy": [120.0, 40.0, 180.0, 90.0]},
        ]

        with tempfile.TemporaryDirectory() as temporary:
            app.dataset_dir = Path(temporary)
            (app.dataset_dir / "images").mkdir()
            (app.dataset_dir / "labels").mkdir()

            result = app.save_dataset_capture(
                frame,
                "starburst candy",
                detections,
            )

            image_path = app.dataset_dir / result["dataset_image"]
            label_path = app.dataset_dir / result["dataset_label"]
            saved = cv2.imread(str(image_path))
            self.assertIsNotNone(saved)
            np.testing.assert_allclose(saved, frame, atol=1)
            self.assertEqual(
                label_path.read_text(encoding="utf-8"),
                "8 0.300000 0.300000 0.400000 0.400000\n"
                "8 0.750000 0.650000 0.300000 0.500000\n",
            )

    def test_dataset_capture_skips_empty_detections(self) -> None:
        app = VisionApplication.__new__(VisionApplication)
        app.dataset_class_id = 8
        app.dataset_prompts = {"starburst candy"}
        with tempfile.TemporaryDirectory() as temporary:
            app.dataset_dir = Path(temporary)

            result = app.save_dataset_capture(
                np.zeros((10, 10, 3), dtype=np.uint8),
                "starburst candy",
                [],
            )

            self.assertEqual(result, {})
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_workspace_projection_draws_green_xy_edges(self) -> None:
        app = VisionApplication.__new__(VisionApplication)
        app.workspace_overlay_lock = threading.Lock()
        app.workspace_overlay = {
            "base_to_camera": np.eye(4),
            "minimum": np.array([-50.0, -50.0, 200.0]),
            "maximum": np.array([50.0, 50.0, 300.0]),
            "updated_monotonic": time.monotonic(),
        }
        app.camera = SimpleNamespace(
            intrinsics=SimpleNamespace(fx=400.0, fy=400.0, ppx=320.0, ppy=240.0)
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        annotated = app.annotate_workspace(frame)

        green = cv2.inRange(
            annotated,
            np.array([0, 200, 0], dtype=np.uint8),
            np.array([20, 255, 20], dtype=np.uint8),
        )
        self.assertGreater(int(np.count_nonzero(green)), 100)
    def test_cache_busted_snapshot_url_returns_jpeg(self) -> None:
        jpeg = b"\xff\xd8test-jpeg\xff\xd9"
        app = SimpleNamespace(
            snapshots={1: {"status": "detected", "slot": 1}},
            snapshot_jpegs={1: jpeg},
        )
        server = create_server(app, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/snapshots/1.jpg?t=123",
                timeout=2,
            ) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers["Content-Type"], "image/jpeg")
                self.assertEqual(response.read(), jpeg)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_rgbd_image_and_deprojection_routes(self) -> None:
        jpeg = b"\xff\xd8rgbd\xff\xd9"
        app = SimpleNamespace(
            rgbd_frames={"token": {"jpeg": jpeg}},
            capture_rgbd=lambda: {
                "status": "ok",
                "token": "token",
                "image_url": "/rgbd/token.jpg",
            },
            deproject=lambda token, pixel: {
                "status": "ok",
                "token": token,
                "pixel_uv": pixel,
                "camera_point_mm": [1.0, 2.0, 3.0],
            },
        )
        server = create_server(app, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/deproject",
                data=json.dumps(
                    {"token": "token", "pixel_uv": [10, 20]}
                ).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                value = json.loads(response.read())
            self.assertEqual(value["camera_point_mm"], [1.0, 2.0, 3.0])
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/rgbd/token.jpg?t=1",
                timeout=2,
            ) as response:
                self.assertEqual(response.read(), jpeg)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
