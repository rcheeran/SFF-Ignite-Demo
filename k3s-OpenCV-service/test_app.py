"""Hardware-free tests for the independent camera and token localization API."""

import json
import os
import threading
import tempfile
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pyrealsense2 as rs

import base_app
from app import VisionApplication, create_server, page_html
from vision import VisionError


def make_application() -> VisionApplication:
    app = VisionApplication.__new__(VisionApplication)
    app.model_name = "unused-camera-metadata"
    app.capture_lock = threading.Lock()
    app.rgbd_lock = threading.Lock()
    app.rgbd_frames = {}
    app.snapshots = {}
    app.snapshot_jpegs = {}
    app.dataset_dir = None
    app.depth_scale_correction = 1.0
    app.depth_offset_mm = 0.0
    app.depth_settings = {
        "inset_fraction": 0.2, "minimum_pixels": 100,
        "minimum_depth_m": 0.05, "maximum_depth_m": 1.0,
        "cluster_bin_mm": 10.0, "cluster_tolerance_mm": 15.0,
        "minimum_cluster_fraction": 0.35,
    }
    app.annotate_workspace = lambda frame: frame
    intrinsics = rs.intrinsics()
    intrinsics.width, intrinsics.height = 640, 480
    intrinsics.fx = intrinsics.fy = 600.0
    intrinsics.ppx, intrinsics.ppy = 320.0, 240.0
    intrinsics.model = rs.distortion.none
    intrinsics.coeffs = [0.0] * 5
    frame = np.full((480, 640, 3), 220, dtype=np.uint8)
    cv2.rectangle(frame, (130, 155), (230, 185), (0, 0, 255), -1)
    cv2.rectangle(frame, (380, 310), (420, 350), (0, 0, 255), -1)
    app.camera = SimpleNamespace(
        name="Intel RealSense D435",
        serial="test-camera", sequence=10, depth_scale=0.001, intrinsics=intrinsics,
        check=mock.Mock(),
        snapshot=mock.Mock(return_value=(
            frame, np.full((480, 640), 300, dtype=np.uint16), 11
        )),
    )
    return app


class CameraHttpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = make_application()
        self.server = create_server(self.app, 0)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.assertFalse(self.thread.is_alive())

    def request(self, path: str, payload=None):
        request = urllib.request.Request(
            self.url + path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            data = response.read()
            return json.loads(data) if response.headers.get_content_type() == "application/json" else data

    def localization(self, captured, detections=None):
        return {
            "token": captured["token"], "slot": 1, "prompt": "starburst",
            "model": "candy-yolo11s",
            "detections": detections if detections is not None else [
                {"score": 0.9, "box_xyxy": [120, 140, 240, 200]}
            ],
        }

    def test_midpoint_pickup_uses_axis_center_and_existing_corrected_box_depth(self) -> None:
        captured = self.request("/api/rgbd-capture", {})
        self.app.depth_scale_correction = 1.1
        self.app.depth_offset_mm = 10
        detection = {
            "score": 0.9, "box_xyxy": [120, 140, 240, 200],
            "long_axis_pixels": [[140, 160], [211, 181]],
            "orientation_aspect_ratio": 3.0, "orientation_mask_fraction": 0.5,
        }
        legacy = self.request("/api/localize", self.localization(captured, [detection]))
        midpoint = self.request("/api/localize", self.localization(
            captured, [{**detection, "pickup_point": "long_axis_midpoint"}]
        ))
        self.assertEqual(midpoint["status"], "detected")
        self.assertEqual(midpoint["pickup_point"], "long_axis_midpoint")
        self.assertEqual(midpoint["pixel_uv"], [175.5, 170.5])
        self.assertNotEqual(midpoint["pixel_uv"], legacy["pixel_uv"])
        self.assertEqual(midpoint["depth_sample_pixel_uv"], legacy["pixel_uv"])
        self.assertAlmostEqual(midpoint["corrected_depth_mm"], 340)
        expected = rs.rs2_deproject_pixel_to_point(
            self.app.camera.intrinsics, [175.5, 170.5], 340
        )
        np.testing.assert_allclose(midpoint["camera_point_mm"], expected)
        self.assertEqual(midpoint["long_axis_camera_unit"], legacy["long_axis_camera_unit"])
        self.assertEqual(midpoint["candidates"][0]["camera_point_mm"], midpoint["camera_point_mm"])
        self.assertTrue(self.request(midpoint["snapshot_url"]).startswith(b"\xff\xd8"))
        self.app.camera.snapshot.assert_called_once()

    def test_midpoint_requires_usable_endpoints_without_fallback(self) -> None:
        captured = self.request("/api/rgbd-capture", {})
        for orientation in (
            {},
            {
                "long_axis_pixels": [[150, 170], [150, 170]],
                "orientation_aspect_ratio": 3, "orientation_mask_fraction": 0.5,
            },
        ):
            with self.subTest(orientation=orientation):
                result = self.request("/api/localize", self.localization(captured, [{
                    "score": 0.9, "box_xyxy": [120, 140, 240, 200],
                    "pickup_point": "long_axis_midpoint", **orientation,
                }]))
                self.assertEqual(result["status"], "rejected")
                self.assertEqual(result["candidate_count"], 0)
                self.assertIn("error", result)

    def test_unknown_pickup_point_is_rejected(self) -> None:
        captured = self.request("/api/rgbd-capture", {})
        result = self.request("/api/localize", self.localization(captured, [{
            "score": 0.9, "box_xyxy": [120, 140, 240, 200], "pickup_point": "unknown",
        }]))
        self.assertEqual(result["status"], "rejected")
        self.assertIn("unsupported pickup_point", result["error"])

    def test_startup_has_no_detector_dependency(self) -> None:
        with mock.patch.object(base_app, "load_depth_correction", return_value=(1.0, 0.0)), \
                mock.patch.object(base_app, "D405FfmpegStream"):
            app = VisionApplication()
        self.assertFalse(hasattr(app, "detector"))
        self.assertFalse(app.camera.show_status_banner)
        health = self.request("/healthz")
        self.assertEqual(health["service"], "camera")
        self.assertNotIn("model", health)
        self.assertNotIn("cuda", health)
        self.app.camera.check.assert_called_once()

    def test_dashboard_and_health_identify_camera_and_owning_service(self) -> None:
        page = self.request("/")
        self.assertIn(
            "Camera View — Intel RealSense D435 | opencv-vision".encode("utf-8"), page
        )
        self.assertIn(b"<title>Piper Camera View</title>", page)
        self.assertNotIn(b"D405", page)
        self.assertNotIn(b"OWL-ViT", page)
        self.assertNotIn(b"Detection runs separately; it does not control the camera.", page)
        self.assertIn(b"Intel RealSense D435", page)
        self.assertNotIn(b"Camera information", page)
        self.assertNotIn(b"Serial number:", page)
        health = self.request("/healthz")
        self.assertEqual(health["camera_name"], "Intel RealSense D435")
        self.assertEqual(health["camera_owner_service"], "opencv-vision")

    def test_dashboard_puts_live_view_beside_stacked_scans_with_details_below(self) -> None:
        page = self.request("/").decode("utf-8")
        top = page.index('<div class="top">')
        details = page.index('<div class="details">')
        self.assertLess(top, page.index('src="/stream.mjpg"'))
        self.assertLess(page.index('src="/stream.mjpg"'), page.index('id="candyshot"'))
        self.assertLess(page.index('id="candyshot"'), page.index('id="handshot"'))
        self.assertLess(page.index('id="handshot"'), details)
        self.assertLess(details, page.index('id="candymeta"'))
        self.assertLess(page.index('id="candymeta"'), page.index('id="handmeta"'))
        self.assertIn("grid-template-columns:minmax(0,7fr) minmax(220px,3fr)", page)
        for legacy in ("Hand scan 4", "Scan 1", "X=100..450 mm"):
            self.assertNotIn(legacy, page)

    def test_dashboard_escapes_camera_identity(self) -> None:
        self.app.camera.name = "<camera>"
        self.app.camera.serial = 'serial<script>'
        page = self.app.dashboard_html()
        self.assertIn(b"&lt;camera&gt;", page)
        self.assertNotIn(b"<camera>", page)

    def test_video_frames_have_no_legacy_banner(self) -> None:
        camera = base_app.D405FfmpegStream()
        camera.show_status_banner = False
        frame = np.full((480, 640, 3), 127, dtype=np.uint8)
        depth = np.full((480, 640), 300, dtype=np.uint16)
        aligned = mock.Mock()
        aligned.get_color_frame.return_value.get_data.return_value = frame
        aligned.get_depth_frame.return_value.get_data.return_value = depth
        camera.pipeline = mock.Mock()
        camera.align = mock.Mock()
        camera.align.process.return_value = aligned
        camera.process = mock.Mock()
        camera.process.poll.return_value = None
        camera.process.stdin.write.side_effect = lambda _data: camera.stop_event.set()
        with mock.patch.object(base_app.cv2, "putText") as text, \
                mock.patch.object(base_app.cv2, "rectangle") as rectangle:
            camera._capture_loop()
        self.assertIsNone(camera.error)
        text.assert_not_called()
        rectangle.assert_not_called()
        camera.process.stdin.write.assert_called_once_with(frame.tobytes())

    def test_preview_startup_does_not_load_or_fabricate_calibration(self) -> None:
        with mock.patch.dict(os.environ, {"CAMERA_PREVIEW_ONLY": "true"}), \
                mock.patch.object(base_app, "load_depth_correction") as correction, \
                mock.patch.object(base_app, "D405FfmpegStream"):
            app = VisionApplication()
            self.assertIn(b"PREVIEW ONLY", page_html())
        correction.assert_not_called()
        self.assertIsNone(app.depth_scale_correction)
        self.assertIsNone(app.depth_offset_mm)
        self.assertFalse(hasattr(app, "detector"))

    def test_preview_exposes_images_but_blocks_all_metric_operations(self) -> None:
        self.app.preview_only = True
        self.app.depth_scale_correction = self.app.depth_offset_mm = None
        health = self.request("/healthz")
        self.assertTrue(health["preview_only"])
        self.assertFalse(health["depth_calibrated"])
        self.assertEqual(health["capabilities"], ["rgbd-capture"])
        captured = self.request("/api/rgbd-capture", {})
        self.assertTrue(captured["preview_only"])
        self.assertTrue(self.request(captured["image_png_url"]).startswith(b"\x89PNG"))
        for path, payload in (
            ("/api/localize", self.localization(captured)),
            ("/api/deproject", {"token": captured["token"], "pixel_uv": [180, 170]}),
            ("/api/workspace-overlay", {
                "base_to_camera": np.eye(4).tolist(),
                "minimum_mm": [0, 0, 0], "maximum_mm": [1, 1, 1],
            }),
        ):
            with self.subTest(path=path), self.assertRaises(urllib.error.HTTPError) as raised:
                self.request(path, payload)
            with raised.exception as response:
                self.assertEqual(response.code, 400)
                self.assertIn("preview-only", json.load(response)["error"])
        self.assertEqual(self.app.snapshots, {})

    def test_calibration_stays_mandatory_without_preview_flag(self) -> None:
        with mock.patch.dict(os.environ, {"CAMERA_PREVIEW_ONLY": "false"}), \
                mock.patch.object(
                    base_app, "load_depth_correction", side_effect=VisionError("not calibrated")
                ), self.assertRaisesRegex(VisionError, "not calibrated"):
            VisionApplication()

    def test_lossless_capture_and_localization_use_stored_depth(self) -> None:
        captured = self.request("/api/rgbd-capture", {})
        png = self.request(captured["image_png_url"])
        decoded = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(decoded, self.app.camera.snapshot.return_value[0])
        self.app.camera.snapshot.return_value = (
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.full((480, 640), 900, dtype=np.uint16), 100
        )
        result = self.request("/api/localize", self.localization(captured))
        self.assertEqual(result["status"], "detected")
        self.assertEqual(result["token"], captured["token"])
        self.assertEqual(result["frame_sequence"], 11)
        self.assertEqual(result["captured_at"], captured["captured_at"])
        self.assertAlmostEqual(result["corrected_depth_mm"], 300.0)
        self.app.camera.snapshot.assert_called_once()
        snapshots = self.request("/api/snapshots")
        self.assertEqual(snapshots["snapshots"]["1"], result)
        self.assertTrue(self.request(result["snapshot_url"]).startswith(b"\xff\xd8"))

    def test_hand_jpeg_and_pixel_deprojection_stay_compatible(self) -> None:
        captured = self.request("/api/rgbd-capture", {})
        self.assertTrue(self.request(captured["image_url"]).startswith(b"\xff\xd8"))
        point = self.request("/api/deproject", {
            "token": captured["token"], "pixel_uv": [180, 170],
        })
        self.assertAlmostEqual(point["corrected_depth_mm"], 300.0)
        self.assertEqual(point["token"], captured["token"])

    def test_empty_detections_are_not_found(self) -> None:
        captured = self.request("/api/rgbd-capture", {})
        result = self.request("/api/localize", self.localization(captured, []))
        self.assertEqual(result["status"], "not_found")
        self.assertEqual(result["candidates"], [])

    def test_best_valid_candidate_and_orientation_are_preserved(self) -> None:
        captured = self.request("/api/rgbd-capture", {})
        result = self.request("/api/localize", self.localization(captured, [
            {"score": 0.8, "box_xyxy": [120, 140, 240, 200],
             "long_axis_pixels": [[140, 170], [220, 170]],
             "orientation_aspect_ratio": 3.0, "orientation_mask_fraction": 0.8},
            {"score": 0.95, "box_xyxy": [370, 300, 430, 360],
             "orientation_error": "ambiguous"},
        ]))
        self.assertEqual(result["detection_count"], 2)
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["score"], 0.8)
        self.assertEqual(result["candidates"][0]["status"], "rejected")
        np.testing.assert_allclose(result["long_axis_camera_unit"], [1, 0, 0])

    def test_unknown_token_and_bad_geometry_are_rejected(self) -> None:
        captured = self.request("/api/rgbd-capture", {})
        for payload in (
            {**self.localization(captured), "token": "expired"},
            self.localization(captured, [{}]),
            self.localization(captured, [{"score": 0.9, "box_xyxy": [-1, 0, 20, 20]}]),
            self.localization(captured, [{"score": 0.9, "box_xyxy": [0, 0, 20, 20],
                                         "long_axis_pixels": [[1, 2], [3, 4]]}]),
        ):
            with self.subTest(payload=payload), self.assertRaises(
                urllib.error.HTTPError
            ) as raised:
                self.request("/api/localize", payload)
            with raised.exception as response:
                self.assertEqual(response.code, 400)
        self.app.camera.snapshot.assert_called_once()
        self.assertEqual(self.app.snapshots, {})

    def test_legacy_capture_is_explicitly_retired(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as raised:
            self.request("/api/capture", {"slot": 1, "prompt": "starburst"})
        with raised.exception as response:
            self.assertEqual(response.code, 410)
            self.assertIn("/api/localize", json.load(response)["error"])
        self.app.camera.snapshot.assert_not_called()

    def test_localization_conflict_does_not_change_snapshot(self) -> None:
        captured = self.request("/api/rgbd-capture", {})
        with self.app.capture_lock:
            with self.assertRaises(urllib.error.HTTPError) as raised:
                self.request("/api/localize", self.localization(captured))
            with raised.exception as response:
                self.assertEqual(response.code, 409)
        self.assertEqual(self.app.snapshots, {})

    def test_frame_cache_evicts_old_tokens_without_recapturing(self) -> None:
        self.app.camera.snapshot.return_value = (
            np.zeros((48, 64, 3), dtype=np.uint8),
            np.full((48, 64), 300, dtype=np.uint16), 11,
        )
        tokens = [self.app.capture_rgbd()["token"] for _ in range(31)]
        self.assertEqual(len(self.app.rgbd_frames), 30)
        self.assertNotIn(tokens[0], self.app.rgbd_frames)
        self.assertIn(tokens[-1], self.app.rgbd_frames)
        with self.assertRaisesRegex(VisionError, "expired RGB-D frame token"):
            self.app.localize(tokens[0], 1, "starburst", "candy-yolo11s", [])
        self.assertEqual(self.app.camera.snapshot.call_count, 31)

    def test_localized_capture_keeps_dataset_images_and_labels(self) -> None:
        self.app.dataset_class_id = 8
        self.app.dataset_prompts = {"starburst"}
        with tempfile.TemporaryDirectory() as directory:
            self.app.dataset_dir = Path(directory)
            (self.app.dataset_dir / "images").mkdir()
            (self.app.dataset_dir / "labels").mkdir()
            captured = self.request("/api/rgbd-capture", {})
            result = self.request("/api/localize", self.localization(captured))
            label = (self.app.dataset_dir / result["dataset_label"]).read_text()
            self.assertEqual(label, "8 0.281250 0.354167 0.187500 0.125000\n")
            saved = cv2.imread(str(self.app.dataset_dir / result["dataset_image"]))
            self.assertEqual(saved.shape, (480, 640, 3))


if __name__ == "__main__":
    unittest.main()
