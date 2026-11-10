"""Demo -> camera -> detection -> token localization, with hardware mocked."""

import base64
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import unittest
import urllib.request
from unittest import mock

import numpy as np

from app import DetectionApplication, create_server
from vision import FoundryYoloDetector, VisionError


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SplitServiceIntegrationTests(unittest.TestCase):
    def test_demo_orchestrates_both_services_with_exact_frame_depth(self) -> None:
        root = Path(__file__).resolve().parents[1]
        helpers = load_module("depth_helpers", root / "k3s-vision-service/vision.py")
        with mock.patch.dict(sys.modules, {"base_vision": helpers}):
            camera_helpers = load_module("camera_helpers", root / "k3s-OpenCV-service/vision.py")
            with mock.patch.dict(sys.modules, {"vision": camera_helpers}):
                base_app = load_module("camera_base", root / "k3s-vision-service/app.py")
            with mock.patch.dict(sys.modules, {"base_app": base_app}):
                camera_app = load_module("camera_app", root / "k3s-OpenCV-service/app.py")
                with mock.patch.dict(sys.modules, {"app": camera_app}):
                    fixtures = load_module("camera_fixtures", root / "k3s-OpenCV-service/test_app.py")
        camera = fixtures.make_application()
        frame = camera.camera.snapshot.return_value[0].copy()
        orchestrator = load_module("demo_http", root / "Demo-Product-Day/vision_client.py")
        output = np.zeros((1, 15, 8400), dtype=np.float32)
        output[0, :4, 0] = [180, 250, 120, 60]
        output[0, 12, 0] = 0.9
        output[0, :4, 1] = [400, 410, 60, 60]
        output[0, 12, 1] = 0.8
        prediction = {"items": [{"data": base64.b64encode(
            json.dumps({"output0": output.tolist()}).encode()
        ).decode()}]}
        metadata = {
            "status": "loaded", "input_shape": [1, 3, 640, 640],
            "execution_provider": "CUDAExecutionProvider",
        }
        tensors = []

        def foundry_request(_detector, path, payload=None):
            if path == "/v1/model":
                return metadata
            self.assertEqual(path, "/v1/predict")
            tensors.append(json.loads(base64.b64decode(payload["items"][0]["data"])))
            camera.camera.sequence = 100
            camera.camera.snapshot.return_value = (
                np.zeros_like(frame), np.full((480, 640), 900, dtype=np.uint16), 101
            )
            return prediction

        with mock.patch.dict(os.environ, {"FOUNDRY_API_KEY": "test-key"}), mock.patch.object(
            FoundryYoloDetector, "_request", foundry_request
        ):
            detection = DetectionApplication()
            servers = [camera_app.create_server(camera, 0), create_server(detection, 0)]
            threads = [threading.Thread(target=server.serve_forever) for server in servers]
            for thread in threads:
                thread.start()
            try:
                camera_url, detection_url = [
                    f"http://127.0.0.1:{server.server_port}" for server in servers
                ]
                client = orchestrator.VisionClient(camera_url, detection_url=detection_url)
                self.assertEqual(client.health()["model"], "candy-yolo11s")
                result = client.capture(1, "starburst candy", 0.25)
                expected = detection.detector.detect(frame, "starburst candy", 0.25)
                self.assertEqual(result["detection_count"], 2)
                self.assertEqual(result["candidate_count"], 1)
                self.assertEqual(result["box_xyxy"], expected[0]["box_xyxy"])
                self.assertEqual(result["long_axis_pixels"], expected[0]["long_axis_pixels"])
                self.assertEqual(result["candidates"][1]["status"], "rejected")
                self.assertAlmostEqual(result["corrected_depth_mm"], 300.0)
                self.assertEqual(result["frame_sequence"], 11)
                self.assertEqual(camera.camera.sequence, 100)
                camera.camera.snapshot.assert_called_once()
                np.testing.assert_array_equal(tensors[0], tensors[1])
                with mock.patch.object(
                    detection.detector, "check", side_effect=VisionError("offline")
                ):
                    with urllib.request.urlopen(camera_url + "/healthz", timeout=5) as response:
                        self.assertEqual(json.load(response)["status"], "ok")
                    with self.assertRaisesRegex(orchestrator.DemoError, "HTTP 503"):
                        client.health()
                with mock.patch.object(
                    detection.detector, "detect", side_effect=VisionError("offline")
                ), self.assertRaisesRegex(orchestrator.DemoError, "HTTP 502"):
                    client.capture(2, "starburst candy", 0.25)
                self.assertNotIn(2, camera.snapshots)

                def expire_token(*_args):
                    camera.rgbd_frames.clear()
                    return expected

                with mock.patch.object(
                    detection.detector, "detect", side_effect=expire_token
                ), self.assertRaisesRegex(orchestrator.DemoError, "expired RGB-D frame token"):
                    client.capture(3, "starburst candy", 0.25)
                self.assertNotIn(3, camera.snapshots)
            finally:
                for server in servers:
                    server.shutdown()
                    server.server_close()
                for thread in threads:
                    thread.join(timeout=5)
                    self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
