"""Hardware-free detection HTTP tests."""

import base64
import json
import struct
import threading
import unittest
import urllib.error
import urllib.request
from unittest import mock

import cv2
import numpy as np

from app import DetectionApplication, MAX_REQUEST_BYTES, create_server
from vision import VisionError


class DetectionHttpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = DetectionApplication.__new__(DetectionApplication)
        self.app.model_name = "candy-yolo11s"
        self.app.detector = mock.Mock()
        self.app.detector.detect.return_value = []
        self.app.inference_lock = threading.Lock()
        self.server = create_server(self.app, 0)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.frame = np.random.default_rng(42).integers(
            0, 256, (48, 64, 3), dtype=np.uint8
        )
        ok, png = cv2.imencode(".png", self.frame)
        self.assertTrue(ok)
        self.payload = {
            "prompt": "starburst candy", "confidence": 0.25,
            "image_png_base64": base64.b64encode(png.tobytes()).decode(),
        }

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.assertFalse(self.thread.is_alive())

    def post(self, payload: object) -> dict:
        request = urllib.request.Request(
            f"{self.url}/api/detect", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.load(response)

    def assert_http_error(self, status: int, payload: object) -> None:
        with self.assertRaises(urllib.error.HTTPError) as raised:
            self.post(payload)
        self.assertEqual(raised.exception.code, status)
        with raised.exception as response:
            self.assertEqual(json.load(response)["status"], "error")

    def test_detection_is_lossless_and_returns_all_fields(self) -> None:
        detections = [{"score": 0.9, "box_xyxy": [1, 2, 30, 20],
                       "orientation_error": "ambiguous"}]
        self.app.detector.detect.return_value = detections
        result = self.post(self.payload)
        self.assertEqual(result, {
            "status": "ok", "model": "candy-yolo11s", "detections": detections
        })
        frame, prompt, confidence = self.app.detector.detect.call_args.args
        np.testing.assert_array_equal(frame, self.frame)
        self.assertEqual((prompt, confidence), ("starburst candy", 0.25))
        self.assertFalse(self.app.inference_lock.locked())

    def test_health_reflects_foundry_failure(self) -> None:
        with urllib.request.urlopen(f"{self.url}/healthz", timeout=5) as response:
            self.assertEqual(json.load(response), {
                "status": "ok", "model": "candy-yolo11s", "cuda": True
            })
        self.app.detector.check.side_effect = VisionError("Foundry offline")
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(f"{self.url}/healthz", timeout=5)
        with raised.exception as response:
            self.assertEqual(response.code, 503)
            self.assertIn("Foundry offline", json.load(response)["error"])

    def test_invalid_inputs_do_not_call_foundry(self) -> None:
        for payload in (
            [], {}, {**self.payload, "prompt": "unknown"},
            {**self.payload, "confidence": None},
            {**self.payload, "confidence": float("nan")},
            {**self.payload, "confidence": 0},
            {**self.payload, "confidence": 1},
            {**self.payload, "image_png_base64": "!invalid!"},
            {**self.payload, "image_png_base64": base64.b64encode(b"JPEG").decode()},
        ):
            with self.subTest(payload=payload):
                self.assert_http_error(400, payload)
        self.app.detector.detect.assert_not_called()

    def test_png_dimensions_are_bounded_before_decoding(self) -> None:
        png = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR"
        png += struct.pack(">II", 100000, 100000)
        with mock.patch("cv2.imdecode") as decode:
            self.assert_http_error(400, {
                **self.payload, "image_png_base64": base64.b64encode(png).decode()
            })
        decode.assert_not_called()

    def test_concurrent_inference_is_rejected(self) -> None:
        self.app.inference_lock.acquire()
        try:
            self.assert_http_error(409, self.payload)
        finally:
            self.app.inference_lock.release()
        self.app.detector.detect.assert_not_called()

    def test_foundry_failure_is_explicit_and_releases_lock(self) -> None:
        self.app.detector.detect.side_effect = VisionError("Foundry offline")
        self.assert_http_error(502, self.payload)
        self.assertFalse(self.app.inference_lock.locked())

    def test_malformed_json_and_oversized_body_are_rejected(self) -> None:
        for data, headers, status in (
            (b"{", {}, 400),
            (b"{}", {"Content-Length": str(MAX_REQUEST_BYTES + 1)}, 413),
        ):
            request = urllib.request.Request(
                f"{self.url}/api/detect", data=data,
                headers={"Content-Type": "application/json", **headers},
            )
            with self.subTest(status=status), self.assertRaises(
                urllib.error.HTTPError
            ) as raised:
                urllib.request.urlopen(request, timeout=5)
            with raised.exception as response:
                self.assertEqual(response.code, status)


if __name__ == "__main__":
    unittest.main()
