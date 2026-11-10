"""Hardware-free tests for the application's detection HTTP adapter."""

import base64
import io
import json
import unittest
import urllib.error
from unittest import mock

import cv2
import numpy as np

from detection_client import DetectionClient, DetectionError


class DetectionClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = DetectionClient("candy-yolo11s", "http://detector:8096")
        self.frame = np.random.default_rng(42).integers(
            0, 256, (48, 64, 3), dtype=np.uint8
        )
        ok, png = cv2.imencode(".png", self.frame)
        self.assertTrue(ok)
        self.png = png.tobytes()

    def response(self, **fields: object) -> io.BytesIO:
        return io.BytesIO(json.dumps({
            "status": "ok", "model": "candy-yolo11s", **fields
        }).encode())

    def test_lossless_image_and_all_detection_fields_are_preserved(self) -> None:
        detections = [{
            "score": 0.9,
            "box_xyxy": [1.0, 2.0, 30.0, 20.0],
            "long_axis_pixels": [[5.0, 10.0], [25.0, 10.0]],
            "orientation_aspect_ratio": 3.0,
            "orientation_mask_fraction": 0.8,
        }, {
            "score": 0.8,
            "box_xyxy": [31.0, 2.0, 50.0, 20.0],
            "orientation_error": "ambiguous",
        }]
        with mock.patch("urllib.request.urlopen", return_value=self.response(
            detections=detections
        )) as request:
            self.assertEqual(self.client.detect_png(self.png, "starburst", 0.25), detections)
        sent = request.call_args.args[0]
        self.assertEqual(sent.full_url, "http://detector:8096/api/detect")
        payload = json.loads(sent.data)
        decoded = cv2.imdecode(np.frombuffer(
            base64.b64decode(payload["image_png_base64"]), dtype=np.uint8
        ), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(decoded, self.frame)
        self.assertEqual(payload["prompt"], "starburst")
        self.assertEqual(payload["confidence"], 0.25)
        self.assertEqual(request.call_args.kwargs["timeout"], 60.0)

    def test_health_requires_expected_model_and_cuda(self) -> None:
        for fields, error in (
            ({"cuda": False}, "CUDA"),
            ({"cuda": True, "model": "wrong-model"}, "model must"),
            ({"cuda": True, "status": "error"}, "invalid detection"),
        ):
            with self.subTest(fields=fields), mock.patch(
                "urllib.request.urlopen", return_value=self.response(**fields)
            ), self.assertRaisesRegex(DetectionError, error):
                self.client.check()
        with mock.patch(
            "urllib.request.urlopen", return_value=self.response(cuda=True)
        ) as request:
            self.client.check()
        self.assertEqual(request.call_args.kwargs["timeout"], 3.0)

    def test_invalid_responses_are_not_reported_as_empty_detections(self) -> None:
        for detections in (
            None, {}, [None], [{}],
            [{"score": float("nan"), "box_xyxy": [1, 2, 3, 4]}],
            [{"score": 0.9, "box_xyxy": [3, 2, 1, 4]}],
            [{"score": 0.9, "box_xyxy": [1, 2, 3, 4],
              "long_axis_pixels": [[float("inf"), 0], [1, 2]]}],
        ):
            with self.subTest(detections=detections), mock.patch(
                "urllib.request.urlopen", return_value=self.response(detections=detections)
            ), self.assertRaises(DetectionError):
                self.client.detect_png(self.png, "starburst", 0.25)

    def test_empty_detection_list_is_a_valid_result(self) -> None:
        with mock.patch(
            "urllib.request.urlopen", return_value=self.response(detections=[])
        ):
            self.assertEqual(self.client.detect_png(self.png, "starburst", 0.25), [])

    def test_connection_and_json_failures_are_explicit(self) -> None:
        for failure in (urllib.error.URLError("offline"), TimeoutError("timed out")):
            with self.subTest(failure=failure), mock.patch(
                "urllib.request.urlopen", side_effect=failure
            ), self.assertRaisesRegex(DetectionError, "request /healthz failed"):
                self.client.check()
        with mock.patch(
            "urllib.request.urlopen", return_value=io.BytesIO(b"not-json")
        ), self.assertRaisesRegex(DetectionError, "request /healthz failed"):
            self.client.check()

    def test_http_failure_preserves_backend_error(self) -> None:
        failure = urllib.error.HTTPError(
            self.client.url, 502, "Bad Gateway", {}, io.BytesIO(b"Foundry unavailable")
        )
        with mock.patch(
            "urllib.request.urlopen", side_effect=failure
        ), self.assertRaisesRegex(DetectionError, "HTTP 502: Foundry unavailable"):
            self.client.check()


if __name__ == "__main__":
    unittest.main()
