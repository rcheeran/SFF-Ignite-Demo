"""Tests for application-owned capture/detection/localization orchestration."""

import unittest
from unittest import mock

from vision_client import DemoError, HandClient, HttpClient, VisionClient

PNG = b"\x89PNG\r\n\x1a\nfake-image"


class VisionClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = VisionClient("http://camera:8094", detection_url="http://detector:8096")
        self.captured = {
            "status": "ok", "token": "abc123", "frame_sequence": 42,
            "image_png_url": "/rgbd/abc123.png",
        }
        self.result = {
            "status": "not_found", "token": "abc123", "frame_sequence": 42, "slot": 1,
        }

    def test_capture_calls_services_in_order_and_preserves_token(self) -> None:
        with mock.patch.object(
            self.client, "_json_request", side_effect=[self.captured, self.result]
        ) as json_request, mock.patch.object(
            self.client, "_request_bytes", return_value=PNG
        ) as image_request, mock.patch.object(
            self.client.detector, "detect_png", return_value=[]
        ) as detect:
            self.assertEqual(self.client.capture(1, "starburst", 0.25), self.result)
        self.assertEqual(json_request.call_args_list, [
            mock.call("/api/rgbd-capture", {}),
            mock.call("/api/localize", {
                "token": "abc123", "slot": 1, "prompt": "starburst",
                "model": "candy-yolo11s", "detections": [],
            }),
        ])
        image_request.assert_called_once_with("/rgbd/abc123.png")
        detect.assert_called_once_with(PNG, "starburst", 0.25)

    def test_missing_png_or_token_does_not_run_detection(self) -> None:
        for captured in ({}, {**self.captured, "image_png_url": "/rgbd/abc123.jpg"}):
            with self.subTest(captured=captured), mock.patch.object(
                self.client, "_json_request", return_value=captured
            ), mock.patch.object(self.client.detector, "detect_png") as detect, \
                    self.assertRaisesRegex(DemoError, "frame token"):
                self.client.capture(1, "starburst", 0.25)
            detect.assert_not_called()

    def test_mismatched_localization_is_not_accepted(self) -> None:
        for mismatch in ({"token": "different"}, {"frame_sequence": 43}, {"slot": 2}):
            with self.subTest(mismatch=mismatch), mock.patch.object(
                self.client, "_json_request",
                side_effect=[self.captured, {**self.result, **mismatch}],
            ), mock.patch.object(
                self.client, "_request_bytes", return_value=PNG
            ), mock.patch.object(
                self.client.detector, "detect_png", return_value=[]
            ), self.assertRaisesRegex(DemoError, "does not match"):
                self.client.capture(1, "starburst", 0.25)

    def test_hand_client_does_not_construct_or_call_detection(self) -> None:
        with mock.patch("vision_client.DetectionClient") as detector:
            hand = HandClient("http://hand:8095")
            with mock.patch.object(hand, "_json_request", return_value={"status": "not_found"}):
                hand.capture_hand(4)
        detector.assert_not_called()

    def test_injected_local_detector_replaces_detection_service(self) -> None:
        local = mock.Mock(model_name="opencv-hsv-candy", uses_cuda=False, url="local-cpu")
        local.detect_png.return_value = []
        with mock.patch("vision_client.DetectionClient") as service:
            client = VisionClient("http://camera:8094", detector=local)
        service.assert_not_called()
        self.assertIs(client.detector, local)
        with mock.patch.object(client, "_json_request", side_effect=[self.captured, self.result]) as request, \
                mock.patch.object(client, "_request_bytes", return_value=PNG):
            client.capture(1, "starburst", 0.25)
        local.detect_png.assert_called_once_with(PNG, "starburst", 0.25)
        self.assertEqual(request.call_args.args[1]["model"], "opencv-hsv-candy")
        with mock.patch.object(HttpClient, "health", return_value={
            "status": "ok", "service": "camera", "capabilities": ["localize"],
        }):
            health = client.health()
        self.assertFalse(health["cuda"])
        local.check.assert_called_once()

    def test_capture_png_returns_metadata_and_lossless_image_only(self) -> None:
        with mock.patch.object(self.client, "_json_request", return_value=self.captured) as request, \
                mock.patch.object(self.client, "_request_bytes", return_value=PNG) as image, \
                mock.patch.object(self.client.detector, "detect_png") as detect:
            self.assertEqual(self.client.capture_png(), (self.captured, PNG))
        request.assert_called_once_with("/api/rgbd-capture", {})
        image.assert_called_once_with("/rgbd/abc123.png")
        detect.assert_not_called()
        with mock.patch.object(self.client, "_json_request", return_value=self.captured), \
                mock.patch.object(self.client, "_request_bytes", return_value=b"<html>"), \
                self.assertRaisesRegex(DemoError, "not a PNG"):
            self.client.capture_png()

    def test_candy_capture_requests_midpoint_for_every_candidate(self) -> None:
        detections = [
            {"score": 0.9, "box_xyxy": [10, 10, 100, 50], "long_axis_pixels": [[20, 20], [80, 40]]},
            {"score": 0.8, "box_xyxy": [110, 10, 200, 50]},
        ]
        result = {
            **self.result, "status": "detected", "pickup_point": "long_axis_midpoint",
            "candidates": [
                {"status": "detected", "pickup_point": "long_axis_midpoint"},
                {"status": "rejected", "error": "missing axis"},
            ],
        }
        with mock.patch.object(
            self.client, "_json_request", side_effect=[self.captured, result]
        ) as request, mock.patch.object(self.client, "_request_bytes", return_value=PNG), \
                mock.patch.object(self.client.detector, "detect_png", return_value=detections):
            self.assertEqual(self.client.capture(1, "starburst", 0.25), result)
        self.assertEqual(
            request.call_args.args[1]["detections"],
            [{**detection, "pickup_point": "long_axis_midpoint"} for detection in detections],
        )
        self.assertNotIn("pickup_point", detections[0])

    def test_old_camera_or_nonmidpoint_candidate_fails_explicitly(self) -> None:
        for result in (
            {**self.result, "status": "detected"},
            {
                **self.result, "status": "detected", "pickup_point": "long_axis_midpoint",
                "candidates": [{"status": "detected", "pickup_point": "depth_cluster"}],
            },
        ):
            with self.subTest(result=result), mock.patch.object(
                self.client, "_json_request", side_effect=[self.captured, result]
            ), mock.patch.object(self.client, "_request_bytes", return_value=PNG), \
                    mock.patch.object(self.client.detector, "detect_png", return_value=[]), \
                    self.assertRaisesRegex(DemoError, "deploy the updated OpenCV"):
                self.client.capture(1, "starburst", 0.25)


if __name__ == "__main__":
    unittest.main()
