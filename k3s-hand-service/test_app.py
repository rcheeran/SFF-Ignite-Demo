"""Hardware-free tests for synchronized hand capture error handling."""

from __future__ import annotations

import io
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import numpy as np

from app import HandLandmarkerStream, VisionServiceRejected


def detected_hand() -> dict:
    return {
        "handedness": "Right",
        "handedness_score": 0.9,
        "palm_center_pixel": [320, 240],
        "is_open_palm": True,
        "landmarks": [
            {"x": 0.5, "y": 0.5, "z": 0.0}
            for _ in range(21)
        ],
    }


class HandCaptureTests(unittest.TestCase):
    def test_capture_accepts_palm_spread_below_fifty_mm(self) -> None:
        stream = HandLandmarkerStream.__new__(HandLandmarkerStream)
        stream.stability_samples = 3
        stream.minimum_stable_samples = 3
        stream.maximum_palm_spread_mm = 50.0
        stream.model_path = Path("hand_landmarker.task")
        stream.snapshots = {}
        stream.snapshot_jpegs = {}
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        stream._synchronized_frame = mock.Mock(
            side_effect=[
                (frame, {"token": f"token-{index}", "frame_sequence": index})
                for index in range(3)
            ]
        )
        stream._detect_frame = mock.Mock(return_value=[detected_hand()])
        stream._json_request = mock.Mock(
            side_effect=[
                {"camera_point_mm": [0.0, 0.0, 300.0]},
                {"camera_point_mm": [32.8, 0.0, 300.0]},
                {"camera_point_mm": [10.0, 0.0, 300.0]},
            ]
        )

        with mock.patch("time.sleep"):
            result = stream.capture_slot(4)

        self.assertEqual(result["status"], "detected")
        self.assertAlmostEqual(result["maximum_temporal_spread_mm"], 22.8)
        self.assertEqual(result["stable_sample_count"], 3)
        self.assertEqual(result["temporal_outlier_count"], 0)
        self.assertEqual(result["maximum_allowed_spread_mm"], 50.0)

    def test_capture_accepts_three_stable_samples_and_two_outliers(self) -> None:
        stream = HandLandmarkerStream.__new__(HandLandmarkerStream)
        stream.stability_samples = 5
        stream.minimum_stable_samples = 3
        stream.maximum_palm_spread_mm = 50.0
        stream.model_path = Path("hand_landmarker.task")
        stream.snapshots = {}
        stream.snapshot_jpegs = {}
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        stream._synchronized_frame = mock.Mock(
            side_effect=[
                (frame, {"token": f"token-{index}", "frame_sequence": index})
                for index in range(5)
            ]
        )
        stream._detect_frame = mock.Mock(return_value=[detected_hand()])
        stream._json_request = mock.Mock(
            side_effect=[
                {"camera_point_mm": [79.0, -62.0, 419.0]},
                {"camera_point_mm": [80.0, -63.0, 420.0]},
                {"camera_point_mm": [81.0, -62.0, 419.0]},
                {"camera_point_mm": [145.0, -62.0, 419.0]},
                {"camera_point_mm": [15.0, -62.0, 419.0]},
            ]
        )

        with mock.patch("time.sleep"):
            result = stream.capture_slot(5)

        self.assertEqual(result["status"], "detected")
        self.assertEqual(result["stable_sample_count"], 3)
        self.assertEqual(result["temporal_outlier_count"], 2)
        self.assertGreater(result["maximum_observed_spread_mm"], 60.0)
        self.assertLess(result["maximum_temporal_spread_mm"], 2.0)
        np.testing.assert_allclose(
            result["camera_point_mm"], [80.0, -62.0, 419.0]
        )

    def test_capture_rejects_when_fewer_than_three_samples_cluster(self) -> None:
        stream = HandLandmarkerStream.__new__(HandLandmarkerStream)
        stream.stability_samples = 5
        stream.minimum_stable_samples = 3
        stream.maximum_palm_spread_mm = 20.0
        stream.model_path = Path("hand_landmarker.task")
        stream.snapshots = {}
        stream.snapshot_jpegs = {}
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        stream._synchronized_frame = mock.Mock(
            side_effect=[
                (frame, {"token": f"token-{index}", "frame_sequence": index})
                for index in range(5)
            ]
        )
        stream._detect_frame = mock.Mock(return_value=[detected_hand()])
        stream._json_request = mock.Mock(
            side_effect=[
                {"camera_point_mm": [0.0, 0.0, 300.0]},
                {"camera_point_mm": [1.0, 0.0, 300.0]},
                {"camera_point_mm": [50.0, 0.0, 300.0]},
                {"camera_point_mm": [100.0, 0.0, 300.0]},
                {"camera_point_mm": [150.0, 0.0, 300.0]},
            ]
        )

        with mock.patch("time.sleep"):
            result = stream.capture_slot(5)

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["stable_sample_count"], 1)
        self.assertIn("need 3 within 20.0mm", result["error"])

    def test_json_request_preserves_vision_rejection_detail(self) -> None:
        stream = HandLandmarkerStream.__new__(HandLandmarkerStream)
        stream.vision_url = "http://vision"
        error = urllib.error.HTTPError(
            "http://vision/api/deproject",
            400,
            "Bad Request",
            {},
            io.BytesIO(
                b'{"status":"rejected","error":"only 12 valid depth pixels"}'
            ),
        )

        with (
            mock.patch("urllib.request.urlopen", side_effect=error),
            self.assertRaisesRegex(
                VisionServiceRejected,
                "only 12 valid depth pixels",
            ),
        ):
            stream._json_request(
                "/api/deproject",
                {"token": "token", "pixel_uv": [320, 240]},
            )

    def test_capture_skips_rejected_depth_sample(self) -> None:
        stream = HandLandmarkerStream.__new__(HandLandmarkerStream)
        stream.stability_samples = 3
        stream.minimum_stable_samples = 2
        stream.maximum_palm_spread_mm = 15.0
        stream.model_path = Path("hand_landmarker.task")
        stream.snapshots = {}
        stream.snapshot_jpegs = {}
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        stream._synchronized_frame = mock.Mock(
            side_effect=[
                (frame, {"token": f"token-{index}", "frame_sequence": index})
                for index in range(3)
            ]
        )
        stream._detect_frame = mock.Mock(return_value=[detected_hand()])
        stream._json_request = mock.Mock(
            side_effect=[
                VisionServiceRejected("no usable depth"),
                {"camera_point_mm": [10.0, 20.0, 300.0]},
                {"camera_point_mm": [11.0, 20.0, 300.0]},
            ]
        )

        with mock.patch("time.sleep"):
            result = stream.capture_slot(4)

        self.assertEqual(result["status"], "detected")
        self.assertEqual(result["detected_sample_count"], 3)
        self.assertEqual(result["depth_sample_count"], 2)
        self.assertEqual(result["depth_rejection_count"], 1)
        self.assertIn("no usable depth", result["depth_rejection_errors"][0])

    def test_capture_does_not_accept_closed_hand(self) -> None:
        stream = HandLandmarkerStream.__new__(HandLandmarkerStream)
        stream.stability_samples = 1
        stream.minimum_stable_samples = 1
        stream.maximum_palm_spread_mm = 15.0
        stream.model_path = Path("hand_landmarker.task")
        stream.snapshots = {}
        stream.snapshot_jpegs = {}
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        stream._synchronized_frame = mock.Mock(
            return_value=(frame, {"token": "token", "frame_sequence": 1})
        )
        hand = detected_hand()
        hand["is_open_palm"] = False
        stream._detect_frame = mock.Mock(return_value=[hand])
        stream._json_request = mock.Mock()

        with mock.patch("time.sleep"):
            result = stream.capture_slot(4)

        self.assertEqual(result["status"], "not_found")
        self.assertEqual(result["visible_hand_sample_count"], 1)
        self.assertEqual(result["detected_sample_count"], 0)
        stream._json_request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
