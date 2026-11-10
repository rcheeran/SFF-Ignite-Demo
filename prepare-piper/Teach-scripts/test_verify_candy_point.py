"""Offline point comparison, token contract, and feedback-only lifecycle tests."""

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

import verify_candy_point as verify
from test_teach_workspaces import feedback_arm


class CandyPointVerificationTests(unittest.TestCase):
    def capture(self):
        return {"status": "ok", "token": "abc123", "frame_sequence": 9, "image_png_url": "/rgbd/abc123.png"}

    def response(self):
        return {"status": "ok", "token": "abc123", "pixel_uv": [20, 30], "camera_point_mm": [246.3, 81.2, -8.3]}

    def test_error_matches_logged_candy_z_without_workspace_rejection(self):
        result = verify.compare_point(
            np.array([246.3, 81.2, 18.0]), self.response(), self.capture(),
            np.eye(4), np.eye(4), [20, 30], 10.0,
        )
        np.testing.assert_allclose(result["camera_minus_reference_mm"], [0, 0, -26.3])
        self.assertAlmostEqual(result["euclidean_error_mm"], 26.3)
        self.assertFalse(result["acceptance"]["quality_passed"])

    def test_transforms_and_threshold_boundary(self):
        response = {**self.response(), "camera_point_mm": [0, 0, 100]}
        wrist, hand_eye = np.eye(4), np.eye(4)
        wrist[:3, 3] = [200, 20, 0]
        hand_eye[:3, 3] = [0, 0, -90]
        result = verify.compare_point(
            np.array([200, 20, 0]), response, self.capture(), wrist, hand_eye, [20, 30], 10,
        )
        self.assertEqual(result["camera_base_point_mm"], [200, 20, 10])
        self.assertTrue(result["acceptance"]["quality_passed"])

    def test_mismatched_token_pixel_and_nonfinite_points_are_rejected(self):
        for field, value in (("token", "wrong"), ("pixel_uv", [1, 2]), ("camera_point_mm", [0, 0, float("nan")])):
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                verify.compare_point(
                    np.zeros(3), {**self.response(), field: value}, self.capture(),
                    np.eye(4), np.eye(4), [20, 30], 10,
                )

    def test_capture_rejects_wrist_motion_and_invalid_tokens(self):
        with patch("sys.argv", ["verify_candy_point.py"]):
            args = verify.parse_args()
        arm, camera = feedback_arm(), Mock()
        camera._json_request.return_value = self.capture()
        before = arm.pose.return_value
        arm.pose.side_effect = [before, before + [5, 0, 0, 0, 0, 0]]
        with self.assertRaisesRegex(RuntimeError, "moved during capture"):
            verify.freeze_capture(arm, camera, args)
        arm.pose.side_effect = None
        camera._json_request.return_value = {**self.capture(), "token": "../x"}
        with self.assertRaisesRegex(RuntimeError, "invalid token"):
            verify.freeze_capture(arm, camera, args)
        camera._request_bytes.assert_not_called()

    def test_frozen_viewer_uses_original_image_dimensions(self):
        success, png = cv2.imencode(".png", np.zeros((60, 80, 3), dtype=np.uint8))
        self.assertTrue(success)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "point.html"
            self.assertEqual(verify.write_pixel_viewer(path, png.tobytes()), (80, 60))
            self.assertIn("naturalWidth", path.read_text())
            self.assertIn("data:image/png;base64", path.read_text())
            with self.assertRaisesRegex(RuntimeError, "decodable"):
                verify.write_pixel_viewer(path, b"not an image")

    def test_main_disconnects_before_pixel_selection_and_saves_no_config(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candy_point_verification.json"
            with patch("sys.argv", ["verify_candy_point.py", "--output", str(output), "--reference-point-mm", "10", "20", "30"]):
                args = verify.parse_args()
            arm, camera = feedback_arm(), Mock()
            camera.health.return_value = {"status": "ok"}
            camera._json_request.return_value = {
                **self.response(), "camera_point_mm": [10, 20, 30],
            }
            success, png = cv2.imencode(".png", np.zeros((60, 80, 3), dtype=np.uint8))
            self.assertTrue(success)

            def prompt(text):
                if "coordinates:" in text:
                    arm.disconnect.assert_called_once()
                    return "20 30"
                if "Save verification" in text:
                    return "y"
                return ""

            with patch.object(verify, "parse_args", return_value=args), patch.object(
                verify.sys.stdin, "isatty", return_value=True
            ), patch.object(verify, "HttpClient", return_value=camera), patch.object(
                verify, "load_calibration", return_value=np.eye(4)
            ), patch.object(verify.teaching, "connect_readonly", return_value=arm), patch.object(
                verify.teaching, "confirm_power_off"
            ) as off, patch.object(verify, "freeze_capture", return_value=(self.capture(), png.tobytes(), np.eye(4))), patch(
                "builtins.input", side_effect=prompt
            ), patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(verify.main(), 0)
            off.assert_called_once()
            saved = json.loads(output.read_text())
            self.assertEqual(saved["euclidean_error_mm"], 0)
            self.assertEqual(saved["reference_method"], "operator-surveyed XYZ")
            camera._json_request.assert_called_once_with("/api/deproject", {"token": "abc123", "pixel_uv": [20, 30]})
            self.assertEqual(sorted(path.name for path in Path(directory).iterdir()), [
                "candy_point_verification.html", "candy_point_verification.json",
            ])

    def test_capture_interrupt_still_powers_off_and_disconnects(self):
        with patch("sys.argv", ["verify_candy_point.py", "--reference-point-mm", "10", "20", "30"]):
            args = verify.parse_args()
        arm = feedback_arm()
        with patch.object(verify, "parse_args", return_value=args), patch.object(
            verify.sys.stdin, "isatty", return_value=True
        ), patch.object(verify, "HttpClient"), patch.object(verify, "load_calibration", return_value=np.eye(4)), patch.object(
            verify.teaching, "connect_readonly", return_value=arm
        ), patch.object(verify.teaching, "confirm_power_off") as off, patch.object(
            verify, "freeze_capture", side_effect=KeyboardInterrupt
        ), patch("builtins.input", return_value=""), patch("sys.stdout", new_callable=io.StringIO), patch(
            "sys.stderr", new_callable=io.StringIO
        ):
            self.assertEqual(verify.main(), 130)
        off.assert_called_once()
        arm.disconnect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
