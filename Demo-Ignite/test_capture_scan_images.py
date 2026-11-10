"""Hardware-free tests for the scan-pose tuning capture tool."""

import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

import capture_scan_images as capture

PNG = b"\x89PNG\r\n\x1a\nfake-image"


class CaptureScanImagesTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "captures"
        self.scan = {"index": 1, "joints_deg": np.array([10.0, 60, -40, 0, 40, 0])}
        self.settings = {
            "verbose": False, "motion_speed_percent": 10, "scan_settle_seconds": 1.0,
            "maximum_scan_position_deviation_mm": 2.0, "maximum_scan_rotation_deviation_deg": 0.75,
            "grasp_tip_z_mm": 18.012, "move_timeout_seconds": 8.0,
        }
        self.hand_eye = {"camera": {"serial": "D435", "intrinsics": {"fx": 600.0}}}
        self.arm = mock.Mock()
        self.arm.joints_deg.return_value = self.scan["joints_deg"]
        self.arm.pose.return_value = np.array([200.0, 0, 300, 180, 20, 180])
        self.objects = mock.Mock()
        self.objects.capture_png.side_effect = lambda: (
            {"token": "abc", "frame_sequence": 7, "captured_at": "now", "width": 640, "height": 480}, PNG,
        )
        self.detector = mock.Mock(model_name="candy-yolo11s")
        self.detector.detect_png.return_value = [{"score": 0.9, "box_xyxy": [1, 2, 30, 20]}]

    def args(self, **fields):
        return argparse.Namespace(**{
            "calibration_dir": Path("/unused/calibration"), "scan_poses": Path("/unused/scans.json"),
            "scan": 1, "frames": 2, "frame_interval": 0.0, "label": "", "output_dir": self.output,
            "vision_url": "http://camera", "detection_url": "http://detector", "confidence": 0.05,
            "no_yolo": False, "can_interface": "test_can", "check_only": False,
            "allow_motion": False, "setup_reviewed": False, "verbose": False, **fields,
        })

    def run_main(self, args, inputs=("MOVE", "q"), tty=True, connect=None, rounds=None):
        connect = connect or mock.Mock(return_value=self.arm)
        patches = [
            mock.patch.object(capture, "parse_args", return_value=args),
            mock.patch.object(capture, "load_hand_eye", return_value=self.hand_eye),
            mock.patch.object(capture, "load_setup", return_value=(self.settings, np.eye(4), [self.scan])),
            mock.patch.object(capture, "check_camera"),
            mock.patch.object(capture, "VisionClient", return_value=self.objects),
            mock.patch.object(capture, "reference_detector", return_value=self.detector),
            mock.patch.object(capture.sys.stdin, "isatty", return_value=tty),
            mock.patch("builtins.input", side_effect=list(inputs)),
            mock.patch.object(capture, "connect_robot", connect),
            mock.patch.object(capture, "prepare_robot_for_scan"),
            mock.patch.object(capture, "return_to_zero"),
            mock.patch.object(capture, "require_healthy"),
            mock.patch.object(capture.time, "sleep"),
            mock.patch("sys.stdout", new_callable=io.StringIO),
            mock.patch("sys.stderr", new_callable=io.StringIO),
        ]
        if rounds is not None:
            patches.append(mock.patch.object(capture, "capture_rounds", side_effect=rounds))
        mocks = [patch.start() for patch in patches]
        try:
            result = capture.main()
        finally:
            for patch in reversed(patches):
                patch.stop()
        named = dict(zip(
            ["parse", "hand_eye", "setup", "camera", "vision", "reference", "tty", "input",
             "connect", "prepare", "zero", "healthy", "sleep", "stdout", "stderr"], mocks,
        ))
        return result, named

    def test_default_check_only_missing_gates_non_tty_and_cancel_never_connect(self):
        cases = [
            ({}, ("q",), True, 0),
            ({"check_only": True, "allow_motion": True, "setup_reviewed": True}, ("MOVE",), True, 0),
            ({"allow_motion": True}, ("MOVE",), True, 2),
            ({"allow_motion": True, "setup_reviewed": True}, ("MOVE",), False, 2),
            ({"allow_motion": True, "setup_reviewed": True}, ("cancel",), True, 0),
            ({"scan": 2, "allow_motion": True, "setup_reviewed": True}, ("MOVE",), True, 2),
            ({"frames": 0}, (), True, 2),
            ({"frames": 21}, (), True, 2),
            ({"label": "bad label"}, (), True, 2),
            ({"confidence": 1.0}, (), True, 2),
        ]
        for fields, inputs, tty, expected in cases:
            connect = mock.Mock(side_effect=AssertionError("CAN connection forbidden"))
            with self.subTest(fields=fields):
                result, mocks = self.run_main(self.args(**fields), inputs, tty, connect)
                self.assertEqual(result, expected)
                connect.assert_not_called()
                mocks["prepare"].assert_not_called()
                mocks["zero"].assert_not_called()

    def test_capture_moves_to_scan_saves_rounds_then_zeroes_and_disconnects(self):
        args = self.args(allow_motion=True, setup_reviewed=True, label="evening")
        connect = mock.Mock(return_value=self.arm)
        result, mocks = self.run_main(args, ("MOVE", "", "hand", "q"), connect=connect)
        self.assertEqual(result, 0)
        connect.assert_called_once_with(iface="test_can", verbose=False)
        mocks["prepare"].assert_called_once_with(self.arm, self.scan, self.settings)
        mocks["zero"].assert_called_once_with(self.arm, context="capture zero return", verbose=False)
        self.arm.disconnect.assert_called_once()
        self.arm.send_gripper.assert_not_called()
        images = sorted(self.output.glob("*.png"))
        self.assertEqual(len(images), 4)
        self.assertEqual(sum("_r01_evening_f" in image.name for image in images), 2)
        self.assertEqual(sum("_r02_evening_hand_f" in image.name for image in images), 2)
        for image in images:
            self.assertEqual(image.read_bytes(), PNG)
            metadata = json.loads(image.with_suffix(".json").read_text())
            self.assertEqual(metadata["image"], image.name)
            self.assertEqual(metadata["camera_serial"], "D435")
            self.assertEqual(metadata["scan_index"], 1)
            self.assertEqual(metadata["yolo_reference"]["detections"][0]["score"], 0.9)
            self.assertEqual(len(metadata["camera_to_base"]), 4)
            self.assertEqual(metadata["workspace"]["grasp_tip_z_mm"], 18.012)
        self.assertEqual(self.detector.detect_png.call_count, 4)

    def test_failures_and_interrupts_still_zero_and_disconnect(self):
        args = self.args(allow_motion=True, setup_reviewed=True)
        for failure, expected in ((capture.DemoError("camera lost"), 2), (KeyboardInterrupt(), 130)):
            self.arm.reset_mock()
            with self.subTest(failure=failure):
                result, mocks = self.run_main(args, ("MOVE",), rounds=failure)
                self.assertEqual(result, expected)
                mocks["zero"].assert_called_once_with(self.arm, context="capture zero return", verbose=False)
                self.arm.disconnect.assert_called_once()

    def test_failed_zero_return_still_disconnects_and_reports_error(self):
        args = self.args(allow_motion=True, setup_reviewed=True)
        connect = mock.Mock(return_value=self.arm)
        with mock.patch.object(capture, "parse_args", return_value=args), \
                mock.patch.object(capture, "load_hand_eye", return_value=self.hand_eye), \
                mock.patch.object(capture, "load_setup", return_value=(self.settings, np.eye(4), [self.scan])), \
                mock.patch.object(capture, "check_camera"), \
                mock.patch.object(capture, "VisionClient", return_value=self.objects), \
                mock.patch.object(capture, "reference_detector", return_value=None), \
                mock.patch.object(capture.sys.stdin, "isatty", return_value=True), \
                mock.patch("builtins.input", side_effect=["MOVE", "q"]), \
                mock.patch.object(capture, "connect_robot", connect), \
                mock.patch.object(capture, "prepare_robot_for_scan"), \
                mock.patch.object(capture, "return_to_zero", side_effect=capture.DemoError("unsafe")), \
                mock.patch.object(capture.time, "sleep"), \
                mock.patch("sys.stdout", new_callable=io.StringIO), \
                mock.patch("sys.stderr", new_callable=io.StringIO) as error:
            patches_result = capture.main()
        self.assertEqual(patches_result, 2)
        self.assertIn("Cannot return to zero", error.getvalue())
        self.arm.disconnect.assert_called_once()

    def test_motion_during_capture_is_rejected_without_saving(self):
        poses = iter([np.array([200.0, 0, 300, 180, 20, 180]), np.array([205.0, 0, 300, 180, 20, 180])])
        self.arm.pose.side_effect = lambda: next(poses)
        with mock.patch.object(capture, "require_healthy"), \
                self.assertRaisesRegex(capture.DemoError, "moved"):
            capture.capture_frame(
                self.arm, self.objects, self.detector, 0.05, np.eye(4), self.settings,
                self.hand_eye, self.scan, self.output / "frame",
            )
        self.assertFalse(self.output.exists())

    def test_yolo_failure_is_recorded_without_stopping_capture(self):
        self.detector.detect_png.side_effect = capture.DetectionError("detector offline")
        with mock.patch.object(capture, "require_healthy"):
            image, reference = capture.capture_frame(
                self.arm, self.objects, self.detector, 0.05, np.eye(4), self.settings,
                self.hand_eye, self.scan, self.output / "frame",
            )
        self.assertTrue(image.exists())
        self.assertEqual(reference["error"], "detector offline")
        self.assertIn("detector offline", image.with_suffix(".json").read_text())

    def test_existing_capture_is_never_overwritten(self):
        self.output.mkdir(parents=True)
        (self.output / "frame.png").write_bytes(b"original")
        with mock.patch.object(capture, "require_healthy"), \
                self.assertRaisesRegex(capture.DemoError, "overwrite"):
            capture.capture_frame(
                self.arm, self.objects, None, 0.05, np.eye(4), self.settings,
                self.hand_eye, self.scan, self.output / "frame",
            )
        self.assertEqual((self.output / "frame.png").read_bytes(), b"original")
        self.objects.capture_png.assert_not_called()

    def test_invalid_round_label_is_reprompted_and_eof_finishes(self):
        args = self.args(frames=1)
        with mock.patch.object(capture, "require_healthy"), \
                mock.patch("builtins.input", side_effect=["bad label", "ok", EOFError]), \
                mock.patch("sys.stdout", new_callable=io.StringIO):
            saved = capture.capture_rounds(
                self.arm, self.objects, None, np.eye(4), self.settings, self.hand_eye, self.scan, args,
            )
        self.assertEqual(saved, 1)
        self.assertEqual(len(list(self.output.glob("*_r01_ok_f1.png"))), 1)

    def test_reference_detector_is_optional(self):
        objects = mock.Mock()
        with mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertIsNone(capture.reference_detector(objects, False))
            objects.detector.check.side_effect = capture.DetectionError("offline")
            self.assertIsNone(capture.reference_detector(objects, True))
            objects.detector.check.side_effect = None
            self.assertIs(capture.reference_detector(objects, True), objects.detector)

    def test_camera_must_match_calibrated_serial_and_support_capture(self):
        camera = mock.Mock(base_url="http://camera")
        for health, valid in (
            ({"service": "camera", "camera_serial": "D435", "capabilities": ["rgbd-capture"]}, True),
            ({"service": "camera", "camera_serial": "other", "capabilities": ["rgbd-capture"]}, False),
            ({"service": "camera", "camera_serial": "D435", "capabilities": []}, False),
        ):
            camera.health.return_value = health
            with self.subTest(health=health), mock.patch("sys.stdout", new_callable=io.StringIO):
                if valid:
                    capture.check_camera(camera, "D435")
                else:
                    with self.assertRaises(capture.DemoError):
                        capture.check_camera(camera, "D435")

    def test_help_works_outside_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(Path(capture.__file__)), "--help"],
                cwd=directory, text=True, capture_output=True, timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--allow-motion", result.stdout)


if __name__ == "__main__":
    unittest.main()
