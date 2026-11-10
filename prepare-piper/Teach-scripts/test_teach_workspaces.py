"""Hardware-free tests for read-only workspace teaching and selective persistence."""

from __future__ import annotations

import argparse
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, call, patch

import numpy as np

import teach_workspaces as teaching


def point(x, y, z) -> dict:
    """Build a minimal recorded workspace corner."""
    return {"tool_tip_base_point_mm": [x, y, z]}


def record() -> dict:
    """Build independent measured workspaces, downward RPY, and reviewed clearances."""
    return {
        "workspaces": {
            "candy": [point(250, -60, 12), point(350, 60, 30)],
            "hand": [point(200, -80, 90), point(300, 80, 140)],
        },
        "delivery_pose": {"fk_end_pose_mm_deg": [300, 0, 250, 180, 0, 0]},
        "clearances": {
            "approach_height_mm": 80,
            "pregrasp_height_mm": 40,
            "hand_approach_height_mm": 30,
        },
    }


def feedback_arm() -> Mock:
    """Expose only feedback/disconnect methods so any motion call fails a test."""
    arm = Mock(spec_set=[
        "in_teach", "teach_status_code", "TEACH_STATUS_STOPPED",
        "joints_deg", "pose", "disconnect", "feedback_hz",
    ])
    arm.TEACH_STATUS_STOPPED = 2
    arm.in_teach.return_value = True
    arm.teach_status_code.return_value = 1
    arm.feedback_hz.return_value = 200.0
    arm.joints_deg.return_value = np.array([5.693, 106.412, -57.882, 12.878, 45.32, 53.939])
    arm.pose.return_value = np.array([313.694, 45.767, 136.432, 174.157, -7.259, 123.142])
    return arm


def arguments(config_path: Path, output_path: Path) -> argparse.Namespace:
    """Use a short stable sampling window without connecting to real CAN."""
    return argparse.Namespace(
        config=config_path, output=output_path, can_interface="test_can",
        samples=3, sample_interval=0.01,
        maximum_joint_stddev_deg=0.25, maximum_position_stddev_mm=0.75,
    )


class WorkspaceGeometryTests(unittest.TestCase):
    def test_bounds_are_measured_envelope_in_any_capture_order(self) -> None:
        points = [point(350, -60, 30), point(250, 60, 12), point(300, 0, 20)]
        original = deepcopy(points)
        self.assertEqual(teaching.workspace_bounds(points), ([250, -60, 12], [350, 60, 30]))
        self.assertEqual(points, original)

    def test_bounds_reject_missing_flat_malformed_and_nonfinite_points(self) -> None:
        for points in (
            [], [point(1, 2, 3)], [point(1, 2, 3), point(2, 3, 3)],
            [point(1, 2, 3), point(float("nan"), 4, 5)],
            [point(1, 2, 3), {"tool_tip_base_point_mm": [1, 2]}],
        ):
            with self.subTest(points=points), self.assertRaises((RuntimeError, ValueError)):
                teaching.workspace_bounds(points)

    def test_palm_bounds_do_not_include_release_offset(self) -> None:
        settings = teaching.measured_settings(record())
        self.assertEqual(settings["hand_workspace_max_mm"], [300, 80, 140])
        self.assertEqual(settings["hand_delivery_rpy_deg"], [180, 0, 0])

    def test_upward_delivery_is_rejected(self) -> None:
        measured = record()
        measured["delivery_pose"]["fk_end_pose_mm_deg"][3:] = [0, 0, 0]
        with self.assertRaisesRegex(RuntimeError, "downward"):
            teaching.measured_settings(measured)

    def test_clearances_reject_nonpositive_nonfinite_and_bad_order(self) -> None:
        for value in (0, -1, float("nan"), float("inf"), 80, 90):
            measured = record()
            measured["clearances"]["pregrasp_height_mm"] = value
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                teaching.measured_settings(measured)

    def test_real_fk_tip_matches_taught_candy_pose(self) -> None:
        arm = feedback_arm()
        with patch("teach_scan_poses.time.sleep"):
            result = teaching.capture_point(arm, arguments(Path("c"), Path("r")))
        np.testing.assert_allclose(
            result["tool_tip_base_point_mm"],
            [315.6753535656766, 65.07355055428283, 18.012104120818933], atol=0.01,
        )
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(arm.joints_deg.call_count, 3)
        self.assertTrue(all(c[0] in {
            "in_teach", "teach_status_code", "joints_deg", "pose", "feedback_hz",
        } for c in arm.method_calls))

    def test_nan_telemetry_and_unstable_samples_are_rejected(self) -> None:
        args = arguments(Path("c"), Path("r"))
        with patch("teach_scan_poses.time.sleep"):
            arm = feedback_arm()
            arm.pose.return_value[0] = float("nan")
            with self.assertRaisesRegex(RuntimeError, "finite"):
                teaching.capture_point(arm, args)
            arm = feedback_arm()
            base = arm.joints_deg.return_value
            arm.joints_deg.side_effect = [base, base + 2, base]
            with self.assertRaisesRegex(RuntimeError, "moved"):
                teaching.capture_point(arm, args)

    def test_stopped_teach_and_teach_exit_are_rejected(self) -> None:
        arm = feedback_arm()
        arm.teach_status_code.return_value = 2
        with self.assertRaisesRegex(RuntimeError, "active physical"):
            teaching.capture_point(arm, arguments(Path("c"), Path("r")))
        arm = feedback_arm()
        arm.in_teach.side_effect = [True, True, False]
        with patch("teach_scan_poses.time.sleep"), self.assertRaisesRegex(RuntimeError, "left active"):
            teaching.capture_point(arm, arguments(Path("c"), Path("r")))

    def test_below_floor_capture_is_rejected(self) -> None:
        with patch("teach_scan_poses.time.sleep"), patch.object(
            teaching, "tip_position", return_value=np.array([300, 0, 8])
        ), self.assertRaisesRegex(RuntimeError, "safety floor"):
            teaching.capture_point(feedback_arm(), arguments(Path("c"), Path("r")))

    def test_missing_or_nonfinite_feedback_is_rejected(self) -> None:
        for hz in (0, float("nan"), float("inf")):
            arm = feedback_arm()
            arm.feedback_hz.return_value = hz
            with self.subTest(hz=hz), self.assertRaisesRegex(RuntimeError, "feedback is unavailable"):
                teaching.capture_point(arm, arguments(Path("c"), Path("r")))


class WorkspacePersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config_path = self.root / "config.json"
        self.output_path = self.root / "calibration/workspace_teaching.json"
        self.original = {
            "camera": {"serial": "D435-test"},
            "hand_eye": {"preserve": True},
            "demo": {"grasp_tip_z_mm": 18.012, "grip_open_mm": 60, "motion_speed_percent": 10},
        }
        self.config_path.write_text(json.dumps(self.original))

    def test_selective_merge_preserves_grip_speed_calibration_and_other_records(self) -> None:
        protected = [self.root / name for name in ("candy_grip.json", "scan_poses.json")]
        for path in protected:
            path.write_text('{"untouched": true}')
        measured = record()
        original_record = deepcopy(measured)
        teaching.save_measurements(self.config_path, self.output_path, measured)
        saved = json.loads(self.config_path.read_text())
        self.assertEqual(saved["camera"], self.original["camera"])
        self.assertEqual(saved["hand_eye"], self.original["hand_eye"])
        for key, value in self.original["demo"].items():
            self.assertEqual(saved["demo"][key], value)
        self.assertEqual(measured, original_record)
        self.assertEqual(json.loads(self.output_path.read_text())["settings"], teaching.measured_settings(measured))
        for path in protected:
            self.assertEqual(path.read_text(), '{"untouched": true}')

    def test_calibration_only_config_is_supported(self) -> None:
        self.config_path.write_text('{"camera": {"serial": "D435"}}')
        teaching.save_measurements(self.config_path, self.output_path, record())
        self.assertNotIn("demo", json.loads(self.config_path.read_text()))
        self.assertEqual(len(json.loads(self.output_path.read_text())["settings"]), 8)

    def test_invalid_measurements_or_conflicting_paths_do_not_write(self) -> None:
        initial = self.config_path.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "paths must differ"):
            teaching.save_measurements(self.config_path, self.config_path, record())
        measured = record()
        measured["clearances"]["approach_height_mm"] = 20
        with self.assertRaisesRegex(RuntimeError, "below"):
            teaching.save_measurements(self.config_path, self.output_path, measured)
        self.assertEqual(self.config_path.read_bytes(), initial)
        self.assertFalse(self.output_path.exists())

    def test_existing_pregrasp_search_constraints_are_preserved_or_rejected(self) -> None:
        for minimum, step in ((45, 1), (35, 10), (float("nan"), 1)):
            initial = self.config_path.read_bytes()
            settings = {**teaching.DEMO_SETTINGS, "minimum_pregrasp_height_mm": minimum, "pregrasp_height_step_mm": step}
            with self.subTest(minimum=minimum, step=step), patch.object(
                teaching, "DEMO_SETTINGS", settings
            ), self.assertRaisesRegex(RuntimeError, "conflicts"):
                teaching.save_measurements(self.config_path, self.output_path, record())
            self.assertEqual(self.config_path.read_bytes(), initial)

    def test_record_write_failure_does_not_change_config(self) -> None:
        initial = self.config_path.read_bytes()
        with patch.object(teaching, "write_json_atomic", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                teaching.save_measurements(self.config_path, self.output_path, record())
        self.assertEqual(self.config_path.read_bytes(), initial)

    def test_saving_writes_only_the_teaching_record(self) -> None:
        with patch.object(teaching, "write_json_atomic") as write:
            teaching.save_measurements(self.config_path, self.output_path, record())
        write.assert_called_once()
        self.assertEqual(write.call_args.args[0], self.output_path)

    def test_other_teaching_and_calibration_output_paths_are_protected(self) -> None:
        for name in ("candy_grip.json", "scan_poses.json", "hand_eye.json", "config.json"):
            output_path = self.root / "protected" / name
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "protected artifact"):
                teaching.save_measurements(self.config_path, output_path, record())
            self.assertFalse(output_path.exists())


class WorkspaceInteractionTests(unittest.TestCase):
    def test_clearance_input_has_no_guessed_defaults(self) -> None:
        with patch("builtins.input", side_effect=["", "nan", "80", "40", "30"]), patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(teaching.prompt_clearances({}), record()["clearances"])

    def test_power_off_confirmation_survives_interrupt(self) -> None:
        with patch("builtins.input", side_effect=[KeyboardInterrupt, "", "OFF"]), patch("sys.stdout", new_callable=io.StringIO):
            teaching.confirm_power_off()

    def test_workspace_capture_retries_incomplete_bounds(self) -> None:
        args = arguments(Path("c"), Path("r"))
        points = record()["workspaces"]["candy"]
        with patch("builtins.input", side_effect=["q", "", "", "q"]), patch.object(
            teaching, "capture_point", side_effect=points
        ), patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(teaching.capture_workspace(feedback_arm(), args, "candy"), points)

    def test_readonly_connection_uses_enable_false(self) -> None:
        robot_module = sys.modules[teaching.connect_readonly.__module__]
        with patch.object(robot_module, "Piper") as piper, patch.object(robot_module, "acquire_lock"):
            teaching.connect_readonly(iface="test_can")
        piper.assert_called_once_with("test_can")
        piper.return_value.connect.assert_called_once_with(enable=False)

    def test_main_saves_only_after_confirmation_and_disconnects_on_cancellation(self) -> None:
        for action in ("n", "yes", KeyboardInterrupt, EOFError):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as directory:
                config_path, output_path = Path(directory) / "config.json", Path(directory) / "record.json"
                config_path.write_text('{"demo": {"motion_speed_percent": 10}}')
                initial = config_path.read_bytes()
                arm, measured = feedback_arm(), record()
                with patch.object(teaching, "parse_args", return_value=arguments(config_path, output_path)), patch.object(
                    teaching.sys.stdin, "isatty", return_value=True
                ), patch.object(teaching, "connect_readonly", return_value=arm) as connect, patch.object(
                    teaching, "capture_workspace", side_effect=list(measured["workspaces"].values())
                ), patch.object(teaching, "capture_delivery_orientation", return_value=measured["delivery_pose"]), patch.object(
                    teaching, "prompt_clearances", return_value=measured["clearances"]
                ), patch("builtins.input", side_effect=["", "OFF", action]), patch("sys.stdout", new_callable=io.StringIO), patch(
                    "sys.stderr", new_callable=io.StringIO
                ):
                    result = teaching.main()
                connect.assert_called_once_with(iface="test_can")
                arm.disconnect.assert_called_once_with()
                self.assertEqual(result, 130 if action is KeyboardInterrupt else 2 if action is EOFError else 0)
                self.assertEqual(output_path.exists(), action == "yes")
                if action != "yes":
                    self.assertEqual(config_path.read_bytes(), initial)
                self.assertEqual(arm.method_calls, [
                    call.feedback_hz(), call.in_teach(), call.teach_status_code(), call.disconnect(),
                ])

    def test_capture_interrupt_still_requires_power_off_before_disconnect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path, output_path = Path(directory) / "config.json", Path(directory) / "record.json"
            config_path.write_text("{}")
            arm = feedback_arm()
            with patch.object(teaching, "parse_args", return_value=arguments(config_path, output_path)), patch.object(
                teaching.sys.stdin, "isatty", return_value=True
            ), patch.object(teaching, "connect_readonly", return_value=arm), patch.object(
                teaching, "capture_workspace", side_effect=KeyboardInterrupt
            ), patch("builtins.input", side_effect=["", "OFF"]) as prompt, patch(
                "sys.stdout", new_callable=io.StringIO
            ), patch("sys.stderr", new_callable=io.StringIO):
                self.assertEqual(teaching.main(), 130)
            self.assertIn("power is OFF", prompt.call_args.args[0])
            arm.disconnect.assert_called_once_with()
            self.assertFalse(output_path.exists())

    def test_full_interactive_flow_uses_only_feedback_and_writes_measured_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path, output_path = Path(directory) / "config.json", Path(directory) / "record.json"
            config_path.write_text('{"camera": {"serial": "D435-test"}, "demo": {"motion_speed_percent": 10}}')
            arm, measured = feedback_arm(), record()
            corners = measured["workspaces"]["candy"] + measured["workspaces"]["hand"]
            positions = [np.array(p["tool_tip_base_point_mm"]) for p in corners] + [np.array([250, 0, 200])]
            responses = ["", "", "", "q", "", "", "q", "", "OFF", "80", "40", "30", "yes"]
            with patch.object(teaching, "parse_args", return_value=arguments(config_path, output_path)), patch.object(
                teaching.sys.stdin, "isatty", return_value=True
            ), patch.object(teaching, "connect_readonly", return_value=arm), patch.object(
                teaching, "tip_position", side_effect=positions
            ), patch.object(teaching, "fk_flange", return_value=np.array([250, 0, 320, 180, 0, 0])), patch(
                "teach_scan_poses.time.sleep"
            ), patch("builtins.input", side_effect=responses), patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(teaching.main(), 0)
            saved = json.loads(config_path.read_text())["demo"]
            self.assertEqual(saved, {"motion_speed_percent": 10})
            persisted = json.loads(output_path.read_text())
            self.assertEqual(persisted["settings"], teaching.measured_settings(measured))
            self.assertEqual(persisted["camera_serial"], "D435-test")
            self.assertEqual(persisted["sampling"]["sample_count"], 3)
            self.assertEqual(persisted["release_clearance_mm"], 76.2)
            self.assertEqual(arm.joints_deg.call_count, 15)
            self.assertEqual(arm.method_calls[-1], call.disconnect())


if __name__ == "__main__":
    unittest.main()
