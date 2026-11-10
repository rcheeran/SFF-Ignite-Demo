"""Hardware-free tests for candy grasp teaching persistence."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import teach_candy_grip
from piper import config as PIPER_C


class CandyGripTeachingTests(unittest.TestCase):
    def test_candy_grasp_floor_is_nine_millimetres(self) -> None:
        self.assertEqual(PIPER_C.TIP_Z_FLOOR_MM, 9.0)

    def test_saves_height_and_widths_without_mutating_input(self) -> None:
        config = json.loads(
            teach_candy_grip.DEFAULT_CONFIG.read_text(encoding="utf-8")
        )
        original = json.loads(json.dumps(config))
        record = {
            "grasp_tip_z_mm": 16.5,
            "grip_open_command_mm": 45.0,
            "grip_close_command_mm": 5.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(config))
            output_path = Path(directory) / "candy_grip.json"
            teach_candy_grip.save_taught_grip(
                config_path,
                output_path,
                record,
            )
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(config_path.read_text()), original)

        self.assertEqual(saved["grasp_tip_z_mm"], 16.5)
        self.assertEqual(saved["grip_open_command_mm"], 45.0)
        self.assertEqual(saved["grip_close_command_mm"], 5.0)
        self.assertEqual(config, original)

    def test_defaults_target_d435_files(self) -> None:
        self.assertEqual(teach_candy_grip.DEFAULT_CONFIG.parent.name, "hand-eye-realsense435")
        self.assertEqual(
            teach_candy_grip.DEFAULT_OUTPUT,
            teach_candy_grip.DEFAULT_CONFIG.parent / "calibration/candy_grip.json",
        )

    def test_calibration_only_config_can_start_teaching(self) -> None:
        defaults = teach_candy_grip.initial_grip_settings({"camera": {"serial": "D435"}})
        self.assertEqual(defaults, {"grip_open_mm": 70.0, "grip_close_mm": 0.0})

    def test_invalid_initial_record_widths_are_rejected(self) -> None:
        for settings in ({"grip_open_command_mm": 75}, {"grip_open_command_mm": float("nan")}):
            with self.subTest(settings=settings), self.assertRaises((teach_candy_grip.DemoError, ValueError)):
                teach_candy_grip.initial_grip_settings(settings)

    def test_save_preserves_other_motion_settings(self) -> None:
        config = {"demo": {"motion_speed_percent": 10, "grasp_tip_z_mm": 15}}
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(config))
            teach_candy_grip.save_taught_grip(
                config_path, Path(directory) / "candy_grip.json",
                {"grasp_tip_z_mm": 16.5, "grip_open_command_mm": 45, "grip_close_command_mm": 5},
            )
            saved = json.loads(config_path.read_text())
        self.assertEqual(saved["demo"]["motion_speed_percent"], 10)
        self.assertEqual(config["demo"]["grasp_tip_z_mm"], 15)

    def test_grasp_below_floor_is_not_saved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            with self.assertRaisesRegex(teach_candy_grip.DemoError, "safety floor"):
                teach_candy_grip.save_taught_grip(
                    config_path, Path(directory) / "candy_grip.json",
                    {"grasp_tip_z_mm": 0, "grip_open_command_mm": 45, "grip_close_command_mm": 5},
                )
            self.assertFalse(config_path.exists())

    def test_rejects_widths_without_clearance(self) -> None:
        with self.assertRaisesRegex(Exception, "at least 5mm"):
            teach_candy_grip.validate_gripper_widths(20.0, 18.0)

    def test_prompt_defaults_use_saved_record(self) -> None:
        self.assertEqual(
            teach_candy_grip.initial_grip_settings(
                {"grip_open_command_mm": 45, "grip_close_command_mm": 5}
            ),
            {"grip_open_mm": 45.0, "grip_close_mm": 5.0},
        )

    def test_configuration_cannot_be_overwritten_by_record(self) -> None:
        with self.assertRaisesRegex(teach_candy_grip.DemoError, "paths must differ"):
            teach_candy_grip.save_taught_grip(Path("config.json"), Path("config.json"), {})

    def test_rejects_width_outside_hardware_range(self) -> None:
        with self.assertRaisesRegex(Exception, "within"):
            teach_candy_grip.validate_gripper_widths(75.0, 0.0)


if __name__ == "__main__":
    unittest.main()
