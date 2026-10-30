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
        original = dict(config["demo"])
        record = {
            "grasp_tip_z_mm": 16.5,
            "grip_open_command_mm": 45.0,
            "grip_close_command_mm": 5.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            output_path = Path(directory) / "candy_grip.json"
            teach_candy_grip.save_taught_grip(
                config_path,
                output_path,
                config,
                record,
            )
            saved = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(saved["demo"]["grasp_tip_z_mm"], 16.5)
        self.assertEqual(saved["demo"]["grip_open_mm"], 45.0)
        self.assertEqual(saved["demo"]["grip_close_mm"], 5.0)
        self.assertEqual(config["demo"], original)

    def test_rejects_widths_without_clearance(self) -> None:
        with self.assertRaisesRegex(Exception, "at least 5mm"):
            teach_candy_grip.validate_gripper_widths(20.0, 18.0)

    def test_rejects_width_outside_hardware_range(self) -> None:
        with self.assertRaisesRegex(Exception, "within"):
            teach_candy_grip.validate_gripper_widths(75.0, 0.0)


if __name__ == "__main__":
    unittest.main()
