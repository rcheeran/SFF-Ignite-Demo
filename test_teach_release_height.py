"""Hardware-free tests for absolute release-height teaching."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import teach_release_height


class ReleaseHeightTeachingTests(unittest.TestCase):
    def test_saves_low_absolute_height_without_mutating_input(self) -> None:
        config = json.loads(
            teach_release_height.DEFAULT_CONFIG.read_text(encoding="utf-8")
        )
        original = config["demo"]["hand_release_tip_z_mm"]
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            output_path = Path(directory) / "release_height.json"
            teach_release_height.save_taught_height(
                config_path,
                output_path,
                config,
                {"saved_release_tip_z_mm": 82.0},
            )
            saved = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(saved["demo"]["hand_release_tip_z_mm"], 82.0)
        self.assertEqual(config["demo"]["hand_release_tip_z_mm"], original)


if __name__ == "__main__":
    unittest.main()
