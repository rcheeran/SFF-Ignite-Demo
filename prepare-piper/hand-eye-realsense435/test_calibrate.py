"""Hardware-free checks for D435 calibration routing and output isolation."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import calibrate


class D435WorkflowTests(unittest.TestCase):
    def test_confirmed_camera_and_marker(self) -> None:
        config = json.loads(calibrate.CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["camera"]["serial"], "263022078260")
        self.assertEqual(config["aruco"]["dictionary"], "DICT_4X4_50")
        self.assertEqual(config["aruco"]["target_marker_id"], 1)
        self.assertEqual(config["aruco"]["marker_size_mm"], 101.6)
        self.assertNotIn("demo", config)

    def test_every_stage_pins_the_d435_configuration(self) -> None:
        for stage in calibrate.STAGES:
            with self.subTest(stage=stage):
                command = calibrate.build_command(stage)
                self.assertEqual(
                    command[command.index("--config") + 1], str(calibrate.CONFIG)
                )
                for flag in (
                    "--samples", "--output", "--calibration",
                    "--hand-eye", "--depth-correction", "--image",
                ):
                    if flag in command:
                        path = Path(command[command.index(flag) + 1])
                        self.assertEqual(path.parent, calibrate.OUTPUT_DIR)
                self.assertNotIn("--allow-failed-quality", command)

    def test_shared_scripts_accept_all_stage_arguments_from_other_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for stage in calibrate.STAGES:
                with self.subTest(stage=stage):
                    result = subprocess.run(
                        [*calibrate.build_command(stage), "--help"],
                        cwd=directory,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)

    def test_verify_saves_its_image_in_the_d435_directory(self) -> None:
        command = calibrate.build_command("verify")
        self.assertEqual(
            command[command.index("--image") + 1],
            str(calibrate.OUTPUT_DIR / "depth_verification.jpg"),
        )
        self.assertIn("--align-in-teach", command)

    def test_missing_hand_eye_solver_is_an_explicit_error(self) -> None:
        with (
            mock.patch.object(
                calibrate.importlib, "import_module",
                return_value=SimpleNamespace(__version__="incomplete"),
            ),
            self.assertRaisesRegex(RuntimeError, "calibrateHandEye"),
        ):
            calibrate.check_dependencies()

    def test_verify_requires_a_passing_depth_correction(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(calibrate, "OUTPUT_DIR", Path(directory)),
        ):
            for name in calibrate.PREREQUISITES["verify"]:
                passed = name != "depth_correction.json"
                (Path(directory) / name).write_text(
                    json.dumps({"acceptance": {"quality_passed": passed}}),
                    encoding="utf-8",
                )
            with self.assertRaisesRegex(RuntimeError, "depth_correction.json"):
                calibrate.require_prerequisites("verify")

    def test_stage_replaces_launcher_and_preserves_exit_status(self) -> None:
        with (
            mock.patch.object(sys, "argv", ["calibrate.py", "solve"]),
            mock.patch.object(calibrate, "check_dependencies"),
            mock.patch.object(
                calibrate.os, "execv", side_effect=SystemExit(7),
            ) as execute,
            self.assertRaises(SystemExit) as exited,
        ):
            calibrate.main()
        self.assertEqual(exited.exception.code, 7)
        command = calibrate.build_command("solve")
        execute.assert_called_once_with(command[0], command)


if __name__ == "__main__":
    unittest.main()
