"""Hardware-free tests for configurable scan-pose teaching."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import teach_scan_poses
from piper import config as C


class ScanPoseTeachingTests(unittest.TestCase):
    def test_pose_count_defaults_to_three_and_accepts_one_to_three(self) -> None:
        with mock.patch.object(sys, "argv", ["teach_scan_poses.py"]):
            self.assertEqual(teach_scan_poses.parse_args().pose_count, 3)
        for count in (1, 2, 3):
            with self.subTest(count=count), mock.patch.object(
                sys, "argv", ["teach_scan_poses.py", "--pose-count", str(count)]
            ):
                self.assertEqual(teach_scan_poses.parse_args().pose_count, count)

    def test_invalid_pose_count_never_connects_to_arm(self) -> None:
        for count in ("0", "-1", "4", "1.5"):
            with self.subTest(count=count), mock.patch.object(
                sys, "argv", ["teach_scan_poses.py", "--pose-count", count]
            ), mock.patch.object(teach_scan_poses, "connect_for_teach") as connect, \
                    mock.patch("sys.stderr"), self.assertRaises(SystemExit) as raised:
                teach_scan_poses.main()
            self.assertEqual(raised.exception.code, 2)
            connect.assert_not_called()

    def test_teaching_captures_requested_count_and_saves_matching_document(self) -> None:
        for count in (1, 2, 3):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "scan_poses.json"
                piper = mock.Mock()
                piper.wait_teach.return_value = True
                piper.wait_teach_stopped.return_value = True
                piper.teach_stop.return_value = True
                piper.in_teach.return_value = False
                piper.joints_deg.return_value = np.array([0, 45, -25, 0, 40, 0])
                poses = [{
                    "joints_deg": [index * 10, 45, -25, 0, 40, 0],
                    "end_pose_mm_deg": [200, 0, 300, 180, 0, 0],
                } for index in range(count)]
                with mock.patch.object(sys, "argv", [
                    "teach_scan_poses.py", "--pose-count", str(count), "--output", str(output),
                ]), mock.patch.object(
                    teach_scan_poses, "connect_for_teach", return_value=piper
                ), mock.patch.object(
                    teach_scan_poses, "collect_stable_pose", side_effect=poses
                ) as collect, mock.patch.object(
                    teach_scan_poses, "move_to_sdk_zero", return_value=np.zeros(6)
                ), mock.patch.object(
                    teach_scan_poses, "replay_poses_and_home", return_value=np.zeros(6)
                ) as replay, mock.patch.object(
                    teach_scan_poses, "hold_until_power_off"
                ) as hold, mock.patch("builtins.input", return_value=""), \
                        mock.patch("builtins.print"), mock.patch.object(teach_scan_poses.time, "sleep"):
                    self.assertEqual(teach_scan_poses.main(), 0)
                self.assertEqual(collect.call_count, count)
                document = json.loads(output.read_text())
                self.assertEqual(document["pose_count"], count)
                self.assertEqual([pose["index"] for pose in document["poses"]], list(range(1, count + 1)))
                self.assertEqual(len(replay.call_args.args[1]), count)
                hold.assert_called_once()
                piper.disconnect.assert_called_once()

    def test_rejects_pose_too_close_to_existing_scan(self) -> None:
        existing = [{"index": 1, "joints_deg": [0, 45, -25, 0, 40, 0]}]
        candidate = {"joints_deg": [0, 46, -25, 0, 40, 0]}
        with self.assertRaisesRegex(RuntimeError, "need 5.00deg"):
            teach_scan_poses.validate_separation(existing, candidate, 5.0)

    def test_accepts_distinct_pose(self) -> None:
        existing = [{"index": 1, "joints_deg": [0, 45, -25, 0, 40, 0]}]
        candidate = {"joints_deg": [15, 60, -30, 0, 35, 0]}
        teach_scan_poses.validate_separation(existing, candidate, 5.0)

    def test_rejects_pose_outside_can_replay_margin(self) -> None:
        pose = {"joints_deg": [-66.3, 51.3, -75.7, 11.7, 75.7, 2.9]}
        with self.assertRaisesRegex(
            RuntimeError,
            r"J5=75\.7deg outside \[-68\.0, 68\.0\]deg",
        ):
            teach_scan_poses.validate_replayable_pose(
                pose,
                C.IK_LIMIT_MARGIN_DEG,
            )

    def test_invalid_pose_aborts_before_any_replay_motion(self) -> None:
        class FakePiper:
            def __init__(self) -> None:
                self.targets = []

            def grip_mm(self):
                return 30.0

            def move_joints(self, target, *_args, **_kwargs):
                self.targets.append(target)
                return True

        piper = FakePiper()
        with self.assertRaisesRegex(RuntimeError, "not safely replayable"):
            teach_scan_poses.replay_poses_and_home(
                piper,
                [{"index": 1, "joints_deg": [0, 45, -25, 0, 75, 0]}],
                12.0,
            )
        self.assertEqual(piper.targets, [])

    def test_sdk_zero_uses_home_target_and_speed(self) -> None:
        class FakePiper:
            def __init__(self) -> None:
                self.joints = np.array([5, 40, -20, 0, 35, 0], dtype=float)
                self.moves = []

            def faults(self):
                return []

            def enabled(self):
                return [True] * 6

            def grip_mm(self):
                return 30.0

            def move_to_zero(self, timeout, **kwargs):
                self.moves.append((np.zeros(6), timeout, kwargs))
                self.joints = np.zeros(6, dtype=float)
                return True

            def joints_deg(self):
                return self.joints.copy()

        piper = FakePiper()
        result = teach_scan_poses.move_to_sdk_zero(
            piper,
            12.0,
            "PRE-TEACH ZERO",
        )

        np.testing.assert_allclose(result, C.HOME_JOINTS_DEG)
        np.testing.assert_allclose(piper.moves[0][0], C.HOME_JOINTS_DEG)
        self.assertEqual(piper.moves[0][2]["speed"], C.SPEED_HOME)

    def test_replays_requested_poses_then_moves_to_zero(self) -> None:
        class FakePiper:
            def __init__(self) -> None:
                self.joints = np.array([5, 40, -20, 0, 35, 0], dtype=float)
                self.targets = []

            def grip_mm(self):
                return 30.0

            def faults(self):
                return []

            def enabled(self):
                return [True] * 6

            def move_joints(self, target, *_args, **_kwargs):
                self.joints = np.asarray(target, dtype=float)
                self.targets.append(self.joints.copy())
                return True

            def move_to_zero(self, *_args, **_kwargs):
                self.joints = np.zeros(6, dtype=float)
                self.targets.append(self.joints.copy())
                return True

            def joints_deg(self):
                return self.joints.copy()

        for count in (1, 2, 3):
            with self.subTest(count=count):
                poses = [
                    {"index": index, "joints_deg": [index * 10, 45, -25, 0, 40, 0]}
                    for index in range(1, count + 1)
                ]
                piper = FakePiper()
                hold = teach_scan_poses.replay_poses_and_home(piper, poses, 12.0)
                self.assertEqual(len(piper.targets), count + 2)
                np.testing.assert_allclose(piper.targets[0], C.HOME_JOINTS_DEG)
                np.testing.assert_allclose(piper.targets[-1], C.HOME_JOINTS_DEG)
                np.testing.assert_allclose(hold, C.HOME_JOINTS_DEG)

    def test_replay_stops_on_unreached_pose(self) -> None:
        class FakePiper:
            def __init__(self) -> None:
                self.joints = np.array([5, 40, -20, 0, 35, 0], dtype=float)

            def grip_mm(self):
                return 30.0

            def faults(self):
                return []

            def enabled(self):
                return [True] * 6

            def move_joints(self, *_args, **_kwargs):
                return False

            def move_to_zero(self, *_args, **_kwargs):
                return False

            def joints_deg(self):
                return self.joints.copy()

        with self.assertRaisesRegex(
            RuntimeError,
            "pre-replay zero was not reached",
        ):
            teach_scan_poses.replay_poses_and_home(
                FakePiper(),
                [{"index": 1, "joints_deg": [10, 45, -25, 0, 40, 0]}],
                12.0,
            )


if __name__ == "__main__":
    unittest.main()
