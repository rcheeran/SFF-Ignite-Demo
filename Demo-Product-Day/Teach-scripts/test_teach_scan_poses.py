"""Hardware-free tests for three-pose teaching validation."""

import unittest

import numpy as np

import teach_scan_poses
from piper import config as C


class ScanPoseTeachingTests(unittest.TestCase):
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

    def test_replays_three_poses_then_moves_to_zero(self) -> None:
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

        poses = [
            {"index": index, "joints_deg": [index * 10, 45, -25, 0, 40, 0]}
            for index in (1, 2, 3)
        ]
        piper = FakePiper()

        hold = teach_scan_poses.replay_poses_and_home(piper, poses, 12.0)

        self.assertEqual(len(piper.targets), 5)
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
