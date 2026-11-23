"""Hardware-free tests for bounded recorded-corner replay."""

import argparse
from copy import deepcopy
import io
import unittest
from unittest.mock import Mock, patch

import numpy as np

import workspace_motion as replay


class WorkspaceReplayTests(unittest.TestCase):
    def setUp(self):
        self.offset_patch = patch.object(replay, "PREVIEW_CORNER_OFFSETS_MM", {})
        self.offset_patch.start()
        self.addCleanup(self.offset_patch.stop)

    def documents(self):
        record = {
            "tool_length_mm": 120.0,
            "workspaces": {"candy": [{"joints_deg": [10, 60, -40, 0, 40, 0], "tool_tip_base_point_mm": [250, 0, 20]}]},
        }
        config = {"demo": {
            "target_workspace_min_mm": [100, -200, 10],
            "target_workspace_max_mm": [500, 200, 300],
            "motion_speed_percent": 20,
            "move_timeout_seconds": 8,
        }}
        return record, config

    def args(self, **fields):
        return argparse.Namespace(**{
            "record": replay.DEFAULT_RECORD, "config": replay.CALIBRATION_DIR / "config.json",
            "corner": None, "corners": None, "can_interface": "test_can", "check_only": False,
            "allow_motion": False, "all_corners": False, "setup_reviewed": False, **fields,
        })

    def test_load_preserves_documents_and_caps_speed(self):
        record, config = self.documents()
        original = deepcopy((record, config))
        with patch.object(replay, "load_json", side_effect=[record, config]), patch.object(
            replay, "tip_position", return_value=np.array([250, 0, 20])
        ), patch.object(replay, "validate_joint_path_safety") as paths:
            corners, speed, timeout = replay.load_corners(self.args().record)
        self.assertEqual(speed, 10)
        self.assertEqual(timeout, 8)
        self.assertEqual(len(corners), 1)
        self.assertEqual((record, config), original)
        paths.assert_called_once()

    def test_changed_tool_length_or_tip_geometry_is_rejected(self):
        for change in ("tool", "tip"):
            record, config = self.documents()
            if change == "tool":
                record["tool_length_mm"] = 100
            else:
                record["workspaces"]["candy"][0]["tool_tip_base_point_mm"][0] += 5
            with self.subTest(change=change), patch.object(
                replay, "load_json", side_effect=[record, config]
            ), patch.object(replay, "tip_position", return_value=np.array([250, 0, 20])), self.assertRaises(RuntimeError):
                replay.load_corners(self.args().record)

    def test_y_offsets_use_real_ik_and_preserve_x_z_orientation_and_record(self):
        record, config = self.documents()
        joints = np.array([30.393, 77.188, -21.962, 24.498, 30.498, -12.893])
        original_tip = replay.tip_position(joints)
        original_rotation = replay.euler_to_rot(replay.fk_flange(joints)[3:])
        record["workspaces"]["candy"][0] = {
            "joints_deg": joints.tolist(), "tool_tip_base_point_mm": original_tip.tolist(),
        }
        config["demo"]["target_workspace_min_mm"][0] = float(original_tip[0])
        for offset in (20, 70):
            original = deepcopy((record, config))
            with self.subTest(offset=offset), patch.object(
                replay, "PREVIEW_CORNER_OFFSETS_MM", {"1": [0, offset, 0]}
            ), patch.object(
                replay, "PREVIEW_MAXIMUM_Y_MM", float(original_tip[1] + offset)
            ), patch.object(replay, "load_json", side_effect=[record, config]):
                corners, _, _ = replay.load_corners(self.args().record)
            requested = original_tip + [0, offset, 0]
            self.assertLessEqual(np.linalg.norm(corners[0]["tool_tip_base_point_mm"] - requested), 1.0)
            achieved_rotation = replay.euler_to_rot(replay.fk_flange(corners[0]["joints_deg"])[3:])
            self.assertLessEqual(np.degrees((original_rotation.inv() * achieved_rotation).magnitude()), 0.5)
            self.assertEqual((record, config), original)

    def test_replay_offsets_reject_unknown_corners_and_nonfinite_values(self):
        for offsets in ({"2": [0, 20, 0]}, {"1": [0, float("nan"), 0]}, []):
            record, config = self.documents()
            with self.subTest(offsets=offsets), patch.object(
                replay, "PREVIEW_CORNER_OFFSETS_MM", offsets
            ), patch.object(
                replay, "load_json", side_effect=[record, config]
            ), patch.object(replay, "tip_position", return_value=np.array([250, 0, 20])), patch.object(
                replay, "validate_joint_path_safety"
            ), self.assertRaises(RuntimeError):
                replay.load_corners(self.args().record)

    def test_c2_preview_target_z_is_eighteen_mm(self):
        record, config = self.documents()
        joints = np.array([30.393, 77.188, -21.962, 24.498, 30.498, -12.893])
        tip = replay.tip_position(joints)
        record["workspaces"]["candy"][0] = {
            "joints_deg": joints.tolist(), "tool_tip_base_point_mm": tip.tolist(),
        }
        with patch.object(
            replay, "PREVIEW_CORNER_OFFSETS_MM", {"1": [0, 70, 18 - float(tip[2])]}
        ), patch.object(
            replay, "PREVIEW_MAXIMUM_Y_MM", float(tip[1] + 70)
        ), patch.object(replay, "load_json", side_effect=[record, config]):
            corners, _, _ = replay.load_corners(self.args().record)
        requested = np.array([tip[0], tip[1] + 70, 18])
        self.assertLessEqual(np.linalg.norm(corners[0]["tool_tip_base_point_mm"] - requested), 1.0)

    def test_c4_preview_target_preserves_taught_orientation(self):
        record, config = self.documents()
        joints = np.array([-39.264, 87.772, -32.652, 2.104, 40.065, -3.598])
        tip = replay.tip_position(joints)
        requested = np.array([144, -170, 18])
        record["workspaces"]["candy"][0] = {
            "joints_deg": joints.tolist(), "tool_tip_base_point_mm": tip.tolist(),
        }
        original_rotation = replay.euler_to_rot(replay.fk_flange(joints)[3:])
        original = deepcopy((record, config))
        with patch.object(
            replay, "PREVIEW_CORNER_OFFSETS_MM", {"1": (requested - tip).tolist()}
        ), patch.object(replay, "load_json", side_effect=[record, config]):
            corners, _, _ = replay.load_corners(self.args().record)
        self.assertLessEqual(np.linalg.norm(corners[0]["tool_tip_base_point_mm"] - requested), 1.0)
        achieved_rotation = replay.euler_to_rot(replay.fk_flange(corners[0]["joints_deg"])[3:])
        self.assertLessEqual(np.degrees((original_rotation.inv() * achieved_rotation).magnitude()), 0.5)
        self.assertEqual((record, config), original)

    def test_c6_and_c8_preview_targets_preserve_taught_orientation(self):
        cases = [
            ([-12.013, 127.581, -95.862, -4.732, 58.696, -3.598], [440, -170, 18]),
            ([19.474, 126.648, -92.894, 8.134, 50.749, 3.141], [440, 206, 18]),
        ]
        for values, coordinates in cases:
            record, config = self.documents()
            joints, requested = np.array(values), np.array(coordinates)
            tip = replay.tip_position(joints)
            record["workspaces"]["candy"][0] = {
                "joints_deg": joints.tolist(), "tool_tip_base_point_mm": tip.tolist(),
            }
            original_rotation = replay.euler_to_rot(replay.fk_flange(joints)[3:])
            original = deepcopy((record, config))
            with self.subTest(target=coordinates), patch.object(
                replay, "PREVIEW_CORNER_OFFSETS_MM", {"1": (requested - tip).tolist()}
            ), patch.object(replay, "load_json", side_effect=[record, config]):
                corners, _, _ = replay.load_corners(self.args().record)
                self.assertLessEqual(np.linalg.norm(corners[0]["tool_tip_base_point_mm"] - requested), 1.0)
                achieved_rotation = replay.euler_to_rot(replay.fk_flange(corners[0]["joints_deg"])[3:])
                self.assertLessEqual(np.degrees((original_rotation.inv() * achieved_rotation).magnitude()), 0.5)
                self.assertEqual((record, config), original)

    def test_replay_checks_current_path_and_preserves_gripper(self):
        corner = {"index": 1, "joints_deg": np.array([10, 60, -40, 0, 40, 0])}
        arm = Mock()
        arm.joints_deg.return_value = corner["joints_deg"]
        with patch.object(replay, "require_healthy"), patch.object(
            replay, "validate_joint_path_safety"
        ) as path, patch.object(replay, "move_robot") as move, patch.object(
            replay, "tip_position", return_value=np.array([250, 0, 20])
        ):
            replay.replay_corner(arm, corner, 10, 8)
        path.assert_called_once()
        move.assert_called_once_with(
            arm, "candy corner 1 (tool-tip base XYZ mm: X=250.00, Y=0.00, Z=20.00)",
            corner["joints_deg"], None, 8, 10,
        )

    def test_unsafe_path_never_moves(self):
        corner = {"index": 1, "joints_deg": np.ones(6)}
        arm = Mock()
        arm.joints_deg.return_value = np.zeros(6)
        with patch.object(replay, "require_healthy"), patch.object(
            replay, "validate_joint_path_safety", side_effect=RuntimeError("unsafe")
        ), patch.object(replay, "move_robot") as move, self.assertRaisesRegex(RuntimeError, "unsafe"):
            replay.replay_corner(arm, corner, 10, 8)
        move.assert_not_called()

    def test_all_eight_corners_visit_in_order_after_whole_route_check(self):
        corners = [{"index": index, "joints_deg": np.full(6, index)} for index in range(1, 9)]
        arm = Mock()
        arm.joints_deg.return_value = np.zeros(6)
        events = Mock()
        with patch.object(replay, "require_healthy"), patch.object(
            replay, "validate_joint_path_safety"
        ) as path, patch.object(replay, "return_to_zero") as zero, \
                patch.object(replay, "replay_corner") as move, patch.object(replay.time, "sleep"):
            events.attach_mock(path, "path")
            events.attach_mock(zero, "zero")
            events.attach_mock(move, "move")
            replay.replay_sequence(arm, corners, 10, 8)
        self.assertEqual(events.mock_calls[0][0], "path")
        self.assertEqual(len(path.call_args.args[1]), 8)
        self.assertEqual([operation[0] for operation in events.mock_calls[:4]], ["path", "path", "path", "zero"])
        np.testing.assert_array_equal(move.call_args_list[0].kwargs["joint_limits"], replay.PIPER_C.JOINT_LIMITS_DEG)
        np.testing.assert_array_equal(move.call_args_list[1].kwargs["joint_limits"], replay.MARGINED_LIMITS)
        self.assertEqual([operation.args[1]["index"] for operation in move.call_args_list], list(range(1, 9)))

    def test_unsafe_later_segment_prevents_even_first_corner_move(self):
        arm = Mock()
        arm.joints_deg.return_value = np.zeros(6)
        with patch.object(replay, "require_healthy"), patch.object(
            replay, "validate_joint_path_safety", side_effect=RuntimeError("later segment unsafe")
        ), patch.object(replay, "replay_corner") as move, self.assertRaisesRegex(RuntimeError, "later segment"):
            replay.replay_sequence(arm, [{"index": 1, "joints_deg": np.ones(6)}], 10, 8)
        move.assert_not_called()

    def test_zero_failure_prevents_any_corner_motion(self):
        arm = Mock()
        arm.joints_deg.return_value = np.zeros(6)
        with patch.object(replay, "require_healthy"), patch.object(
            replay, "validate_joint_path_safety"
        ), patch.object(replay, "return_to_zero", side_effect=RuntimeError("zero failed")), patch.object(
            replay, "replay_corner"
        ) as move, self.assertRaisesRegex(RuntimeError, "zero failed"):
            replay.replay_sequence(arm, [{"index": 1, "joints_deg": np.ones(6)}], 10, 8)
        move.assert_not_called()

    def test_default_check_only_missing_gate_and_cancel_never_connect(self):
        cases = [
            ({}, 0), ({"check_only": True, "allow_motion": True}, 0),
            ({"allow_motion": True, "corner": 1}, 2),
            ({"allow_motion": True, "setup_reviewed": True}, 2),
            ({"allow_motion": True, "setup_reviewed": True, "corner": 2}, 2),
            ({"allow_motion": True, "setup_reviewed": True, "corner": 1}, 0),
            ({"allow_motion": True, "all_corners": True}, 2),
            ({"corners": [1, 1]}, 2),
            ({"corners": [0, 1]}, 2),
        ]
        corner = {"index": 1, "joints_deg": np.ones(6), "tool_tip_base_point_mm": np.array([250, 0, 20])}
        for fields, expected in cases:
            with self.subTest(fields=fields), patch.object(
                replay, "parse_args", return_value=self.args(**fields)
            ), patch.object(replay, "load_corners", return_value=([corner], 10, 8)), patch.object(
                replay.sys.stdin, "isatty", return_value=True
            ), patch("builtins.input", return_value="cancel"), patch.object(
                replay, "connect_for_teach"
            ) as connect, patch("sys.stdout", new_callable=io.StringIO), patch("sys.stderr", new_callable=io.StringIO):
                self.assertEqual(replay.main(), expected)
            connect.assert_not_called()

    def test_single_replay_holds_and_disconnects_even_on_failure(self):
        corner = {"index": 1, "joints_deg": np.ones(6), "tool_tip_base_point_mm": np.array([250, 0, 20])}
        for failure, expected in ((None, 0), (RuntimeError("unsafe"), 2), (KeyboardInterrupt(), 130)):
            arm = Mock()
            with self.subTest(failure=failure), patch.object(
                replay, "parse_args", return_value=self.args(allow_motion=True, setup_reviewed=True, corner=1)
            ), patch.object(replay, "load_corners", return_value=([corner], 10, 8)), patch.object(
                replay.sys.stdin, "isatty", return_value=True
            ), patch("builtins.input", return_value="MOVE"), patch.object(
                replay, "connect_for_teach", return_value=arm
            ), patch.object(replay, "replay_sequence", side_effect=failure) as move, patch.object(
                replay, "hold_until_power_off"
            ) as hold, patch.object(replay.time, "sleep"), patch("sys.stdout", new_callable=io.StringIO), patch(
                "sys.stderr", new_callable=io.StringIO
            ):
                self.assertEqual(replay.main(), expected)
            move.assert_called_once_with(arm, [corner], 10, 8)
            hold.assert_called_once()
            arm.disconnect.assert_called_once()

    def test_all_corners_mode_connects_once_and_powers_off_only_at_end(self):
        corners = [
            {"index": index, "joints_deg": np.ones(6), "tool_tip_base_point_mm": np.array([250, 0, 20])}
            for index in range(1, 9)
        ]
        arm = Mock()
        with patch.object(
            replay, "parse_args", return_value=self.args(allow_motion=True, setup_reviewed=True, all_corners=True)
        ), patch.object(replay, "load_corners", return_value=(corners, 10, 8)), patch.object(
            replay.sys.stdin, "isatty", return_value=True
        ), patch("builtins.input", return_value="MOVE"), patch.object(
            replay, "connect_for_teach", return_value=arm
        ) as connect, patch.object(replay, "replay_sequence") as sequence, patch.object(
            replay, "hold_until_power_off"
        ) as hold, patch.object(replay.time, "sleep"), patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(replay.main(), 0)
        connect.assert_called_once()
        sequence.assert_called_once_with(arm, corners, 10, 8)
        hold.assert_called_once()
        arm.disconnect.assert_called_once()

    def test_selected_even_corners_preserve_requested_order(self):
        corners = [
            {"index": index, "joints_deg": np.ones(6), "tool_tip_base_point_mm": np.array([250, 0, 20])}
            for index in range(1, 9)
        ]
        arm = Mock()
        with patch.object(
            replay, "parse_args", return_value=self.args(allow_motion=True, setup_reviewed=True, corners=[2, 4, 6, 8])
        ), patch.object(replay, "load_corners", return_value=(corners, 10, 8)), patch.object(
            replay.sys.stdin, "isatty", return_value=True
        ), patch("builtins.input", return_value="MOVE"), patch.object(
            replay, "connect_for_teach", return_value=arm
        ), patch.object(replay, "replay_sequence") as sequence, patch.object(
            replay, "hold_until_power_off"
        ) as hold, patch.object(replay.time, "sleep"), patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(replay.main(), 0)
        self.assertEqual([corner["index"] for corner in sequence.call_args.args[1]], [2, 4, 6, 8])
        hold.assert_called_once()


if __name__ == "__main__":
    unittest.main()
