"""Hardware-free regression tests for Ignite's local geometry and planners."""

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation as Rot

import demo
import test_demo as fixtures
from piper import config as C
from piper.kinematics import fk, ik, MARGINED_LIMITS
from piper.transforms import euler_to_rot, fk_flange, tip_position


class MotionTests(unittest.TestCase):
    def test_grasp_plans_candies_at_any_angle_with_retaught_height(self):
        settings = demo.validate_demo_settings(demo.DEMO_SETTINGS)
        np.testing.assert_array_equal(settings["grasp_rpy_deg"][:2], [180, 20])
        self.assertEqual(settings["approach_height_mm"], 100)
        self.assertEqual(settings["grasp_tip_z_mm"], 16.584)
        scan = demo.load_scan_poses(demo.DEFAULT_SCAN_POSES)[0]
        arm = mock.Mock()
        arm.joints_deg.return_value = scan["joints_deg"]
        for point, yaw, tilt in (
            # Candies pointing along X: unreachable with the hand-held ~9 degree tilt, and
            # the 20 degree grasp would lean forward, so they are grasped vertically.
            ([386.0694617188344, -56.00213572056944, 13.53736843490475], 169.3, 0.0),
            ([401.4164737470595, 164.94562584745404, 12.231306135758473], 209.0, 0.0),
            ([303.507, 65.829, 11.399], 178.2, 0.0),
            # Candies lying across the line from the base keep the 20 degree tilt.
            ([386.0694617188344, -56.00213572056944, 13.53736843490475], 90.0, 20.0),
            ([350.0, 0.0, 12.0], 120.0, 20.0),
        ):
            axis = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])
            candidate = {
                "status": "detected", "base_point_mm": point, "score": 0.9,
                "slot": 1, "long_axis_base_unit_xy": axis.tolist(),
                "long_axis_base_yaw_deg": yaw,
            }
            with self.subTest(point=point, yaw=yaw), mock.patch("builtins.print"):
                target, plan = demo.pick_plan(arm, [candidate], scan, settings)
                self.assertEqual(target["base_point_mm"], point)
                demo.validate_joint_path_safety(
                    "regression full pickup route",
                    [scan["joints_deg"]] + [joints for _, joints in plan],
                )
                waypoints = dict(plan)
                np.testing.assert_allclose(
                    demo.tip_position(waypoints["grasp"]), [point[0], point[1], 16.584], atol=2,
                )
                approach = demo.tip_position(waypoints["pick approach"])
                np.testing.assert_allclose(approach[:2], point[:2], atol=2)
                self.assertTrue(16.584 + 30 - 2 <= approach[2] <= 16.584 + 100 + 2)
                rotation = fk(waypoints["grasp"])[:3, :3]
                self.assertAlmostEqual(np.degrees(np.arccos(-rotation[2, 2])), tilt, delta=0.5)
                jaw = rotation[:2, 1] / np.linalg.norm(rotation[:2, 1])
                self.assertGreaterEqual(np.degrees(np.arccos(abs(jaw @ axis))), 70.0)
        arm.move_joints.assert_not_called()

    def test_configured_delivery_tilt_reaches_reported_palm_and_workspace_edge(self):
        """The 20-degree delivery pitch keeps J5 inside its limit out to X 500 mm."""
        settings = demo.validate_demo_settings(demo.DEMO_SETTINGS)
        settings["hand_delivery_rpy_deg"] = np.asarray(settings["hand_delivery_rpy_deg"], dtype=float)
        self.assertEqual(settings["hand_delivery_rpy_deg"][1], 20.0)
        tilt = np.degrees(np.arccos(-demo.euler_to_rot(settings["hand_delivery_rpy_deg"]).apply([0, 0, 1])[2]))
        self.assertAlmostEqual(tilt, 20.0, delta=0.5)
        scan = demo.load_scan_poses(demo.DEFAULT_SCAN_POSES)[0]
        arm = mock.Mock()
        arm.joints_deg.return_value = scan["joints_deg"]
        for point in (
            (450.5, -136.3, 29.1),  # reported palm that failed hand-approach IK at 10 degrees
            (500.0, -136.3, 29.1), (500.0, 0.0, 29.1), (500.0, 150.0, 29.1),
            (300.0, 0.0, 29.1),
        ):
            with self.subTest(point=point):
                plan = dict(demo.plan_delivery(fixtures.palm(point), arm, scan, settings, 10))
                np.testing.assert_allclose(
                    tip_position(plan["hand release"]), [point[0], point[1], point[2] + 76.2], atol=2,
                )
                np.testing.assert_allclose(
                    tip_position(plan["hand approach"]), [point[0], point[1], point[2] + 96.2], atol=2,
                )
                for name in ("hand approach", "hand release"):
                    self.assertLessEqual(abs(plan[name][4]), 68.0)

    def test_measured_start_tolerates_settling_past_planning_limit_only(self):
        lift = np.array([11.93, 62.59, -35.1, -21.76, 68.0, 95.37])  # planner output: J5 on +68 margin
        arm = mock.Mock()
        for measured_j5, accepted in ((67.9, True), (68.3, True), (69.99, True), (70.01, False), (71.5, False)):
            arm.joints_deg.return_value = np.array([*lift[:4], measured_j5, lift[5]])
            with self.subTest(j5=measured_j5):
                if accepted:
                    start = demo.measured_start_joints(arm, "return to scan for palm reacquisition")
                    self.assertEqual(start[4], min(measured_j5, 68.0))
                    np.testing.assert_array_equal(start[:4], lift[:4])
                else:
                    with self.assertRaisesRegex(demo.DemoError, r"measured J5=7\d\.\d\ddeg is more than 2deg outside \[-68\.0, 68\.0\]"):
                        demo.measured_start_joints(arm, "return to scan for palm reacquisition")
        arm.joints_deg.return_value = np.array([-0.4, -0.3, 0.2, 0, 0, 0])  # zero-pose encoder noise
        start = demo.measured_start_joints(arm, "startup first scan", np.asarray(C.JOINT_LIMITS_DEG, float))
        np.testing.assert_array_equal(start, [-0.4, 0.0, 0.0, 0, 0, 0])

    def test_reported_lift_overshoot_no_longer_blocks_scan_return_or_candy_return(self):
        """Reproduce: J5 planned at +68 deg, measured 68.3 deg after lift (inside the 70 deg factory limit)."""
        settings = demo.validate_demo_settings(demo.DEMO_SETTINGS)
        scan = demo.load_scan_poses(demo.DEFAULT_SCAN_POSES)[0]
        lift = np.array([11.93, 62.59, -35.1, -21.76, 68.0, 95.37])
        measured = lift.copy()
        measured[4] = 68.3
        with self.assertRaisesRegex(
            demo.DemoError, r"segment 1 crosses a joint limit at the segment start: J5=68\.30deg outside \[-68\.0, 68\.0\]deg",
        ):
            demo.validate_joint_path_safety("return to scan for palm reacquisition", [measured, scan["joints_deg"]])
        state = {"joints": measured.copy()}
        arm = mock.Mock()
        arm.joints_deg.side_effect = lambda: state["joints"].copy()
        with mock.patch.object(demo, "move_robot") as move:
            demo.checked_move(arm, "return to scan for palm reacquisition", scan["joints_deg"], 5.0, settings)
        move.assert_called_once()
        grasp = lift.copy()
        grasp[1] += 8
        grasp[2] -= 4
        grasp[4] = 60.0
        carry = demo.CandyCarry()
        carry.gripped_at(grasp)
        carry.reached("lift", lift)
        plan = [("pick approach", lift), ("grasp", grasp), ("lift", lift), ("scan 1", scan["joints_deg"])]
        state["joints"] = measured.copy()

        def command(_arm, _name, joints, *_args, **_kwargs):
            state["joints"] = np.asarray(joints, dtype=float)

        with mock.patch.object(demo, "move_robot", side_effect=command) as move, \
                mock.patch.object(demo, "hold_gripper") as release:
            demo.return_candy_to_pickup(arm, plan, carry, settings)
        self.assertEqual(move.call_args_list[0].args[1], "return candy: pickup location")
        release.assert_called_once()
        np.testing.assert_allclose(state["joints"], scan["joints_deg"])

    def test_candy_return_from_real_palm_release_passes_real_path_checks(self):
        """Pick a reported candy, carry it to a real planned palm release, then return it."""
        settings = demo.validate_demo_settings(demo.DEMO_SETTINGS)
        settings["hand_delivery_rpy_deg"] = np.asarray(settings["hand_delivery_rpy_deg"], dtype=float)
        scan = demo.load_scan_poses(demo.DEFAULT_SCAN_POSES)[0]
        for point, yaw in (
            ([386.0694617188344, -56.00213572056944, 13.53736843490475], 90.0),
            ([300.0, 100.0, 12.0], 90.0),
        ):
            axis = [float(np.cos(np.radians(yaw))), float(np.sin(np.radians(yaw)))]
            candidate = {
                "status": "detected", "base_point_mm": point, "score": 0.9, "slot": 1,
                "long_axis_base_unit_xy": axis, "long_axis_base_yaw_deg": yaw,
            }
            state = {"joints": np.asarray(scan["joints_deg"], dtype=float)}
            arm = mock.Mock()
            arm.joints_deg.side_effect = lambda: state["joints"].copy()
            _, pickup = demo.pick_plan(arm, [candidate], scan, settings)
            pickup_waypoints = dict(pickup)
            carry = demo.CandyCarry()
            carry.gripped_at(pickup_waypoints["grasp"])
            for name in ("lift", "lift transit"):
                if name in pickup_waypoints:
                    carry.reached(name, pickup_waypoints[name])
            carry.reached("scan pose", scan["joints_deg"])
            palm = fixtures.palm((270, 80, 40))
            delivery = dict(demo.plan_delivery(palm, arm, scan, settings, 10))
            carry.reached("hand approach", delivery["hand approach"])
            carry.reached("hand release height", delivery["hand release"])
            state["joints"] = np.asarray(delivery["hand release"], dtype=float)

            def command(_arm, _name, joints, *_args, **_kwargs):
                state["joints"] = np.asarray(joints, dtype=float)

            with self.subTest(point=point), mock.patch.object(demo, "move_robot", side_effect=command) as move, \
                    mock.patch.object(demo, "hold_gripper") as release:
                demo.return_candy_to_pickup(arm, pickup, carry, settings)
            np.testing.assert_allclose(
                demo.tip_position(release.call_args.args[1]), [point[0], point[1], 16.584], atol=2,
            )
            names = [call.args[1] for call in move.call_args_list]
            self.assertEqual(names[:2], ["return candy: retrace hand approach", "return candy: retrace scan pose"])
            self.assertEqual(names[-1], "return candy: back to scan")
            np.testing.assert_allclose(state["joints"], scan["joints_deg"])
            self.assertFalse(carry.holding)
            arm.move_joints.assert_not_called()

    def test_startup_zero_precedes_first_scan_without_gripper_commands(self):
        arm = mock.Mock()
        arm.joints_deg.return_value = np.zeros(6)
        scan = {"joints_deg": np.array([10, 60, -40, 0, 40, 0])}
        settings = {"move_timeout_seconds": 8, "motion_speed_percent": 10, "grip_open_mm": 60.0, "gripper_hold_seconds": 1.0}
        events = mock.Mock()
        with mock.patch.object(demo, "validate_joint_path_safety") as paths, \
                mock.patch.object(demo, "return_to_zero") as zero, \
                mock.patch.object(demo, "move_robot") as move:
            events.attach_mock(paths, "path")
            events.attach_mock(zero, "zero")
            events.attach_mock(move, "move")
            demo.prepare_robot_for_scan(arm, scan, settings)
        self.assertEqual([operation[0] for operation in events.mock_calls], ["path", "zero", "path", "move"])
        zero.assert_called_once_with(arm, context="startup zero", verbose=False, open_grip_mm=60.0, open_seconds=1.0)
        self.assertIsNone(move.call_args.args[3])
        self.assertFalse(move.call_args.kwargs["verbose"])
        for operation in paths.call_args_list:
            np.testing.assert_array_equal(operation.kwargs["joint_limits"], C.JOINT_LIMITS_DEG)

    def test_failed_startup_zero_never_commands_first_scan(self):
        arm = mock.Mock()
        scan = {"joints_deg": np.ones(6)}
        with mock.patch.object(demo, "validate_joint_path_safety"), mock.patch.object(
            demo, "return_to_zero", side_effect=demo.DemoError("zero failed")
        ), mock.patch.object(demo, "move_robot") as move, self.assertRaisesRegex(demo.DemoError, "zero failed"):
            demo.prepare_robot_for_scan(arm, scan, {"move_timeout_seconds": 8, "motion_speed_percent": 10, "grip_open_mm": 60.0, "gripper_hold_seconds": 1.0})
        move.assert_not_called()

    def test_startup_forwards_verbose_setting_to_zero_return(self):
        arm = mock.Mock()
        arm.joints_deg.return_value = np.zeros(6)
        settings = {"verbose": True, "move_timeout_seconds": 8, "motion_speed_percent": 10, "grip_open_mm": 60.0, "gripper_hold_seconds": 1.0}
        with mock.patch.object(demo, "validate_joint_path_safety"), mock.patch.object(
            demo, "return_to_zero"
        ) as zero, mock.patch.object(demo, "move_robot") as move:
            demo.prepare_robot_for_scan(arm, {"joints_deg": np.ones(6)}, settings)
        zero.assert_called_once_with(arm, context="startup zero", verbose=True, open_grip_mm=60.0, open_seconds=1.0)
        self.assertTrue(move.call_args.kwargs["verbose"])

    def zero_return_arm(self):
        arm = mock.Mock()
        arm.faults.return_value = []
        arm.enabled.return_value = [True] * 6
        arm.j6_temp.return_value = 32
        arm.joints_deg.return_value = np.asarray(C.READY_JOINTS_DEG)
        arm.move_to_zero.return_value = True
        return arm

    def test_zero_return_uses_sdk_sequence_and_preserves_gripper(self):
        arm = self.zero_return_arm()
        arm.joints_deg.side_effect = [np.asarray(C.READY_JOINTS_DEG), np.zeros(6)]
        demo.return_to_zero(arm)
        arm.move_to_zero.assert_called_once_with(secs=C.HOME_TIMEOUT_S, speed=C.SPEED_REST)
        arm.send_gripper.assert_not_called()
        arm.send_joints.assert_not_called()

    def test_zero_return_opens_gripper_only_after_reaching_zero(self):
        arm = self.zero_return_arm()
        arm.joints_deg.side_effect = [np.asarray(C.READY_JOINTS_DEG), np.zeros(6)]
        arm.grip_mm.return_value = 59.5
        events = mock.Mock()
        events.attach_mock(arm.move_to_zero, "zero")
        with mock.patch.object(demo, "hold_gripper") as release, mock.patch("builtins.print") as output:
            events.attach_mock(release, "open")
            demo.return_to_zero(arm, context="shutdown zero return", open_grip_mm=60.0, open_seconds=1.0)
        self.assertEqual([operation[0] for operation in events.mock_calls], ["zero", "open"])
        joints, width, seconds = release.call_args.args[1:]
        np.testing.assert_array_equal(joints, np.zeros(6))
        self.assertEqual((width, seconds), (60.0, 1.0))
        self.assertIn(mock.call("[OK] Gripper open at zero: 59.5mm.", flush=True), output.call_args_list)

    def test_gripper_must_measure_open_at_zero(self):
        for width in (54.9, float("nan"), 20.0):
            arm = self.zero_return_arm()
            arm.joints_deg.side_effect = [np.asarray(C.READY_JOINTS_DEG), np.zeros(6)]
            arm.grip_mm.return_value = width
            with self.subTest(width=width), mock.patch.object(demo, "hold_gripper"), \
                    mock.patch("builtins.print"), self.assertRaisesRegex(demo.DemoError, "did not open at zero"):
                demo.return_to_zero(arm, context="startup zero", open_grip_mm=60.0)
        arm = self.zero_return_arm()
        arm.joints_deg.side_effect = [np.asarray(C.READY_JOINTS_DEG), np.zeros(6)]
        arm.grip_mm.return_value = 55.0
        with mock.patch.object(demo, "hold_gripper"), mock.patch("builtins.print"):
            demo.return_to_zero(arm, context="startup zero", open_grip_mm=60.0)

    def test_gripper_never_opens_when_zero_is_not_reached(self):
        for problem in ("unsafe path", "zero not reached", "outside tolerance"):
            arm = self.zero_return_arm()
            arm.joints_deg.side_effect = [np.asarray(C.READY_JOINTS_DEG), np.full(6, 5.0)]
            if problem == "zero not reached":
                arm.move_to_zero.return_value = False
            with self.subTest(problem=problem), mock.patch.object(
                demo, "validate_joint_path_safety",
                side_effect=demo.DemoError("unsafe") if problem == "unsafe path" else None,
            ), mock.patch.object(demo, "hold_gripper") as release, mock.patch("builtins.print"), \
                    self.assertRaises(demo.DemoError):
                demo.return_to_zero(arm, context="shutdown zero return", open_grip_mm=60.0)
            release.assert_not_called()

    def test_zero_motion_diagnostic_is_verbose_only(self):
        for verbose in (False, True):
            arm = self.zero_return_arm()
            arm.joints_deg.side_effect = [np.asarray(C.READY_JOINTS_DEG), np.zeros(6)]
            with self.subTest(verbose=verbose), mock.patch("builtins.print") as output:
                demo.return_to_zero(arm, context="startup zero", verbose=verbose)
            messages = [call.args[0] for call in output.call_args_list]
            self.assertEqual(
                any("startup zero: moving to SDK zero" in message for message in messages),
                verbose,
            )
            arm.move_to_zero.assert_called_once_with(secs=C.HOME_TIMEOUT_S, speed=C.SPEED_REST)

    def test_motion_announcement_can_be_suppressed_without_skipping_move_or_health_checks(self):
        for verbose in (False, True):
            arm = mock.Mock()
            arm.move_joints.return_value = True
            target = np.zeros(6)
            with self.subTest(verbose=verbose), mock.patch.object(
                demo, "require_healthy"
            ) as health, mock.patch("builtins.print") as output:
                demo.move_robot(
                    arm, "initial scan pose after zero", target, None, 8, 10, verbose=verbose,
                )
            arm.move_joints.assert_called_once_with(
                target, 8, grip_mm=None, speed=10, tol=2.0, settle=0.4,
            )
            self.assertEqual(health.call_args_list, [
                mock.call(arm, "before initial scan pose after zero"),
                mock.call(arm, "after initial scan pose after zero"),
            ])
            self.assertEqual(output.call_count, int(verbose))
            if verbose:
                self.assertEqual(output.call_args.args[0], "motion: initial scan pose after zero")

    def test_scan_pose_motion_announcement_follows_verbose_setting(self):
        scan = {"index": 1, "joints_deg": np.ones(6)}
        for verbose in (False, True):
            settings = {
                "verbose": verbose, "grip_open_mm": 60.0, "move_timeout_seconds": 8.0,
                "motion_speed_percent": 10, "scan_settle_seconds": 0.0,
            }
            with self.subTest(verbose=verbose), \
                    mock.patch.object(demo, "move_robot", side_effect=RuntimeError("stop after move")) as move, \
                    self.assertRaisesRegex(RuntimeError, "stop after move"):
                demo.capture_scan_pair(
                    mock.Mock(), mock.Mock(), mock.Mock(), scan, 4, "starburst candy", 0.5, np.eye(4), settings,
                )
            move.assert_called_once()
            self.assertEqual(move.call_args.args[1], "scan 1")
            self.assertEqual(move.call_args.kwargs["verbose"], verbose)

    def test_enable_diagnostics_follow_verbose_setting(self):
        import piper.robot as robot

        for verbose in (False, True):
            arm = robot.Piper("test_can")
            sdk = mock.Mock()
            with self.subTest(verbose=verbose), mock.patch.object(
                robot, "can_link", return_value={"up": True}
            ), mock.patch.dict(sys.modules, {"piper_sdk": sdk}), mock.patch.object(
                arm, "feedback_hz", return_value=100
            ), mock.patch.object(arm, "joints_deg", return_value=np.zeros(6)), mock.patch.object(
                arm, "_enable_motors", return_value=True
            ) as enable, mock.patch.object(arm, "restore_joint_limits") as limits, mock.patch(
                "builtins.print"
            ) as output:
                self.assertIs(arm.connect(verbose=verbose), arm)
            enable.assert_called_once()
            limits.assert_called_once_with(verbose=verbose)
            self.assertEqual(output.call_count, int(verbose))
            if verbose:
                self.assertIn("enable: joints before", output.call_args.args[0])

    def test_joint_limit_summary_is_verbose_but_restoration_warning_is_always_visible(self):
        import piper.robot as robot

        factory = [list(pair) for pair in C.JOINT_LIMITS_DEG]
        for verbose in (False, True):
            for restore in (False, True):
                stored = [list(pair) for pair in factory]
                if restore:
                    stored[5] = [-179, -60]
                arm = robot.Piper("test_can")
                arm._p = mock.Mock()
                with self.subTest(verbose=verbose, restore=restore), mock.patch.object(
                    arm, "joint_limits", side_effect=[stored, factory]
                ), mock.patch.object(robot.time, "sleep"), mock.patch("builtins.print") as output:
                    arm.restore_joint_limits(verbose=verbose)
                messages = [call.args[0] for call in output.call_args_list]
                self.assertEqual(any("joint limits (deg):" in message for message in messages), verbose)
                self.assertEqual(any(" - restoring" in message for message in messages), restore)
                self.assertEqual(arm._p.MotorAngleLimitMaxSpdSet.call_count, int(restore))

    def test_connection_status_summary_follows_verbose_setting(self):
        import piper.robot as robot

        for verbose in (False, True):
            arm = mock.Mock()
            arm.in_teach.return_value = False
            arm.joints_deg.return_value = np.zeros(6)
            with self.subTest(verbose=verbose), mock.patch.object(
                robot, "acquire_lock"
            ), mock.patch.object(robot, "can_receivers", return_value=0), mock.patch.object(
                robot, "Piper", return_value=arm
            ), mock.patch("builtins.print") as output:
                self.assertIs(robot.connect_robot(iface="test_can", verbose=verbose), arm)
            arm.connect.assert_called_once_with(enable=True, verbose=verbose)
            self.assertEqual(output.call_count, int(verbose))
            if verbose:
                self.assertTrue(output.call_args.args[0].startswith("enabled "))
            arm.disconnect.assert_not_called()

    def test_zero_return_rejects_unhealthy_arm_and_unsafe_paths(self):
        for problem in ("fault", "limit", "floor"):
            arm = self.zero_return_arm()
            if problem == "fault":
                arm.faults.return_value = ["EMERGENCY_STOP"]
            if problem == "limit":
                arm.joints_deg.return_value = np.array([0, 45, -25, 0, 90, 0])
            with self.subTest(problem=problem), mock.patch.object(
                demo, "tip_position", return_value=np.array([200, 0, 0 if problem == "floor" else 100])
            ), self.assertRaises(demo.DemoError):
                demo.return_to_zero(arm)
            arm.move_to_zero.assert_not_called()

    def test_zero_return_requires_success_and_measured_convergence(self):
        for result in (False, True):
            arm = self.zero_return_arm()
            arm.move_to_zero.return_value = result
            with self.subTest(result=result), self.assertRaises(demo.DemoError):
                demo.return_to_zero(arm)

    def test_current_pose_hand_capture_does_not_command_a_scan_move(self):
        arm, hands, objects = mock.Mock(), mock.Mock(), mock.Mock()
        arm.pose.return_value = np.array([200, 0, 250, 180, 20, 180])
        hands.capture_hand.return_value = {"status": "not_found"}
        settings = demo.validate_demo_settings(fixtures.IgniteTests.example_config()["demo"])
        with mock.patch.object(demo, "require_healthy") as health, \
                mock.patch.object(demo, "move_robot") as move:
            result = demo.capture_hand_at_current_pose(
                arm, hands, objects, {"index": 1}, 4, np.eye(4), settings,
            )
        self.assertEqual(result["status"], "not_found")
        hands.capture_hand.assert_called_once_with(4)
        health.assert_called_once()
        move.assert_not_called()
        arm.move_joints.assert_not_called()

    def test_scan_pair_uses_local_object_and_hand_processing(self):
        arm, hands, objects = mock.Mock(), mock.Mock(), mock.Mock()
        arm.pose.return_value = np.array([200, 0, 250, 180, 20, 180])
        objects.capture.return_value = {"status": "not_found", "snapshot_url": "/snapshots/1.jpg"}
        hands.capture_hand.return_value = {"status": "not_found"}
        settings = demo.validate_demo_settings(fixtures.IgniteTests.example_config()["demo"])
        with mock.patch.object(demo, "move_robot"), mock.patch.object(demo.time, "sleep"):
            candy, hand = demo.capture_scan_pair(
                arm, objects, hands, {"index": 1, "joints_deg": np.ones(6)}, 4,
                "starburst candy", 0.25, np.eye(4), settings,
            )
        self.assertEqual((candy["status"], hand["status"]), ("not_found", "not_found"))
        objects.capture.assert_called_once_with(1, "starburst candy", 0.25)
        hands.capture_hand.assert_called_once_with(4)

    def test_fk_matches_sdk_and_ik_round_trips(self):
        for joints in (
            np.asarray(C.READY_JOINTS_DEG),
            np.array([-5.73, 35.966, -53.772, 3.918, 67.018, -7.811]),
        ):
            expected = demo.piper_pose_to_transform(fk_flange(joints))
            np.testing.assert_allclose(fk(joints), expected, atol=1e-8)
            solved, converged = ik(
                expected[:3, 3], euler_to_rot(fk_flange(joints)[3:]),
                joints + 0.1, limits=MARGINED_LIMITS,
            )
            self.assertTrue(converged)
            np.testing.assert_allclose(fk(solved)[:3, 3], expected[:3, 3], atol=C.SIM_IK_TOL_MM)

    def test_hand_eye_transform_composes_base_gripper_and_camera(self):
        base_gripper = demo.piper_pose_to_transform([100, 200, 300, 0, 0, 90])
        gripper_camera = demo.piper_pose_to_transform([10, 0, 20, 0, 0, 0])
        actual = demo.transform_camera_point_to_base(
            np.array([30, 0, 40]), base_gripper, gripper_camera,
        )
        np.testing.assert_allclose(actual, [100, 240, 360], atol=1e-8)
        mean = demo.average_transforms([base_gripper, base_gripper])
        np.testing.assert_allclose(mean, base_gripper, atol=1e-8)
        self.assertAlmostEqual(demo.rotation_error_degrees(mean, base_gripper), 0.0)
        with self.assertRaises(demo.HandEyeError):
            demo.rigid_transform({"camera_to_gripper": [1, 2]}, "camera_to_gripper")

    def test_relative_palm_release_uses_real_ik(self):
        settings = {
            "hand_workspace_min_mm": np.array([100, -400, -30]),
            "hand_workspace_max_mm": np.array([500, 400, 100]),
            "hand_delivery_rpy_deg": np.array([180, 20, 180]),
            "hand_approach_height_mm": 20.0,
        }
        arm = mock.Mock()
        arm.joints_deg.return_value = np.asarray(C.READY_JOINTS_DEG)
        scan = {"joints_deg": np.asarray(C.READY_JOINTS_DEG)}
        plan = dict(demo.plan_delivery(fixtures.palm((260, 40, 0)), arm, scan, settings, 10))
        np.testing.assert_allclose(
            tip_position(plan["hand release"]), [260, 40, 76.2], atol=2.0,
        )
        np.testing.assert_allclose(plan["hand retreat"], plan["hand approach"])

    def test_verbose_transform_reports_intermediate_points_without_changing_result(self):
        base_gripper = demo.piper_pose_to_transform([100, 200, 300, 0, 0, 90])
        gripper_camera = demo.piper_pose_to_transform([10, 0, 20, 0, 0, 0])
        point = np.array([30, 0, 40])
        with mock.patch("builtins.print") as output:
            normal = demo.transform_camera_point_to_base(point, base_gripper, gripper_camera)
        output.assert_not_called()
        with mock.patch("builtins.print") as output:
            verbose = demo.transform_camera_point_to_base(
                point, base_gripper, gripper_camera, verbose=True, label="test",
            )
        np.testing.assert_array_equal(verbose, normal)
        output.assert_called_once()
        text = output.call_args.args[0]
        for expected in (
            "[TRANSFORM test]", "camera XYZ (mm): [30.0, 0.0, 40.0]",
            "gripper XYZ (mm): [40.0, 0.0, 60.0]",
            "robot base XYZ (mm): [100.0, 240.0, 360.0]",
            np.array2string(gripper_camera, precision=3, suppress_small=True, floatmode="fixed"),
            np.array2string(base_gripper, precision=3, suppress_small=True, floatmode="fixed"),
        ):
            self.assertIn(expected, text)

    def test_candy_and_palm_transforms_follow_verbose_setting(self):
        candidate = {
            "status": "detected", "camera_point_mm": [250, 20, 30],
            "long_axis_camera_unit": [1, 0, 0],
        }
        hand = {
            **candidate, "score": 0.9, "handedness": "Right",
            "snapshot_url": "/snapshots/4.jpg",
        }
        for verbose in (False, True):
            settings = {
                "verbose": verbose,
                "target_workspace_min_mm": np.zeros(3),
                "target_workspace_max_mm": np.ones(3) * 500,
                "hand_workspace_min_mm": np.zeros(3),
                "hand_workspace_max_mm": np.ones(3) * 500,
                "candy_min_z_mm": -50.0, "candy_max_z_mm": 50.0,
                "candy_xy_correction_mm": np.zeros(2),
            }
            with self.subTest(verbose=verbose), mock.patch("builtins.print") as output:
                candy_result = demo.transform_object_candidate(
                    candidate, 1, np.eye(4), np.eye(4), settings,
                )
                hand_result = demo.process_hand_capture(
                    hand, 1, 4, mock.Mock(base_url="http://hand"),
                    np.zeros(6), np.zeros(6), np.eye(4), np.eye(4), np.eye(4), settings,
                )
            self.assertEqual(candy_result["status"], "detected")
            self.assertEqual(hand_result["status"], "detected")
            np.testing.assert_array_equal(candy_result["base_point_mm"], [250, 20, 30])
            np.testing.assert_array_equal(hand_result["base_point_mm"], [250, 20, 30])
            traces = [
                call.args[0] for call in output.call_args_list
                if call.args[0].startswith("[TRANSFORM ")
            ]
            self.assertEqual(len(traces), 2 if verbose else 0)
            if verbose:
                self.assertIn("[TRANSFORM candy scan 1]", traces[0])
                self.assertIn("[TRANSFORM palm scan 1]", traces[1])

    def test_motion_checks_use_local_constants(self):
        import piper.robot as robot
        import piper.transforms as transforms
        import piper.kinematics as kinematics

        self.assertIs(demo.PIPER_C, robot.C)
        self.assertIs(demo.PIPER_C, transforms.C)
        self.assertIs(demo.PIPER_C, kinematics.C)
        unsafe = np.asarray(C.READY_JOINTS_DEG)
        unsafe[4] = 90
        with self.assertRaisesRegex(demo.DemoError, "joint limit"):
            demo.validate_joint_path_safety("unsafe", [unsafe, unsafe])


    def test_scan_file_accepts_one_to_three_poses(self) -> None:
        for count in (1, 2, 3):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "poses.json"
                path.write_text(json.dumps({
                    "pose_count": count,
                    "poses": [{
                        "index": index,
                        "joints_deg": [index * 10, 45, -25, 0, 40, 0],
                        "end_pose_mm_deg": [200, 0, 200, 180, 20, 180],
                    } for index in range(1, count + 1)],
                }))
                parsed = demo.load_scan_poses(path)
                self.assertEqual(len(parsed), count)
                self.assertEqual([pose["index"] for pose in parsed], list(range(1, count + 1)))

    def test_execute_pickup_uses_adaptive_transit_sequence(self) -> None:
        arm = mock.Mock()
        arm.grip_mm.return_value = 21.0
        plan = [
            ("pick transit", np.zeros(6)),
            ("pick approach", np.ones(6)),
            ("grasp", np.full(6, 2.0)),
            ("lift", np.ones(6)),
            ("lift transit", np.zeros(6)),
            ("scan 1", np.full(6, 3.0)),
        ]
        config = {
            "motion_speed_percent": 20,
            "pick_motion_speed_percent": 15,
            "move_timeout_seconds": 8.0,
            "grip_open_mm": 70.0,
            "grip_close_mm": 0.0,
            "minimum_held_grip_mm": 8.0,
            "gripper_hold_seconds": 1.0,
        }
        moves = []
        with (
            mock.patch.object(
                demo,
                "move_robot",
                side_effect=lambda _arm, name, *_args: moves.append(
                    (name, _args[-1])
                ),
            ),
            mock.patch.object(demo, "hold_gripper"),
        ):
            held_width = demo.execute_pickup(arm, plan, config)

        self.assertEqual(held_width, 21.0)
        self.assertEqual(
            [name for name, _speed in moves],
            [
                "pick transit",
                "pick approach",
                "grasp",
                "lift",
                "lift transit",
            ],
        )
        self.assertEqual([speed for _name, speed in moves], [15] * 5)

    def test_failed_grasp_reopens_and_returns_to_scan_for_retry(self) -> None:
        arm = mock.Mock()
        arm.grip_mm.return_value = 0.0
        plan = [
            ("pick approach", np.zeros(6)),
            ("grasp", np.ones(6)),
            ("lift", np.full(6, 2.0)),
            ("scan 1", np.full(6, 3.0)),
        ]
        config = {
            "motion_speed_percent": 20,
            "move_timeout_seconds": 8.0,
            "grip_open_mm": 70.0,
            "grip_close_mm": 0.0,
            "minimum_held_grip_mm": 8.0,
            "gripper_hold_seconds": 1.0,
        }
        moves = []
        holds = []
        with (
            mock.patch.object(
                demo,
                "move_robot",
                side_effect=lambda _arm, name, *_args: moves.append(name),
            ),
            mock.patch.object(
                demo,
                "hold_gripper",
                side_effect=lambda _arm, _joints, grip, _seconds: holds.append(grip),
            ),
            self.assertRaises(demo.GraspVerificationError) as raised,
        ):
            demo.execute_pickup(arm, plan, config)

        self.assertAlmostEqual(raised.exception.measured_width_mm, 0.0)
        self.assertEqual(
            moves,
            ["pick approach", "grasp", "lift", "scan 1 after missed grasp"],
        )
        self.assertEqual(holds, [0.0, 70.0])

    def test_held_candy_requires_minimum_post_lift_width(self) -> None:
        plan = [
            ("pick transit", np.full(6, 5.0)),
            ("pick approach", np.zeros(6)),
            ("grasp", np.ones(6)),
            ("lift", np.full(6, 2.0)),
            ("lift transit", np.full(6, 4.0)),
            ("scan 1", np.full(6, 3.0)),
        ]
        config = {
            "motion_speed_percent": 20,
            "move_timeout_seconds": 8.0,
            "grip_open_mm": 60.0,
            "grip_close_mm": 5.0,
            "minimum_held_grip_mm": 8.0,
            "gripper_hold_seconds": 1.0,
        }
        for width, held in (
            (0.0, False), (0.4, False), (5.0, False), (7.99, False), (float("nan"), False),
            (8.0, True), (22.4, True),
        ):
            arm = mock.Mock()
            arm.grip_mm.return_value = width
            carry = demo.CandyCarry()
            moves, holds = [], []
            with self.subTest(width=width), mock.patch.object(
                demo, "move_robot", side_effect=lambda _arm, name, *_args: moves.append(name),
            ), mock.patch.object(
                demo, "hold_gripper",
                side_effect=lambda _arm, joints, grip, _seconds: holds.append((joints[0], grip)),
            ), mock.patch("builtins.print") as output:
                if held:
                    self.assertEqual(demo.execute_pickup(arm, plan, config, carry), width)
                else:
                    with self.assertRaises(demo.GraspVerificationError) as raised:
                        demo.execute_pickup(arm, plan, config, carry)
            pickup_moves = ["pick transit", "pick approach", "grasp", "lift", "lift transit"]
            if held:
                self.assertEqual(moves, pickup_moves)
                self.assertEqual(holds, [(1.0, 5.0)])
                self.assertTrue(carry.holding)
            else:
                # Open at the lift transit pose (candy, if any, drops), then go back to scan.
                self.assertEqual(moves, pickup_moves + ["scan 1 after missed grasp"])
                self.assertEqual(holds, [(1.0, 5.0), (4.0, 60.0)])
                self.assertFalse(carry.holding)
                self.assertEqual(raised.exception.minimum_held_mm, 8.0)
                self.assertIn("below 8.0mm", str(raised.exception))
                self.assertIn("No candy in gripper", str(output.call_args_list))

    def test_forward_leaning_grasps_switch_to_vertical_when_reachable(self):
        settings = demo.validate_demo_settings(demo.DEMO_SETTINGS)
        settings["grasp_rpy_deg"] = np.asarray(settings["grasp_rpy_deg"], dtype=np.float64)
        scan = demo.load_scan_poses(demo.DEFAULT_SCAN_POSES)[0]
        arm = mock.Mock()
        arm.joints_deg.return_value = scan["joints_deg"]

        def tilt_and_lean(point, yaw):
            target = {
                "status": "detected", "base_point_mm": point, "score": 0.9, "slot": 1,
                "long_axis_base_unit_xy": [math.cos(math.radians(yaw)), math.sin(math.radians(yaw))],
                "long_axis_base_yaw_deg": yaw,
            }
            with mock.patch("builtins.print"):
                _, plan = demo.pick_plan(arm, [{"object_candidates": [target]}], scan, settings)
            tool_axis = demo.fk(dict(plan)["grasp"])[:3, 2]
            return math.degrees(math.acos(-tool_axis[2])), demo.forward_lean_x(dict(plan)["grasp"])

        for point, yaw, vertical in (
            ([299.0, 103.6, 5.0], 175.5, True),   # along X: 20 degrees would lean forward
            ([229.4, 116.4, -2.0], -153.4, True),  # diagonal, forward lean X 0.89
            ([309.9, 121.8, 5.0], 87.9, False),   # along Y: sideways lean is kept
            ([480.0, 80.0, 5.0], 175.0, False),   # vertical unreachable: keep tilted grasp
        ):
            with self.subTest(point=point, yaw=yaw):
                tilt, lean_x = tilt_and_lean(point, yaw)
                if vertical:
                    self.assertLess(tilt, 1.0)
                    self.assertEqual(lean_x, 0.0)
                else:
                    self.assertAlmostEqual(tilt, 20.0, delta=0.5)
        with mock.patch.object(demo, "select_reachable_object_target", wraps=demo.select_reachable_object_target) as select:
            settings["forward_lean_vertical_threshold"] = 1.0
            self.assertAlmostEqual(tilt_and_lean([299.0, 103.6, 5.0], 175.5)[0], 20.0, delta=0.5)
        self.assertEqual(select.call_count, 1)

    def test_verbose_pickup_diagnostics_report_frame_grasp_geometry_and_arrival(self):
        settings = demo.validate_demo_settings(demo.DEMO_SETTINGS)
        scan = demo.load_scan_poses(demo.DEFAULT_SCAN_POSES)[0]
        arm = mock.Mock()
        arm.joints_deg.return_value = scan["joints_deg"]
        yaw = 90.0
        target_in = {
            "status": "detected", "base_point_mm": [350.0, 0.0, 8.0], "score": 0.9, "slot": 1,
            "long_axis_base_unit_xy": [0.0, 1.0], "long_axis_base_yaw_deg": yaw,
            "length_mm": 84.0, "width_mm": 33.0,
        }
        capture = {"object_candidates": [target_in], "dataset_image": "images/frame.jpg",
                   "captured_at": "2026-11-04T23:00:00+00:00"}
        target, plan = demo.pick_plan(arm, [capture], scan, settings)
        line = demo.pick_diagnostic(capture, target, plan)
        self.assertIn("PICK PLAN: frame=images/frame.jpg captured_at=2026-11-04T23:00:00+00:00", line)
        self.assertIn("size 84x33mm", line)
        self.assertIn("candy axis yaw 90.0deg", line)
        jaw_angle = float(line.split("jaws at ")[1].split("deg")[0])
        self.assertGreaterEqual(jaw_angle, 70.0)
        self.assertIn("tool tilt 20.0deg leaning [", line)
        vertical_settings = {**settings, "grasp_rpy_deg": np.array([180.0, 0.0, 0.0])}
        _, vertical_plan = demo.pick_plan(arm, [capture], scan, vertical_settings)
        self.assertIn("tool tilt 0.0deg (vertical)", demo.pick_diagnostic(capture, target, vertical_plan))
        waypoints = dict(plan)
        overshoot = waypoints["grasp"].copy()
        overshoot[0] += 0.5
        for verbose, expected in ((True, 1), (False, 0)):
            pick_arm = mock.Mock()
            pick_arm.joints_deg.return_value = overshoot
            pick_arm.grip_mm.return_value = 20.9
            with self.subTest(verbose=verbose), mock.patch.object(demo, "move_robot"), \
                    mock.patch.object(demo, "hold_gripper"), mock.patch("builtins.print") as output:
                demo.execute_pickup(pick_arm, plan, {**settings, "verbose": verbose})
            arrivals = [call.args[0] for call in output.call_args_list if str(call.args[0]).startswith("GRASP ARRIVAL")]
            self.assertEqual(len(arrivals), expected)
            if verbose:
                self.assertIn("max joint error 0.50deg", arrivals[0])

    def test_minimum_held_width_must_lie_between_close_and_open_commands(self) -> None:
        settings = fixtures.IgniteTests.example_config()["demo"]
        self.assertEqual(demo.DEMO_SETTINGS["minimum_held_grip_mm"], 8.0)
        demo.validate_demo_settings(demo.DEMO_SETTINGS)
        for value in (0.0, -1.0, float("nan"), 70.0, 80.0):
            with self.subTest(value=value), self.assertRaises(demo.DemoError):
                demo.validate_demo_settings({**settings, "minimum_held_grip_mm": value})
        with self.assertRaisesRegex(demo.DemoError, "between the close and open"):
            demo.validate_demo_settings({**settings, "grip_close_mm": 10.0, "minimum_held_grip_mm": 8.0})

    def test_flange_for_tip_round_trip(self) -> None:
        rotation = euler_to_rot([180.0, 20.0, 180.0])
        tip = np.array([300.0, 0.0, 20.0])
        flange = demo.flange_for_tip(tip, rotation)
        reconstructed = flange + rotation.apply([0.0, 0.0, C.TOOL_LEN_MM])
        np.testing.assert_allclose(reconstructed, tip)

    def test_plan_pickup_contains_only_pickup_and_return_waypoints(self) -> None:
        grasp_rpy = np.array([180.0, 20.0, 180.0])

        plan = demo.plan_pickup(
            target_base_mm=np.array([300.0, 0.0, 20.0]),
            start_joints=np.asarray(C.READY_JOINTS_DEG),
            return_joints=np.asarray(C.READY_JOINTS_DEG),
            grasp_rpy=grasp_rpy,
            grasp_tip_z_mm=20.0,
            approach_height_mm=100.0,
        )

        self.assertEqual(
            [name for name, _ in plan],
            [
                "pick approach",
                "grasp",
                "lift",
                "scan 1",
            ],
        )
        waypoints = dict(plan)
        np.testing.assert_allclose(waypoints["lift"], waypoints["pick approach"])
        for _, joints in plan:
            self.assertTrue(np.all(np.isfinite(joints)))
            self.assertGreaterEqual(float(tip_position(joints)[2]), C.TIP_Z_FLOOR_MM)

    def test_grasp_finger_axis_aligns_with_candy_length(self) -> None:
        aligned = demo.align_grasp_rpy_to_axis(
            np.array([180.0, 20.0, 180.0]),
            np.array([0.0, 1.0]),
        )

        np.testing.assert_allclose(aligned, [180.0, 20.0, 90.0])
        rotation = euler_to_rot(aligned).as_matrix()
        projected_tool_x = rotation[:2, 0]
        projected_tool_x /= np.linalg.norm(projected_tool_x)
        self.assertAlmostEqual(abs(float(np.dot(projected_tool_x, [0.0, 1.0]))), 1.0)

    def test_equivalent_grasp_yaws_preserve_candy_alignment(self) -> None:
        candidates = demo.equivalent_grasp_rpys(
            np.array([180.0, 20.0, 180.0]),
            np.array([0.3, -0.95]),
        )

        self.assertEqual(len(candidates), 2)
        yaw_difference = (candidates[1][2] - candidates[0][2]) % 360.0
        self.assertAlmostEqual(yaw_difference, 180.0)
        for candidate in candidates:
            tool_x = euler_to_rot(candidate).as_matrix()[:2, 0]
            tool_x /= np.linalg.norm(tool_x)
            axis = np.array([0.3, -0.95])
            axis /= np.linalg.norm(axis)
            self.assertAlmostEqual(abs(float(np.dot(tool_x, axis))), 1.0)

    def test_hand_delivery_releases_above_palm_and_retreats(self) -> None:
        palm = np.array([260.0, 40.0, 60.0])
        plan = demo.plan_hand_delivery(
            palm_base_mm=palm,
            start_joints=np.asarray(C.READY_JOINTS_DEG),
            return_joints=np.asarray(C.READY_JOINTS_DEG),
            delivery_rpy=np.array([180.0, 20.0, 180.0]),
            release_tip_z_mm=73.0,
            approach_height_mm=20.0,
        )

        waypoints = dict(plan)
        self.assertEqual(
            [name for name, _ in plan],
            ["hand approach", "hand release", "hand retreat", "scan 1"],
        )
        np.testing.assert_allclose(
            waypoints["hand retreat"],
            waypoints["hand approach"],
        )
        self.assertGreaterEqual(
            float(tip_position(waypoints["hand release"])[2]),
            73.0 - 2.0,
        )

    def test_choose_target_uses_consensus_and_highest_score_slot(self) -> None:
        captures = [
            {
                "status": "detected",
                "slot": 1,
                "score": 0.7,
                "base_point_mm": [300.0, 10.0, 20.0],
            },
            {"status": "not_found", "slot": 2},
            {
                "status": "detected",
                "slot": 3,
                "score": 0.9,
                "base_point_mm": [304.0, 12.0, 18.0],
            },
        ]

        target = demo.choose_target(captures, maximum_difference=10.0)

        self.assertEqual(target["selected_from_slot"], 3)
        self.assertEqual(target["detected_view_count"], 2)
        np.testing.assert_allclose(target["base_point_mm"], [304.0, 12.0, 18.0])
        np.testing.assert_allclose(
            target["consensus_base_point_mm"], [302.0, 11.0, 19.0]
        )
        self.assertTrue(target["cross_view_agreement_passed"])

    def test_choose_target_reports_cross_view_disagreement(self) -> None:
        captures = [
            {
                "status": "detected",
                "slot": 1,
                "score": 0.8,
                "base_point_mm": [200.0, 0.0, 20.0],
            },
            {
                "status": "detected",
                "slot": 2,
                "score": 0.7,
                "base_point_mm": [300.0, 0.0, 20.0],
            },
        ]
        target = demo.choose_target(captures, maximum_difference=25.0)

        self.assertEqual(target["selected_from_slot"], 1)
        np.testing.assert_allclose(target["base_point_mm"], [200.0, 0.0, 20.0])
        self.assertFalse(target["cross_view_agreement_passed"])
        self.assertEqual(target["maximum_cross_view_difference_mm"], 100.0)

    def test_choose_target_rejects_cross_view_orientation_disagreement(self) -> None:
        captures = [
            {
                "status": "detected",
                "slot": 1,
                "score": 0.8,
                "base_point_mm": [300.0, 0.0, 20.0],
                "long_axis_base_yaw_deg": 5.0,
            },
            {
                "status": "detected",
                "slot": 2,
                "score": 0.7,
                "base_point_mm": [302.0, 1.0, 20.0],
                "long_axis_base_yaw_deg": 42.0,
            },
        ]

        target = demo.choose_target(
            captures,
            maximum_difference=25.0,
            maximum_orientation_difference_deg=20.0,
        )

        self.assertFalse(target["cross_view_orientation_agreement_passed"])
        self.assertFalse(target["cross_view_agreement_passed"])
        self.assertEqual(
            target["maximum_cross_view_orientation_difference_deg"],
            37.0,
        )

    def test_choose_target_ignores_rejected_out_of_workspace_view(self) -> None:
        captures = [
            {"status": "not_found", "slot": 1},
            {
                "status": "detected",
                "slot": 2,
                "score": 0.054,
                "base_point_mm": [210.0, 220.0, 12.0],
            },
            {
                "status": "rejected",
                "slot": 3,
                "score": 0.061,
                "base_point_mm": [149.6, 253.2, 10.5],
                "error": "outside workspace",
            },
        ]

        target = demo.choose_target(captures, maximum_difference=25.0)

        self.assertEqual(target["selected_from_slot"], 2)
        np.testing.assert_allclose(target["base_point_mm"], [210.0, 220.0, 12.0])

    def test_distinct_object_targets_clusters_views_and_sorts_nearest(self) -> None:
        def candidate(slot, score, point, yaw):
            radians = np.deg2rad(yaw)
            return {
                "status": "detected",
                "slot": slot,
                "score": score,
                "base_point_mm": point,
                "long_axis_base_yaw_deg": yaw,
                "long_axis_base_unit_xy": [
                    float(np.cos(radians)),
                    float(np.sin(radians)),
                ],
            }

        captures = [
            {
                "object_candidates": [
                    candidate(1, 0.8, [205.0, 5.0, -30.0], 10.0),
                    candidate(1, 0.9, [350.0, 20.0, -28.0], 80.0),
                ]
            },
            {
                "object_candidates": [
                    candidate(2, 0.85, [200.0, 0.0, -29.0], 12.0),
                    candidate(2, 0.75, [346.0, 18.0, -30.0], 78.0),
                ]
            },
        ]

        targets = demo.distinct_object_targets(captures, 25.0, 20.0)

        self.assertEqual(len(targets), 2)
        self.assertLess(targets[0]["base_distance_mm"], targets[1]["base_distance_mm"])
        self.assertEqual(targets[0]["detected_view_count"], 2)
        self.assertTrue(targets[0]["cross_view_agreement_passed"])
        np.testing.assert_allclose(targets[0]["base_point_mm"], [200.0, 0.0, -29.0])

if __name__ == "__main__":
    unittest.main()
