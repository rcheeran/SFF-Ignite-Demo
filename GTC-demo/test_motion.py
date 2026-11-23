"""Hardware-free regression tests for the motion, zero-return, and geometry code shared with Ignite."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

import demo
from piper import config as C
from piper.kinematics import fk, ik, MARGINED_LIMITS
from piper.transforms import euler_to_rot, fk_flange, tip_position


class MotionTests(unittest.TestCase):
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

    def test_zero_return_tolerates_resting_slightly_past_zero_limits(self):
        tolerance = demo.ZERO_RETURN_START_TOLERANCE_DEG
        for start, allowed in (
            ([-1.6, -2.0, 2.3, 6.5, 25.0, -12.8], True),   # after power cycle, 2026-11-11
            ([0.0, 0.3, 0.48, 0.0, 0.0, 0.0], True),       # J3 just past its 0 deg limit
            ([0.0, -tolerance, tolerance, 0.0, 0.0, 0.0], True),
            ([0.0, -(tolerance + 0.1), 0.0, 0.0, 0.0, 0.0], False),
            ([0.0, 0.0, tolerance + 0.1, 0.0, 0.0, 0.0], False),
        ):
            arm = self.zero_return_arm()
            arm.joints_deg.side_effect = [np.array(start), np.zeros(6)]
            with self.subTest(start=start), mock.patch("builtins.print"):
                if allowed:
                    demo.return_to_zero(arm, context="startup zero")
                    arm.move_to_zero.assert_called_once()
                else:
                    with self.assertRaisesRegex(demo.DemoError, "more than 5deg outside"):
                        demo.return_to_zero(arm, context="startup zero")
                    arm.move_to_zero.assert_not_called()

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

    def test_flange_for_tip_round_trip(self) -> None:
        rotation = euler_to_rot([180.0, 20.0, 180.0])
        tip = np.array([300.0, 0.0, 20.0])
        flange = demo.flange_for_tip(tip, rotation)
        reconstructed = flange + rotation.apply([0.0, 0.0, C.TOOL_LEN_MM])
        np.testing.assert_allclose(reconstructed, tip)

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


if __name__ == "__main__":
    unittest.main()
