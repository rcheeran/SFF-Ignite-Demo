"""Hardware-free tests for the GTC bowl-to-hand flow in demo.py."""

import argparse
from datetime import datetime, timedelta, timezone
import io
import itertools
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

import bowl_vision as bv
import demo
from piper.kinematics import fk
from piper.transforms import rot_to_euler, tip_position
from test_bowl_vision import GOLD, GREEN, INTRINSICS, RED, render

# The taught scan pose (prepare-piper/scan-poses/scan_poses.json is not in git).
SCAN = {
    "index": 1,
    "joints_deg": np.array([-5.73, 35.966, -53.772, 3.918, 67.018, -7.811]),
    "end_pose_mm_deg": np.array([82.029, -2.477, 374.27, -177.127, 45.876, -176.709]),
}
# Inside the band the scan pose can see whole and straight-down grasps can reach.
BOWL_XY = (290.0, 20.0)
BOWL = bv.BOWL_SETTINGS
RIM_Z = BOWL["table_z_mm"] + BOWL["bowl_height_mm"]


def settings(**overrides):
    values = demo.validate_demo_settings(demo.DEMO_SETTINGS)
    values.update(camera_intrinsics=INTRINSICS, verbose=False, pause_before_grip=False)
    values.update(overrides)
    return values


def healthy_arm(joints=None, grip_mm=22.0):
    """Mock arm whose joints follow every commanded move."""
    arm = mock.Mock()
    arm.faults.return_value = []
    arm.enabled.return_value = [True] * 6
    arm.j6_temp.return_value = 30
    state = {"joints": np.asarray(SCAN["joints_deg"] if joints is None else joints, dtype=float)}
    arm.joints_deg.side_effect = lambda: state["joints"].copy()
    arm.grip_mm.return_value = grip_mm
    arm.state = state
    return arm


def follow_moves(arm):
    """Patch demo.move_robot so the mock arm's joints become each target; return the patcher."""
    def move(robot, name, joints, *args, **kwargs):
        robot.state["joints"] = np.asarray(joints, dtype=float).copy()
    return mock.patch.object(demo, "move_robot", side_effect=move)


def pose_for(transform):
    """Piper pose (mm, deg) whose transform equals a 4x4 camera-to-base (camera_to_gripper = I)."""
    return np.array([*transform[:3, 3], *rot_to_euler(Rot.from_matrix(transform[:3, :3]))])


class FakeCamera:
    """Camera service with a rendered photo and depth given by a height per rim-plane region."""

    def __init__(self, frame, c2b, height_at):
        self.frame, self.c2b, self.height_at = frame, c2b, height_at
        self.view = bv.bowl_view(INTRINSICS, c2b)

    def capture_png(self):
        captured = {"token": "token1", "frame_sequence": 1, "captured_at": datetime.now(timezone.utc).isoformat()}
        return captured, cv2.imencode(".png", self.frame)[1].tobytes()

    def deproject(self, token, pixel):
        plane_xy = self.view.image_to_plane([pixel])[0]
        height = self.height_at(plane_xy)
        if height is None:
            raise demo.DemoError("no usable depth")
        point = bv.point_at_height(self.view, plane_xy, height)
        return (np.linalg.inv(self.c2b) @ np.append(point, 1.0))[:3]


CANDIES = ((290, 20, 0, RED), (300, 48, 30, GOLD))
FLOOR_Z = 5.0
TIP_ABOVE_TOP = demo.DEMO_SETTINGS["grasp_tip_above_candy_top_mm"]


def bowl_depth(rim_z=RIM_Z - 4.0, candy_z=45.0, candies=CANDIES):
    """Depth model on the rim plane: inner wall top near the rim, candy tops on candies, floor elsewhere."""
    def height_at(plane_xy):
        point = np.asarray(plane_xy)
        if np.linalg.norm(point - BOWL_XY) > BOWL["bowl_diameter_mm"] / 2.0 - 10.0:
            return rim_z
        if any(np.linalg.norm(point - candy[:2]) <= 12.7 for candy in candies):
            return candy_z
        return FLOOR_Z
    return height_at


class BowlScanTests(unittest.TestCase):
    def scan(self, height_at, candies=CANDIES):
        frame, c2b = render(bowl_center=BOWL_XY, candies=candies)
        arm = healthy_arm()
        arm.pose.return_value = pose_for(c2b)
        camera = FakeCamera(frame, c2b, height_at)
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(demo, "LOG_DIR", Path(folder)), \
                mock.patch.object(demo, "BOWL_SCAN_IMAGE", Path(folder) / "scan.jpg"), \
                mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
            result = demo.scan_bowl(arm, camera, np.eye(4), settings())
            saved = (Path(folder) / "scan.jpg").exists()
        return result, arm, camera, stdout.getvalue(), saved

    def test_bowl_is_confirmed_by_rim_depth_and_candies_located_from_depth(self):
        result, arm, camera, output, saved = self.scan(bowl_depth())
        self.assertEqual(result["status"], "detected")
        self.assertTrue(saved)
        self.assertIn("above the table", result["bowl"]["rim_check"])
        self.assertIn("BOWL SCAN: detected", output)
        self.assertEqual(len(result["pickable"]), 2)
        for candy in result["pickable"]:
            expected = bv.point_at_height(camera.view, candy["plane_center_mm"], 45.0)
            np.testing.assert_allclose(candy["base_point_mm"], expected, atol=0.5)  # pixel rounded to 0.1 px
            self.assertAlmostEqual(candy["true_long_mm"], bv.true_size(camera.view, candy["long_mm"], 45.0))
            self.assertAlmostEqual(candy["true_long_mm"], 25.4 * (418 - 45) / (418 - RIM_Z), delta=2.5)
        # The red candy's fingers cannot close along Y: one would land on the gold candy.
        red = min(result["pickable"], key=lambda c: np.linalg.norm(np.subtract(c["plane_center_mm"], (290, 20))))
        self.assertTrue(all(min(abs(yaw - 90.0), abs(yaw - 270.0)) > 15.0 for yaw in red["closing_yaw_deg"]))
        # The best candy has a reachable straight-down grasp just above its top surface.
        candy, plan = demo.pick_first_reachable(arm, result, SCAN, settings())
        np.testing.assert_allclose(
            tip_position(dict(plan)["grasp"]), np.add(candy["base_point_mm"], [0, 0, TIP_ABOVE_TOP]), atol=1.0,
        )

    def test_flat_circle_and_candies_without_depth_are_not_picked(self):
        flat, *_ = self.scan(bowl_depth(rim_z=BOWL["table_z_mm"]))
        self.assertEqual(flat["status"], "flat_circle")
        self.assertIn("flat circle", flat["bowl"]["rim_check"])
        self.assertEqual(flat["pickable"], [])

        def no_candy_depth(plane_xy):
            return bowl_depth()(plane_xy) if np.linalg.norm(np.asarray(plane_xy) - BOWL_XY) > 66.0 else None

        packed, *_ = self.scan(bowl_depth(candies=CANDIES + ((265, 20, 0, GREEN), (315, 20, 0, GREEN), (290, -5, 0, GREEN), (290, 45, 0, GREEN))))
        packed_red = [item for item in packed["unpickable"] if np.linalg.norm(np.subtract(item["plane_center_mm"], (290, 20))) < 3.0]
        self.assertEqual(len(packed_red), 1, "red candy boxed in by neighbours on all four sides")
        self.assertTrue(any("finger would land" in reason for reason in packed_red[0]["reasons"]))

        blind, *_ = self.scan(no_candy_depth)
        self.assertEqual(blind["status"], "no_pickable")
        no_depth = [item for item in blind["rejected_candies"] if any("no depth" in reason for reason in item["reasons"])]
        self.assertEqual(len(no_depth), 2)


class BowlPickPlanTests(unittest.TestCase):
    def candy(self, xy=BOWL_XY, top_z=45.0, closing=(0.0, 90.0)):
        return {"base_point_mm": [*xy, top_z], "closing_yaw_deg": list(closing), "edge_yaw_deg": 0.0}

    def plan(self, candy, bowl_xy=BOWL_XY, values=None):
        return demo.plan_bowl_pick(candy, {"plane_center_mm": list(bowl_xy)}, SCAN["joints_deg"], SCAN["joints_deg"], values or settings())

    def test_pick_approaches_above_the_rim_then_moves_straight_down_and_up(self):
        plan = self.plan(self.candy(closing=(90.0,)))
        names = [name for name, _ in plan]
        self.assertEqual(names[0], "above bowl")
        self.assertEqual(names[-2:], ["lift above bowl", "scan"])
        descents = [name for name in names if name.startswith("descend")]
        lifts = [name for name in names if name.startswith("lift ") and name != "lift above bowl"]
        self.assertGreater(len(descents), 0)
        self.assertEqual(len(lifts), len(descents))
        self.assertEqual(names.index("grasp"), len(descents) + 1)
        waypoints = dict(plan)
        np.testing.assert_allclose(tip_position(waypoints["above bowl"]), [*BOWL_XY, RIM_Z + 30.0], atol=0.5)
        np.testing.assert_allclose(tip_position(waypoints["grasp"]), [*BOWL_XY, 45.0 + TIP_ABOVE_TOP], atol=0.5)
        for name in descents:
            np.testing.assert_allclose(tip_position(waypoints[name])[:2], BOWL_XY, atol=0.5)
        # Lifts retrace the descent in reverse.
        for lift, descent in zip(lifts, reversed(descents)):
            np.testing.assert_array_equal(waypoints[lift], waypoints[descent])
        # The jaws close along the requested direction (90 degrees = base +Y).
        jaw_axis = fk(waypoints["grasp"])[:3, 1]
        self.assertAlmostEqual(abs(jaw_axis[1]), 1.0, delta=0.01)
        np.testing.assert_array_equal(waypoints["scan"], SCAN["joints_deg"])

    def test_grasp_height_never_goes_below_the_bowl_floor_limit(self):
        plan = self.plan(self.candy(top_z=5.0))
        self.assertAlmostEqual(tip_position(dict(plan)["grasp"])[2], settings()["minimum_grasp_tip_z_mm"], delta=0.5)

    def test_candies_at_the_wall_or_out_of_reach_have_no_plan(self):
        with self.assertRaisesRegex(demo.DemoError, "clear of the bowl wall"):
            self.plan(self.candy(xy=(BOWL_XY[0] + 60.0, BOWL_XY[1])))
        with self.assertRaisesRegex(demo.DemoError, "no reachable bowl grasp at any allowed tilt"):
            self.plan(self.candy(xy=(640.0, 0.0)), bowl_xy=(640.0, 0.0))

    def test_far_bowl_uses_the_smallest_tilt_and_descends_along_the_tool_axis(self):
        far = (420.0, 0.0)
        vertical_only = settings(grasp_tilt_options_deg=[0.0])
        with self.assertRaisesRegex(demo.DemoError, "no reachable bowl grasp"):
            self.plan(self.candy(xy=far), bowl_xy=far, values=vertical_only)
        plan = dict(self.plan(self.candy(xy=far), bowl_xy=far))
        self.assertAlmostEqual(demo.grasp_tilt_deg(plan["grasp"]), 10.0, delta=0.5)
        grasp_tip = np.array([*far, 45.0 + TIP_ABOVE_TOP])
        np.testing.assert_allclose(tip_position(plan["grasp"]), grasp_tip, atol=0.5)
        frame = fk(plan["grasp"])
        up = -frame[:3, 2]
        self.assertLess(abs(frame[2, 1]), 0.01)  # the jaws still close level
        for name, joints in plan.items():
            if name.startswith("descend") or name == "above bowl":
                offset = np.cross(tip_position(joints) - grasp_tip, up)
                self.assertLess(np.linalg.norm(offset), 0.5, name)
        self.assertAlmostEqual(tip_position(plan["above bowl"])[2], RIM_Z + 30.0, delta=0.5)
        self.assertTrue(bv.gripper_clear_of_bowl(
            grasp_tip, frame[:3, 2], 0.0, far, RIM_Z, demo.FINGER_EXTENSION_BELOW_TIP_MM, settings()["bowl"],
        ) or bv.gripper_clear_of_bowl(
            grasp_tip, frame[:3, 2], 90.0, far, RIM_Z, demo.FINGER_EXTENSION_BELOW_TIP_MM, settings()["bowl"],
        ))

    def test_descent_that_strays_from_its_line_is_rejected(self):
        plan = dict(self.plan(self.candy()))
        stray = demo.plan_bowl_pick(
            self.candy(xy=(BOWL_XY[0] + 12.0, BOWL_XY[1])), {"plane_center_mm": list(BOWL_XY)},
            SCAN["joints_deg"], SCAN["joints_deg"], settings(),
        )
        mixed = [("above bowl", plan["above bowl"]), ("grasp", dict(stray)["grasp"])]
        with self.assertRaisesRegex(demo.DemoError, "strays"):
            demo.validate_descent_line(mixed, [*BOWL_XY, 45.0 + TIP_ABOVE_TOP], [0.0, 0.0, 1.0], RIM_Z + 15.0, 3.0)


class PickupExecutionTests(unittest.TestCase):
    def setUp(self):
        self.settings = settings()
        self.candy = {"base_point_mm": [*BOWL_XY, 45.0], "closing_yaw_deg": [0.0], "edge_yaw_deg": 90.0}
        self.plan = demo.plan_bowl_pick(
            self.candy, {"plane_center_mm": list(BOWL_XY)}, SCAN["joints_deg"], SCAN["joints_deg"], self.settings,
        )
        self.names = [name for name, _ in self.plan]

    def execute(self, grip_mm):
        arm = healthy_arm(grip_mm=grip_mm)
        carry = demo.CandyCarry()
        with follow_moves(arm) as move, mock.patch.object(demo, "hold_gripper") as hold:
            try:
                width = demo.execute_bowl_pickup(arm, self.plan, self.settings, carry, self.candy)
            except demo.GraspVerificationError as exc:
                width = exc
        return width, carry, move, hold

    def test_held_candy_is_lifted_clear_of_the_rim_and_carried_to_scan(self):
        width, carry, move, hold = self.execute(22.0)
        self.assertEqual(width, 22.0)
        moved = [call.args[1] for call in move.call_args_list]
        self.assertEqual(moved, self.names)
        grips = [call.args[3] for call in move.call_args_list]
        grasp = self.names.index("grasp")
        self.assertTrue(all(grip == 40.0 for grip in grips[:grasp + 1]))
        self.assertTrue(all(grip == 1.0 for grip in grips[grasp + 1:]))
        hold.assert_called_once()
        np.testing.assert_array_equal(hold.call_args.args[1], dict(self.plan)["grasp"])
        self.assertEqual(hold.call_args.args[2], 1.0)
        self.assertTrue(carry.holding)
        self.assertEqual([name for name, _ in carry.path], ["grasp", *self.names[grasp + 1:]])

    def test_missed_grasp_opens_above_the_bowl_and_returns_to_scan(self):
        error, carry, move, hold = self.execute(1.0)
        self.assertIsInstance(error, demo.GraspVerificationError)
        self.assertFalse(carry.holding)
        opened = hold.call_args_list[-1]
        np.testing.assert_array_equal(opened.args[1], dict(self.plan)["lift above bowl"])
        self.assertEqual(opened.args[2], 40.0)
        self.assertEqual(move.call_args.args[1], "return to scan pose after missed grasp")
        self.assertEqual(move.call_args.args[3], 40.0)

    def test_return_after_leaving_the_bowl_retraces_and_drops_above_it(self):
        waypoints = dict(self.plan)
        carry = demo.CandyCarry()
        carry.gripped_at(waypoints["grasp"])
        for name in self.names[self.names.index("grasp") + 1:]:
            carry.reached(name, waypoints[name])
        arm = healthy_arm(joints=waypoints["scan"])
        with follow_moves(arm) as move, mock.patch.object(demo, "hold_gripper") as hold, \
                mock.patch("builtins.print"):
            demo.return_candy_to_bowl(arm, self.plan, carry, self.settings)
        self.assertEqual(
            [call.args[1] for call in move.call_args_list],
            ["return candy: above bowl", "return candy: back to scan"],
        )
        self.assertEqual(move.call_args_list[0].args[3], 1.0)  # still holding on the way
        np.testing.assert_array_equal(hold.call_args.args[1], waypoints["lift above bowl"])
        self.assertEqual(hold.call_args.args[2], 40.0)
        self.assertFalse(carry.holding)

    def test_return_during_the_lift_finishes_the_lift_first(self):
        waypoints = dict(self.plan)
        carry = demo.CandyCarry()
        carry.gripped_at(waypoints["grasp"])
        carry.reached("lift 1", waypoints["lift 1"])
        arm = healthy_arm(joints=waypoints["lift 1"])
        with follow_moves(arm) as move, mock.patch.object(demo, "hold_gripper"), mock.patch("builtins.print"):
            demo.return_candy_to_bowl(arm, self.plan, carry, self.settings)
        remaining = [name for name in self.names if name.startswith("lift ") and name != "lift above bowl"][1:]
        self.assertEqual(
            [call.args[1] for call in move.call_args_list],
            [f"return candy: {name}" for name in remaining] + ["return candy: above bowl", "return candy: back to scan"],
        )


def hand_capture(palm_status="visible", finger_status="visible", palm=(300.0, 0.0, 40.0), finger=(340.0, 0.0, 40.0), age=0.0):
    captured_at = (datetime.now(timezone.utc) - timedelta(seconds=age)).isoformat()
    return {
        "status": "detected", "captured_at": captured_at,
        "points": {
            "palm_center": {"status": palm_status, "camera_point_mm": list(palm)},
            "finger_base": {"status": finger_status, "camera_point_mm": list(finger)},
        },
    }


class HandDropTests(unittest.TestCase):
    """With camera_to_gripper = I and the arm pose at the origin, camera points are base points."""

    def arm(self):
        arm = healthy_arm()
        arm.pose.return_value = np.zeros(6)
        return arm

    def test_waits_reporting_each_new_reason_once_then_plans_the_drop(self):
        hands = mock.Mock()
        hands.capture_hand.side_effect = [
            {"status": "not_found"}, {"status": "not_found"},
            hand_capture(finger_status="inconsistent"), hand_capture(palm_status="no_depth"),
        ]
        with mock.patch.object(demo.time, "sleep"), mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
            capture, decision, plan = demo.wait_for_hand(self.arm(), hands, SCAN, np.eye(4), settings())
        output = stdout.getvalue()
        self.assertEqual(output.count("no open palm"), 1)
        self.assertEqual(output.count("finger base not visible (inconsistent)"), 1)
        self.assertIn("drop on the finger base at [340.0, 0.0, 40.0]mm", output)
        self.assertEqual(decision["target"], "finger_base")
        np.testing.assert_allclose(tip_position(plan["hand release"]), [340.0, 0.0, 75.0], atol=1.0)
        np.testing.assert_allclose(tip_position(plan["hand approach"]), [340.0, 0.0, 95.0], atol=1.0)
        hands.capture_hand.assert_called_with(demo.HAND_SLOT)

    def test_delivery_releases_above_the_palm_only_while_the_capture_is_fresh(self):
        for name, ages, released in (("fresh", itertools.repeat(0.5), True), ("stale before release", [0.5, 0.5, 11.0], False)):
            with self.subTest(name):
                hands = mock.Mock()
                hands.capture_hand.return_value = hand_capture()
                carry = demo.CandyCarry()
                carry.gripped_at(SCAN["joints_deg"])
                with mock.patch.object(demo, "checked_move") as move, mock.patch.object(demo, "hold_gripper") as hold, \
                        mock.patch.object(demo, "capture_age_seconds", side_effect=ages), \
                        mock.patch("builtins.print"):
                    try:
                        demo.deliver_to_hand(self.arm(), hands, SCAN, np.eye(4), settings(), carry)
                    except demo.DemoError as exc:
                        self.assertIn("stale", str(exc))
                moves = [call.args[1] for call in move.call_args_list]
                if released:
                    self.assertEqual(moves, ["hand approach", "hand release height (+35mm)", "hand retreat", "return to scan"])
                    self.assertEqual(hold.call_args.args[2], 40.0)
                    self.assertFalse(carry.holding)
                else:
                    self.assertEqual(moves, ["hand approach", "hand release height (+35mm)"])
                    hold.assert_not_called()
                    self.assertTrue(carry.holding)

    def test_drop_plan_reaches_palms_out_to_the_workspace_edge(self):
        """The 20-degree delivery pitch keeps J5 inside its limit out to X 500 mm."""
        arm = healthy_arm()
        for point in ((450.5, -136.3, 29.1), (500.0, -136.3, 29.1), (500.0, 0.0, 29.1), (500.0, 150.0, 29.1), (300.0, 0.0, 29.1)):
            with self.subTest(point=point):
                plan = demo.plan_drop({"base_point_mm": list(point)}, arm, SCAN, settings())
                np.testing.assert_allclose(tip_position(plan["hand release"]), [point[0], point[1], point[2] + 35.0], atol=2)
                for name in ("hand approach", "hand release"):
                    self.assertLessEqual(abs(plan[name][4]), 68.0)


class CycleTests(unittest.TestCase):
    def run_cycle(self, scan_result, pick_only=False, execute=None, deliver=None):
        arm, camera, hands = healthy_arm(), mock.Mock(), mock.Mock()
        candy = {"base_point_mm": [*BOWL_XY, 45.0]}
        with mock.patch.object(demo, "go_to_scan"), \
                mock.patch.object(demo, "scan_bowl", return_value=scan_result), \
                mock.patch.object(demo, "pick_first_reachable", return_value=(candy, [("grasp", np.zeros(6))])) as plan, \
                mock.patch.object(demo, "execute_bowl_pickup", side_effect=execute or (lambda *a: 22.0)) as pick, \
                mock.patch.object(demo, "deliver_to_hand", side_effect=deliver) as delivery, \
                mock.patch.object(demo, "return_candy_to_bowl") as back, \
                mock.patch("builtins.print"):
            result = demo.run_cycle(arm, camera, None if pick_only else hands, SCAN, np.eye(4), settings(), pick_only)
        return result, plan, pick, delivery, back

    def test_no_pickable_bowl_means_no_pick(self):
        result, plan, pick, *_ = self.run_cycle({"status": "empty"})
        self.assertFalse(result)
        plan.assert_not_called()
        pick.assert_not_called()

    def test_missed_grasp_rescans_without_waiting_for_a_hand(self):
        def miss(*args):
            raise demo.GraspVerificationError(1.0, 12.0)
        result, _, _, delivery, back = self.run_cycle({"status": "detected"}, execute=miss)
        self.assertFalse(result)
        delivery.assert_not_called()
        back.assert_not_called()

    def test_error_while_holding_returns_the_candy_to_the_bowl(self):
        def holding(arm, plan, values, carry, candy):
            carry.gripped_at(np.zeros(6))
            return 22.0
        result, _, _, _, back = self.run_cycle(
            {"status": "detected"}, execute=holding, deliver=demo.DemoError("hand approach did not reach its target"),
        )
        self.assertFalse(result)
        back.assert_called_once()

    def test_error_with_nothing_held_stops_the_loop(self):
        with self.assertRaisesRegex(demo.DemoError, "faults"):
            self.run_cycle({"status": "detected"}, execute=demo.DemoError("before grasp: Piper faults: [1]"))

    def test_pick_only_drops_the_candy_back_without_a_hand(self):
        def holding(arm, plan, values, carry, candy):
            carry.gripped_at(np.zeros(6))
            return 22.0
        result, _, _, delivery, back = self.run_cycle({"status": "detected"}, pick_only=True, execute=holding)
        self.assertTrue(result)
        delivery.assert_not_called()
        back.assert_called_once()


class MainTests(unittest.TestCase):
    def args(self, **overrides):
        values = dict(
            calibration_dir=Path("calibration"), scan_poses=Path("scan_poses.json"), vision_url="http://camera:8094",
            hand_url="http://hand:8095", can_interface="can_piper", check_only=False, scan_only=False,
            pick_only=False, verbose=False, pause_before_grip=False, allow_motion=True, max_cycles=2,
        )
        values.update(overrides)
        return argparse.Namespace(**values)

    def run_main(self, args):
        arm = healthy_arm()
        with mock.patch.object(demo, "parse_args", return_value=args), \
                mock.patch.object(demo, "load_hand_eye", return_value={"camera": {"serial": "D435"}}), \
                mock.patch.object(demo, "ensure_k3s_address"), \
                mock.patch.object(demo, "check_services") as services, \
                mock.patch.object(demo, "load_setup", return_value=(settings(), np.eye(4), [SCAN])), \
                mock.patch.object(demo, "connect_robot", return_value=arm) as connect, \
                mock.patch.object(demo, "prepare_robot_for_scan") as startup, \
                mock.patch.object(demo, "run_cycle", return_value=True) as cycle, \
                mock.patch.object(demo, "run_scan_only") as scan_only, \
                mock.patch.object(demo, "return_to_zero") as zero, \
                mock.patch.object(demo.time, "sleep"), mock.patch("sys.stdout", new_callable=io.StringIO), \
                mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
            code = demo.main()
        return code, arm, services, connect, startup, cycle, scan_only, zero, stderr.getvalue()

    def test_check_only_and_missing_allow_motion_never_connect(self):
        code, _, _, connect, *_ = self.run_main(self.args(check_only=True))
        self.assertEqual(code, 0)
        connect.assert_not_called()
        code, *_, stderr = self.run_main(self.args(allow_motion=False))
        self.assertEqual(code, 2)
        self.assertIn("motion requires --allow-motion", stderr)

    def test_loop_runs_requested_cycles_then_returns_to_zero_and_disconnects(self):
        code, arm, services, _, startup, cycle, scan_only, zero, _ = self.run_main(self.args())
        self.assertEqual(code, 0)
        startup.assert_called_once()
        self.assertEqual(cycle.call_count, 2)
        scan_only.assert_not_called()
        self.assertIsNotNone(services.call_args.args[1])  # hand service checked
        zero.assert_called_once()
        self.assertEqual(zero.call_args.kwargs["open_grip_mm"], 40.0)
        arm.disconnect.assert_called_once()

    def test_pick_only_skips_the_hand_service_and_scan_only_never_picks(self):
        _, _, services, _, _, cycle, _, _, _ = self.run_main(self.args(pick_only=True, max_cycles=1))
        self.assertIsNone(services.call_args.args[1])
        self.assertTrue(cycle.call_args.kwargs["pick_only"])
        _, _, _, _, _, cycle, scan_only, _, _ = self.run_main(self.args(scan_only=True, max_cycles=3))
        self.assertEqual(scan_only.call_count, 3)
        cycle.assert_not_called()

    def test_scan_only_and_pick_only_are_exclusive(self):
        with mock.patch.object(sys, "argv", ["demo.py", "--scan-only", "--pick-only"]), \
                mock.patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
            demo.parse_args()

    def test_error_stops_the_loop_then_returns_to_zero(self):
        args = self.args(max_cycles=0)
        arm = healthy_arm()
        with mock.patch.object(demo, "parse_args", return_value=args), \
                mock.patch.object(demo, "load_hand_eye", return_value={"camera": {"serial": "D435"}}), \
                mock.patch.object(demo, "ensure_k3s_address"), mock.patch.object(demo, "check_services"), \
                mock.patch.object(demo, "load_setup", return_value=(settings(), np.eye(4), [SCAN])), \
                mock.patch.object(demo, "connect_robot", return_value=arm), \
                mock.patch.object(demo, "prepare_robot_for_scan"), \
                mock.patch.object(demo, "run_cycle", side_effect=demo.DemoError("candy return failed")), \
                mock.patch.object(demo, "return_to_zero") as zero, mock.patch.object(demo.time, "sleep"), \
                mock.patch("sys.stdout", new_callable=io.StringIO), \
                mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
            self.assertEqual(demo.main(), 2)
        self.assertIn("GTC demo stopped: candy return failed", stderr.getvalue())
        zero.assert_called_once()
        arm.disconnect.assert_called_once()


class PreflightTests(unittest.TestCase):
    def k3s_commands(self, *check_results, fix_result=None):
        """Patch subprocess.run with scripted k3s-node-ip-sync.sh --check results."""
        checks = iter(check_results)

        def run(command, **_):
            if command[-1] == "--check":
                code, status = next(checks)
                return subprocess.CompletedProcess(command, code, stdout=status + "\n", stderr="")
            return fix_result or subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        return mock.patch.object(demo.subprocess, "run", side_effect=run)

    def test_k3s_address_in_sync_or_unknown_never_restarts_k3s(self):
        for code, status, tag in (
            (0, "in sync: k3s uses 10.0.0.5, which this host has", "[OK]"),
            (2, "unknown: cannot read the k3s node address", "[WARN]"),
        ):
            with self.subTest(code=code), self.k3s_commands((code, status)) as run, \
                    mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
                demo.ensure_k3s_address()
            self.assertEqual(run.call_count, 1)
            self.assertIn(f"{tag} k3s address {status}", stdout.getvalue())

    def test_stale_k3s_address_restarts_k3s_or_stops_preflight(self):
        stale = (1, "OUT OF SYNC: k3s uses 10.0.0.5, but this host has 10.0.0.9")
        with self.k3s_commands(stale, (0, "in sync: k3s uses 10.0.0.9, which this host has")) as run, \
                mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
            demo.ensure_k3s_address()
        self.assertEqual(run.call_args_list[1].args[0], ["sudo", "-n", "systemctl", "start", "k3s-node-ip-sync.service"])
        self.assertIn("[OK] k3s address in sync: k3s uses 10.0.0.9", stdout.getvalue())
        not_installed = subprocess.CompletedProcess([], 5, stdout="", stderr="Unit k3s-node-ip-sync.service not found.")
        with self.k3s_commands(stale, stale, fix_result=not_installed), \
                mock.patch("sys.stdout", new_callable=io.StringIO) as stdout, self.assertRaises(demo.DemoError):
            demo.ensure_k3s_address()
        self.assertIn("restart failed: Unit k3s-node-ip-sync.service not found.", stdout.getvalue())

    def services(self, camera_health, hand_health=None):
        camera = mock.Mock(base_url="http://camera:8094")
        camera.health.return_value = camera_health
        hands = None
        if hand_health is not None:
            hands = mock.Mock()
            hands.health.return_value = hand_health
        with mock.patch("builtins.print"):
            demo.check_services(camera, hands, "D435")

    def test_services_need_the_calibrated_camera_and_a_hand_stream_from_it(self):
        good = {"camera_serial": "D435", "preview_only": False, "depth_calibrated": True}
        self.services(good, {"source_url": "http://camera:8094/stream.mjpg"})
        self.services(good)  # pick-only: no hand service
        for camera, hand in (
            ({**good, "camera_serial": "other"}, None),
            ({**good, "preview_only": True}, None),
            (good, {"source_url": "http://elsewhere/stream.mjpg"}),
        ):
            with self.subTest(camera=camera, hand=hand), self.assertRaises(demo.DemoError):
                self.services(camera, hand)

    def test_settings_require_ordered_gripper_widths_and_share_them_with_the_bowl_check(self):
        values = demo.validate_demo_settings(demo.DEMO_SETTINGS)
        self.assertEqual(values["bowl"]["grip_open_mm"], values["grip_open_mm"])
        for key, value in (("minimum_held_grip_mm", 0.5), ("minimum_held_grip_mm", 45.0), ("grip_open_mm", 90.0)):
            with self.subTest(key=key, value=value), self.assertRaisesRegex(demo.DemoError, "gripper settings"):
                demo.validate_demo_settings({**demo.DEMO_SETTINGS, key: value})

    def test_pause_before_grip_needs_an_interactive_terminal(self):
        args = MainTests().args(pause_before_grip=True)
        with mock.patch.object(demo, "parse_args", return_value=args), \
                mock.patch.object(demo, "load_hand_eye", return_value={"camera": {"serial": "D435"}}), \
                mock.patch.object(demo, "ensure_k3s_address"), mock.patch.object(demo, "check_services"), \
                mock.patch.object(demo, "load_setup", return_value=(settings(pause_before_grip=True), np.eye(4), [SCAN])), \
                mock.patch.object(demo, "connect_robot") as connect, mock.patch("sys.stdin.isatty", return_value=False), \
                mock.patch("sys.stdout", new_callable=io.StringIO), \
                mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
            self.assertEqual(demo.main(), 2)
        self.assertIn("interactive terminal", stderr.getvalue())
        connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
