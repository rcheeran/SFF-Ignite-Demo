"""Hardware-free tests for the GTC table-to-hand flow in demo.py."""

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
import table_vision as tv
from piper.kinematics import fk
from piper.transforms import rot_to_euler, tip_position
from test_bowl_vision import GOLD, GREEN, INTRINSICS, RED
from test_table_vision import WHITE, render_table

# The taught scan pose (prepare-piper/scan-poses/scan_poses.json is not in git).
SCAN = {
    "index": 1,
    "joints_deg": np.array([-5.73, 35.966, -53.772, 3.918, 67.018, -7.811]),
    "end_pose_mm_deg": np.array([82.029, -2.477, 374.27, -177.127, 45.876, -176.709]),
}
TABLE_Z = tv.TABLE_SETTINGS["table_z_mm"]
TOP_Z = TABLE_Z + tv.TABLE_SETTINGS["candy_top_above_table_mm"]
TIP_ABOVE_TOP = demo.DEMO_SETTINGS["grasp_tip_above_candy_top_mm"]
APPROACH = demo.DEMO_SETTINGS["pick_approach_height_mm"]
# Inside the scan-pose view and within straight-down reach.
NEAR = (300.0, 0.0)


def settings(**overrides):
    """Validated demo settings; the camera-to-robot correction is off unless overridden."""
    values = demo.validate_demo_settings(demo.DEMO_SETTINGS)
    values.update(
        camera_intrinsics=INTRINSICS, verbose=False, pause_before_grip=False,
        candy_xy_correction_mm=np.zeros(2), candy_xy_correction_matrix=np.eye(2),
    )
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
    """Camera service with a rendered photo and depth given by a height per candy-top-plane point."""

    def __init__(self, frame, c2b, height_at):
        self.frame, self.c2b, self.height_at = frame, c2b, height_at
        self.view = tv.table_view(INTRINSICS, c2b, settings()["table"])

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


CANDIES = ((300, 0, 0, RED, GOLD), (300, 36, 0, GREEN, WHITE), (420, -80, 30, GOLD, RED))


def table_depth(candies=CANDIES, top_z=TOP_Z):
    """Depth model on the candy-top plane: candy tops on candies, the table elsewhere."""
    def height_at(plane_xy):
        if any(np.max(np.abs(np.asarray(plane_xy) - candy[:2])) <= 12.7 for candy in candies):
            return top_z
        return TABLE_Z
    return height_at


class TableScanTests(unittest.TestCase):
    def scan(self, height_at, candies=CANDIES):
        frame, c2b = render_table(candies)
        arm = healthy_arm()
        arm.pose.return_value = pose_for(c2b)
        camera = FakeCamera(frame, c2b, height_at)
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(demo, "LOG_DIR", Path(folder)), \
                mock.patch.object(demo, "CANDY_SCAN_IMAGE", Path(folder) / "scan.jpg"), \
                mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
            result = demo.scan_table(arm, camera, np.eye(4), settings())
            saved = (Path(folder) / "scan.jpg").exists()
        return result, arm, camera, stdout.getvalue(), saved

    def test_candies_are_located_from_depth_and_ranked_nearest_the_robot_first(self):
        result, arm, camera, output, saved = self.scan(table_depth())
        self.assertEqual(result["status"], "detected")
        self.assertTrue(saved)
        self.assertIn("TABLE SCAN: detected, 3 candies, 3 pickable", output)
        distances = [np.linalg.norm(candy["base_point_mm"][:2]) for candy in result["pickable"]]
        self.assertEqual(distances, sorted(distances))
        for candy in result["pickable"]:
            expected = bv.point_at_height(camera.view, candy["plane_center_mm"], TOP_Z)
            np.testing.assert_allclose(candy["base_point_mm"], expected, atol=0.5)  # pixel rounded to 0.1 px
            self.assertAlmostEqual(candy["true_long_mm"], 25.4, delta=3.0)
        # The red candy's fingers cannot close along Y: one would land on the green candy beside it.
        red = min(result["pickable"], key=lambda c: np.linalg.norm(np.subtract(c["plane_center_mm"], (300, 0))))
        self.assertTrue(all(min(abs(yaw - 90.0), abs(yaw - 270.0)) > 15.0 for yaw in red["closing_yaw_deg"]))
        # The nearest candy has a reachable straight-down grasp just above its top surface.
        candy, plan = demo.pick_first_reachable(arm, result, SCAN, settings())
        np.testing.assert_allclose(
            tip_position(dict(plan)["grasp"]), np.add(candy["base_point_mm"], [0, 0, TIP_ABOVE_TOP]), atol=1.0,
        )
        self.assertAlmostEqual(demo.grasp_tilt_deg(dict(plan)["grasp"]), 0.0, delta=0.5)

    def test_prints_and_candies_without_depth_are_not_picked(self):
        printed, *_ = self.scan(table_depth(top_z=TABLE_Z))
        self.assertEqual(printed["status"], "no_pickable")
        self.assertTrue(all("above the table" in " ".join(item["reasons"]) for item in printed["unpickable"]))

        def no_candy_depth(plane_xy):
            return None if table_depth()(plane_xy) == TOP_Z else TABLE_Z

        blind, *_ = self.scan(no_candy_depth)
        self.assertEqual(blind["status"], "no_pickable")
        no_depth = [item for item in blind["rejected_candies"] if any("no depth" in reason for reason in item["reasons"])]
        self.assertEqual(len(no_depth), 3)


class TablePickPlanTests(unittest.TestCase):
    def candy(self, xy=NEAR, top_z=TOP_Z, closing=(0.0, 90.0)):
        return {"base_point_mm": [*xy, top_z], "closing_yaw_deg": list(closing), "edge_yaw_deg": 0.0}

    def plan(self, candy, values=None):
        return demo.plan_table_pick(candy, SCAN["joints_deg"], SCAN["joints_deg"], values or settings())

    def test_pick_approaches_above_the_candy_then_moves_straight_down_and_up(self):
        plan = self.plan(self.candy(closing=(90.0,)))
        names = [name for name, _ in plan]
        self.assertEqual(names[0], "above candy")
        self.assertEqual(names[-2:], ["lift above candy", "scan"])
        descents = [name for name in names if name.startswith("descend")]
        lifts = [name for name in names if name.startswith("lift ") and name != "lift above candy"]
        self.assertGreater(len(descents), 0)
        self.assertEqual(len(lifts), len(descents))
        self.assertEqual(names.index("grasp"), len(descents) + 1)
        waypoints = dict(plan)
        grasp_z = TOP_Z + TIP_ABOVE_TOP
        np.testing.assert_allclose(tip_position(waypoints["above candy"]), [*NEAR, grasp_z + APPROACH], atol=0.5)
        np.testing.assert_allclose(tip_position(waypoints["grasp"]), [*NEAR, grasp_z], atol=0.5)
        for name in descents:
            np.testing.assert_allclose(tip_position(waypoints[name])[:2], NEAR, atol=0.5)
        # Lifts retrace the descent in reverse.
        for lift, descent in zip(lifts, reversed(descents)):
            np.testing.assert_array_equal(waypoints[lift], waypoints[descent])
        # The jaws close along the requested direction (90 degrees = base +Y).
        jaw_axis = fk(waypoints["grasp"])[:3, 1]
        self.assertAlmostEqual(abs(jaw_axis[1]), 1.0, delta=0.01)
        np.testing.assert_array_equal(waypoints["scan"], SCAN["joints_deg"])

    def test_grasp_height_never_goes_below_the_table_limit(self):
        plan = self.plan(self.candy(top_z=5.0))
        self.assertAlmostEqual(tip_position(dict(plan)["grasp"])[2], settings()["minimum_grasp_tip_z_mm"], delta=0.5)

    def test_far_candy_uses_the_smallest_tilt_and_moves_along_the_tool_axis(self):
        far = (430.0, 0.0)
        with self.assertRaisesRegex(demo.DemoError, "no reachable grasp at any allowed tilt"):
            self.plan(self.candy(xy=far), values=settings(grasp_tilt_options_deg=[0.0]))
        plan = dict(self.plan(self.candy(xy=far)))
        self.assertAlmostEqual(demo.grasp_tilt_deg(plan["grasp"]), 10.0, delta=0.5)
        grasp_tip = np.array([*far, TOP_Z + TIP_ABOVE_TOP])
        np.testing.assert_allclose(tip_position(plan["grasp"]), grasp_tip, atol=0.5)
        frame = fk(plan["grasp"])
        up = -frame[:3, 2]
        self.assertLess(abs(frame[2, 1]), 0.01)  # the jaws still close level
        for name, joints in plan.items():
            if name.startswith("descend") or name == "above candy":
                offset = np.cross(tip_position(joints) - grasp_tip, up)
                self.assertLess(np.linalg.norm(offset), 0.5, name)
        self.assertAlmostEqual(tip_position(plan["above candy"])[2], grasp_tip[2] + APPROACH, delta=0.5)

    def test_out_of_reach_candy_has_no_plan(self):
        with self.assertRaisesRegex(demo.DemoError, "no reachable grasp at any allowed tilt"):
            self.plan(self.candy(xy=(640.0, 0.0)))

    def test_descent_that_strays_or_a_move_that_dips_toward_the_table_is_rejected(self):
        plan = dict(self.plan(self.candy()))
        stray = dict(self.plan(self.candy(xy=(NEAR[0] + 12.0, NEAR[1]))))
        mixed = [("above candy", plan["above candy"]), ("grasp", stray["grasp"])]
        grasp_tip = [*NEAR, TOP_Z + TIP_ABOVE_TOP]
        with self.assertRaisesRegex(demo.DemoError, "strays"):
            demo.validate_descent_line(mixed, grasp_tip, [0.0, 0.0, 1.0], grasp_tip[2] + APPROACH + 1.0, 3.0)
        with self.assertRaisesRegex(demo.DemoError, "dips to tip Z"):
            demo.validate_transit_height("move above the candy", SCAN["joints_deg"], plan["grasp"], grasp_tip[2] + APPROACH - 1.0)


class PickupExecutionTests(unittest.TestCase):
    def setUp(self):
        self.settings = settings()
        self.candy = {"base_point_mm": [*NEAR, TOP_Z], "closing_yaw_deg": [0.0], "edge_yaw_deg": 90.0}
        self.plan = demo.plan_table_pick(self.candy, SCAN["joints_deg"], SCAN["joints_deg"], self.settings)
        self.names = [name for name, _ in self.plan]

    def execute(self, grip_mm):
        arm = healthy_arm(grip_mm=grip_mm)
        carry = demo.CandyCarry()
        with follow_moves(arm) as move, mock.patch.object(demo, "hold_gripper") as hold:
            try:
                width = demo.execute_table_pickup(arm, self.plan, self.settings, carry, self.candy)
            except demo.GraspVerificationError as exc:
                width = exc
        return width, carry, move, hold

    def test_held_candy_is_lifted_back_above_its_spot_and_carried_to_scan(self):
        width, carry, move, hold = self.execute(22.0)
        self.assertEqual(width, 22.0)
        moved = [call.args[1] for call in move.call_args_list]
        self.assertEqual(moved, self.names)
        grips = [call.args[3] for call in move.call_args_list]
        grasp = self.names.index("grasp")
        self.assertEqual(self.settings["pick_open_mm"], 60.0)
        self.assertTrue(all(grip == 60.0 for grip in grips[:grasp + 1]))
        self.assertTrue(all(grip == 1.0 for grip in grips[grasp + 1:]))
        hold.assert_called_once()
        np.testing.assert_array_equal(hold.call_args.args[1], dict(self.plan)["grasp"])
        self.assertEqual(hold.call_args.args[2], 1.0)
        self.assertTrue(carry.holding)
        self.assertEqual([name for name, _ in carry.path], ["grasp", *self.names[grasp + 1:]])

    def test_missed_grasp_opens_above_the_spot_and_returns_to_scan(self):
        error, carry, move, hold = self.execute(1.0)
        self.assertIsInstance(error, demo.GraspVerificationError)
        self.assertFalse(carry.holding)
        opened = hold.call_args_list[-1]
        np.testing.assert_array_equal(opened.args[1], dict(self.plan)["lift above candy"])
        self.assertEqual(opened.args[2], 40.0)
        self.assertEqual(move.call_args.args[1], "return to scan pose after missed grasp")
        self.assertEqual(move.call_args.args[3], 40.0)

    def test_fingers_land_where_the_wider_pick_opening_puts_them(self):
        landed = []

        def to_pixel(_transform, _intrinsics, point):
            landed.append(np.asarray(point, dtype=float))
            return np.array([320.0, 240.0])

        camera = mock.Mock()
        with mock.patch.object(demo, "base_to_pixel", side_effect=to_pixel), \
                mock.patch.object(demo, "to_base_point", return_value=np.array([0.0, 0.0, TABLE_Z])):
            self.assertIsNone(demo.finger_landing_problem(self.candy, 0.0, camera, "token", np.eye(4), self.settings))
        reach = self.settings["pick_open_mm"] / 2.0 + self.settings["finger_thickness_mm"] / 2.0
        np.testing.assert_allclose([point[0] - NEAR[0] for point in landed], [-reach, reach])
        np.testing.assert_allclose([point[1] for point in landed], [0.0, 0.0])

    def test_return_after_leaving_retraces_and_drops_the_candy_above_its_spot(self):
        waypoints = dict(self.plan)
        carry = demo.CandyCarry()
        carry.gripped_at(waypoints["grasp"])
        for name in self.names[self.names.index("grasp") + 1:]:
            carry.reached(name, waypoints[name])
        arm = healthy_arm(joints=waypoints["scan"])
        with follow_moves(arm) as move, mock.patch.object(demo, "hold_gripper") as hold, \
                mock.patch("builtins.print"):
            demo.return_candy_to_table(arm, self.plan, carry, self.settings)
        self.assertEqual(
            [call.args[1] for call in move.call_args_list],
            ["return candy: above its spot", "return candy: back to scan"],
        )
        self.assertEqual(move.call_args_list[0].args[3], 1.0)  # still holding on the way
        np.testing.assert_array_equal(hold.call_args.args[1], waypoints["lift above candy"])
        self.assertEqual(hold.call_args.args[2], 40.0)
        self.assertFalse(carry.holding)

    def test_return_during_the_lift_finishes_the_lift_first(self):
        waypoints = dict(self.plan)
        carry = demo.CandyCarry()
        carry.gripped_at(waypoints["grasp"])
        carry.reached("lift 1", waypoints["lift 1"])
        arm = healthy_arm(joints=waypoints["lift 1"])
        with follow_moves(arm) as move, mock.patch.object(demo, "hold_gripper"), mock.patch("builtins.print"):
            demo.return_candy_to_table(arm, self.plan, carry, self.settings)
        remaining = [name for name in self.names if name.startswith("lift ") and name != "lift above candy"][1:]
        self.assertEqual(
            [call.args[1] for call in move.call_args_list],
            [f"return candy: {name}" for name in remaining] + ["return candy: above its spot", "return candy: back to scan"],
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
    def run_cycle(self, scan_result, pick_only=False, execute=None, deliver=None, hand=None):
        arm, camera, hands = healthy_arm(), mock.Mock(), mock.Mock()
        candy = {"base_point_mm": [*NEAR, TOP_Z]}
        with mock.patch.object(demo, "go_to_scan"), \
                mock.patch.object(demo, "scan_table", return_value=scan_result), \
                mock.patch.object(demo, "pick_first_reachable", return_value=(candy, [("grasp", np.zeros(6))])) as plan, \
                mock.patch.object(demo, "hand_in_view", return_value=hand) as interlock, \
                mock.patch.object(demo, "execute_table_pickup", side_effect=execute or (lambda *a: 22.0)) as pick, \
                mock.patch.object(demo, "deliver_to_hand", side_effect=deliver) as delivery, \
                mock.patch.object(demo, "return_candy_to_table") as back, \
                mock.patch("builtins.print"):
            result = demo.run_cycle(arm, camera, hands, SCAN, np.eye(4), settings(), pick_only)
        self.interlock = interlock
        return result, plan, pick, delivery, back

    def test_no_pickable_candy_means_no_pick(self):
        result, plan, pick, *_ = self.run_cycle({"status": "no_pickable"})
        self.assertFalse(result)
        plan.assert_not_called()
        pick.assert_not_called()

    def test_no_pick_starts_while_a_hand_is_in_view(self):
        result, plan, pick, delivery, _ = self.run_cycle({"status": "detected"}, hand="a hand is in view")
        self.assertFalse(result)
        plan.assert_called_once()
        self.interlock.assert_called_once()
        pick.assert_not_called()
        delivery.assert_not_called()

    def test_missed_grasp_rescans_without_waiting_for_a_hand(self):
        def miss(*args):
            raise demo.GraspVerificationError(1.0, 12.0)
        result, _, _, delivery, back = self.run_cycle({"status": "detected"}, execute=miss)
        self.assertFalse(result)
        delivery.assert_not_called()
        back.assert_not_called()

    def test_error_while_holding_puts_the_candy_back(self):
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
        self.interlock.assert_called_once()
        delivery.assert_not_called()
        back.assert_called_once()


class HandInViewTests(unittest.TestCase):
    def check(self, capture=None, error=None):
        hands = mock.Mock()
        hands.capture_hand.side_effect = error or (lambda slot: capture)
        return demo.hand_in_view(hands)

    def test_only_a_fresh_capture_with_no_hand_in_any_sample_allows_a_pick(self):
        now = datetime.now(timezone.utc).isoformat()
        clear = {"status": "not_found", "captured_at": now, "visible_hand_sample_count": 0, "sample_count": 5}
        self.assertIsNone(self.check(clear))
        for name, capture in (
            ("closed hand", {**clear, "visible_hand_sample_count": 2}),
            ("open palm", {**clear, "status": "detected", "visible_hand_sample_count": 5}),
            ("unstable palm", {**clear, "status": "rejected", "visible_hand_sample_count": 3}),
            ("older hand service", {"status": "not_found", "captured_at": now}),
            ("stale", {**clear, "captured_at": (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()}),
        ):
            with self.subTest(name):
                self.assertIsNotNone(self.check(capture))
        self.assertIn("cannot confirm", self.check(error=demo.DemoError("cannot reach service")))


class PositionCorrectionTests(unittest.TestCase):
    def test_correction_moves_candy_and_hand_targets_alike(self):
        matrix = np.array([[0.9, 0.0], [0.0, 1.0]])
        values = settings(candy_xy_correction_mm=np.array([10.0, -2.0]), candy_xy_correction_matrix=matrix)
        np.testing.assert_allclose(demo.correct_xy([300.0, 50.0, 16.0], values), [280.0, 48.0, 16.0])
        arm = healthy_arm()
        arm.pose.return_value = np.zeros(6)
        hands = mock.Mock()
        hands.capture_hand.return_value = hand_capture(palm=(400.0, 0.0, 40.0), finger=(440.0, 0.0, 40.0))
        _, decision = demo.capture_hand_target(arm, hands, np.eye(4), values)
        np.testing.assert_allclose(decision["base_point_mm"], [370.0, -2.0, 40.0])

    def test_default_correction_is_valid_and_a_matrix_far_from_identity_is_rejected(self):
        values = demo.validate_demo_settings(demo.DEMO_SETTINGS)
        self.assertEqual(values["candy_xy_correction_matrix"].shape, (2, 2))
        with self.assertRaisesRegex(demo.DemoError, "within 0.3 of the identity"):
            demo.validate_demo_settings({**demo.DEMO_SETTINGS, "candy_xy_correction_matrix": [[1.5, 0.0], [0.0, 1.0]]})


class MeasureOffsetTests(unittest.TestCase):
    def test_hover_keeps_the_fingertips_above_the_candy_and_never_grips(self):
        values = demo.measure_offset_settings(settings())
        candy = {"base_point_mm": [*NEAR, TOP_Z], "closing_yaw_deg": [0.0], "edge_yaw_deg": 90.0}
        plan = demo.plan_table_pick(candy, SCAN["joints_deg"], SCAN["joints_deg"], values)
        fingertips = tip_position(dict(plan)["grasp"])[2] - demo.FINGER_EXTENSION_BELOW_TIP_MM
        self.assertAlmostEqual(fingertips, TOP_Z + demo.MEASURE_HOVER_MM, delta=0.5)
        arm = healthy_arm()
        with follow_moves(arm) as move, mock.patch.object(demo, "hold_gripper") as hold, \
                mock.patch.object(demo, "pause_for_grasp_offset", return_value=np.array([-20.0, 0.0])) as pause:
            offset = demo.hover_and_measure(arm, plan, values, candy)
        np.testing.assert_array_equal(offset, [-20.0, 0.0])
        hold.assert_not_called()
        self.assertFalse(pause.call_args.kwargs["closing"])
        self.assertTrue(all(call.args[3] == values["grip_open_mm"] for call in move.call_args_list))
        self.assertEqual(move.call_args_list[-1].args[1], "scan")

    def test_each_candy_from_one_scan_is_measured_once_no_hand_is_in_view(self):
        candies = [{"base_point_mm": [x, 0.0, TOP_Z], "closing_yaw_deg": [0.0], "edge_yaw_deg": 90.0} for x in (280.0, 320.0)]
        arm = healthy_arm()
        with mock.patch.object(demo, "go_to_scan"), \
                mock.patch.object(demo, "scan_table", return_value={"pickable": candies}), \
                mock.patch.object(demo, "hand_in_view", side_effect=["a hand is in view", None, None]) as interlock, \
                mock.patch.object(demo, "hover_and_measure", return_value=np.zeros(2)) as hover, \
                mock.patch.object(demo.time, "sleep"), mock.patch("builtins.print"):
            recorded = demo.run_measure_offsets(arm, mock.Mock(), mock.Mock(), SCAN, np.eye(4), settings())
        self.assertEqual(recorded, 2)
        self.assertEqual(interlock.call_count, 3)
        self.assertEqual([call.args[3]["base_point_mm"][0] for call in hover.call_args_list], [280.0, 320.0])


class MainTests(unittest.TestCase):
    def args(self, **overrides):
        values = dict(
            calibration_dir=Path("calibration"), scan_poses=Path("scan_poses.json"), vision_url="http://camera:8094",
            hand_url="http://hand:8095", can_interface="can_piper", check_only=False, scan_only=False,
            pick_only=False, measure_offsets=False, verbose=False, pause_before_grip=False, allow_motion=True, max_cycles=2,
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

    def test_pick_only_still_checks_the_hand_service_and_scan_only_never_picks(self):
        _, _, services, _, _, cycle, _, _, _ = self.run_main(self.args(pick_only=True, max_cycles=1))
        self.assertIsNotNone(services.call_args.args[1])
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
        self.services(good)  # camera only
        for camera, hand in (
            ({**good, "camera_serial": "other"}, None),
            ({**good, "preview_only": True}, None),
            (good, {"source_url": "http://elsewhere/stream.mjpg"}),
        ):
            with self.subTest(camera=camera, hand=hand), self.assertRaises(demo.DemoError):
                self.services(camera, hand)

    def test_settings_require_ordered_gripper_widths_and_look_for_candies_in_the_workspace(self):
        values = demo.validate_demo_settings(demo.DEMO_SETTINGS)
        self.assertEqual(values["table"]["view_x_mm"], [float(demo.WORKSPACE_MIN_MM[0]), float(demo.WORKSPACE_MAX_MM[0])])
        self.assertEqual(values["table"]["view_y_mm"], [float(demo.WORKSPACE_MIN_MM[1]), float(demo.WORKSPACE_MAX_MM[1])])
        for key, value in (
            ("minimum_held_grip_mm", 0.5), ("minimum_held_grip_mm", 45.0), ("grip_open_mm", 90.0),
            ("pick_open_mm", 30.0), ("pick_open_mm", 75.0),
        ):
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
