"""Hardware-free tests for the Ignite loop; no CAN or live service requests."""

import argparse
from datetime import datetime, timedelta, timezone
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

import demo


def palm(point=(250.0, 0.0, 15.0), **fields):
    return {
        "status": "detected", "base_point_mm": list(point), "handedness": "Right",
        "gesture": "open_palm", "stable_sample_count": 3,
        "maximum_temporal_spread_mm": 1.0, "maximum_allowed_spread_mm": 10.0,
        "captured_at": datetime.now(timezone.utc).isoformat(), **fields,
    }


class IgniteTests(unittest.TestCase):
    @staticmethod
    def example_config():
        return {"demo": {
            "motion_speed_percent": 20, "pick_motion_speed_percent": 15,
            "detection_confidence_threshold": 0.25,
            "candy_min_z_mm": -30.0, "candy_max_z_mm": 30.0, "candy_xy_correction_mm": [0.0, 0.0],
            "maximum_scan_position_deviation_mm": 2.0,
            "maximum_scan_rotation_deviation_deg": 0.75,
            "maximum_cross_view_target_difference_mm": 25.0,
            "maximum_cross_view_orientation_difference_deg": 20.0,
            "scan_settle_seconds": 1.0,
            "target_workspace_min_mm": [100, -400, -50],
            "target_workspace_max_mm": [500, 400, 100],
            "hand_workspace_min_mm": [100, -400, -30],
            "hand_workspace_max_mm": [500, 400, 100],
            "hand_approach_height_mm": 20.0,
            "hand_delivery_rpy_deg": [180, 20, 180],
            "grasp_rpy_deg": [180, 20, 180], "grasp_tip_z_mm": 20.0,
            "forward_lean_vertical_threshold": 0.7,
            "approach_height_mm": 100.0, "pregrasp_height_mm": 50.0,
            "minimum_pregrasp_height_mm": 30.0, "pregrasp_height_step_mm": 5.0,
            "maximum_approach_inward_offset_mm": 120.0, "approach_inward_step_mm": 20.0,
            "maximum_grasp_yaw_offset_deg": 20.0, "grasp_yaw_offset_step_deg": 5.0,
            "grip_open_mm": 70.0, "grip_close_mm": 0.0, "minimum_held_grip_mm": 8.0,
            "move_timeout_seconds": 8.0,
            "gripper_hold_seconds": 1.0, "retry_delay_seconds": 1.0,
        }}

    def test_runs_without_either_older_demo_folder(self):
        source = Path(demo.__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(
                source, root / "Demo-Ignite",
                ignore=shutil.ignore_patterns("__pycache__", "captures"),
            )
            detection = root / "k3s-vision-detection"
            detection.mkdir()
            shutil.copy2(
                demo.ROOT / "k3s-vision-detection/detection_client.py",
                detection / "detection_client.py",
            )
            script = """
import json
from pathlib import Path
import sys
from unittest import mock
import demo
from test_demo import IgniteTests

root = Path.cwd()
accepted = {"acceptance": {"quality_passed": True}}
calibration = root / "calibration"
calibration.mkdir()
for name in ("hand_eye", "validation", "depth_correction", "depth_verification"):
    document = {**accepted, "camera": {"serial": "test-camera"}}
    if name == "hand_eye":
        document["camera_to_gripper"] = demo.np.eye(4).tolist()
    (calibration / (name + ".json")).write_text(json.dumps(document))
config = root / "config.json"
config.write_text(json.dumps(IgniteTests.example_config()))
scans = root / "scan_poses.json"
scans.write_text(json.dumps({"pose_count": 1, "poses": [{
    "index": 1, "joints_deg": [0, 45, -25, 0, 40, 0],
    "end_pose_mm_deg": [200, 0, 200, 180, 20, 180],
}]}))
sys.argv = ["demo.py", "--check-only",
            "--calibration-dir", str(calibration), "--scan-poses", str(scans)]
with mock.patch.object(demo, "check_services") as services, mock.patch.object(
    demo, "connect_robot", side_effect=AssertionError("CAN connection forbidden")
):
    assert demo.main() == 0
    assert services.call_args.args[-1] == "test-camera"
for module in tuple(sys.modules.values()):
    filename = getattr(module, "__file__", None)
    if filename:
        assert not any(part in Path(filename).parts for part in (
            "Demo-Product-Day", "Demo-VLA-On-Piper",
        )), filename
assert demo.select_reachable_object_target.__module__ == "demo"
assert demo.PIPER_C.__file__.startswith(str(root / "Demo-Ignite"))
print("Standalone preflight passed without older demo folders")
"""
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=root, env={**os.environ, "PYTHONPATH": str(root / "Demo-Ignite")},
                text=True, capture_output=True, timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Standalone preflight passed", result.stdout)

    def test_robot_connection_is_loaded_from_ignite_folder(self):
        robot_module = sys.modules[demo.connect_robot.__module__]
        self.assertEqual(
            Path(robot_module.__file__).resolve(),
            Path(demo.__file__).resolve().parent / "piper/robot.py",
        )
        self.assertEqual(
            Path(robot_module.C.__file__).resolve(),
            Path(demo.__file__).resolve().parent / "piper/config.py",
        )
        transforms_module = sys.modules[robot_module.tip_position.__module__]
        self.assertEqual(
            Path(transforms_module.__file__).resolve(),
            Path(demo.__file__).resolve().parent / "piper/transforms.py",
        )
        self.assertIs(transforms_module.C, robot_module.C)
        np.testing.assert_allclose(
            robot_module.tip_position(np.zeros(6)),
            demo.tip_position(np.zeros(6)),
        )

    def test_vision_client_is_loaded_from_ignite_folder(self):
        client_module = sys.modules[demo.VisionClient.__module__]
        self.assertEqual(
            Path(client_module.__file__).resolve(),
            Path(demo.__file__).resolve().with_name("vision_client.py"),
        )
        self.assertEqual(demo.capture_scan_pair.__module__, demo.__name__)

    def setUp(self):
        startup = mock.patch.object(demo, "prepare_robot_for_scan")
        self.startup = startup.start()
        self.addCleanup(startup.stop)
        zero = mock.patch.object(demo, "return_to_zero")
        self.zero = zero.start()
        self.addCleanup(zero.stop)
        self.arm = mock.Mock()
        self.arm.joints_deg.return_value = np.zeros(6)
        self.arm.faults.return_value = []
        self.arm.enabled.return_value = [True] * 6
        self.arm.j6_temp.return_value = 30
        self.scan = {"index": 1, "joints_deg": np.ones(6)}
        self.settings = {
            "grip_open_mm": 40.0, "grip_close_mm": 0.0, "minimum_held_grip_mm": 8.0,
            "move_timeout_seconds": 10.0, "motion_speed_percent": 10,
            "gripper_hold_seconds": 0.2, "scan_settle_seconds": 0.1,
            "retry_delay_seconds": 0.1, "hand_approach_height_mm": 40.0,
            "hand_delivery_rpy_deg": np.array([180.0, 0.0, 0.0]),
            "hand_workspace_min_mm": np.array([100.0, -300.0, -50.0]),
            "hand_workspace_max_mm": np.array([450.0, 300.0, 100.0]),
            "detection_confidence_threshold": 0.25,
        }
        self.objects, self.hands = mock.Mock(), mock.Mock()
        self.objects.detector.check.return_value = None
        self.hands.health.return_value = {"source_url": "http://camera:8094/stream.mjpg"}

    def test_release_point_is_three_inches_above_palm_not_absolute_z(self):
        for z in (-10.0, 0.0, 30.0):
            with self.subTest(z=z), mock.patch.object(
                demo, "plan_hand_delivery", return_value=[("hand release", np.zeros(6))]
            ) as planner, mock.patch.object(demo, "validate_joint_path_safety"):
                demo.plan_delivery(palm((250, 20, z)), self.arm, self.scan, self.settings, 10)
            self.assertAlmostEqual(planner.call_args.args[4], z + 76.2)
            np.testing.assert_array_equal(planner.call_args.args[0], [250, 20, z])

    def test_invalid_missing_stale_and_future_palms_fail(self):
        for value in (
            palm(status="not_found"), palm(status="rejected"), palm(captured_at="bad"),
            palm(captured_at=datetime.now().isoformat()),
            palm(captured_at=(datetime.now(timezone.utc) - timedelta(seconds=20)).isoformat()),
            palm(captured_at=(datetime.now(timezone.utc) + timedelta(seconds=20)).isoformat()),
            palm((float("nan"), 0, 0)),
            palm(gesture="closed_fist"),
        ):
            with self.subTest(value=value), self.assertRaises(demo.DemoError):
                demo.verify_palm(value, 10)

    def test_palm_verification_ignores_handedness_and_stability_metadata(self):
        for value in (
            palm(handedness="unknown", stable_sample_count=1, maximum_temporal_spread_mm=100),
            {
                "status": "detected", "gesture": "open_palm",
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "base_point_mm": [250, 0, 15],
            },
        ):
            with self.subTest(value=value):
                np.testing.assert_array_equal(demo.verify_palm(value, 10), [250, 0, 15])

    def test_delivery_validates_workspace_before_ik_or_movement(self):
        for point in ((900, 0, 0), (250, 0, -51), (250, 0, 101)):
            with self.subTest(point=point), mock.patch.object(
                demo, "plan_hand_delivery"
            ) as planner, self.assertRaises(demo.DemoError):
                demo.plan_delivery(palm(point), self.arm, self.scan, self.settings, 10)
            planner.assert_not_called()

    def test_each_move_validates_path_before_command(self):
        with mock.patch.object(
            demo, "validate_joint_path_safety", side_effect=demo.DemoError("unsafe path")
        ), mock.patch.object(demo, "move_robot") as move, \
                self.assertRaisesRegex(demo.DemoError, "unsafe path"):
            demo.checked_move(self.arm, "approach", np.ones(6), 0.0, self.settings)
        move.assert_not_called()

    def run_delivery(self, captured_palms):
        plan = [
            ("hand approach", np.ones(6) * 2),
            ("hand release", np.ones(6) * 3),
            ("hand retreat", np.ones(6) * 2),
            ("scan 1", np.ones(6)),
        ]
        with mock.patch.object(demo, "capture_hand_at_current_pose", side_effect=captured_palms) as capture, \
                mock.patch.object(demo, "checked_move") as moves, \
                mock.patch.object(demo, "plan_hand_delivery", return_value=plan), \
                mock.patch.object(demo, "validate_joint_path_safety"), \
                mock.patch.object(demo, "plan_delivery", wraps=demo.plan_delivery) as planner, \
                mock.patch.object(demo, "hold_gripper") as release, \
                mock.patch.object(demo.time, "sleep"):
            demo.deliver_to_palm(
                self.arm, self.objects, self.hands, self.scan, np.eye(4),
                self.settings, 10,
            )
        capture.assert_called_once()
        return moves, planner, release

    def test_delivery_uses_updated_coordinates_and_returns_to_scan(self):
        updated = palm((270, 10, 20), handedness="Left", stable_sample_count=1,
                       maximum_temporal_spread_mm=100)
        moves, planner, release = self.run_delivery([updated])
        self.assertEqual(planner.call_count, 1)
        self.assertEqual(planner.call_args.args[0], updated)
        names = [call.args[1] for call in moves.call_args_list]
        self.assertEqual(names, [
            "return to scan for palm reacquisition", "hand approach",
            "hand release height (palm Z + 76.2mm)", "hand retreat", "return to scan",
        ])
        release.assert_called_once()
        self.assertEqual(release.call_args.args[2], 40.0)

    def test_delivery_validates_each_plan_and_before_approach_and_release(self):
        updated = palm()
        with mock.patch.object(demo, "verify_palm", wraps=demo.verify_palm) as validate:
            self.run_delivery([updated])
        self.assertEqual(validate.call_args_list, [
            mock.call(updated, 10),
            mock.call(updated, 10),
            mock.call(updated, 10),
            mock.call(updated, 10),
        ])

    def delivery_plan(self):
        return [
            ("hand approach", np.full(6, 7.0)), ("hand release", np.full(6, 8.0)),
            ("hand retreat", np.full(6, 7.0)), ("scan 1", np.ones(6)),
        ]

    def test_missing_or_unreachable_palm_waits_at_scan_then_delivers(self):
        ik_failure = demo.DemoError("no joint-limit-safe IK solution for hand approach at [331.3, -16.8, 323.7]mm")
        good = palm((260, 10, 20))
        cases = {
            "hand service error": {"captures": [demo.DemoError("hand service unavailable"), good]},
            "palm not found": {"captures": [palm(status="not_found"), good]},
            "not an open palm": {"captures": [palm(gesture="closed_fist"), good]},
            "nonfinite palm": {"captures": [palm((float("nan"), 0, 15)), good]},
            "outside workspace": {"captures": [palm((900, 0, 15)), good]},
            "stale palm": {"captures": [
                palm(captured_at=(datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()), good,
            ]},
            "hand approach IK": {"captures": [good, good], "planner": [ik_failure, self.delivery_plan()]},
            "path rejected": {"captures": [good, good], "paths": [demo.DemoError("unsafe path"), None]},
            "several failures": {"captures": [
                palm(status="not_found"), palm((900, 0, 15)), good, good,
            ], "planner": [ik_failure, self.delivery_plan()]},
        }
        for name, case in cases.items():
            carry = self.carry_after_pickup()
            planner = case.get("planner", [self.delivery_plan()] * 4)
            paths = case.get("paths", [None]) + [None] * 4
            with self.subTest(name=name), mock.patch.object(
                demo, "capture_hand_at_current_pose", side_effect=case["captures"]
            ) as capture, mock.patch.object(demo, "plan_hand_delivery", side_effect=planner), \
                    mock.patch.object(demo, "validate_joint_path_safety", side_effect=paths), \
                    mock.patch.object(demo, "checked_move") as moves, \
                    mock.patch.object(demo, "hold_gripper") as release, \
                    mock.patch.object(demo.time, "sleep") as sleep, \
                    mock.patch("builtins.print") as output:
                demo.deliver_to_palm(
                    self.arm, self.objects, self.hands, self.scan, np.eye(4), self.settings, 10, carry,
                )
            attempts = len(case["captures"])
            self.assertEqual(capture.call_count, attempts)
            self.assertEqual(sleep.call_count, attempts)  # one settle, then one retry delay per failure
            self.assertEqual([call.args[1] for call in moves.call_args_list], [
                "return to scan for palm reacquisition", "hand approach",
                "hand release height (palm Z + 76.2mm)", "hand retreat", "return to scan",
            ])
            release.assert_called_once()
            self.assertFalse(carry.holding)
            waits = [call for call in output.call_args_list if "Waiting at scan pose with candy" in str(call)]
            self.assertEqual(len(waits), attempts - 1)
            self.assertEqual(self.arm.faults.call_count >= attempts, True)
            self.arm.reset_mock()

    def test_waiting_for_palm_stops_on_unhealthy_arm_or_interrupt_without_motion(self):
        for name, health, sleep in (
            ("arm fault", [[], ["EMERGENCY_STOP"]], None),
            ("ctrl+c while waiting", [[]] * 3, KeyboardInterrupt()),
        ):
            carry = self.carry_after_pickup()
            self.arm.faults.side_effect = health
            expected = KeyboardInterrupt if isinstance(sleep, KeyboardInterrupt) else demo.DemoError

            def pause(seconds, interrupt=sleep):
                if interrupt is not None and seconds == self.settings["retry_delay_seconds"] and pause.settled:
                    raise interrupt
                pause.settled = True

            pause.settled = False
            with self.subTest(name=name), mock.patch.object(
                demo, "capture_hand_at_current_pose", return_value=palm(status="not_found")
            ), mock.patch.object(demo, "checked_move") as moves, \
                    mock.patch.object(demo, "hold_gripper") as release, \
                    mock.patch.object(demo.time, "sleep", side_effect=pause), \
                    mock.patch("builtins.print"), self.assertRaises(expected):
                demo.deliver_to_palm(
                    self.arm, self.objects, self.hands, self.scan, np.eye(4), self.settings, 10, carry,
                )
            self.assertEqual(
                [call.args[1] for call in moves.call_args_list], ["return to scan for palm reacquisition"],
            )
            release.assert_not_called()
            self.assertTrue(carry.holding)
            self.assertEqual([waypoint for waypoint, _ in carry.path], ["grasp", "lift", "scan pose"])
            self.arm.faults.side_effect = None

    def test_stale_hand_after_approach_or_descent_cannot_release(self):
        for stale_at in (
            "hand approach", "hand release height (palm Z + 76.2mm)",
        ):
            detected = palm()
            old = (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()

            def move(_arm, name, *_args):
                if name == stale_at:
                    detected["captured_at"] = old

            with self.subTest(stale_at=stale_at), mock.patch.object(
                demo, "capture_hand_at_current_pose", return_value=detected
            ) as capture, mock.patch.object(demo, "checked_move", side_effect=move), \
                    mock.patch.object(demo, "plan_delivery", return_value=[
                        ("hand approach", np.zeros(6)), ("hand release", np.zeros(6)),
                        ("hand retreat", np.zeros(6)),
                    ]), mock.patch.object(demo, "hold_gripper") as release, \
                    mock.patch.object(demo.time, "sleep"), self.assertRaisesRegex(demo.DemoError, "stale"):
                demo.deliver_to_palm(
                    self.arm, self.objects, self.hands, self.scan, np.eye(4),
                    self.settings, 10,
                )
            release.assert_not_called()
            capture.assert_called_once()

    def carry_after_pickup(self):
        carry = demo.CandyCarry()
        carry.gripped_at(np.full(6, 3.0))
        carry.reached("lift", np.full(6, 4.0))
        return carry

    def run_tracked_delivery(self, carry, failing_step=None, capture=None, planner=None):
        plan = [
            ("hand approach", np.full(6, 7.0)), ("hand release", np.full(6, 8.0)),
            ("hand retreat", np.full(6, 7.0)), ("scan 1", np.ones(6)),
        ]

        def move(_arm, name, *_args):
            if name == failing_step:
                raise demo.DemoError(f"{name} did not reach its planned joint target")

        def release(*_args):
            if failing_step == "open":
                raise demo.DemoError("gripper hold fault")

        capture_kwargs = (
            {"side_effect": capture} if isinstance(capture, Exception) else {"return_value": capture or palm()}
        )
        with mock.patch.object(demo, "capture_hand_at_current_pose", **capture_kwargs), \
                mock.patch.object(demo, "plan_hand_delivery", side_effect=planner, return_value=plan), \
                mock.patch.object(demo, "validate_joint_path_safety"), \
                mock.patch.object(demo, "checked_move", side_effect=move), \
                mock.patch.object(demo, "hold_gripper", side_effect=release), \
                mock.patch.object(demo.time, "sleep"):
            demo.deliver_to_palm(
                self.arm, self.objects, self.hands, self.scan, np.eye(4), self.settings, 10, carry,
            )

    def test_carry_tracks_reached_waypoints_until_gripper_opens_at_palm(self):
        cases = (
            ("return to scan fails", {"failing_step": "return to scan for palm reacquisition"},
             True, ["grasp", "lift"]),
            ("hand approach move fails", {"failing_step": "hand approach"}, True, ["grasp", "lift", "scan pose"]),
            ("release move fails", {"failing_step": "hand release height (palm Z + 76.2mm)"}, True,
             ["grasp", "lift", "scan pose", "hand approach"]),
            ("gripper opening fails", {"failing_step": "open"}, False, None),
            ("retreat after release fails", {"failing_step": "hand retreat"}, False, None),
            ("success", {}, False, None),
        )
        for name, options, holding, path in cases:
            carry = self.carry_after_pickup()
            with self.subTest(name=name):
                if name == "success":
                    self.run_tracked_delivery(carry, **options)
                else:
                    with self.assertRaises(demo.DemoError):
                        self.run_tracked_delivery(carry, **options)
                self.assertEqual(carry.holding, holding)
                if path is not None:
                    self.assertEqual([waypoint for waypoint, _ in carry.path], path)

    def test_pickup_marks_candy_held_only_after_gripper_closes(self):
        plan = [
            ("pick transit", np.full(6, 1.0)), ("pick approach", np.full(6, 2.0)), ("grasp", np.full(6, 3.0)),
            ("lift", np.full(6, 2.0)), ("lift transit", np.full(6, 1.0)), ("scan 1", np.zeros(6)),
        ]
        for failing, holding, path in (
            ("grasp", False, []),
            ("lift", True, ["grasp"]),
            ("lift transit", True, ["grasp", "lift"]),
            (None, True, ["grasp", "lift", "lift transit"]),
        ):
            carry = demo.CandyCarry()

            def move(_arm, name, *_args):
                if name == failing:
                    raise demo.DemoError(f"{name} failed")

            self.arm.grip_mm.return_value = 10.0
            with self.subTest(failing=failing), mock.patch.object(demo, "move_robot", side_effect=move), \
                    mock.patch.object(demo, "hold_gripper"):
                if failing is None:
                    demo.execute_pickup(self.arm, plan, self.return_settings(), carry)
                else:
                    with self.assertRaises(demo.DemoError):
                        demo.execute_pickup(self.arm, plan, self.return_settings(), carry)
            self.assertEqual(carry.holding, holding)
            self.assertEqual([waypoint for waypoint, _ in carry.path], path)
        carry = demo.CandyCarry()
        self.arm.grip_mm.return_value = 0.0
        with mock.patch.object(demo, "move_robot"), mock.patch.object(demo, "hold_gripper"), \
                self.assertRaises(demo.GraspVerificationError):
            demo.execute_pickup(self.arm, plan, self.return_settings(), carry)
        self.assertFalse(carry.holding)

    def return_settings(self):
        return {**self.settings, "pick_motion_speed_percent": 7, "grip_close_mm": 5.0}

    def test_return_retraces_carried_path_then_releases_only_at_grasp(self):
        plan = [
            ("pick transit", np.full(6, 10.0)), ("pick approach", np.full(6, 20.0)), ("grasp", np.full(6, 30.0)),
            ("lift", np.full(6, 40.0)), ("lift transit", np.full(6, 50.0)), ("scan 1", np.full(6, 60.0)),
        ]
        carry = demo.CandyCarry()
        carry.gripped_at(np.full(6, 30.0))
        for name, value in (("lift", 40), ("lift transit", 50), ("scan pose", 60), ("hand approach", 70),
                            ("hand release height", 80)):
            carry.reached(name, np.full(6, float(value)))
        state = {"joints": np.full(6, 75.0)}  # stopped between hand approach and release
        self.arm.joints_deg.side_effect = lambda: state["joints"].copy()
        events = mock.Mock()

        def command(_arm, _name, joints, *_args):
            state["joints"] = np.asarray(joints, dtype=float)

        with mock.patch.object(demo, "validate_joint_path_safety") as path, \
                mock.patch.object(demo, "measured_start_joints", side_effect=lambda arm, *_args: arm.joints_deg()), \
                mock.patch.object(demo, "move_robot", side_effect=command) as move, \
                mock.patch.object(demo, "hold_gripper") as release:
            events.attach_mock(move, "move")
            events.attach_mock(release, "release")
            demo.return_candy_to_pickup(self.arm, plan, carry, self.return_settings())
        moves = [(call.args[1], call.args[2][0], call.args[3], call.args[5]) for call in move.call_args_list]
        self.assertEqual(moves, [
            ("return candy: retrace hand release height", 80.0, 5.0, 7),
            ("return candy: retrace hand approach", 70.0, 5.0, 7),
            ("return candy: retrace scan pose", 60.0, 5.0, 7),
            ("return candy: retrace lift transit", 50.0, 5.0, 7),
            ("return candy: retrace lift", 40.0, 5.0, 7),
            ("return candy: pickup location", 30.0, 5.0, 7),
            ("return candy: retreat above pickup", 40.0, 40.0, 7),
            ("return candy: retreat transit", 50.0, 40.0, 7),
            ("return candy: back to scan", 60.0, 40.0, 7),
        ])
        order = [operation[0] for operation in events.mock_calls]
        self.assertEqual(order.index("release"), 6)
        np.testing.assert_array_equal(release.call_args.args[1], np.full(6, 30.0))
        self.assertEqual(release.call_args.args[2], 40.0)
        self.assertFalse(carry.holding)
        whole_route = path.call_args_list[0].args[1]
        np.testing.assert_array_equal(whole_route[0], np.full(6, 75.0))
        self.assertEqual(len(whole_route), 1 + len(moves))

    def test_return_skips_waypoints_already_reached(self):
        plan = [("pick approach", np.full(6, 20.0)), ("grasp", np.full(6, 30.0)), ("lift", np.full(6, 20.0)),
                ("scan 1", np.full(6, 60.0))]
        carry = demo.CandyCarry()
        carry.gripped_at(np.full(6, 30.0))
        carry.reached("lift", np.full(6, 20.0))
        carry.reached("scan pose", np.full(6, 60.0))
        state = {"joints": np.full(6, 61.0)}  # within 2 degrees of the scan pose
        self.arm.joints_deg.side_effect = lambda: state["joints"].copy()

        def command(_arm, _name, joints, *_args):
            state["joints"] = np.asarray(joints, dtype=float)

        with mock.patch.object(demo, "validate_joint_path_safety"), \
                mock.patch.object(demo, "measured_start_joints", side_effect=lambda arm, *_args: arm.joints_deg()), \
                mock.patch.object(demo, "move_robot", side_effect=command) as move, \
                mock.patch.object(demo, "hold_gripper"):
            demo.return_candy_to_pickup(self.arm, plan, carry, self.return_settings())
        self.assertEqual([call.args[1] for call in move.call_args_list], [
            "return candy: retrace lift", "return candy: pickup location",
            "return candy: retreat above pickup", "return candy: back to scan",
        ])

    def test_unsafe_return_route_or_missing_carry_moves_nothing(self):
        plan = [("pick approach", np.ones(6)), ("grasp", np.full(6, 2.0)), ("lift", np.ones(6)), ("scan 1", np.zeros(6))]
        carry = demo.CandyCarry()
        carry.gripped_at(np.full(6, 2.0))
        with mock.patch.object(
            demo, "validate_joint_path_safety", side_effect=demo.DemoError("unsafe return")
        ), mock.patch.object(demo, "move_robot") as move, mock.patch.object(demo, "hold_gripper") as release, \
                self.assertRaisesRegex(demo.DemoError, "unsafe return"):
            demo.return_candy_to_pickup(self.arm, plan, carry, self.return_settings())
        self.assertTrue(carry.holding)
        with mock.patch.object(demo, "move_robot") as move_again, \
                self.assertRaisesRegex(demo.DemoError, "no tracked held candy"):
            demo.return_candy_to_pickup(self.arm, plan, demo.CandyCarry(), self.return_settings())
        for mocked in (move, release, move_again):
            mocked.assert_not_called()

    def run_cycle_with(self, pickup=None, delivery=None, returned=None):
        plan = [("pick approach", np.ones(6)), ("grasp", np.full(6, 2.0)), ("lift", np.ones(6)), ("scan 1", np.zeros(6))]

        def pick(_arm, _plan, _settings, carry, _target=None):
            carry.gripped_at(np.full(6, 2.0))
            if pickup is not None:
                raise pickup
            return 10.0

        with mock.patch.object(demo, "validate_joint_path_safety"), \
                mock.patch.object(demo, "capture_scan_pair", return_value=({"status": "detected"}, palm())), \
                mock.patch.object(demo, "pick_plan", return_value=({"base_point_mm": [250, 0, 0]}, plan)), \
                mock.patch.object(demo, "execute_pickup", side_effect=pick), \
                mock.patch.object(demo, "deliver_to_palm", side_effect=delivery) as delivered, \
                mock.patch.object(demo, "return_candy_to_pickup", side_effect=returned) as returning, \
                mock.patch("builtins.print") as output:
            try:
                result = demo.run_cycle(
                    self.arm, self.objects, self.hands, [self.scan], np.eye(4), self.settings, 10,
                )
            finally:
                self.output = " ".join(str(call) for call in output.call_args_list)
        return result, plan, delivered, returning

    def test_cycle_returns_held_candy_for_any_demo_error_and_continues(self):
        for failure in (
            demo.DemoError("no joint-limit-safe IK solution for hand approach at [331.3, -16.8, 323.7]mm"),
            demo.DemoError("palm capture is stale"),
            demo.DemoError("hand approach did not reach its planned joint target"),
            demo.HandEyeError("camera point must contain three finite values"),
            OSError("hand service unavailable"),
            KeyError("status"),
        ):
            with self.subTest(failure=failure):
                result, plan, _, returning = self.run_cycle_with(delivery=failure)
                self.assertFalse(result)
                returning.assert_called_once()
                self.assertIs(returning.call_args.args[1], plan)
                self.assertTrue(returning.call_args.args[2].holding)
                self.assertIn("ERROR while holding candy", self.output)
                self.assertIn("continuing scan loop", self.output)
        result, _, delivered, returning = self.run_cycle_with(pickup=demo.DemoError("lift failed"))
        self.assertFalse(result)
        delivered.assert_not_called()
        returning.assert_called_once()

    def test_cycle_stops_for_interrupts_failed_returns_and_errors_without_candy(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_cycle_with(delivery=KeyboardInterrupt())
        with self.assertRaisesRegex(demo.DemoError, "candy return failed after 'approach failed': unsafe"):
            self.run_cycle_with(delivery=demo.DemoError("approach failed"), returned=demo.DemoError("unsafe"))

        def after_release(*args):
            args[-1].holding = False
            raise demo.DemoError("hand retreat failed")

        # Raised unchanged (not wrapped as a failed return): nothing was carried back.
        with self.assertRaisesRegex(demo.DemoError, "^hand retreat failed$"):
            self.run_cycle_with(delivery=after_release)

    def test_cycle_does_not_return_candy_after_a_missed_grasp(self):
        result, _, delivered, returning = self.run_cycle_with(pickup=demo.GraspVerificationError(0.0))
        self.assertFalse(result)
        delivered.assert_not_called()
        returning.assert_not_called()

    def test_main_loop_continues_after_returned_candy(self):
        with mock.patch.object(demo, "parse_args", return_value=self.args(max_cycles=3)), \
                mock.patch.object(demo, "load_hand_eye", return_value={"camera": {"serial": "D435"}}), \
                mock.patch.object(demo, "check_services"), mock.patch.object(
                    demo, "load_setup", return_value=(self.settings, np.eye(4), [self.scan])
                ), mock.patch.object(demo, "connect_robot", return_value=self.arm), \
                mock.patch.object(demo, "run_cycle", side_effect=[False, True, False]) as cycle, \
                mock.patch.object(demo.time, "sleep"):
            self.assertEqual(demo.main(), 0)
        self.assertEqual(cycle.call_count, 3)
        self.zero.assert_called_once_with(self.arm, context="shutdown zero return", verbose=False, open_grip_mm=40.0, open_seconds=0.2)

    def candy_capture(self, *points):
        return {"object_candidates": [
            {
                "status": "detected", "score": 0.9 - 0.1 * index, "slot": 1,
                "base_point_mm": list(point), "long_axis_base_unit_xy": [1.0, 0.0],
                "long_axis_base_yaw_deg": 0.0,
            }
            for index, point in enumerate(points)
        ]}

    def test_unreachable_nearest_candy_is_skipped_for_the_next_one(self):
        settings = demo.validate_demo_settings(demo.DEMO_SETTINGS)
        near, far = (250.0, 0.0, 8.0), (380.0, 50.0, 8.0)
        good_plan = [("pick approach", np.ones(6)), ("grasp", np.full(6, 2.0)), ("lift", np.ones(6)), ("scan 1", np.zeros(6))]

        def planner(target, *_args):
            if np.allclose(target[:2], near[:2]):
                raise demo.DemoError("no joint-limit-safe IK solution for pick approach at [1, 2, 3]mm")
            return good_plan

        self.arm.joints_deg.return_value = np.array([0, 45, -25, 0, 40, 0])
        with mock.patch.object(demo, "plan_pickup", side_effect=planner) as planned, \
                mock.patch.object(demo, "validate_joint_path_safety"), \
                mock.patch.object(demo, "move_robot") as move, mock.patch("builtins.print") as output:
            target, plan = demo.pick_plan(self.arm, [self.candy_capture(far, near)], self.scan, settings)
        self.assertEqual(target["base_point_mm"], list(far))
        self.assertIs(plan, good_plan)
        tried = [tuple(np.round(call.args[0][:2], 1)) for call in planned.call_args_list]
        self.assertEqual(tried[0], near[:2])  # nearest first
        self.assertIn(far[:2], tried)
        self.assertIn("Skipping unreachable candy at [250.0, 0.0, 8.0]mm", str(output.call_args_list))
        move.assert_not_called()

    def test_no_reachable_candy_raises_dedicated_error_with_targets(self):
        settings = demo.validate_demo_settings(demo.DEMO_SETTINGS)
        self.arm.joints_deg.return_value = np.array([0, 45, -25, 0, 40, 0])
        with mock.patch.object(demo, "plan_pickup", side_effect=demo.DemoError("no IK")), \
                self.assertRaises(demo.NoReachableCandyError) as error:
            demo.pick_plan(self.arm, [self.candy_capture((250, 0, 8), (380, 50, 8))], self.scan, settings)
        self.assertIsInstance(error.exception, demo.DemoError)
        self.assertEqual(len(error.exception.targets), 2)
        self.assertIn("no accepted reachable pickup plan", str(error.exception))

    def test_cycle_rescans_instead_of_stopping_when_no_candy_is_reachable(self):
        scans = [self.scan, {**self.scan, "index": 2}]
        failure = demo.NoReachableCandyError(
            "detected candy has no accepted reachable pickup plan: ...",
            [{"base_point_mm": [303.507, 65.829, 11.399], "planning_rejection_reason": (
                "all bounded candy-aligned grasp orientations failed: yaw 178.2deg: no joint-limit-safe IK "
                "solution for pick approach at [318.2, 53.2, 236.4]mm; adaptive inward/pre-grasp search also "
                "failed: ... | yaw -1.8deg: ..."
            )}],
        )
        with mock.patch.object(demo, "validate_joint_path_safety"), \
                mock.patch.object(demo, "capture_scan_pair", return_value=({"status": "detected"}, palm())) as scanned, \
                mock.patch.object(demo, "pick_plan", side_effect=failure), \
                mock.patch.object(demo, "execute_pickup") as pickup, mock.patch("builtins.print") as output:
            self.assertFalse(demo.run_cycle(
                self.arm, self.objects, self.hands, scans, np.eye(4), self.settings, 10,
            ))
        self.assertEqual(scanned.call_count, 2)
        pickup.assert_not_called()
        printed = " ".join(str(call) for call in output.call_args_list)
        self.assertIn("No reachable candy in this scan; scanning again", printed)
        self.assertIn(
            "candy at [303.5, 65.8, 11.4]mm: no joint-limit-safe IK solution for pick approach at [318.2, 53.2, 236.4]mm",
            printed,
        )
        self.assertNotIn("yaw -1.8deg", printed)

    def test_main_loop_keeps_running_through_unreachable_scans(self):
        with mock.patch.object(demo, "parse_args", return_value=self.args(max_cycles=3)), \
                mock.patch.object(demo, "load_hand_eye", return_value={"camera": {"serial": "D435"}}), \
                mock.patch.object(demo, "check_services"), mock.patch.object(
                    demo, "load_setup", return_value=(self.settings, np.eye(4), [self.scan])
                ), mock.patch.object(demo, "connect_robot", return_value=self.arm), \
                mock.patch.object(demo, "validate_joint_path_safety"), \
                mock.patch.object(demo, "capture_scan_pair", return_value=({"status": "detected"}, palm())), \
                mock.patch.object(demo, "pick_plan", side_effect=demo.NoReachableCandyError("none", [])) as planned, \
                mock.patch.object(demo.time, "sleep"), mock.patch("builtins.print"):
            self.assertEqual(demo.main(), 0)
        self.assertEqual(planned.call_count, 3)

    def test_direct_pickup_route_failure_falls_back_to_adaptive_search(self):
        def path_check(name, *_args, **_kwargs):
            if name == "direct pick approach":
                raise demo.DemoError("direct pick approach segment 3 leaves the flange workspace")

        with mock.patch.object(demo, "solve_waypoint", side_effect=lambda name, *_args: np.full(6, float(len(name)))), \
                mock.patch.object(demo, "validate_joint_path_safety", side_effect=path_check) as checks:
            plan = demo.plan_pickup(
                np.array([300.0, 0.0, 8.0]), np.zeros(6), np.ones(6), np.array([180.0, 20.0, 0.0]),
                16.584, 100.0,
            )
        self.assertEqual([name for name, _ in plan][0], "pick transit")
        self.assertEqual([call.args[0] for call in checks.call_args_list][:2],
                         ["direct pick approach", "adaptive pick approach"])

    def test_rejection_summary_keeps_only_the_first_failure(self):
        self.assertEqual(
            demo.summarize_rejection(
                "all bounded candy-aligned grasp orientations failed: yaw 178.2deg: no joint-limit-safe IK "
                "solution for pick approach at [318.2, 53.2, 236.4]mm; adaptive inward/pre-grasp search also "
                "failed: x | yaw -1.8deg: y"
            ),
            "no joint-limit-safe IK solution for pick approach at [318.2, 53.2, 236.4]mm",
        )
        self.assertEqual(demo.summarize_rejection("cross-view position or orientation disagreement"),
                         "cross-view position or orientation disagreement")
        self.assertLessEqual(len(demo.summarize_rejection("x" * 500)), 160)

    def test_pick_plan_reports_specific_rejection_reasons_without_motion(self):
        settings = demo.validate_demo_settings(demo.DEMO_SETTINGS)
        for targets, expected in (
            ([], "no detected candy targets were available"),
            ([{
                "base_point_mm": [250, 0, -8],
                "planning_rejection_reason": "no joint-limit-safe IK solution for grasp",
            }], "target 1 at [250, 0, -8]mm: no joint-limit-safe IK solution for grasp"),
            ([
                {
                    "base_point_mm": [250, 0, 18],
                    "planning_rejection_reason": "cross-view position or orientation disagreement",
                },
                {
                    "base_point_mm": [300, 0, 18],
                    "planning_rejection_reason": "grasp would put the tool tip below the floor",
                },
            ], "target 2 at [300, 0, 18]mm: grasp would put the tool tip below the floor"),
        ):
            with self.subTest(targets=targets), mock.patch.object(
                demo, "distinct_object_targets", return_value=targets
            ), mock.patch.object(demo, "select_reachable_object_target", return_value=None), \
                    mock.patch.object(demo, "move_robot") as move, \
                    self.assertRaises(demo.DemoError) as error:
                demo.pick_plan(self.arm, [], self.scan, settings)
            self.assertIn(expected, str(error.exception))
            move.assert_not_called()

    def test_no_candy_or_hand_skips_pickup(self):
        for candy, hand in (
            ({"status": "not_found"}, palm()), ({"status": "detected"}, palm(status="not_found")),
        ):
            with self.subTest(candy=candy, hand=hand), mock.patch.object(
                demo, "validate_joint_path_safety"
            ), mock.patch.object(
                demo, "capture_scan_pair", return_value=(candy, hand)
            ), mock.patch.object(demo, "execute_pickup") as pickup:
                self.assertFalse(demo.run_cycle(
                    self.arm, self.objects, self.hands, [self.scan], np.eye(4), self.settings, 10
                ))
            pickup.assert_not_called()

    def test_cycle_picks_before_hand_reacquisition_and_delivery(self):
        events = []
        with mock.patch.object(demo, "validate_joint_path_safety"), \
                mock.patch.object(demo, "capture_scan_pair", return_value=(
                    {"status": "detected"}, palm()
                )), mock.patch.object(demo, "pick_plan", return_value=(
                    {"base_point_mm": [250, 0, 0]}, []
                )), mock.patch.object(
                    demo, "execute_pickup",
                    side_effect=lambda *_args: events.append("pickup") or 10.0,
                ), mock.patch.object(
                    demo, "deliver_to_palm", side_effect=lambda *_args: events.append("delivery")
                ):
            self.assertTrue(demo.run_cycle(
                self.arm, self.objects, self.hands, [self.scan], np.eye(4), self.settings, 10
            ))
        self.assertEqual(events, ["pickup", "delivery"])

    def test_missed_grasp_does_not_attempt_hand_delivery(self):
        with mock.patch.object(demo, "validate_joint_path_safety"), \
                mock.patch.object(demo, "capture_scan_pair", return_value=(
                    {"status": "detected"}, palm()
                )), mock.patch.object(demo, "pick_plan", return_value=(
                    {"base_point_mm": [250, 0, 0]}, []
                )), mock.patch.object(
                    demo, "execute_pickup", side_effect=demo.GraspVerificationError(0.0)
                ), mock.patch.object(demo, "deliver_to_palm") as delivery:
            self.assertFalse(demo.run_cycle(
                self.arm, self.objects, self.hands, [self.scan], np.eye(4), self.settings, 10
            ))
        delivery.assert_not_called()

    def test_all_services_report_even_when_one_fails(self):
        camera = mock.Mock(base_url="http://camera:8094")
        camera.health.side_effect = demo.DemoError("offline")
        with mock.patch("builtins.print") as output, self.assertRaises(demo.DemoError):
            demo.check_services(camera, self.objects, self.hands, "263022078260")
        self.objects.detector.check.assert_called_once()
        self.hands.health.assert_called_once()
        self.assertTrue(any("[FAILED] OpenCV" in str(call) for call in output.call_args_list))
        self.assertNotIn(
            mock.call("[OK] All required services are running and available.", flush=True),
            output.call_args_list,
        )

    def test_preview_mode_and_wrong_camera_are_rejected(self):
        for fields in ({"preview_only": True}, {"camera_serial": "wrong"}):
            camera = mock.Mock(base_url="http://camera:8094")
            camera.health.return_value = {
                "camera_serial": "263022078260", "preview_only": False,
                "service": "camera",
                "depth_calibrated": True, "capabilities": ["localize", "deproject"], **fields,
            }
            with self.subTest(fields=fields), self.assertRaises(demo.DemoError):
                with mock.patch("builtins.print") as output:
                    demo.check_services(camera, self.objects, self.hands, "263022078260")
                self.assertEqual(
                    output.call_args,
                    mock.call("[OK] All required services are running and available.", flush=True),
                )

    def test_services_pass_only_for_matching_calibrated_camera_and_hand_source(self):
        camera = mock.Mock(base_url="http://camera:8094")
        camera.health.return_value = {
            "status": "ok", "service": "camera", "camera_serial": "263022078260",
            "preview_only": False, "depth_calibrated": True,
            "capabilities": ["localize", "deproject"],
        }
        self.objects.detector.check.return_value = None
        self.hands.health.return_value = {"source_url": "http://camera:8094/stream.mjpg"}
        demo.check_services(camera, self.objects, self.hands, "263022078260")
        self.hands.health.return_value = {"source_url": "http://different-camera/stream.mjpg"}
        with self.assertRaisesRegex(demo.DemoError, "hand stream"):
            demo.check_services(camera, self.objects, self.hands, "263022078260")

    def test_setup_requires_every_quality_gate_and_matching_camera(self):
        config = self.example_config()
        accepted = {"acceptance": {"quality_passed": True}}
        documents = {
            "/unused/calibration/hand_eye.json": {
                **accepted, "camera": {"serial": "263022078260"},
                "camera_to_gripper": np.eye(4).tolist(),
            },
            "/unused/calibration/validation.json": dict(accepted),
            "/unused/calibration/depth_correction.json": dict(accepted),
            "/unused/calibration/depth_verification.json": {
                **accepted, "camera": {"serial": "263022078260"},
            },
        }
        with mock.patch.object(
            demo, "load_json", side_effect=lambda path: documents[str(path)]
        ), mock.patch.object(demo, "load_scan_poses", return_value=[self.scan]):
            settings, transform, scans = demo.load_setup(self.args())
            self.assertEqual(settings["camera_serial"], "263022078260")
            self.assertFalse(settings["verbose"])
            verbose_settings, _, _ = demo.load_setup(self.args(verbose=True))
            self.assertTrue(verbose_settings["verbose"])
            np.testing.assert_array_equal(transform, np.eye(4))
            self.assertEqual(scans, [self.scan])
            for name in (
                "hand_eye.json", "validation.json", "depth_correction.json", "depth_verification.json"
            ):
                key = "/unused/calibration/" + name
                original = documents[key]
                documents[key] = {**original, "acceptance": {"quality_passed": False}}
                with self.subTest(name=name), self.assertRaises(demo.DemoError):
                    demo.load_setup(self.args())
                documents[key] = original
            for name in ("hand_eye.json", "depth_verification.json"):
                key = "/unused/calibration/" + name
                original = documents[key]
                documents[key] = {**original, "camera": {"serial": "wrong-camera"}}
                with self.subTest(name=name), self.assertRaisesRegex(demo.DemoError, "different camera"):
                    demo.load_setup(self.args())
                documents[key] = original
            for serial in (None, "", "   ", 263022078260):
                key = "/unused/calibration/hand_eye.json"
                original = documents[key]
                documents[key] = {**original, "camera": {"serial": serial}}
                with self.subTest(serial=serial), self.assertRaisesRegex(demo.DemoError, "nonempty camera serial"):
                    demo.load_setup(self.args())
                documents[key] = original

    def test_motion_config_needs_no_legacy_settings_or_hand_movement_limit(self):
        config = self.example_config()
        parsed = demo.validate_demo_settings(config["demo"])
        for key in (
            "hand_release_tip_z_mm", "drop_approach_height_mm",
            "maximum_grasp_retries", "enforce_hand_reacquisition_limit",
            "maximum_hand_reacquisition_difference_mm",
        ):
            self.assertNotIn(key, parsed)
        self.assertNotIn("maximum_hand_reacquisition_difference_mm", demo.DEMO_SETTINGS)

    def test_candy_workspace_is_defined_in_code_not_config(self):
        config = self.example_config()
        del config["demo"]["target_workspace_min_mm"]
        del config["demo"]["target_workspace_max_mm"]
        parsed = demo.validate_demo_settings(config["demo"])
        np.testing.assert_allclose(parsed["target_workspace_min_mm"], [144, -170, 17.423])
        np.testing.assert_allclose(parsed["target_workspace_max_mm"], [demo.WORKSPACE_MAX_MM[0], 206.317, 263.323])
        config["demo"]["target_workspace_min_mm"] = [0, 0, -999]
        config["demo"]["target_workspace_max_mm"] = [999, 999, 999]
        parsed_again = demo.validate_demo_settings(config["demo"])
        np.testing.assert_array_equal(parsed_again["target_workspace_min_mm"], parsed["target_workspace_min_mm"])
        np.testing.assert_array_equal(parsed_again["target_workspace_max_mm"], parsed["target_workspace_max_mm"])
        np.testing.assert_array_equal(parsed_again["hand_workspace_min_mm"], parsed["target_workspace_min_mm"])
        np.testing.assert_array_equal(parsed_again["hand_workspace_max_mm"], parsed["target_workspace_max_mm"])
        parsed_again["hand_workspace_min_mm"][0] = 0
        self.assertEqual(parsed_again["target_workspace_min_mm"][0], 144)
        parsed_again["target_workspace_min_mm"][0] = 0
        self.assertEqual(demo.WORKSPACE_MIN_MM[0], 144)

    def test_workspace_limits_are_derived_from_corners_with_three_decimals(self):
        corners = np.asarray(demo.WORKSPACE_CORNERS_MM)
        np.testing.assert_array_equal(demo.WORKSPACE_MIN_MM, [*corners[:, :2].min(axis=0), 17.423])
        np.testing.assert_array_equal(demo.WORKSPACE_MAX_MM, [*corners[:, :2].max(axis=0), 263.323])
        np.testing.assert_array_equal(demo.WORKSPACE_MIN_MM[:2], [144.0, -170.0])
        self.assertEqual(demo.WORKSPACE_MAX_MM[1], 206.317)
        self.assertGreaterEqual(demo.WORKSPACE_MAX_MM[0], 500.0)
        for value in (*np.ravel(demo.WORKSPACE_CORNERS_MM), demo.WORKSPACE_MIN_Z_MM, demo.WORKSPACE_MAX_Z_MM):
            self.assertEqual(round(float(value), 3), float(value))

    def test_workspace_rejection_message_uses_three_decimals(self):
        minimum = np.array([144.0, -170.0, 17.423054878247285])
        maximum = np.array([500.0, 206.3166357672802, 263.3228203692904])
        for check_z, expected in (
            (True, "[144.0, -170.0, 17.423]..[500.0, 206.317, 263.323]mm"),
            (False, "XY workspace bounds [144.0, -170.0]..[500.0, 206.317]mm"),
        ):
            with self.subTest(check_z=check_z), self.assertRaises(demo.DemoError) as error:
                demo.validate_target_workspace(np.array([600.0, 0, 20]), minimum, maximum, check_z=check_z)
            self.assertIn(expected, str(error.exception))
            self.assertNotIn("17.423054878247285", str(error.exception))
            self.assertNotIn("206.3166357672802", str(error.exception))

    def test_scan_summary_reports_candies_and_picking_point_is_rounded(self):
        settings = demo.validate_demo_settings(demo.DEMO_SETTINGS)
        candidate = {
            "status": "detected", "score": 0.9, "box_xyxy": [1, 2, 30, 40],
            "camera_point_mm": [300.12345678, 10.98765432, 18.5], "long_axis_camera_unit": [1, 0, 0],
        }
        with mock.patch("builtins.print") as output:
            demo.process_object_capture(
                {"status": "detected", "candidates": [candidate], "snapshot_url": "/snapshots/1.jpg"},
                1, 0.05, mock.Mock(base_url="http://camera"), np.zeros(6), np.zeros(6),
                np.eye(4), np.eye(4), np.eye(4), settings,
            )
        summary = output.call_args.args[0]
        self.assertIn("CANDY SCAN: detected 1 candy(ies)", summary)
        self.assertNotIn("candidate(s)", summary)
        plan = [("pick approach", np.ones(6)), ("grasp", np.ones(6)), ("lift", np.ones(6)), ("scan 1", np.zeros(6))]
        with mock.patch.object(demo, "validate_joint_path_safety"), \
                mock.patch.object(demo, "capture_scan_pair", return_value=({"status": "detected"}, palm())), \
                mock.patch.object(demo, "pick_plan", return_value=(
                    {"base_point_mm": [400.4122290024606, 162.75170327231274, 10.450959714083695]}, plan,
                )), mock.patch.object(demo, "execute_pickup", side_effect=demo.GraspVerificationError(0.0)), \
                mock.patch("builtins.print") as output:
            demo.run_cycle(self.arm, self.objects, self.hands, [self.scan], np.eye(4), self.settings, 10)
        self.assertIn(
            mock.call("Picking Starburst at [400.412, 162.752, 10.451]mm", flush=True), output.call_args_list,
        )

    def test_candy_corners_are_accepted_and_only_xy_limits_are_enforced(self):
        settings = demo.validate_demo_settings(self.example_config()["demo"])
        minimum, maximum = settings["target_workspace_min_mm"], settings["target_workspace_max_mm"]
        for corner in demo.WORKSPACE_CORNERS_MM:
            demo.validate_target_workspace(np.array(corner), minimum, maximum, check_z=False)
        for point in ([246.3, 81.2, -8.3], [299.1, 3.0, -1.0], [250, 0, 264]):
            with self.subTest(point=point):
                demo.validate_target_workspace(np.array(point), minimum, maximum, check_z=False)
        for point in (
            [143, 0, 18], [demo.WORKSPACE_MAX_MM[0] + 1, 0, 18], [250, -171, 18], [250, 207, 18],
            [250, 0, float("nan")], [250, 0, float("inf")],
        ):
            with self.subTest(point=point), self.assertRaises(demo.DemoError):
                demo.validate_target_workspace(np.array(point), minimum, maximum, check_z=False)

    def test_opencv_candidates_must_be_under_four_by_two_inches(self):
        settings = demo.validate_demo_settings(demo.DEMO_SETTINGS)
        settings["camera_intrinsics"] = {"fx": 500.0, "fy": 500.0, "ppx": 320.0, "ppy": 240.0}
        # Camera 500 mm above the base looking straight down at (300, 0); candy surface Z = 12 mm.
        gripper_to_base = np.array([[1, 0, 0, 300], [0, -1, 0, 0], [0, 0, -1, 500], [0, 0, 0, 1]], float)
        millimetres_per_pixel = (500 - 12) / 500

        def candidate(length_mm, width_mm):
            half_axis_px = 0.4 * length_mm / millimetres_per_pixel
            return {
                "status": "detected", "camera_point_mm": [0, 0, 488],
                "long_axis_camera_unit": [1, 0, 0],
                "long_axis_pixels": [[320 - half_axis_px, 240], [320 + half_axis_px, 240]],
                "orientation_aspect_ratio": length_mm / width_mm,
            }

        for detector, length, width, accepted in (
            ("opencv", 86, 33, True),
            ("opencv", 101, 50, True),
            ("opencv", 102, 30, False),
            ("opencv", 90, 52, False),
            ("opencv", 200, 80, False),
            ("yolo", 200, 80, True),
        ):
            settings["detector"] = detector
            with self.subTest(detector=detector, length=length, width=width):
                result = demo.transform_object_candidate(
                    candidate(length, width), 1, gripper_to_base, np.eye(4), settings,
                )
                self.assertEqual(result["status"], "detected" if accepted else "rejected")
                np.testing.assert_allclose(result.get("base_point_mm", [300, 0, 12]), [300, 0, 12], atol=1e-6)
                if detector == "opencv":
                    self.assertAlmostEqual(result["length_mm"], length, places=4)
                    self.assertAlmostEqual(result["width_mm"], width, places=4)
                    if not accepted:
                        self.assertIn("101.6 x 50.8 mm candy limit", result["error"])
                else:
                    self.assertNotIn("length_mm", result)

    def test_grasp_tilt_override_replaces_only_pitch_within_limits(self):
        accepted = {"acceptance": {"quality_passed": True}}
        hand_eye = {**accepted, "camera": {"serial": "D435"}, "camera_to_gripper": np.eye(4).tolist()}
        documents = {
            "validation.json": accepted, "depth_correction.json": accepted,
            "depth_verification.json": {**accepted, "camera": {"serial": "D435"}},
        }
        nominal = list(demo.DEMO_SETTINGS["grasp_rpy_deg"])
        with mock.patch.object(demo, "load_json", side_effect=lambda path: documents[Path(path).name]), \
                mock.patch.object(demo, "load_scan_poses", return_value=[self.scan]), \
                mock.patch("builtins.print"):
            for tilt in (None, 0.0, 12.5, demo.MAXIMUM_GRASP_TILT_OVERRIDE_DEG):
                with self.subTest(tilt=tilt):
                    settings, _, _ = demo.load_setup(self.args(grasp_tilt_deg=tilt), hand_eye)
                    expected = nominal if tilt is None else [nominal[0], tilt, nominal[2]]
                    np.testing.assert_allclose(settings["grasp_rpy_deg"], expected)
            for tilt in (-1.0, demo.MAXIMUM_GRASP_TILT_OVERRIDE_DEG + 0.1, float("nan")):
                with self.subTest(tilt=tilt), self.assertRaisesRegex(demo.DemoError, "--grasp-tilt-deg"):
                    demo.load_setup(self.args(grasp_tilt_deg=tilt), hand_eye)
        self.assertEqual(demo.DEMO_SETTINGS["grasp_rpy_deg"], nominal)

    def test_opencv_setup_requires_camera_intrinsics(self):
        accepted = {"acceptance": {"quality_passed": True}}
        hand_eye = {**accepted, "camera": {"serial": "D435"}, "camera_to_gripper": np.eye(4).tolist()}
        documents = {
            "validation.json": accepted, "depth_correction.json": accepted,
            "depth_verification.json": {**accepted, "camera": {"serial": "D435"}},
        }
        intrinsics = {"fx": 606.0, "fy": 605.5, "ppx": 314.8, "ppy": 255.7}
        for detector, camera_intrinsics, valid in (
            ("yolo", None, True), ("opencv", intrinsics, True), ("opencv", None, False),
            ("opencv", {**intrinsics, "fx": 0}, False), ("opencv", {"fx": 606.0}, False),
        ):
            camera = {"serial": "D435", **({} if camera_intrinsics is None else {"intrinsics": camera_intrinsics})}
            with self.subTest(detector=detector, intrinsics=camera_intrinsics), mock.patch.object(
                demo, "load_json", side_effect=lambda path: documents[Path(path).name]
            ), mock.patch.object(demo, "load_scan_poses", return_value=[self.scan]):
                if valid:
                    settings, _, _ = demo.load_setup(self.args(detector=detector), {**hand_eye, "camera": camera})
                    self.assertEqual(settings["detector"], detector)
                else:
                    with self.assertRaisesRegex(demo.DemoError, "intrinsics"):
                        demo.load_setup(self.args(detector=detector), {**hand_eye, "camera": camera})

    def test_candy_transformation_checks_xy_workspace_and_plausible_table_height(self):
        settings = demo.validate_demo_settings(demo.DEMO_SETTINGS)
        robot_transform = np.eye(4)
        robot_transform[2, 3] = -100
        for x, y, z, status in (
            (246.3, 81.2, -8.3, "detected"),
            (299.1, 3, -1, "detected"),
            (250, 0, demo.DEMO_SETTINGS["candy_min_z_mm"], "detected"),
            (250, 0, demo.DEMO_SETTINGS["candy_max_z_mm"], "detected"),
            (250, 0, demo.DEMO_SETTINGS["candy_max_z_mm"] + 0.1, "rejected"),
            (250, 0, demo.DEMO_SETTINGS["candy_min_z_mm"] - 0.1, "rejected"),
            (250, 0, -348.1, "rejected"),
            (250, 0, 264, "rejected"),
            (143, 0, -8.3, "rejected"),
            (demo.WORKSPACE_MAX_MM[0] + 1, 0, -8.3, "rejected"),
            (250, -171, -8.3, "rejected"),
            (250, 207, -8.3, "rejected"),
            (250, 0, float("nan"), "rejected"),
            (250, 0, float("inf"), "rejected"),
        ):
            with self.subTest(x=x, y=y, z=z):
                result = demo.transform_object_candidate(
                    {
                        "status": "detected", "camera_point_mm": [x, y, z + 100],
                        "long_axis_camera_unit": [1, 0, 0],
                    },
                    1, robot_transform, np.eye(4), settings,
                )
                self.assertEqual(result["status"], status)
                if status == "detected":
                    np.testing.assert_allclose(result["base_point_mm"], [x, y, z])
                else:
                    self.assertIn("error", result)
                    if math.isfinite(z) and 143 < x < demo.WORKSPACE_MAX_MM[0] and -171 < y < 207:
                        self.assertIn("plausible table range", result["error"])

    def test_candy_xy_correction_shifts_planned_point_before_workspace_check(self):
        settings = demo.validate_demo_settings(
            {**demo.DEMO_SETTINGS, "candy_xy_correction_mm": [7.0, -34.0]}
        )
        robot_transform = np.eye(4)
        robot_transform[2, 3] = -100
        for x, y, status in ((250.0, 80.0, "detected"), (250.0, -150.0, "rejected")):
            with self.subTest(x=x, y=y):
                result = demo.transform_object_candidate(
                    {"status": "detected", "camera_point_mm": [x, y, 100.0], "long_axis_camera_unit": [1, 0, 0]},
                    1, robot_transform, np.eye(4), settings,
                )
                self.assertEqual(result["status"], status)
                np.testing.assert_allclose(result["uncorrected_base_point_mm"], [x, y, 0.0])
                if status == "detected":
                    np.testing.assert_allclose(result["base_point_mm"], [x + 7.0, y - 34.0, 0.0])
        uncorrected = demo.transform_object_candidate(
            {"status": "detected", "camera_point_mm": [250.0, 80.0, 100.0], "long_axis_camera_unit": [1, 0, 0]},
            1, robot_transform, np.eye(4), demo.validate_demo_settings(demo.DEMO_SETTINGS),
        )
        self.assertNotIn("uncorrected_base_point_mm", uncorrected)

    def test_candy_correction_and_height_window_are_validated(self):
        self.assertEqual(demo.DEMO_SETTINGS["candy_xy_correction_mm"], [0.0, 0.0])
        self.assertGreaterEqual(demo.DEMO_SETTINGS["detection_confidence_threshold"], 0.5)
        for overrides in (
            {"candy_xy_correction_mm": [0.0]},
            {"candy_xy_correction_mm": [float("nan"), 0.0]},
            {"candy_xy_correction_mm": [demo.MAXIMUM_CANDY_XY_CORRECTION_MM + 1, 0.0]},
            {"candy_min_z_mm": 30.0, "candy_max_z_mm": -30.0},
            {"candy_min_z_mm": float("nan")},
            {"candy_max_z_mm": "high"},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(demo.DemoError):
                demo.validate_demo_settings({**demo.DEMO_SETTINGS, **overrides})
        settings = demo.validate_demo_settings(
            {**demo.DEMO_SETTINGS, "candy_xy_correction_mm": [-60.0, 60.0]}
        )
        np.testing.assert_allclose(settings["candy_xy_correction_mm"], [-60.0, 60.0])

    def test_verbose_scan_lists_rejected_candies_alongside_accepted_ones(self):
        settings = demo.validate_demo_settings(demo.DEMO_SETTINGS)
        robot_transform = np.eye(4)
        robot_transform[2, 3] = -100
        service_result = {
            "status": "detected", "snapshot_url": "/snapshots/1.jpg",
            "candidates": [
                {"status": "detected", "score": 0.9, "camera_point_mm": [250, 80, 100],
                 "long_axis_camera_unit": [1, 0, 0]},
                {"status": "detected", "score": 0.49, "camera_point_mm": [295, 46, -248],
                 "long_axis_camera_unit": [1, 0, 0]},
            ],
        }
        for verbose in (False, True):
            settings["verbose"] = verbose
            with self.subTest(verbose=verbose), mock.patch("builtins.print") as output:
                result = demo.process_object_capture(
                    service_result, 1, 0.5, mock.Mock(base_url="http://camera"), np.zeros(6), np.zeros(6),
                    robot_transform, robot_transform, np.eye(4), settings,
                )
            text = "\n".join(str(call.args[0]) for call in output.call_args_list)
            self.assertEqual(result["candidate_count"], 1)
            self.assertEqual(result["score"], 0.9)
            self.assertEqual("rejected candy score=0.490" in text, verbose)
            self.assertEqual("plausible table range" in text, verbose)

    def test_forward_lean_threshold_is_validated(self):
        for value in (0.0, -0.1, 1.01, float("nan")):
            with self.subTest(value=value), self.assertRaises(demo.DemoError):
                demo.validate_demo_settings({**demo.DEMO_SETTINGS, "forward_lean_vertical_threshold": value})
        self.assertEqual(
            demo.validate_demo_settings({**demo.DEMO_SETTINGS, "forward_lean_vertical_threshold": 1})[
                "forward_lean_vertical_threshold"
            ],
            1.0,
        )

    def test_grasp_offset_parsing(self):
        for text, expected in (("", None), ("   ", None), ("0 -30", [0.0, -30.0]), ("7.5, 12", [7.5, 12.0])):
            with self.subTest(text=text):
                result = demo.parse_grasp_offset(text)
                if expected is None:
                    self.assertIsNone(result)
                else:
                    np.testing.assert_allclose(result, expected)
        for text in ("5", "1 2 3", "a b", "nan 0", "0 101", "q"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                demo.parse_grasp_offset(text)

    def test_pause_before_grip_holds_open_jaws_then_records_offset_before_closing(self):
        plan = [("pick approach", np.full(6, 2.0)), ("grasp", np.full(6, 3.0)), ("lift", np.full(6, 2.0)),
                ("scan 1", np.zeros(6))]
        target = {"base_point_mm": [250.0, 80.0, 1.0], "long_axis_base_yaw_deg": 90.0, "score": 0.9}
        events = []
        self.arm.grip_mm.return_value = 20.0
        self.arm.send_joints.side_effect = lambda joints, grip, speed: events.append(("hold", grip))
        answers = iter(["bad", "", "0 -30"])
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "logs" / "grasp_offsets.jsonl"
            for expected_records in (0, 1):
                settings = {**self.return_settings(), "pause_before_grip": True,
                            "candy_xy_correction_mm": np.array([5.0, 0.0]), "detector": "yolo"}
                events.clear()

                def answer(_prompt):
                    time.sleep(0.05)
                    return next(answers)

                with self.subTest(expected_records=expected_records), \
                        mock.patch.object(demo, "GRASP_OFFSET_LOG", log), \
                        mock.patch.object(demo, "record_grasp_offset", wraps=demo.record_grasp_offset) as record, \
                        mock.patch("builtins.input", side_effect=answer), mock.patch("builtins.print"), \
                        mock.patch.object(demo, "move_robot", side_effect=lambda _a, name, *_: events.append(name)), \
                        mock.patch.object(demo, "hold_gripper", side_effect=lambda _a, _j, grip, _s: events.append(("grip", grip))):
                    demo.execute_pickup(self.arm, plan, settings, demo.CandyCarry(), target)
                self.assertEqual(events[:2], ["pick approach", "grasp"])
                self.assertIn(("hold", settings["grip_open_mm"]), events)
                close = events.index(("grip", settings["grip_close_mm"]))
                self.assertTrue(all(event[0] == "hold" for event in events[2:close]))
                self.assertEqual(record.call_count, expected_records)
            records = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["offset_mm"], [0.0, -30.0])
        self.assertEqual(records[0]["correction_in_effect_mm"], [5.0, 0.0])
        self.assertEqual(records[0]["candy_base_point_mm"], [250.0, 80.0, 1.0])

    def test_recorded_offsets_suggest_correction_from_run_mean(self):
        settings = {"candy_xy_correction_mm": np.array([5.0, -10.0])}
        with tempfile.TemporaryDirectory() as directory, mock.patch("builtins.print"):
            log = Path(directory) / "offsets.jsonl"
            first = demo.record_grasp_offset(
                np.array([0.0, -30.0]), {}, np.zeros(3), np.array([1.0, 0.0]), settings, log,
            )
            second = demo.record_grasp_offset(
                np.array([10.0, -20.0]), {}, np.zeros(3), np.array([0.0, 1.0]), settings, log,
            )
            records = [json.loads(line) for line in log.read_text().splitlines()]
        np.testing.assert_allclose(first, [5.0, -40.0])
        np.testing.assert_allclose(second, [10.0, -35.0])
        self.assertEqual([record["across_jaw_mm"] for record in records], [0.0, -20.0])
        self.assertEqual([record["along_candy_mm"] for record in records], [-30.0, -10.0])

    def test_pause_before_grip_requires_interactive_terminal(self):
        settings = {**self.settings, "pause_before_grip": True}
        with mock.patch.object(demo, "parse_args", return_value=self.args(pause_before_grip=True)), \
                mock.patch.object(demo, "load_hand_eye", return_value={"camera": {"serial": "D435"}}), \
                mock.patch.object(demo, "VisionClient", return_value=self.objects), \
                mock.patch.object(demo, "HandClient", return_value=self.hands), \
                mock.patch.object(demo, "check_services"), \
                mock.patch.object(demo, "load_setup", return_value=(settings, np.eye(4), [self.scan])), \
                mock.patch.object(demo.sys.stdin, "isatty", return_value=False), \
                mock.patch.object(demo, "connect_robot") as connect, \
                mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
            self.assertEqual(demo.main(), 2)
        connect.assert_not_called()
        self.assertIn("interactive terminal", stderr.getvalue())

    def test_palm_capture_and_delivery_use_shared_workspace_including_z(self):
        settings = demo.validate_demo_settings(demo.DEMO_SETTINGS)
        settings["grasp_rpy_deg"] = np.array(settings["grasp_rpy_deg"])
        self.assertNotIn("hand_workspace_min_mm", demo.DEMO_SETTINGS)
        self.assertNotIn("hand_workspace_max_mm", demo.DEMO_SETTINGS)
        for point, accepted in (
            ([305.5, -165.9, 27.3], True),
            ([144, -170, demo.WORKSPACE_MIN_Z_MM], True),
            ([demo.WORKSPACE_MAX_MM[0], 206.317, demo.WORKSPACE_MAX_Z_MM], True),
            ([470, 0, 27.3], True),
            ([143, 0, 27.3], False),
            ([demo.WORKSPACE_MAX_MM[0] + 1, 0, 27.3], False),
            ([305.5, -171, 27.3], False),
            ([305.5, 207, 27.3], False),
            ([305.5, 0, 17], False),
            ([305.5, 0, 264], False),
        ):
            with self.subTest(point=point), mock.patch("builtins.print"):
                result = demo.process_hand_capture(
                    {
                        **palm(point), "camera_point_mm": point,
                        "score": 0.9, "snapshot_url": "/palm.jpg",
                    },
                    1, 4, mock.Mock(base_url="http://hand"),
                    np.zeros(6), np.zeros(6), np.eye(4), np.eye(4), np.eye(4), settings,
                )
                self.assertEqual(result["status"], "detected" if accepted else "rejected")
                with mock.patch.object(demo, "plan_hand_delivery", return_value=[]) as planner, \
                        mock.patch.object(demo, "validate_joint_path_safety"):
                    if accepted:
                        demo.plan_delivery(result, self.arm, self.scan, settings, 10)
                        planner.assert_called_once()
                    else:
                        with self.assertRaises(demo.DemoError):
                            demo.plan_delivery(palm(point), self.arm, self.scan, settings, 10)
                        planner.assert_not_called()

    def args(self, **fields):
        return argparse.Namespace(
            **{
                "calibration_dir": Path("/unused/calibration"),
                "scan_poses": Path("/unused/scan_poses.json"), "vision_url": "http://camera",
                "detection_url": "http://detector", "hand_url": "http://hand",
                "can_interface": "can_piper", "max_cycles": 2,
                "check_only": False, "allow_motion": True, "allow_pick": True,
                "detector": "yolo",
                "verbose": False,
                "allow_hand_delivery": True, "setup_reviewed": True, **fields,
            }
        )

    def test_hand_age_is_not_a_command_line_argument(self):
        with mock.patch.object(sys, "argv", ["demo.py"]):
            self.assertFalse(hasattr(demo.parse_args(), "maximum_hand_age_seconds"))
        with mock.patch.object(sys, "argv", ["demo.py", "--maximum-hand-age-seconds", "3"]):
            with self.assertRaises(SystemExit) as error:
                demo.parse_args()
        self.assertEqual(error.exception.code, 2)

    def test_verbose_is_opt_in(self):
        with mock.patch.object(sys, "argv", ["demo.py"]):
            self.assertFalse(demo.parse_args().verbose)
        with mock.patch.object(sys, "argv", ["demo.py", "--verbose"]):
            self.assertTrue(demo.parse_args().verbose)

    def test_detector_defaults_to_yolo_and_accepts_only_known_choices(self):
        with mock.patch.object(sys, "argv", ["demo.py"]):
            self.assertEqual(demo.parse_args().detector, "yolo")
        with mock.patch.object(sys, "argv", ["demo.py", "--detector", "opencv"]):
            self.assertEqual(demo.parse_args().detector, "opencv")
        with mock.patch.object(sys, "argv", ["demo.py", "--detector", "owl"]), \
                mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            demo.parse_args()

    def test_selected_detector_is_used_for_candy_detection(self):
        for choice, expected in (("yolo", type(None)), ("opencv", demo.OpenCVCandyDetector)):
            with self.subTest(choice=choice), mock.patch.object(
                demo, "parse_args", return_value=self.args(check_only=True, detector=choice)
            ), mock.patch.object(
                demo, "load_hand_eye", return_value={"camera": {"serial": "D435"}}
            ), mock.patch.object(demo, "VisionClient") as vision, \
                    mock.patch.object(demo, "check_services"), mock.patch.object(
                        demo, "load_setup", return_value=(self.settings, np.eye(4), [self.scan])
                    ), mock.patch.object(demo, "connect_robot") as connect:
                self.assertEqual(demo.main(), 0)
            self.assertIsInstance(vision.call_args.kwargs["detector"], expected)
            connect.assert_not_called()

    def test_service_check_reports_selected_detector(self):
        camera = mock.Mock(base_url="http://camera:8094")
        camera.health.return_value = {
            "status": "ok", "service": "camera", "camera_serial": "D435",
            "preview_only": False, "depth_calibrated": True,
            "capabilities": ["localize", "deproject"],
        }
        self.hands.health.return_value = {"source_url": "http://camera:8094/stream.mjpg"}
        for detector, label in (
            (demo.OpenCVCandyDetector(), "[OK] OpenCV candy detection (local CPU colour detector)"),
            (self.objects.detector, "[OK] YOLO / Foundry detection (CUDA model loaded)"),
        ):
            self.objects.detector = detector
            with self.subTest(label=label), mock.patch("builtins.print") as output:
                demo.check_services(camera, self.objects, self.hands, "D435")
            self.assertIn(mock.call(label, flush=True), output.call_args_list)
        broken = demo.OpenCVCandyDetector({**demo.OpenCVCandyDetector().settings, "minimum_fill_ratio": 2})
        self.objects.detector = broken
        with mock.patch("builtins.print"), self.assertRaisesRegex(demo.DemoError, "OpenCV candy detection"):
            demo.check_services(camera, self.objects, self.hands, "D435")

    def test_check_only_never_connects_to_can(self):
        with mock.patch.object(demo, "parse_args", return_value=self.args(check_only=True)), \
                mock.patch.object(demo, "load_hand_eye", return_value={"camera": {"serial": "D435"}}), \
                mock.patch.object(demo, "check_services") as services, mock.patch.object(
                    demo, "load_setup", return_value=(self.settings, np.eye(4), [self.scan])
                ), mock.patch.object(demo, "connect_robot") as connect:
            self.assertEqual(demo.main(), 0)
        connect.assert_not_called()
        self.assertEqual(services.call_args.args[-1], "D435")

    def test_each_motion_gate_prevents_can_connection(self):
        for flag in ("allow_motion", "allow_pick", "allow_hand_delivery", "setup_reviewed"):
            with self.subTest(flag=flag), mock.patch.object(
                demo, "parse_args", return_value=self.args(**{flag: False})
            ), mock.patch.object(
                demo, "load_hand_eye", return_value={"camera": {"serial": "D435"}}
            ), mock.patch.object(demo, "check_services"), mock.patch.object(
                demo, "load_setup", return_value=(self.settings, np.eye(4), [self.scan])
            ), mock.patch.object(demo, "connect_robot") as connect:
                self.assertEqual(demo.main(), 2)
            connect.assert_not_called()

    def test_removed_scan_option_is_rejected(self):
        with mock.patch.object(sys, "argv", ["demo.py", "--scan-only"]), \
                self.assertRaises(SystemExit) as error:
            demo.parse_args()
        self.assertEqual(error.exception.code, 2)

    def test_loop_runs_requested_cycles_then_returns_to_zero_and_disconnects(self):
        shutdown = mock.Mock()
        shutdown.attach_mock(self.zero, "zero")
        shutdown.attach_mock(self.arm.disconnect, "disconnect")
        with mock.patch.object(demo, "parse_args", return_value=self.args()), \
                mock.patch.object(demo, "load_hand_eye", return_value={"camera": {"serial": "D435"}}), \
                mock.patch.object(demo, "check_services"), mock.patch.object(
                    demo, "load_setup", return_value=(self.settings, np.eye(4), [self.scan])
                ), mock.patch.object(demo, "connect_robot", return_value=self.arm), \
                mock.patch.object(demo, "run_cycle", return_value=True) as cycle, \
                mock.patch.object(demo.time, "sleep"):
            self.assertEqual(demo.main(), 0)
        self.assertEqual(cycle.call_count, 2)
        self.startup.assert_called_once_with(self.arm, self.scan, self.settings)
        self.assertEqual(cycle.call_args.args[-1], demo.MAXIMUM_HAND_AGE_SECONDS)
        self.assertEqual(shutdown.mock_calls, [
            mock.call.zero(self.arm, context="shutdown zero return", verbose=False, open_grip_mm=40.0, open_seconds=0.2),
            mock.call.disconnect(),
        ])
        self.arm.disconnect.assert_called_once()

    def test_main_forwards_verbosity_to_connection_and_shutdown(self):
        for verbose in (False, True):
            self.zero.reset_mock()
            with self.subTest(verbose=verbose), mock.patch.object(
                demo, "parse_args", return_value=self.args(verbose=verbose, max_cycles=1)
            ), mock.patch.object(
                demo, "load_hand_eye", return_value={"camera": {"serial": "D435"}}
            ), mock.patch.object(demo, "check_services"), mock.patch.object(
                demo, "load_setup", return_value=(self.settings, np.eye(4), [self.scan])
            ), mock.patch.object(demo, "connect_robot", return_value=self.arm) as connect, \
                    mock.patch.object(demo, "run_cycle"), mock.patch.object(demo.time, "sleep"):
                self.assertEqual(demo.main(), 0)
            connect.assert_called_once_with(iface="can_piper", verbose=verbose)
            self.zero.assert_called_once_with(self.arm, context="shutdown zero return", verbose=verbose, open_grip_mm=40.0, open_seconds=0.2)

    def test_failed_startup_stops_before_cycle_then_attempts_zero_and_disconnects(self):
        self.startup.side_effect = demo.DemoError("startup zero failed")
        with mock.patch.object(demo, "parse_args", return_value=self.args()), \
                mock.patch.object(demo, "load_hand_eye", return_value={"camera": {"serial": "D435"}}), \
                mock.patch.object(demo, "check_services"), mock.patch.object(
                    demo, "load_setup", return_value=(self.settings, np.eye(4), [self.scan])
                ), mock.patch.object(demo, "connect_robot", return_value=self.arm), \
                mock.patch.object(demo, "run_cycle") as cycle, \
                mock.patch.object(demo.time, "sleep"):
            self.assertEqual(demo.main(), 2)
        cycle.assert_not_called()
        self.zero.assert_called_once_with(self.arm, context="shutdown zero return", verbose=False, open_grip_mm=40.0, open_seconds=0.2)
        self.arm.disconnect.assert_called_once()

    def test_error_or_interrupt_stops_loop_without_release(self):
        for failure, code in ((demo.DemoError("palm lost"), 2), (KeyboardInterrupt(), 130)):
            with self.subTest(failure=failure), mock.patch.object(
                demo, "parse_args", return_value=self.args()
            ), mock.patch.object(
                demo, "load_hand_eye", return_value={"camera": {"serial": "D435"}}
            ), mock.patch.object(demo, "check_services"), mock.patch.object(
                demo, "load_setup", return_value=(self.settings, np.eye(4), [self.scan])
            ), mock.patch.object(demo, "connect_robot", return_value=self.arm), \
                    mock.patch.object(demo, "run_cycle", side_effect=failure) as cycle, \
                    mock.patch.object(demo, "hold_gripper") as release, \
                    mock.patch.object(demo, "return_to_zero") as zero, \
                    mock.patch.object(demo.time, "sleep"):
                self.assertEqual(demo.main(), code)
            self.assertEqual(cycle.call_count, 1)
            release.assert_not_called()
            zero.assert_called_once_with(self.arm, context="shutdown zero return", verbose=False, open_grip_mm=40.0, open_seconds=0.2)

    def test_interrupted_or_failed_zero_return_reports_failure_and_disconnects(self):
        for cycle_failure, failure, code in (
            (None, demo.DemoError("unsafe path"), 2),
            (None, KeyboardInterrupt(), 130),
            (KeyboardInterrupt(), demo.DemoError("unsafe path"), 130),
            (KeyboardInterrupt(), KeyboardInterrupt(), 130),
        ):
            with self.subTest(cycle_failure=cycle_failure, failure=failure), mock.patch.object(
                demo, "parse_args", return_value=self.args()
            ), mock.patch.object(
                demo, "load_hand_eye", return_value={"camera": {"serial": "D435"}}
            ), mock.patch.object(demo, "check_services"), mock.patch.object(
                demo, "load_setup", return_value=(self.settings, np.eye(4), [self.scan])
            ), mock.patch.object(demo, "connect_robot", return_value=self.arm), \
                    mock.patch.object(demo, "run_cycle", side_effect=cycle_failure), \
                    mock.patch.object(demo, "return_to_zero", side_effect=failure) as zero, \
                    mock.patch.object(demo.time, "sleep"), \
                    mock.patch("builtins.print") as output:
                self.assertEqual(demo.main(), code)
            zero.assert_called_once_with(self.arm, context="shutdown zero return", verbose=False, open_grip_mm=40.0, open_seconds=0.2)
            self.assertTrue(any("disconnecting without further motion" in str(call) for call in output.call_args_list))
        self.assertEqual(self.arm.disconnect.call_count, 4)

    def test_interrupt_before_connection_does_not_attempt_zero_return(self):
        with mock.patch.object(demo, "parse_args", return_value=self.args()), \
                mock.patch.object(demo, "load_hand_eye", side_effect=KeyboardInterrupt()), \
                mock.patch.object(demo, "connect_robot") as connect, \
                mock.patch.object(demo, "return_to_zero") as zero:
            self.assertEqual(demo.main(), 130)
        connect.assert_not_called()
        zero.assert_not_called()

    def test_runtime_settings_exclude_obsolete_candy_bounds_and_config_argument(self):
        self.assertNotIn("target_workspace_min_mm", demo.DEMO_SETTINGS)
        self.assertNotIn("target_workspace_max_mm", demo.DEMO_SETTINGS)
        with mock.patch.object(sys, "argv", ["demo.py"]):
            self.assertFalse(hasattr(demo.parse_args(), "config"))
        with mock.patch.object(sys, "argv", ["demo.py", "--config", "config.json"]), self.assertRaises(SystemExit):
            demo.parse_args()

    def test_help_works_outside_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(Path(demo.__file__)), "--help"],
                cwd=directory, text=True, capture_output=True, timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--check-only", result.stdout)
        self.assertIn("--allow-hand-delivery", result.stdout)


if __name__ == "__main__":
    unittest.main()
