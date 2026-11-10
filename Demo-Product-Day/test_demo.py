"""Hardware-free tests for demo.py perception geometry and motion planning."""

import json
import math
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import demo
from piper import config as C
from piper.transforms import euler_to_rot, tip_position

sys.path.insert(0, str(demo.PROJECT_DIR / "k3s-vision-service"))
from vision import VisionError, robust_box_depth


class DemoDepthTests(unittest.TestCase):
    def test_shared_paths_follow_repository_layout(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(demo.PROJECT_DIR, root)
        self.assertEqual(
            demo.PIPER_DIR, root / "Demo-VLA-On-Piper" / "piper-openpi-V2"
        )
        self.assertTrue(demo.PIPER_DIR.is_dir())
        self.assertEqual(
            demo.DEFAULT_CONFIG, root / "prepare-piper" / "hand-eye-realsense405" / "config.json"
        )
        self.assertTrue(demo.DEFAULT_CONFIG.is_file())
        self.assertEqual(
            demo.DEFAULT_SCAN_POSES,
            root / "prepare-piper" / "scan-poses" / "scan_poses.json",
        )

    def test_entrypoints_show_help_outside_repository(self) -> None:
        demo_dir = Path(__file__).resolve().parent
        scripts = [demo_dir / "demo.py"] + [
            demo.PROJECT_DIR / "prepare-piper" / "Teach-scripts" / name
            for name in (
                "teach_scan_poses.py",
                "teach_candy_grip.py",
                "teach_release_height.py",
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            for script in scripts:
                with self.subTest(script=script.name):
                    result = subprocess.run(
                        [sys.executable, str(script), "--help"],
                        cwd=directory,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)

    def test_active_grasp_height_is_above_floor(self) -> None:
        config = demo.validate_demo_config(
            json.loads(demo.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        )

        self.assertEqual(config["grasp_tip_z_mm"], 9.5)
        self.assertGreater(config["grasp_tip_z_mm"], C.TIP_Z_FLOOR_MM)

    def test_object_and_hand_workspaces_use_expanded_y_range(self) -> None:
        config = demo.validate_demo_config(
            json.loads(demo.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        )

        np.testing.assert_allclose(
            config["target_workspace_min_mm"], [100.0, -400.0, -50.0]
        )
        np.testing.assert_allclose(
            config["target_workspace_max_mm"], [500.0, 400.0, 100.0]
        )
        np.testing.assert_allclose(
            config["hand_workspace_min_mm"], [100.0, -400.0, -30.0]
        )
        np.testing.assert_allclose(
            config["hand_workspace_max_mm"], [500.0, 400.0, 30.0]
        )

    def test_pick_motion_speed_is_fifteen_percent(self) -> None:
        config = json.loads(demo.DEFAULT_CONFIG.read_text(encoding="utf-8"))

        parsed = demo.validate_demo_config(config)

        self.assertEqual(parsed["pick_motion_speed_percent"], 15)

    def test_demo_allows_one_grasp_retry(self) -> None:
        config = json.loads(demo.DEFAULT_CONFIG.read_text(encoding="utf-8"))

        parsed = demo.validate_demo_config(config)

        self.assertEqual(parsed["maximum_grasp_retries"], 1)

    def test_hand_reacquisition_displacement_limit_is_disabled(self) -> None:
        config = json.loads(demo.DEFAULT_CONFIG.read_text(encoding="utf-8"))

        parsed = demo.validate_demo_config(config)

        self.assertFalse(parsed["enforce_hand_reacquisition_limit"])

    def test_demo_motion_speed_is_twenty_percent(self) -> None:
        config = json.loads(demo.DEFAULT_CONFIG.read_text(encoding="utf-8"))

        parsed = demo.validate_demo_config(config)

        self.assertEqual(parsed["motion_speed_percent"], 20)

    def test_rejects_motion_speed_above_one_hundred_percent(self) -> None:
        config = json.loads(demo.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        config["demo"]["motion_speed_percent"] = 101

        with self.assertRaisesRegex(demo.DemoError, "integer from 1 to 100"):
            demo.validate_demo_config(config)

    def test_scan_pair_captures_object_and_hand_concurrently(self) -> None:
        object_started = threading.Event()
        hand_started = threading.Event()

        class ObjectClient:
            base_url = "http://vision"

            def set_workspace_overlay(self, *_args) -> None:
                pass

            def capture(self, slot, prompt, confidence):
                object_started.set()
                self.overlapped = hand_started.wait(timeout=1.0)
                return {
                    "status": "not_found",
                    "slot": slot,
                    "prompt": prompt,
                    "confidence": confidence,
                    "snapshot_url": "/snapshots/1.jpg",
                }

        class HandClient:
            base_url = "http://hand"

            def capture_hand(self, slot):
                hand_started.set()
                self.overlapped = object_started.wait(timeout=1.0)
                return {
                    "status": "not_found",
                    "slot": slot,
                    "snapshot_url": "/snapshots/4.jpg",
                }

        arm = mock.Mock()
        arm.pose.return_value = np.zeros(6)
        object_client = ObjectClient()
        hand_client = HandClient()
        settings = json.loads(
            demo.DEFAULT_CONFIG.read_text(encoding="utf-8")
        )["demo"]
        settings["scan_settle_seconds"] = 0.0

        with mock.patch.object(demo, "move_checked") as move:
            object_result, hand_result = demo.capture_scan_pair(
                arm,
                object_client,
                hand_client,
                {"index": 1, "joints_deg": np.zeros(6)},
                4,
                "starburst candy",
                0.05,
                np.eye(4),
                settings,
            )

        move.assert_called_once()
        self.assertEqual(move.call_args.args[-1], 20)
        self.assertTrue(object_client.overlapped)
        self.assertTrue(hand_client.overlapped)
        self.assertEqual(object_result["status"], "not_found")
        self.assertEqual(hand_result["status"], "not_found")
        self.assertEqual(arm.pose.call_count, 2)
        timing = object_result["scan_timing_seconds"]
        self.assertIs(timing, hand_result["scan_timing_seconds"])
        self.assertGreaterEqual(timing["move_seconds"], 0.0)
        self.assertGreaterEqual(timing["settle_seconds"], 0.0)
        self.assertGreaterEqual(timing["object_inference_seconds"], 0.0)
        self.assertGreaterEqual(timing["hand_inference_seconds"], 0.0)
        self.assertGreaterEqual(timing["concurrent_inference_seconds"], 0.0)
        self.assertGreaterEqual(timing["total_pose_seconds"], 0.0)

    def test_allows_recorded_absolute_hand_release_height_in_config(self) -> None:
        config = json.loads(demo.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        config["demo"]["hand_release_tip_z_mm"] = 82.0

        parsed = demo.validate_demo_config(config)

        self.assertEqual(parsed["hand_release_tip_z_mm"], 82.0)

    def test_rejects_unsafe_clearance_from_actual_detected_palm(self) -> None:
        with self.assertRaisesRegex(demo.DemoError, "is not above"):
            demo.validate_detected_palm_release_clearance(73.0, 80.0)

    def test_accepts_taught_height_above_detected_palm(self) -> None:
        clearance = demo.validate_detected_palm_release_clearance(73.0, -7.0)

        self.assertEqual(clearance, 80.0)

    def test_robust_box_depth_selects_dominant_cluster(self) -> None:
        depth = np.full((40, 50), 600, dtype=np.uint16)
        depth[10:30, 15:35] = 300
        depth[16:18, 22:24] = 900

        pixel, depth_mm, diagnostics = robust_box_depth(
            depth,
            [10, 5, 40, 35],
            depth_scale=0.001,
            inset_fraction=0.2,
            minimum_pixels=100,
            minimum_depth_m=0.05,
            maximum_depth_m=1.0,
            cluster_bin_mm=10.0,
            cluster_tolerance_mm=15.0,
            minimum_cluster_fraction=0.5,
        )

        np.testing.assert_allclose(pixel, (24.5, 19.5), atol=0.5)
        self.assertAlmostEqual(depth_mm, 300.0)
        self.assertGreater(diagnostics["depth_cluster_fraction"], 0.9)

    def test_robust_box_depth_rejects_too_few_pixels(self) -> None:
        depth = np.zeros((20, 20), dtype=np.uint16)
        depth[9:11, 9:11] = 300

        with self.assertRaisesRegex(VisionError, "valid depth pixels"):
            robust_box_depth(
                depth,
                [0, 0, 20, 20],
                depth_scale=0.001,
                inset_fraction=0.1,
                minimum_pixels=10,
                minimum_depth_m=0.05,
                maximum_depth_m=1.0,
                cluster_bin_mm=10.0,
                cluster_tolerance_mm=15.0,
                minimum_cluster_fraction=0.5,
            )

    def test_target_workspace_is_enforced(self) -> None:
        minimum = np.array([150.0, -250.0, -30.0])
        maximum = np.array([450.0, 250.0, 100.0])
        demo.validate_target_workspace(np.array([300.0, 0.0, 20.0]), minimum, maximum)
        with self.assertRaisesRegex(demo.DemoError, "outside"):
            demo.validate_target_workspace(
                np.array([500.0, 0.0, 20.0]), minimum, maximum
            )

    def test_scan_file_rejects_pose_outside_can_margin(self) -> None:
        document = {
            "pose_count": 3,
            "poses": [
                {
                    "index": index,
                    "joints_deg": [index * 10, 45, -25, 0, 75, 0],
                    "end_pose_mm_deg": [200, 0, 200, 180, 20, 180],
                }
                for index in (1, 2, 3)
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(
                demo.DemoError, "scan pose 1 is not safely replayable"
            ):
                demo.load_scan_poses(path)

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

    def test_scan_file_rejects_invalid_counts_and_mismatches(self) -> None:
        pose = {
            "index": 1, "joints_deg": [10, 45, -25, 0, 40, 0],
            "end_pose_mm_deg": [200, 0, 200, 180, 20, 180],
        }
        for count, poses in (
            (0, []), (4, [pose] * 4), (True, [pose]), (1.0, [pose]),
            (2, [pose]), (1, [pose, pose]), (1, None),
        ):
            with self.subTest(count=count, poses=poses), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "poses.json"
                path.write_text(json.dumps({"pose_count": count, "poses": poses}))
                with self.assertRaisesRegex(demo.DemoError, "1..3 taught scan poses"):
                    demo.load_scan_poses(path)

    def test_two_scan_poses_still_require_separation_and_order(self) -> None:
        first = {
            "index": 1, "joints_deg": [10, 45, -25, 0, 40, 0],
            "end_pose_mm_deg": [200, 0, 200, 180, 20, 180],
        }
        for second, error in (
            ({**first, "index": 2}, "not distinct"),
            ({**first, "index": 3}, "out of order"),
        ):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "poses.json"
                path.write_text(json.dumps({"pose_count": 2, "poses": [first, second]}))
                with self.assertRaisesRegex(demo.DemoError, error):
                    demo.load_scan_poses(path)


class DemoPlanningTests(unittest.TestCase):
    def test_adaptive_inward_transit_reaches_j5_limited_candy(self) -> None:
        config = demo.validate_demo_config(
            json.loads(demo.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        )
        scans = demo.load_scan_poses(demo.DEFAULT_SCAN_POSES)
        target = np.array([384.0359, -271.6435, -23.4940])

        plan = demo.plan_pick_and_replace(
            target,
            scans[-1]["joints_deg"],
            scans[0]["joints_deg"],
            np.array([180.0, 20.0, 214.5121]),
            float(config["grasp_tip_z_mm"]),
            float(config["approach_height_mm"]),
            float(config["pregrasp_height_mm"]),
            float(config["minimum_pregrasp_height_mm"]),
            float(config["pregrasp_height_step_mm"]),
            float(config["maximum_approach_inward_offset_mm"]),
            float(config["approach_inward_step_mm"]),
        )

        self.assertEqual(
            [name for name, _ in plan],
            [
                "pick transit",
                "pick approach",
                "grasp",
                "lift",
                "lift transit",
                "replace",
                "retreat",
                "retreat transit",
                "scan 1",
            ],
        )
        waypoints = dict(plan)
        transit_tip = tip_position(waypoints["pick transit"])
        approach_tip = tip_position(waypoints["pick approach"])
        grasp_tip = tip_position(waypoints["grasp"])
        self.assertLess(
            np.linalg.norm(transit_tip[:2]),
            np.linalg.norm(target[:2]),
        )
        grasp_height = float(config["grasp_tip_z_mm"])
        self.assertAlmostEqual(transit_tip[2], grasp_height + 100.0, delta=1.0)
        self.assertAlmostEqual(approach_tip[2], grasp_height + 50.0, delta=1.0)
        self.assertAlmostEqual(grasp_tip[2], grasp_height, delta=1.0)
        np.testing.assert_allclose(
            waypoints["lift transit"],
            waypoints["pick transit"],
        )
        np.testing.assert_allclose(
            waypoints["retreat transit"],
            waypoints["pick transit"],
        )

    def test_adaptive_pregrasp_lowers_for_outer_workspace_candy(self) -> None:
        config = demo.validate_demo_config(
            json.loads(demo.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        )
        scans = demo.load_scan_poses(demo.DEFAULT_SCAN_POSES)

        plan = demo.plan_pick_and_replace(
            np.array([441.3, -341.5, -42.2]),
            scans[-1]["joints_deg"],
            scans[0]["joints_deg"],
            np.array([180.0, 20.0, 180.76]),
            float(config["grasp_tip_z_mm"]),
            float(config["approach_height_mm"]),
            float(config["pregrasp_height_mm"]),
            float(config["minimum_pregrasp_height_mm"]),
            float(config["pregrasp_height_step_mm"]),
            float(config["maximum_approach_inward_offset_mm"]),
            float(config["approach_inward_step_mm"]),
        )

        waypoints = dict(plan)
        self.assertAlmostEqual(
            tip_position(waypoints["pick approach"])[2],
            float(config["grasp_tip_z_mm"]) + 40.0,
            delta=1.0,
        )
        self.assertLess(
            np.linalg.norm(tip_position(waypoints["pick transit"])[:2]),
            np.linalg.norm(np.array([441.3, -341.5])),
        )
        np.testing.assert_allclose(waypoints["lift"], waypoints["pick approach"])
        np.testing.assert_allclose(
            waypoints["retreat"], waypoints["pick approach"]
        )

    def test_equivalent_grasp_yaw_reaches_positive_y_candy(self) -> None:
        config = demo.validate_demo_config(
            json.loads(demo.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        )
        scans = demo.load_scan_poses(demo.DEFAULT_SCAN_POSES)
        target = {
            "base_point_mm": [382.0103, 297.3192, -22.2622],
            "base_distance_mm": 484.0771,
            "cross_view_agreement_passed": True,
            "long_axis_base_unit_xy": [0.2985727, -0.9543869],
        }

        selected = demo.select_reachable_object_target(
            [target],
            scans[-1]["joints_deg"],
            scans[0]["joints_deg"],
            np.asarray(config["grasp_rpy_deg"]),
            float(config["grasp_tip_z_mm"]),
            float(config["approach_height_mm"]),
            float(config["pregrasp_height_mm"]),
            float(config["minimum_pregrasp_height_mm"]),
            float(config["pregrasp_height_step_mm"]),
            float(config["maximum_approach_inward_offset_mm"]),
            float(config["approach_inward_step_mm"]),
        )

        self.assertIsNotNone(selected)
        selected_target, aligned_rpy, plan, rejections = selected
        self.assertIs(selected_target, target)
        self.assertAlmostEqual(aligned_rpy[2], -72.6281, places=3)
        self.assertIn("pick transit", dict(plan))
        self.assertEqual(rejections, [])

    def test_thirty_mm_pregrasp_reaches_outer_positive_y_candy(self) -> None:
        config = demo.validate_demo_config(
            json.loads(demo.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        )
        scans = demo.load_scan_poses(demo.DEFAULT_SCAN_POSES)

        plan = demo.plan_pick_and_replace(
            np.array([386.76, 299.55, -32.21]),
            scans[-1]["joints_deg"],
            scans[0]["joints_deg"],
            np.array([180.0, 20.0, -71.67]),
            float(config["grasp_tip_z_mm"]),
            float(config["approach_height_mm"]),
            float(config["pregrasp_height_mm"]),
            float(config["minimum_pregrasp_height_mm"]),
            float(config["pregrasp_height_step_mm"]),
            float(config["maximum_approach_inward_offset_mm"]),
            float(config["approach_inward_step_mm"]),
        )

        approach_tip = tip_position(dict(plan)["pick approach"])
        self.assertAlmostEqual(
            approach_tip[2],
            float(config["grasp_tip_z_mm"]) + 30.0,
            delta=1.0,
        )

    def test_bounded_yaw_offset_reaches_far_positive_y_candy(self) -> None:
        config = demo.validate_demo_config(
            json.loads(demo.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        )
        scans = demo.load_scan_poses(demo.DEFAULT_SCAN_POSES)
        target = {
            "base_point_mm": [409.65, 295.45, -30.60],
            "base_distance_mm": 505.1,
            "cross_view_agreement_passed": True,
            "long_axis_base_unit_xy": [
                math.cos(math.radians(-63.468)),
                math.sin(math.radians(-63.468)),
            ],
        }

        selected = demo.select_reachable_object_target(
            [target],
            scans[-1]["joints_deg"],
            scans[0]["joints_deg"],
            np.asarray(config["grasp_rpy_deg"]),
            float(config["grasp_tip_z_mm"]),
            float(config["approach_height_mm"]),
            float(config["pregrasp_height_mm"]),
            float(config["minimum_pregrasp_height_mm"]),
            float(config["pregrasp_height_step_mm"]),
            float(config["maximum_approach_inward_offset_mm"]),
            float(config["approach_inward_step_mm"]),
            float(config["maximum_grasp_yaw_offset_deg"]),
            float(config["grasp_yaw_offset_step_deg"]),
        )

        self.assertIsNotNone(selected)
        _target, aligned_rpy, plan, rejections = selected
        self.assertAlmostEqual(aligned_rpy[2], -83.468, places=3)
        self.assertIn("pick transit", dict(plan))
        self.assertEqual(rejections, [])

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
            "gripper_hold_seconds": 1.0,
        }
        moves = []
        with (
            mock.patch.object(
                demo,
                "move_checked",
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

    def test_return_held_object_reverses_adaptive_transit(self) -> None:
        plan = [
            ("lift", np.ones(6)),
            ("lift transit", np.zeros(6)),
            ("replace", np.full(6, 2.0)),
            ("retreat", np.ones(6)),
            ("retreat transit", np.zeros(6)),
            ("scan 1", np.full(6, 3.0)),
        ]
        config = {
            "motion_speed_percent": 20,
            "move_timeout_seconds": 8.0,
            "grip_open_mm": 70.0,
            "grip_close_mm": 0.0,
            "gripper_hold_seconds": 1.0,
        }
        moves = []
        with (
            mock.patch.object(
                demo,
                "move_checked",
                side_effect=lambda _arm, name, *_args: moves.append(name),
            ),
            mock.patch.object(demo, "hold_gripper"),
        ):
            demo.return_held_object(object(), plan, config, "retry")

        self.assertEqual(
            moves,
            [
                "retry transit approach",
                "retry approach",
                "retry replace",
                "retry retreat",
                "retry retreat transit",
                "return to scan 1",
            ],
        )

    def test_highest_confidence_object_capture_selects_valid_slot(self) -> None:
        captures = [
            {"status": "detected", "candidate_count": 1, "score": 0.4, "slot": 1},
            {"status": "rejected", "candidate_count": 0, "score": 0.9, "slot": 2},
            {"status": "detected", "candidate_count": 2, "score": 0.8, "slot": 3},
        ]

        selected = demo.highest_confidence_object_capture(captures)

        self.assertEqual(selected["slot"], 3)

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
            "gripper_hold_seconds": 1.0,
        }
        moves = []
        holds = []
        with (
            mock.patch.object(
                demo,
                "move_checked",
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

    def test_submillimeter_grasp_width_continues(self) -> None:
        arm = mock.Mock()
        arm.grip_mm.return_value = 0.4
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
            "gripper_hold_seconds": 1.0,
        }

        with (
            mock.patch.object(demo, "move_checked"),
            mock.patch.object(demo, "hold_gripper"),
        ):
            held_width = demo.execute_pickup(arm, plan, config)

        self.assertEqual(held_width, 0.4)

    def test_flange_for_tip_round_trip(self) -> None:
        rotation = euler_to_rot([180.0, 20.0, 180.0])
        tip = np.array([300.0, 0.0, 20.0])
        flange = demo.flange_for_tip(tip, rotation)
        reconstructed = flange + rotation.apply([0.0, 0.0, C.TOOL_LEN_MM])
        np.testing.assert_allclose(reconstructed, tip)

    def test_plan_pick_and_replace_reuses_exact_grasp_location(self) -> None:
        grasp_rpy = np.array([180.0, 20.0, 180.0])

        plan = demo.plan_pick_and_replace(
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
                "replace",
                "retreat",
                "scan 1",
            ],
        )
        waypoints = dict(plan)
        np.testing.assert_allclose(waypoints["replace"], waypoints["grasp"])
        np.testing.assert_allclose(waypoints["retreat"], waypoints["pick approach"])
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

    def test_plan_rejects_unreachable_target(self) -> None:
        with self.assertRaises(demo.DemoError):
            demo.plan_pick_and_replace(
                target_base_mm=np.array([300.0, 0.0, 20.0]),
                start_joints=np.asarray(C.READY_JOINTS_DEG),
                return_joints=np.asarray(C.READY_JOINTS_DEG),
                grasp_rpy=np.array([180.0, 20.0, 180.0]),
                grasp_tip_z_mm=-1000.0,
                approach_height_mm=100.0,
            )

    def test_execute_pick_and_replace_sequence(self) -> None:
        class FakeArm:
            def grip_mm(self):
                return 25.0

        approach = np.zeros(6, dtype=float)
        grasp = np.ones(6, dtype=float)
        plan = [
            ("pick approach", approach),
            ("grasp", grasp),
            ("lift", approach.copy()),
            ("replace", grasp.copy()),
            ("retreat", approach.copy()),
            ("scan 1", np.full(6, 2.0)),
        ]
        config = {
            "motion_speed_percent": 20,
            "move_timeout_seconds": 8.0,
            "grip_open_mm": 70.0,
            "grip_close_mm": 0.0,
            "gripper_hold_seconds": 1.0,
        }
        moves = []
        holds = []
        with (
            mock.patch.object(
                demo,
                "move_checked",
                side_effect=lambda _arm, name, joints, grip, _timeout, speed: moves.append(
                    (name, joints.copy(), grip, speed)
                ),
            ),
            mock.patch.object(
                demo,
                "hold_gripper",
                side_effect=lambda _arm, joints, grip, _seconds: holds.append(
                    (joints.copy(), grip)
                ),
            ),
        ):
            result = demo.execute_pick_and_replace(FakeArm(), plan, config)

        self.assertEqual(
            [name for name, _, _, _ in moves],
            ["pick approach", "grasp", "lift", "replace", "retreat", "return to scan 1"],
        )
        self.assertEqual([speed for _, _, _, speed in moves], [20] * 6)
        self.assertEqual([grip for _, grip in holds], [0.0, 70.0])
        np.testing.assert_allclose(moves[3][1], dict(plan)["grasp"])
        self.assertEqual(result["gripper_width_after_lift_mm"], 25.0)

    def test_return_held_object_replaces_and_returns_to_scan(self) -> None:
        plan = [
            ("lift", np.zeros(6)),
            ("replace", np.ones(6)),
            ("retreat", np.full(6, 2.0)),
            ("scan 1", np.full(6, 3.0)),
        ]
        config = {
            "motion_speed_percent": 20,
            "move_timeout_seconds": 8.0,
            "grip_open_mm": 70.0,
            "grip_close_mm": 0.0,
            "gripper_hold_seconds": 1.0,
        }
        moves = []
        holds = []
        released = []
        with (
            mock.patch.object(
                demo,
                "move_checked",
                side_effect=lambda _arm, name, joints, grip, _timeout, speed: moves.append(
                    (name, joints.copy(), grip, speed)
                ),
            ),
            mock.patch.object(
                demo,
                "hold_gripper",
                side_effect=lambda _arm, joints, grip, _seconds: holds.append(
                    (joints.copy(), grip)
                ),
            ),
        ):
            demo.return_held_object(
                object(),
                plan,
                config,
                "no-palm return",
                on_release=lambda: released.append(True),
            )

        self.assertEqual(
            [name for name, _, _, _ in moves],
            [
                "no-palm return approach",
                "no-palm return replace",
                "no-palm return retreat",
                "return to scan 1",
            ],
        )
        self.assertEqual([speed for _, _, _, speed in moves], [20] * 4)
        self.assertEqual([grip for _, grip in holds], [70.0])
        self.assertEqual(released, [True])
        np.testing.assert_allclose(holds[0][0], dict(plan)["replace"])

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

    def test_hand_delivery_marks_release_before_retreat(self) -> None:
        plan = [
            ("hand approach", np.zeros(6)),
            ("hand release", np.ones(6)),
            ("hand retreat", np.full(6, 2.0)),
            ("scan 1", np.full(6, 3.0)),
        ]
        config = {
            "motion_speed_percent": 20,
            "move_timeout_seconds": 8.0,
            "grip_open_mm": 70.0,
            "grip_close_mm": 0.0,
            "gripper_hold_seconds": 1.0,
        }
        released = []
        moves = []
        with (
            mock.patch.object(
                demo,
                "move_checked",
                side_effect=lambda _arm, name, *_args: moves.append(name),
            ),
            mock.patch.object(demo, "hold_gripper"),
        ):
            demo.execute_hand_delivery(
                object(),
                plan,
                config,
                on_release=lambda: released.append(list(moves)),
            )

        self.assertEqual(released, [["hand approach", "hand release"]])
        self.assertEqual(
            moves,
            ["hand approach", "hand release", "hand retreat", "return to scan 1"],
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

    def test_palm_cross_view_agreement_ignores_z_difference(self) -> None:
        captures = [
            {
                "status": "detected",
                "slot": 4,
                "score": 0.928,
                "base_point_mm": [428.8, -169.9, -25.0],
            },
            {
                "status": "detected",
                "slot": 5,
                "score": 0.958,
                "base_point_mm": [415.2, -171.1, 17.2],
            },
        ]

        target = demo.choose_target_if_detected(
            captures,
            maximum_difference=25.0,
            label="stable open palm",
            comparison_axes=(0, 1),
        )

        self.assertIsNotNone(target)
        self.assertTrue(target["cross_view_agreement_passed"])
        self.assertEqual(target["cross_view_comparison_axes"], ["x", "y"])
        self.assertAlmostEqual(
            target["maximum_cross_view_difference_mm"],
            13.65,
            delta=0.1,
        )

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

    def test_choose_target_if_detected_returns_none_for_scan_miss(self) -> None:
        target = demo.choose_target_if_detected(
            [
                {"status": "not_found", "slot": 1},
                {"status": "rejected", "slot": 2},
                {"status": "not_found", "slot": 3},
            ],
            maximum_difference=25.0,
            label="Starburst candy",
        )

        self.assertIsNone(target)

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

    def test_select_reachable_target_skips_unreachable_nearest(self) -> None:
        targets = [
            {
                "base_point_mm": [200.0, 0.0, -30.0],
                "base_distance_mm": 200.0,
                "cross_view_agreement_passed": True,
                "long_axis_base_unit_xy": [1.0, 0.0],
            },
            {
                "base_point_mm": [300.0, 0.0, -30.0],
                "base_distance_mm": 300.0,
                "cross_view_agreement_passed": True,
                "long_axis_base_unit_xy": [1.0, 0.0],
            },
        ]
        reachable_plan = [("grasp", np.ones(6))]

        def plan(target_base_mm, *_args):
            if target_base_mm[0] < 250.0:
                raise demo.DemoError("unreachable")
            return reachable_plan

        with mock.patch.object(demo, "plan_pick_and_replace", side_effect=plan):
            selected = demo.select_reachable_object_target(
                targets,
                np.zeros(6),
                np.zeros(6),
                np.array([180.0, 20.0, 180.0]),
                10.5,
                100.0,
            )

        self.assertIsNotNone(selected)
        target, _rpy, plan_result, rejections = selected
        self.assertEqual(target["base_point_mm"][0], 300.0)
        self.assertIs(plan_result, reachable_plan)
        self.assertEqual(len(rejections), 1)
        self.assertIn("unreachable", rejections[0]["reason"])
        self.assertIn(
            "all bounded candy-aligned grasp orientations failed",
            targets[0]["planning_rejection_reason"],
        )


if __name__ == "__main__":
    unittest.main()
