#!/usr/bin/env python3
"""Supervised D435 Starburst pickup and delivery loop."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable

import numpy as np
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

#vision_client interfaces with the camera service, the model service, and provides convenient clients for both.
from vision_client import (  # noqa: E402
    DEFAULT_DETECTION_URL,
    DEFAULT_VISION_URL,
    DemoError,
    HandClient,
    HttpClient,
    VisionClient,
)
from opencv_detector import OpenCVCandyDetector, candy_size_mm, check_candy_size  # noqa: E402

#The piper module provides interfaces for controlling the robotic arm, including kinematics, robot state, and transformations.
from piper import config as PIPER_C  # noqa: E402
from piper.kinematics import MARGINED_LIMITS, fk, ik  # noqa: E402
from piper.robot import Piper, connect_robot  # noqa: E402
from piper.transforms import euler_to_rot, fk_flange, tip_position  # noqa: E402

#camera to robot calbibrations
CALIBRATION_DIR = ROOT / "prepare-piper/hand-eye-realsense435"
RELEASE_CLEARANCE_MM = 76.2
# A move counts as arrived when every joint is within this many degrees of its target.
ARRIVAL_TOLERANCE_DEG = 2.0
MAXIMUM_HAND_AGE_SECONDS = 10.0
MAXIMUM_CANDY_XY_CORRECTION_MM = 60.0
MAXIMUM_GRASP_TILT_OVERRIDE_DEG = 30.0
GRASP_OFFSET_LOG = Path(__file__).resolve().parent / "logs" / "grasp_offsets.jsonl"
PROMPT = "starburst candy"
DEFAULT_SCAN_POSES = ROOT / "prepare-piper/scan-poses/scan_poses.json"
WORKSPACE_CORNERS_MM = (
    (144.644, 206.317, 18.0),  # C2
    (144.0, -170.0, 18.0),  # C4
    (570.0, -170.0, 18.0),  # C6
    (570.0, 206.0, 18.0),  # C8
)
WORKSPACE_MIN_Z_MM = 17.423
WORKSPACE_MAX_Z_MM = 263.323
WORKSPACE_MIN_MM = np.array([
    min(point[0] for point in WORKSPACE_CORNERS_MM),
    min(point[1] for point in WORKSPACE_CORNERS_MM),
    WORKSPACE_MIN_Z_MM,
])
WORKSPACE_MAX_MM = np.array([
    max(point[0] for point in WORKSPACE_CORNERS_MM),
    max(point[1] for point in WORKSPACE_CORNERS_MM),
    WORKSPACE_MAX_Z_MM,
])
DEMO_SETTINGS = {
    # Re-taught 2026-11-04 (candy_grip.json): candy centred between the open jaws.
    "grasp_tip_z_mm": 16.584,
    "grip_open_mm": 60.000,
    "grip_close_mm": 5.000,
    # Measured jaw width after lift; below this the gripper is assumed empty.
    # An empty gripper closes to ~5 mm; a held Starburst measured 20.9-22.4 mm.
    "minimum_held_grip_mm": 8.000,
    # Roll/pitch 180/20 (20 degrees from vertical) instead of the hand-held teaching
    # tilt (~9 degrees): offline planning reaches ~98% of candy angles out to X 500 mm
    # versus ~54% for the taught tilt. Yaw is replaced by candy alignment at runtime.
    "grasp_rpy_deg": [180.000, 20.000, 123.142],
    # When the tilted grasp's fingertips would point within ~45 degrees of +X (away
    # from the base; candies lying along X), grasp vertically if reachable instead.
    # 2026-11-05/06: forward leans picked 2/18 and toppled candies; vertical picked 3/3.
    "forward_lean_vertical_threshold": 0.700,
    "motion_speed_percent": 10,
    "pick_motion_speed_percent": 10,
    "approach_height_mm": 100.000,
    "pregrasp_height_mm": 50.000,
    "hand_approach_height_mm": 20.000,
    # Pitch raised from the taught 10.012 degrees: at 10 degrees J5 (wrist pitch,
    # +/-70 degrees) limited palm reach to X ~410-440 mm; 20 degrees reaches X 500 mm.
    "hand_delivery_rpy_deg": [178.517, 20.000, 171.532],
    "approach_inward_step_mm": 20.000,
    # YOLO picks scored 0.05-0.49 were gripper-occluded or bad-depth detections while
    # real candies scored 0.84-0.99. OpenCV's fill score is already >= 0.6.
    "detection_confidence_threshold": 0.500,
    # Candy centres sit near the table plane (Z ~ -3 mm); real candies measured
    # -12..+7 mm while a bad depth reading put one detection at Z -348 mm.
    "candy_min_z_mm": -30.000,
    "candy_max_z_mm": 30.000,
    # Added to every detected candy's base XY before planning. Measure it with
    # --pause-before-grip, which prints the suggested value after each reading.
    "candy_xy_correction_mm": [0.000, 0.000],
    "grasp_yaw_offset_step_deg": 5.000,
    "gripper_hold_seconds": 1.000,
    "maximum_approach_inward_offset_mm": 120.000,
    "maximum_cross_view_orientation_difference_deg": 20.000,
    "maximum_cross_view_target_difference_mm": 25.000,
    "maximum_grasp_yaw_offset_deg": 20.000,
    "maximum_scan_position_deviation_mm": 2.000,
    "maximum_scan_rotation_deviation_deg": 0.750,
    "minimum_pregrasp_height_mm": 30.000,
    "move_timeout_seconds": 8.000,
    "pregrasp_height_step_mm": 5.000,
    "retry_delay_seconds": 1.000,
    "scan_settle_seconds": 1.000,
}



class CalibrationError(ValueError):
    """Raised when calibration input is missing, malformed, or degenerate."""


def load_json(path: Path) -> dict[str, Any]:
    """Read a JSON object, reporting missing files or malformed data explicitly."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CalibrationError(f"file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CalibrationError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CalibrationError(f"expected a JSON object in {path}")
    return data


class HandEyeError(ValueError):
    """Raised when hand-eye inputs are malformed or geometrically inadequate."""


def transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Combine rotation and translation in millimetres into a 4x4 transform."""
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return matrix


def piper_pose_to_transform(pose_mm_deg: Iterable[float]) -> np.ndarray:
    """Convert finite Piper XYZ/RPY telemetry (mm/degrees) to a base-frame transform."""
    pose = np.asarray(list(pose_mm_deg), dtype=np.float64)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise HandEyeError("Piper pose must contain six finite values")
    return transform(euler_to_rot(pose[3:]).as_matrix(), pose[:3])


def average_rotations(rotations: Iterable[np.ndarray]) -> np.ndarray:
    """Average rotation matrices with SVD while preserving a proper rotation."""
    matrices = np.stack([np.asarray(rotation).reshape(3, 3) for rotation in rotations])
    u, _, vt = np.linalg.svd(np.sum(matrices, axis=0))
    mean = u @ vt
    if np.linalg.det(mean) < 0:
        u[:, -1] *= -1
        mean = u @ vt
    return mean


def rotation_error_degrees(left: np.ndarray, right: np.ndarray) -> float:
    """Measure the angular difference between two transforms, ignoring translation."""
    relative = np.asarray(left)[:3, :3].T @ np.asarray(right)[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def average_transforms(transforms: Iterable[np.ndarray]) -> np.ndarray:
    """Average translation and rotation separately for before/after capture telemetry."""
    matrices = np.stack([np.asarray(value).reshape(4, 4) for value in transforms])
    return transform(
        average_rotations(matrices[:, :3, :3]),
        np.mean(matrices[:, :3, 3], axis=0),
    )


def rigid_transform(document: dict, key: str) -> np.ndarray:
    """Read a finite 4x4 calibration matrix and require a homogeneous bottom row."""
    matrix = np.asarray(document.get(key), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise HandEyeError(f"{key} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8):
        raise HandEyeError(f"{key} is not a rigid transform")
    return matrix


def transform_camera_point_to_base(
    camera_point_mm: np.ndarray,
    gripper_to_base: np.ndarray,
    camera_to_gripper: np.ndarray,
    *,
    verbose: bool = False,
    label: str = "point",
) -> np.ndarray:
    """Map a camera point in mm through camera-to-gripper and gripper-to-base transforms."""
    point = np.asarray(camera_point_mm, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise HandEyeError("camera point must contain three finite values")
    homogeneous = np.append(point, 1.0)
    base_point = (gripper_to_base @ camera_to_gripper @ homogeneous)[:3]
    if verbose:
        gripper_point = camera_to_gripper @ homogeneous
        print(
            f"[TRANSFORM {label}]\n"
            f"  camera XYZ (mm): {np.round(point, 3).tolist()}\n"
            "  camera -> gripper matrix (translation in mm):\n"
            f"{np.array2string(camera_to_gripper, precision=3, suppress_small=True, floatmode='fixed')}\n"
            f"  gripper XYZ (mm): {np.round(gripper_point[:3], 3).tolist()}\n"
            "  gripper -> robot base matrix (translation in mm):\n"
            f"{np.array2string(gripper_to_base, precision=3, suppress_small=True, floatmode='fixed')}\n"
            f"  robot base XYZ (mm): {np.round(base_point, 3).tolist()}",
            flush=True,
        )
    return base_point


class GraspVerificationError(DemoError):
    """The measured post-lift jaw width shows no object was retained."""

    def __init__(self, measured_width_mm: float, minimum_held_mm: float | None = None) -> None:
        """Record the measured post-lift jaw width in the grasp-failure exception."""
        self.measured_width_mm = measured_width_mm
        self.minimum_held_mm = minimum_held_mm
        limit = "" if minimum_held_mm is None else f" (below {minimum_held_mm:.1f}mm)"
        super().__init__(
            "grasp verification failed: measured gripper width "
            f"{measured_width_mm:.1f}mm{limit}"
        )


class NoReachableCandyError(DemoError):
    """No detected candy in a scan has an accepted, path-checked pickup plan."""

    def __init__(self, message: str, targets: list[dict] | None = None) -> None:
        super().__init__(message)
        self.targets = list(targets or [])


def summarize_rejection(reason: str) -> str:
    """Shorten a planner rejection to its first underlying failure for console output."""
    text = str(reason)
    prefix = "all bounded candy-aligned grasp orientations failed: "
    if text.startswith(prefix):
        first = text[len(prefix):].split(" | ")[0]
        text = first.split(": ", 1)[-1].split("; adaptive")[0]
    return text if len(text) <= 160 else text[:157] + "..."


class CandyCarry:
    """Track whether a candy may be gripped and the waypoints reached since grasping it.

    The path starts at the grasp pose when the gripper closes and records each
    successfully reached waypoint, so recovery can retrace it in reverse.
    """

    def __init__(self) -> None:
        self.holding = False
        self.path: list[tuple[str, np.ndarray]] = []

    def gripped_at(self, joints: np.ndarray) -> None:
        """Start tracking after the gripper closes at the grasp pose."""
        self.holding = True
        self.path = [("grasp", np.asarray(joints, dtype=np.float64).copy())]

    def reached(self, name: str, joints: np.ndarray) -> None:
        """Record a reached waypoint while a candy may be held."""
        if self.holding:
            self.path.append((name, np.asarray(joints, dtype=np.float64).copy()))


def finite_vector(value: Any, length: int, name: str) -> np.ndarray:
    """Convert a value to a finite float vector with the required length."""
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise DemoError(f"{name} must contain {length} finite numbers")
    return result


def require_quality(document: dict, name: str, path: Path) -> None:
    """Reject a calibration document that has not passed its quality checks."""
    acceptance = document.get("acceptance")
    if not isinstance(acceptance, dict) or acceptance.get("quality_passed") is not True:
        raise DemoError(f"{name} has not passed quality checks: {path}")


def validate_demo_settings(settings: dict) -> dict:
    """Validate code-defined runtime settings and return normalized workspace bounds."""
    try:
        demo = dict(settings)
        target_min = finite_vector(
            WORKSPACE_MIN_MM, 3, "code-defined workspace minimum"
        ).copy()
        target_max = finite_vector(
            WORKSPACE_MAX_MM, 3, "code-defined workspace maximum"
        ).copy()
        finite_vector(demo["hand_delivery_rpy_deg"], 3, "demo.hand_delivery_rpy_deg")
        correction = finite_vector(
            demo["candy_xy_correction_mm"], 2, "demo.candy_xy_correction_mm"
        ).copy()
        candy_min_z, candy_max_z = float(demo["candy_min_z_mm"]), float(demo["candy_max_z_mm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DemoError(f"invalid demo configuration: {exc}") from exc
    if np.any(target_min >= target_max):
        raise DemoError("demo target workspace minimum must be below maximum")
    if np.any(np.abs(correction) > MAXIMUM_CANDY_XY_CORRECTION_MM):
        raise DemoError(
            f"demo.candy_xy_correction_mm components must be within "
            f"+/-{MAXIMUM_CANDY_XY_CORRECTION_MM:.0f}mm"
        )
    if not (math.isfinite(candy_min_z) and math.isfinite(candy_max_z) and candy_min_z < candy_max_z):
        raise DemoError("demo.candy_min_z_mm must be finite and below demo.candy_max_z_mm")
    demo["candy_xy_correction_mm"] = correction
    demo["candy_min_z_mm"], demo["candy_max_z_mm"] = candy_min_z, candy_max_z
    positive_keys = (
        "minimum_held_grip_mm",
        "motion_speed_percent",
        "pick_motion_speed_percent",
        "maximum_scan_position_deviation_mm",
        "maximum_scan_rotation_deviation_deg",
        "maximum_cross_view_target_difference_mm",
        "maximum_cross_view_orientation_difference_deg",
        "scan_settle_seconds",
        "approach_height_mm",
        "pregrasp_height_mm",
        "minimum_pregrasp_height_mm",
        "pregrasp_height_step_mm",
        "maximum_approach_inward_offset_mm",
        "approach_inward_step_mm",
        "maximum_grasp_yaw_offset_deg",
        "grasp_yaw_offset_step_deg",
        "move_timeout_seconds",
        "gripper_hold_seconds",
        "retry_delay_seconds",
        "hand_approach_height_mm",
    )
    for key in positive_keys:
        value = float(demo[key])
        if not math.isfinite(value) or value <= 0:
            raise DemoError(f"demo.{key} must be positive")
    if not float(demo["grip_close_mm"]) < float(demo["minimum_held_grip_mm"]) < float(demo["grip_open_mm"]):
        raise DemoError("demo.minimum_held_grip_mm must lie between the close and open gripper commands")
    forward_lean = float(demo["forward_lean_vertical_threshold"])
    if not 0 < forward_lean <= 1:
        raise DemoError("demo.forward_lean_vertical_threshold must be above 0 and at most 1")
    demo["forward_lean_vertical_threshold"] = forward_lean
    speed = int(demo["motion_speed_percent"])
    if speed != float(demo["motion_speed_percent"]) or speed > 100:
        raise DemoError("demo.motion_speed_percent must be an integer from 1 to 100")
    demo["motion_speed_percent"] = speed
    pick_speed = int(demo["pick_motion_speed_percent"])
    if pick_speed != float(demo["pick_motion_speed_percent"]) or pick_speed > 100:
        raise DemoError(
            "demo.pick_motion_speed_percent must be an integer from 1 to 100"
        )
    demo["pick_motion_speed_percent"] = pick_speed
    if float(demo["pregrasp_height_mm"]) >= float(demo["approach_height_mm"]):
        raise DemoError(
            "demo.pregrasp_height_mm must be below demo.approach_height_mm"
        )
    if float(demo["minimum_pregrasp_height_mm"]) > float(
        demo["pregrasp_height_mm"]
    ):
        raise DemoError(
            "demo.minimum_pregrasp_height_mm must not exceed "
            "demo.pregrasp_height_mm"
        )
    if float(demo["pregrasp_height_step_mm"]) > (
        float(demo["pregrasp_height_mm"])
        - float(demo["minimum_pregrasp_height_mm"])
    ):
        raise DemoError(
            "demo.pregrasp_height_step_mm must not exceed the configured "
            "pre-grasp height range"
        )
    if float(demo["approach_inward_step_mm"]) > float(
        demo["maximum_approach_inward_offset_mm"]
    ):
        raise DemoError(
            "demo.approach_inward_step_mm must not exceed "
            "demo.maximum_approach_inward_offset_mm"
        )
    if float(demo["grasp_yaw_offset_step_deg"]) > float(
        demo["maximum_grasp_yaw_offset_deg"]
    ):
        raise DemoError(
            "demo.grasp_yaw_offset_step_deg must not exceed "
            "demo.maximum_grasp_yaw_offset_deg"
        )
    confidence = float(demo["detection_confidence_threshold"])
    if not 0 < confidence < 1:
        raise DemoError("demo detection confidence must be between zero and one")
    demo["target_workspace_min_mm"] = target_min
    demo["target_workspace_max_mm"] = target_max
    demo["hand_workspace_min_mm"] = target_min.copy()
    demo["hand_workspace_max_mm"] = target_max.copy()
    return demo


def load_scan_poses(path: Path) -> list[dict[str, Any]]:
    """Load one to three distinct, joint-limit-safe taught scan poses."""
    document = load_json(path)
    poses = document.get("poses")
    count = document.get("pose_count")
    if (
        type(count) is not int or count not in (1, 2, 3)
        or not isinstance(poses, list) or len(poses) != count
    ):
        raise DemoError(f"{path} must contain 1..3 taught scan poses matching pose_count")
    parsed = []
    limits = np.asarray(PIPER_C.JOINT_LIMITS_DEG, dtype=np.float64)
    safe_minimum = limits[:, 0] + PIPER_C.IK_LIMIT_MARGIN_DEG
    safe_maximum = limits[:, 1] - PIPER_C.IK_LIMIT_MARGIN_DEG
    for expected_index, value in enumerate(poses, start=1):
        if not isinstance(value, dict) or value.get("index") != expected_index:
            raise DemoError(f"scan pose {expected_index} is missing or out of order")
        joints = finite_vector(
            value.get("joints_deg"),
            6,
            f"scan pose {expected_index} joints",
        )
        violations = [
            f"J{joint_index + 1}={joints[joint_index]:.1f}deg outside "
            f"[{safe_minimum[joint_index]:.1f}, "
            f"{safe_maximum[joint_index]:.1f}]deg"
            for joint_index in range(6)
            if joints[joint_index] < safe_minimum[joint_index]
            or joints[joint_index] > safe_maximum[joint_index]
        ]
        if violations:
            raise DemoError(
                f"scan pose {expected_index} is not safely replayable: "
                + "; ".join(violations)
                + ". Rerun prepare-piper/Teach-scripts/teach_scan_poses.py"
            )
        parsed.append(
            {
                **value,
                "joints_deg": joints,
                "end_pose_mm_deg": finite_vector(
                    value.get("end_pose_mm_deg"),
                    6,
                    f"scan pose {expected_index} end pose",
                ),
            }
        )
    for left in range(len(parsed)):
        for right in range(left + 1, len(parsed)):
            if (
                np.linalg.norm(
                    parsed[left]["joints_deg"] - parsed[right]["joints_deg"]
                )
                < 5.0
            ):
                raise DemoError(
                    f"scan poses {left + 1} and {right + 1} are not distinct"
                )
    return parsed


def validate_target_workspace(
    point_mm: np.ndarray, minimum_mm: np.ndarray, maximum_mm: np.ndarray,
    *, check_z: bool = True,
) -> None:
    """Require finite XYZ and check workspace bounds, optionally using XY only."""
    point = finite_vector(point_mm, 3, "detected base point")
    dimensions = 3 if check_z else 2
    minimum, maximum = minimum_mm[:dimensions], maximum_mm[:dimensions]
    if np.any(point[:dimensions] < minimum) or np.any(point[:dimensions] > maximum):
        bounds_label = "" if check_z else "XY workspace bounds "
        raise DemoError(
            f"detected base point {np.round(point, 1).tolist()}mm is outside "
            f"{bounds_label}{np.round(minimum, 3).tolist()}..{np.round(maximum, 3).tolist()}mm"
        )


def require_healthy(arm: Piper, context: str) -> None:
    """Stop motion when faults, disabled joints, or excessive heat are present."""
    faults = arm.faults()
    if faults:
        raise DemoError(f"{context}: Piper faults: {faults}")
    if not all(arm.enabled()):
        raise DemoError(f"{context}: one or more Piper joints are not enabled")
    if arm.j6_temp() >= PIPER_C.J6_TEMP_PAUSE_C:
        raise DemoError(
            f"{context}: J6 temperature {arm.j6_temp()}C is at or above "
            f"{PIPER_C.J6_TEMP_PAUSE_C}C"
        )


def move_robot(
    arm: Piper,
    name: str,
    joints: np.ndarray,
    grip_mm: float | None,
    timeout: float,
    speed: int,
    *,
    verbose: bool = True,
) -> None:
    """Execute a joint move with health checks before and after convergence."""
    require_healthy(arm, f"before {name}")
    if verbose:
        print(f"motion: {name}")
    if not arm.move_joints(
        joints,
        timeout,
        grip_mm=grip_mm,
        speed=speed,
        tol=ARRIVAL_TOLERANCE_DEG,
        settle=0.4,
    ):
        raise DemoError(f"{name} did not reach its planned joint target")
    require_healthy(arm, f"after {name}")


def hold_gripper(
    arm: Piper, joints: np.ndarray, grip_mm: float, seconds: float
) -> None:
    """Continuously command a stationary pose and gripper width for a duration."""
    deadline = time.monotonic() + seconds
    next_tick = time.monotonic()
    while time.monotonic() < deadline:
        require_healthy(arm, "gripper hold")
        arm.send_joints(joints, grip_mm, speed=PIPER_C.SPEED_REST)
        next_tick += 1.0 / PIPER_C.CONTROL_HZ
        time.sleep(max(0.0, next_tick - time.monotonic()))


def flange_for_tip(tip_xyz: np.ndarray, rotation: Rot) -> np.ndarray:
    """Convert a desired tool-tip point into the corresponding flange point."""
    return np.asarray(tip_xyz, dtype=np.float64) - rotation.apply(
        [0.0, 0.0, PIPER_C.TOOL_LEN_MM]
    )


def solve_waypoint(
    name: str, flange_xyz: np.ndarray, rotation: Rot, seed_joints: np.ndarray
) -> np.ndarray:
    """Solve and safety-check an IK waypoint from a seeded joint position."""
    xyz = finite_vector(flange_xyz, 3, f"{name} flange position")
    if np.any(xyz < np.asarray(PIPER_C.WORKSPACE_MIN_MM)) or np.any(
        xyz > np.asarray(PIPER_C.WORKSPACE_MAX_MM)
    ):
        raise DemoError(
            f"{name} flange {np.round(xyz, 1).tolist()}mm is outside the "
            "configured Piper workspace"
        )
    joints, converged = ik(xyz, rotation, seed_joints, limits=MARGINED_LIMITS)
    if not converged:
        raise DemoError(
            f"no joint-limit-safe IK solution for {name} at "
            f"{np.round(xyz, 1).tolist()}mm"
        )
    tool_tip_z = float(tip_position(joints)[2])
    if tool_tip_z < PIPER_C.TIP_Z_FLOOR_MM:
        raise DemoError(
            f"{name} would put the tool tip at {tool_tip_z:.1f}mm, below "
            f"the {PIPER_C.TIP_Z_FLOOR_MM:.1f}mm floor"
        )
    return joints


def validate_joint_path_safety(
    name: str,
    waypoints: list[np.ndarray],
    samples_per_segment: int = 20,
    *,
    joint_limits: np.ndarray = MARGINED_LIMITS,
) -> None:
    """Sample joint paths against the supplied limits, flange bounds, and tip floor.

    Normal moves use inward-margined limits; SDK zero return explicitly uses
    factory limits because zero lies on the J2/J3 boundaries.
    """
    lower = joint_limits[:, 0]
    upper = joint_limits[:, 1]
    for segment_index, (start, end) in enumerate(
        zip(waypoints, waypoints[1:]),
        start=1,
    ):
        for fraction in np.linspace(0.0, 1.0, samples_per_segment + 1):
            joints = (1.0 - fraction) * start + fraction * end
            outside = (joints < lower) | (joints > upper)
            if np.any(outside):
                joint = int(np.flatnonzero(outside)[0])
                where = " at the segment start" if fraction == 0.0 else ""
                raise DemoError(
                    f"{name} segment {segment_index} crosses a joint limit{where}: "
                    f"J{joint + 1}={joints[joint]:.2f}deg outside "
                    f"[{lower[joint]:.1f}, {upper[joint]:.1f}]deg"
                )
            flange = fk_flange(joints)[:3]
            if np.any(flange < np.asarray(PIPER_C.WORKSPACE_MIN_MM)) or np.any(
                flange > np.asarray(PIPER_C.WORKSPACE_MAX_MM)
            ):
                raise DemoError(
                    f"{name} segment {segment_index} leaves the flange workspace"
                )
            tip_z = float(tip_position(joints)[2])
            if tip_z < PIPER_C.TIP_Z_FLOOR_MM:
                raise DemoError(
                    f"{name} segment {segment_index} puts the tool tip at "
                    f"{tip_z:.1f}mm, below the "
                    f"{PIPER_C.TIP_Z_FLOOR_MM:.1f}mm floor"
                )


def measured_start_joints(
    arm: Any, name: str, joint_limits: np.ndarray = MARGINED_LIMITS,
) -> np.ndarray:
    """Return measured joints as a path-check start, tolerating arrival settling.

    A joint may settle up to the 2-degree arrival tolerance past a planned
    waypoint, so a waypoint planned exactly on an inward-margined limit can read
    slightly beyond it. Such overshoot is clamped back onto the limit for path
    sampling; anything further out raises. With the default margined limits this
    allows readings up to, but never past, the factory joint limits. Planned
    waypoints themselves are not relaxed.
    """
    current = finite_vector(arm.joints_deg(), 6, f"{name} measured joints")
    limits = np.asarray(joint_limits, dtype=np.float64)
    overshoot = np.maximum(limits[:, 0] - current, current - limits[:, 1])
    if np.any(overshoot > ARRIVAL_TOLERANCE_DEG):
        joint = int(np.argmax(overshoot))
        raise DemoError(
            f"{name}: measured J{joint + 1}={current[joint]:.2f}deg is more than "
            f"{ARRIVAL_TOLERANCE_DEG:.0f}deg outside "
            f"[{limits[joint, 0]:.1f}, {limits[joint, 1]:.1f}]deg"
        )
    return np.clip(current, limits[:, 0], limits[:, 1])


def plan_pickup(
    target_base_mm: np.ndarray,
    start_joints: np.ndarray,
    return_joints: np.ndarray,
    grasp_rpy: np.ndarray,
    grasp_tip_z_mm: float,
    approach_height_mm: float,
    pregrasp_height_mm: float = 50.0,
    minimum_pregrasp_height_mm: float = 30.0,
    pregrasp_height_step_mm: float = 5.0,
    maximum_inward_offset_mm: float = 120.0,
    inward_step_mm: float = 20.0,
) -> list[tuple[str, np.ndarray]]:
    """Plan pickup, lift, and return using a direct or adaptive inward approach."""
    target = finite_vector(target_base_mm, 3, "target base point")
    grasp_rotation = euler_to_rot(grasp_rpy)
    grasp_tip = np.array([target[0], target[1], grasp_tip_z_mm])
    approach_tip = grasp_tip + [0.0, 0.0, approach_height_mm]
    try:
        q_approach = solve_waypoint(
            "pick approach",
            flange_for_tip(approach_tip, grasp_rotation),
            grasp_rotation,
            start_joints,
        )
        q_grasp = solve_waypoint(
            "grasp",
            flange_for_tip(grasp_tip, grasp_rotation),
            grasp_rotation,
            q_approach,
        )
        # Check the whole direct route here so a failure falls through to the
        # adaptive search, other orientations, and other candies.
        validate_joint_path_safety(
            "direct pick approach",
            [
                np.asarray(start_joints, dtype=np.float64), q_approach, q_grasp, q_approach,
                np.asarray(return_joints, dtype=np.float64),
            ],
        )
        return [
            ("pick approach", q_approach),
            ("grasp", q_grasp),
            ("lift", q_approach.copy()),
            ("scan 1", np.asarray(return_joints, dtype=np.float64).copy()),
        ]
    except DemoError as direct_error:
        radial_distance = float(np.linalg.norm(target[:2]))
        if radial_distance <= 1e-6:
            raise direct_error
        inward_direction = target[:2] / radial_distance
        last_error: DemoError = direct_error
        pregrasp_heights = np.arange(
            pregrasp_height_mm,
            minimum_pregrasp_height_mm - pregrasp_height_step_mm * 0.5,
            -pregrasp_height_step_mm,
        )
        offsets = np.arange(
            inward_step_mm,
            maximum_inward_offset_mm + inward_step_mm * 0.5,
            inward_step_mm,
        )
        for pregrasp_height in pregrasp_heights:
            pregrasp_tip = grasp_tip + [0.0, 0.0, pregrasp_height]
            for inward_offset in offsets:
                transit_tip = approach_tip.copy()
                transit_tip[:2] -= inward_direction * inward_offset
                try:
                    q_transit = solve_waypoint(
                        "pick transit",
                        flange_for_tip(transit_tip, grasp_rotation),
                        grasp_rotation,
                        start_joints,
                    )
                    q_pregrasp = solve_waypoint(
                        "pick approach",
                        flange_for_tip(pregrasp_tip, grasp_rotation),
                        grasp_rotation,
                        q_transit,
                    )
                    q_grasp = solve_waypoint(
                        "grasp",
                        flange_for_tip(grasp_tip, grasp_rotation),
                        grasp_rotation,
                        q_pregrasp,
                    )
                    validate_joint_path_safety(
                        "adaptive pick approach",
                        [
                            np.asarray(start_joints, dtype=np.float64),
                            q_transit,
                            q_pregrasp,
                            q_grasp,
                            q_pregrasp,
                            q_transit,
                            np.asarray(return_joints, dtype=np.float64),
                        ],
                    )
                except DemoError as exc:
                    last_error = exc
                    continue
                return [
                    ("pick transit", q_transit),
                    ("pick approach", q_pregrasp),
                    ("grasp", q_grasp),
                    ("lift", q_pregrasp.copy()),
                    ("lift transit", q_transit.copy()),
                    ("scan 1", np.asarray(return_joints, dtype=np.float64).copy()),
                ]
        raise DemoError(
            f"{direct_error}; adaptive inward/pre-grasp search also failed: "
            f"{last_error}"
        )


def align_grasp_rpy_to_axis(
    nominal_rpy_deg: np.ndarray,
    long_axis_base_unit_xy: np.ndarray,
) -> np.ndarray:
    """Align the gripper finger axis with the candy while preserving roll/pitch."""
    nominal = finite_vector(nominal_rpy_deg, 3, "nominal grasp RPY")
    axis = finite_vector(long_axis_base_unit_xy, 2, "candy long axis")
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-6:
        raise DemoError("candy long axis is degenerate")
    axis /= axis_norm
    axis_yaw = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
    equivalent_yaws = [
        axis_yaw + 180.0 * turns for turns in range(-2, 4)
    ]
    aligned = nominal.copy()
    aligned[2] = min(
        equivalent_yaws,
        key=lambda value: abs(value - float(nominal[2])),
    )
    return aligned


def equivalent_grasp_rpys(
    nominal_rpy_deg: np.ndarray,
    long_axis_base_unit_xy: np.ndarray,
) -> list[np.ndarray]:
    """Return both 180-degree-equivalent candy-aligned grasp orientations."""
    primary = align_grasp_rpy_to_axis(
        nominal_rpy_deg,
        long_axis_base_unit_xy,
    )
    flipped = primary.copy()
    flipped[2] = (float(primary[2]) + 360.0) % 360.0 - 180.0
    return [primary, flipped]


def grasp_rpy_candidates(
    nominal_rpy_deg: np.ndarray,
    long_axis_base_unit_xy: np.ndarray,
    maximum_yaw_offset_deg: float,
    yaw_offset_step_deg: float,
) -> list[np.ndarray]:
    """Return exact then progressively offset candy-aligned grasp orientations."""
    exact = equivalent_grasp_rpys(
        nominal_rpy_deg,
        long_axis_base_unit_xy,
    )
    candidates = [orientation.copy() for orientation in exact]
    offsets = np.arange(
        yaw_offset_step_deg,
        maximum_yaw_offset_deg + yaw_offset_step_deg * 0.5,
        yaw_offset_step_deg,
    )
    for offset in offsets:
        for signed_offset in (-offset, offset):
            for orientation in exact:
                candidate = orientation.copy()
                candidate[2] += signed_offset
                candidates.append(candidate)
    return candidates


def parse_grasp_offset(text: str) -> np.ndarray | None:
    """Parse an operator's 'dx dy' candy offset in mm; a blank line records nothing."""
    fields = text.replace(",", " ").split()
    if not fields:
        return None
    if len(fields) != 2:
        raise ValueError("enter two numbers 'dx dy' in mm, or press Enter to grip without recording")
    try:
        offset = np.array([float(value) for value in fields])
    except ValueError as exc:
        raise ValueError(f"not a number: {exc}") from exc
    if not np.all(np.isfinite(offset)) or np.any(np.abs(offset) > 100.0):
        raise ValueError("each offset must be a finite number within +/-100mm")
    return offset


def hold_for_operator_input(
    arm: Piper, joints: np.ndarray, grip_mm: float, prompt: str,
    parse: Callable[[str], Any],
) -> Any:
    """Hold a pose and gripper width until the operator types a line that parses.

    Invalid lines re-prompt; end of input counts as a blank line. Health checks
    run on every hold tick, and Ctrl+C propagates from the hold loop.
    """
    answer: dict[str, Any] = {}
    answered = threading.Event()

    def ask() -> None:
        """Read operator lines until one parses, then signal the hold loop."""
        while True:
            try:
                text = input(prompt)
            except EOFError:
                text = ""
            try:
                answer["value"] = parse(text)
            except ValueError as exc:
                print(f"  {exc}", flush=True)
                continue
            answered.set()
            return

    threading.Thread(target=ask, daemon=True).start()
    next_tick = time.monotonic()
    while not answered.is_set():
        require_healthy(arm, "grasp pause")
        arm.send_joints(joints, grip_mm, speed=PIPER_C.SPEED_REST)
        next_tick += 1.0 / PIPER_C.CONTROL_HZ
        answered.wait(timeout=max(0.0, next_tick - time.monotonic()))
    return answer["value"]


def record_grasp_offset(
    offset: np.ndarray, target: dict, jaw_point: np.ndarray, jaw_axis_xy: np.ndarray,
    demo: dict, path: Path | None = None,
) -> np.ndarray:
    """Log one measured candy-minus-jaw offset and print the suggested XY correction.

    The suggestion is the correction in effect plus the mean of this run's
    offsets; it is appended to a JSON-lines file (GRASP_OFFSET_LOG by default).
    """
    path = GRASP_OFFSET_LOG if path is None else path
    correction = np.asarray(demo["candy_xy_correction_mm"], dtype=np.float64)
    across = float(offset @ jaw_axis_xy)
    along = float(offset @ np.array([-jaw_axis_xy[1], jaw_axis_xy[0]]))
    offsets = demo.setdefault("recorded_grasp_offsets_mm", [])
    offsets.append(np.asarray(offset, dtype=np.float64))
    suggested = correction + np.mean(offsets, axis=0)
    print(
        f"  Recorded candy offset {np.round(offset, 1).tolist()}mm: {across:+.1f}mm along the jaw "
        f"closing direction, {along:+.1f}mm along the candy",
        flush=True,
    )
    print(
        f"  GRASP OFFSET SUMMARY: {len(offsets)} reading(s), mean "
        f"{np.round(np.mean(offsets, axis=0), 1).tolist()}mm -> suggested "
        f"\"candy_xy_correction_mm\": {np.round(suggested, 1).tolist()} "
        f"(current {np.round(correction, 1).tolist()})",
        flush=True,
    )
    record = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "detector": demo.get("detector"),
        "candy_base_point_mm": target.get("base_point_mm"),
        "uncorrected_base_point_mm": target.get("uncorrected_base_point_mm", target.get("base_point_mm")),
        "candy_axis_yaw_deg": target.get("long_axis_base_yaw_deg"),
        "score": target.get("score"),
        "jaw_point_mm": np.round(jaw_point, 3).tolist(),
        "jaw_axis_xy": np.round(jaw_axis_xy, 4).tolist(),
        "offset_mm": np.round(offset, 3).tolist(),
        "across_jaw_mm": round(across, 3),
        "along_candy_mm": round(along, 3),
        "correction_in_effect_mm": np.round(correction, 3).tolist(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError as exc:
        print(f"  WARNING: could not append grasp offset to {path}: {exc}", flush=True)
    return suggested


def pause_for_grasp_offset(
    arm: Piper, grasp_joints: np.ndarray, open_mm: float, target: dict, demo: dict,
) -> np.ndarray | None:
    """Hold the open-jaw grasp pose and ask the operator where the candy really is."""
    measured = finite_vector(arm.joints_deg(), 6, "measured grasp joints")
    jaw_point = tip_position(measured)
    jaw_axis = fk(grasp_joints)[:2, 1]
    jaw_axis = jaw_axis / max(float(np.linalg.norm(jaw_axis)), 1e-9)
    candy = target.get("base_point_mm")
    print(
        "\nPAUSED AT GRASP: jaws open and arm holding position (Ctrl+C stops the demo).\n"
        f"  candy estimate {np.round(candy, 1).tolist() if candy is not None else '?'}mm, candy axis yaw "
        f"{float(target.get('long_axis_base_yaw_deg', math.nan)):.1f}deg, "
        f"jaw point {np.round(jaw_point[:2], 1).tolist()}mm\n"
        "  Measure the candy centre relative to the midpoint between the open fingertips,\n"
        "  in robot base directions:\n"
        "    +X = straight out from the robot base (toward the far edge), -X = back toward the base\n"
        "    +Y = to your left when standing behind the robot facing +X, -Y = to your right",
        flush=True,
    )
    offset = hold_for_operator_input(
        arm, grasp_joints, open_mm,
        "  Candy offset 'dx dy' in mm (e.g. '0 -30'), or Enter to grip without recording: ",
        parse_grasp_offset,
    )
    if offset is None:
        print("  No offset recorded", flush=True)
        return None
    record_grasp_offset(offset, target, jaw_point, jaw_axis, demo)
    return offset


def execute_pickup(
    arm: Piper,
    plan: list[tuple[str, np.ndarray]],
    demo: dict,
    carry: CandyCarry | None = None,
    target: dict | None = None,
) -> float:
    """Execute approach and grasp, then verify an object is held after lifting.

    When a carry tracker is supplied, mark the candy as held once the gripper
    closes and record each reached lift waypoint for recovery. With
    pause_before_grip set, hold the open-jaw grasp pose for an operator offset
    reading before closing.
    """
    waypoints = dict(plan)
    timeout = float(demo["move_timeout_seconds"])
    open_mm = float(demo["grip_open_mm"])
    close_mm = float(demo["grip_close_mm"])
    hold_seconds = float(demo["gripper_hold_seconds"])
    speed = int(
        demo.get("pick_motion_speed_percent", demo["motion_speed_percent"])
    )
    if "pick transit" in waypoints:
        move_robot(
            arm,
            "pick transit",
            waypoints["pick transit"],
            open_mm,
            timeout,
            speed,
        )
    move_robot(
        arm, "pick approach", waypoints["pick approach"], open_mm, timeout, speed
    )
    move_robot(arm, "grasp", waypoints["grasp"], open_mm, timeout, speed)
    if demo.get("verbose"):
        measured = finite_vector(arm.joints_deg(), 6, "measured grasp joints")
        planned_tip, measured_tip = tip_position(waypoints["grasp"]), tip_position(measured)
        print(
            f"GRASP ARRIVAL: planned tip {np.round(planned_tip, 1).tolist()}mm, "
            f"measured tip {np.round(measured_tip, 1).tolist()}mm "
            f"(difference {np.round(measured_tip - planned_tip, 1).tolist()}mm, "
            f"max joint error {np.max(np.abs(measured - waypoints['grasp'])):.2f}deg)",
            flush=True,
        )
    if demo.get("pause_before_grip"):
        pause_for_grasp_offset(arm, waypoints["grasp"], open_mm, target or {}, demo)
    print("motion: close gripper")
    hold_gripper(arm, waypoints["grasp"], close_mm, hold_seconds)
    if carry is not None:
        carry.gripped_at(waypoints["grasp"])
    move_robot(arm, "lift", waypoints["lift"], close_mm, timeout, speed)
    if carry is not None:
        carry.reached("lift", waypoints["lift"])
    if "lift transit" in waypoints:
        move_robot(
            arm,
            "lift transit",
            waypoints["lift transit"],
            close_mm,
            timeout,
            speed,
        )
        if carry is not None:
            carry.reached("lift transit", waypoints["lift transit"])
    held_width = float(arm.grip_mm())
    minimum_held = float(demo["minimum_held_grip_mm"])
    if not math.isfinite(held_width) or held_width < minimum_held:
        if carry is not None:
            carry.holding = False
        print(
            f"No candy in gripper: measured width {held_width:.1f}mm is below "
            f"{minimum_held:.1f}mm; opening gripper and returning to scan",
            flush=True,
        )
        missed_grasp_pose = waypoints.get("lift transit", waypoints["lift"])
        hold_gripper(arm, missed_grasp_pose, open_mm, hold_seconds)
        move_robot(
            arm,
            "scan 1 after missed grasp",
            waypoints["scan 1"],
            open_mm,
            timeout,
            speed,
        )
        raise GraspVerificationError(held_width, minimum_held)
    return held_width


def plan_hand_delivery(
    palm_base_mm: np.ndarray,
    start_joints: np.ndarray,
    return_joints: np.ndarray,
    delivery_rpy: np.ndarray,
    release_tip_z_mm: float,
    approach_height_mm: float,
) -> list[tuple[str, np.ndarray]]:
    """Plan a fixed-height approach, release above a palm, and safe retreat."""
    palm = finite_vector(palm_base_mm, 3, "palm base point")
    rotation = euler_to_rot(delivery_rpy)
    release_tip = np.array([palm[0], palm[1], float(release_tip_z_mm)])
    if not np.all(np.isfinite(release_tip)):
        raise DemoError("hand release tip must contain finite coordinates")
    approach_tip = release_tip + [0.0, 0.0, approach_height_mm]
    q_approach = solve_waypoint(
        "hand approach",
        flange_for_tip(approach_tip, rotation),
        rotation,
        start_joints,
    )
    q_release = solve_waypoint(
        "hand release",
        flange_for_tip(release_tip, rotation),
        rotation,
        q_approach,
    )
    return [
        ("hand approach", q_approach),
        ("hand release", q_release),
        ("hand retreat", q_approach.copy()),
        ("scan 1", np.asarray(return_joints, dtype=np.float64).copy()),
    ]


def transform_object_candidate(
    candidate: dict[str, Any],
    scan_index: int,
    robot_transform: np.ndarray,
    camera_to_gripper: np.ndarray,
    demo: dict,
) -> dict[str, Any]:
    """Transform one localized candy point and long axis into the base frame."""
    transformed = {**candidate, "slot": scan_index}
    if candidate.get("status") != "detected":
        return transformed
    try:
        camera_point = finite_vector(
            candidate.get("camera_point_mm"),
            3,
            f"scan {scan_index} camera point",
        )
        base_point = transform_camera_point_to_base(
            camera_point,
            robot_transform,
            camera_to_gripper,
            verbose=demo.get("verbose", False),
            label="candy scan",
        )
        camera_axis = finite_vector(
            candidate.get("long_axis_camera_unit"),
            3,
            f"scan {scan_index} Starburst long axis",
        )
        base_axis = (
            robot_transform[:3, :3]
            @ camera_to_gripper[:3, :3]
            @ camera_axis
        )
        base_axis_xy = base_axis[:2]
        base_axis_xy_norm = float(np.linalg.norm(base_axis_xy))
        if base_axis_xy_norm < 0.75:
            raise DemoError(
                "Starburst long axis is too steep to define a safe table yaw"
            )
        base_axis_xy /= base_axis_xy_norm
        candy_z = float(base_point[2])
        minimum_z, maximum_z = float(demo["candy_min_z_mm"]), float(demo["candy_max_z_mm"])
        if not minimum_z <= candy_z <= maximum_z:
            raise DemoError(
                f"candy height Z {candy_z:.1f}mm is outside the plausible table range "
                f"{minimum_z:.1f}..{maximum_z:.1f}mm (likely a bad depth reading)"
            )
        correction = np.asarray(demo["candy_xy_correction_mm"], dtype=np.float64)
        if np.any(correction != 0):
            transformed["uncorrected_base_point_mm"] = base_point.tolist()
            base_point = base_point.copy()
            base_point[:2] += correction
            if demo.get("verbose"):
                print(
                    f"candy scan: applied candy XY correction "
                    f"{np.round(correction, 3).tolist()}mm -> {np.round(base_point, 3).tolist()}mm",
                    flush=True,
                )
        validate_target_workspace(
            base_point,
            demo["target_workspace_min_mm"],
            demo["target_workspace_max_mm"],
            check_z=False,
        )
        if demo.get("detector") == "opencv":
            length_mm, width_mm = candy_size_mm(
                candidate.get("long_axis_pixels"),
                candidate.get("orientation_aspect_ratio", math.nan),
                demo["camera_intrinsics"],
                robot_transform @ camera_to_gripper,
                float(base_point[2]),
            )
            transformed.update(length_mm=length_mm, width_mm=width_mm)
            check_candy_size(length_mm, width_mm)
        transformed.update(
            {
                "base_point_mm": base_point.tolist(),
                "long_axis_base_unit_xy": base_axis_xy.tolist(),
                "long_axis_base_yaw_deg": math.degrees(
                    math.atan2(
                        float(base_axis_xy[1]),
                        float(base_axis_xy[0]),
                    )
                ),
            }
        )
    except (DemoError, HandEyeError, ValueError) as exc:
        transformed["status"] = "rejected"
        transformed["error"] = str(exc)
    return transformed


def process_object_capture(
    service_result: dict[str, Any],
    index: int,
    confidence: float,
    client: VisionClient,
    before_pose: np.ndarray,
    after_pose: np.ndarray,
    before_transform: np.ndarray,
    after_transform: np.ndarray,
    camera_to_gripper: np.ndarray,
    demo: dict,
) -> dict[str, Any]:
    """Transform and report one object-service result using shared telemetry."""
    position_deviation = float(
        np.linalg.norm(after_pose[:3] - before_pose[:3])
    )
    rotation_deviation = rotation_error_degrees(
        before_transform, after_transform
    )
    result = {
        **service_result,
        "robot_pose_before_mm_deg": before_pose.tolist(),
        "robot_pose_after_mm_deg": after_pose.tolist(),
        "robot_position_deviation_mm": position_deviation,
        "robot_rotation_deviation_deg": rotation_deviation,
    }
    source_candidates = service_result.get("candidates")
    if not isinstance(source_candidates, list):
        source_candidates = [service_result]
    robot_transform = average_transforms([before_transform, after_transform])
    object_candidates = [
        transform_object_candidate(
            candidate,
            index,
            robot_transform,
            camera_to_gripper,
            demo,
        )
        for candidate in source_candidates
    ]
    valid_candidates = sorted(
        (
            candidate
            for candidate in object_candidates
            if candidate.get("status") == "detected"
        ),
        key=lambda candidate: float(candidate["score"]),
        reverse=True,
    )
    result["object_candidates"] = object_candidates
    result["candidate_count"] = len(valid_candidates)
    if valid_candidates:
        result.update(valid_candidates[0])
        size = (
            f", size={result['length_mm']:.0f}x{result['width_mm']:.0f}mm"
            if "length_mm" in result else ""
        )
        print(
            f"CANDY SCAN: detected {len(valid_candidates)} candy(ies), "
            f"best score={float(result['score']):.3f}, "
            f"camera={np.round(result['camera_point_mm'], 1).tolist()}mm{size}, "
            f"snapshot={client.base_url}{result['snapshot_url']}"
        )
        if demo.get("verbose"):
            for candidate in object_candidates:
                if candidate.get("status") == "rejected":
                    print(
                        f"  rejected candy score={float(candidate.get('score', math.nan)):.3f}: "
                        f"{candidate.get('error', 'candidate rejected')}",
                        flush=True,
                    )
    else:
        if object_candidates and service_result["status"] != "not_found":
            result["status"] = "rejected"
            result["error"] = "; ".join(
                str(candidate.get("error", "candidate rejected"))
                for candidate in object_candidates
            )
        reason = (
            f": {result['error']}"
            if result.get("error")
            else ""
        )
        print(
            f"CANDY SCAN: {result['status']} at "
            f"confidence={confidence:.3f}{reason}, "
            f"snapshot={client.base_url}{result['snapshot_url']}"
        )
    return result


def capture_scan_pair(
    arm: Piper,
    object_client: VisionClient,
    hand_client: HandClient,
    scan: dict[str, Any],
    hand_slot: int,
    prompt: str,
    confidence: float,
    camera_to_gripper: np.ndarray,
    demo: dict,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Capture object and palm concurrently at one settled scan pose."""
    pose_started_at = time.perf_counter()
    scan_index = int(scan["index"])
    move_started_at = time.perf_counter()
    move_robot(
        arm,
        f"scan {scan_index}",
        scan["joints_deg"],
        float(demo["grip_open_mm"]),
        float(demo["move_timeout_seconds"]),
        int(demo["motion_speed_percent"]),
        verbose=bool(demo.get("verbose", False)),
    )
    move_seconds = time.perf_counter() - move_started_at
    settle_started_at = time.perf_counter()
    time.sleep(float(demo["scan_settle_seconds"]))
    settle_seconds = time.perf_counter() - settle_started_at
    before_pose = arm.pose()
    before_transform = piper_pose_to_transform(before_pose)
    object_client.set_workspace_overlay(
        before_transform @ camera_to_gripper,
        demo["target_workspace_min_mm"],
        demo["target_workspace_max_mm"],
    )

    def timed_call(call: Callable, *args: Any) -> tuple[Any, float]:
        """Run one service request and return its result plus elapsed seconds."""
        started_at = time.perf_counter()
        result = call(*args)
        return result, time.perf_counter() - started_at

    inference_started_at = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        object_future = executor.submit(
            timed_call,
            object_client.capture,
            scan_index,
            prompt,
            confidence,
        )
        hand_future = executor.submit(
            timed_call,
            hand_client.capture_hand,
            hand_slot,
        )
        object_service_result, object_seconds = object_future.result()
        hand_service_result, hand_seconds = hand_future.result()
    concurrent_inference_seconds = time.perf_counter() - inference_started_at
    after_pose = arm.pose()
    after_transform = piper_pose_to_transform(after_pose)
    position_deviation = float(
        np.linalg.norm(after_pose[:3] - before_pose[:3])
    )
    rotation_deviation = rotation_error_degrees(
        before_transform,
        after_transform,
    )
    if position_deviation > float(demo["maximum_scan_position_deviation_mm"]):
        raise DemoError(
            f"scan {scan_index} moved {position_deviation:.2f}mm "
            "during concurrent inference"
        )
    if rotation_deviation > float(demo["maximum_scan_rotation_deviation_deg"]):
        raise DemoError(
            f"scan {scan_index} rotated {rotation_deviation:.2f}deg "
            "during concurrent inference"
        )

    object_result = process_object_capture(
        object_service_result,
        scan_index,
        confidence,
        object_client,
        before_pose,
        after_pose,
        before_transform,
        after_transform,
        camera_to_gripper,
        demo,
    )
    hand_result = process_hand_capture(
        hand_service_result,
        scan_index,
        hand_slot,
        hand_client,
        before_pose,
        after_pose,
        before_transform,
        after_transform,
        camera_to_gripper,
        demo,
    )
    timing = {
        "move_seconds": move_seconds,
        "settle_seconds": settle_seconds,
        "object_inference_seconds": object_seconds,
        "hand_inference_seconds": hand_seconds,
        "concurrent_inference_seconds": concurrent_inference_seconds,
        "total_pose_seconds": time.perf_counter() - pose_started_at,
    }
    object_result["scan_timing_seconds"] = timing
    hand_result["scan_timing_seconds"] = timing
    print(
        f"SCAN TIMING: move={move_seconds:.2f}s, "
        f"settle={settle_seconds:.2f}s, object={object_seconds:.2f}s, "
        f"hand={hand_seconds:.2f}s, "
        f"concurrent={concurrent_inference_seconds:.2f}s, "
        f"total={timing['total_pose_seconds']:.2f}s"
    )
    return object_result, hand_result


def capture_hand_at_current_pose(
    arm: Piper,
    client: HandClient,
    overlay_client: VisionClient,
    scan: dict[str, Any],
    slot: int,
    camera_to_gripper: np.ndarray,
    demo: dict,
) -> dict[str, Any]:
    """Capture a palm at the current pose, rejecting arm movement during inference."""
    scan_index = int(scan["index"])
    require_healthy(arm, f"before open-palm scan {scan_index}")
    before_pose = arm.pose()
    before_transform = piper_pose_to_transform(before_pose)
    overlay_client.set_workspace_overlay(
        before_transform @ camera_to_gripper,
        demo["target_workspace_min_mm"],
        demo["target_workspace_max_mm"],
    )
    service_result = client.capture_hand(slot)
    after_pose = arm.pose()
    after_transform = piper_pose_to_transform(after_pose)
    position_deviation = float(np.linalg.norm(after_pose[:3] - before_pose[:3]))
    rotation_deviation = rotation_error_degrees(
        before_transform,
        after_transform,
    )
    if position_deviation > float(demo["maximum_scan_position_deviation_mm"]):
        raise DemoError(
            f"hand scan {scan_index} moved {position_deviation:.2f}mm "
            "during inference"
        )
    if rotation_deviation > float(demo["maximum_scan_rotation_deviation_deg"]):
        raise DemoError(
            f"hand scan {scan_index} rotated {rotation_deviation:.2f}deg "
            "during inference"
        )
    return process_hand_capture(
        service_result,
        scan_index,
        slot,
        client,
        before_pose,
        after_pose,
        before_transform,
        after_transform,
        camera_to_gripper,
        demo,
    )


def process_hand_capture(
    service_result: dict[str, Any],
    scan_index: int,
    slot: int,
    client: HandClient,
    before_pose: np.ndarray,
    after_pose: np.ndarray,
    before_transform: np.ndarray,
    after_transform: np.ndarray,
    camera_to_gripper: np.ndarray,
    demo: dict,
) -> dict[str, Any]:
    """Transform and report one hand-service result using shared telemetry."""
    position_deviation = float(
        np.linalg.norm(after_pose[:3] - before_pose[:3])
    )
    rotation_deviation = rotation_error_degrees(
        before_transform, after_transform
    )
    result = {
        **service_result,
        "scan_pose_index": scan_index,
        "robot_pose_before_mm_deg": before_pose.tolist(),
        "robot_pose_after_mm_deg": after_pose.tolist(),
        "robot_position_deviation_mm": position_deviation,
        "robot_rotation_deviation_deg": rotation_deviation,
    }
    if service_result["status"] == "detected":
        camera_point = finite_vector(
            service_result.get("camera_point_mm"),
            3,
            f"hand scan {scan_index} camera point",
        )
        robot_transform = average_transforms(
            [before_transform, after_transform]
        )
        base_point = transform_camera_point_to_base(
            camera_point,
            robot_transform,
            camera_to_gripper,
            verbose=demo.get("verbose", False),
            label="palm scan",
        )
        result["base_point_mm"] = base_point.tolist()
        try:
            validate_target_workspace(
                base_point,
                demo["hand_workspace_min_mm"],
                demo["hand_workspace_max_mm"],
            )
        except DemoError as exc:
            result["status"] = "rejected"
            result["error"] = str(exc)
            print(
                f"OPEN PALM SCAN: rejected "
                f"base={np.round(base_point, 1).tolist()}mm: {exc}"
            )
            return result
        print(
            f"OPEN PALM SCAN: detected "
            f"{service_result.get('handedness', 'unknown')} "
            f"score={float(service_result['score']):.3f}, "
            f"base={np.round(base_point, 1).tolist()}mm, "
            f"snapshot={client.base_url}{service_result['snapshot_url']}"
        )
    else:
        print(
            f"OPEN PALM SCAN: {service_result['status']}, "
            f"snapshot={client.base_url}{service_result.get('snapshot_url', '')}"
        )
    return result


def choose_target(
    captures: list[dict[str, Any]],
    maximum_difference: float,
    label: str = "requested object",
    maximum_orientation_difference_deg: float = 180.0,
    comparison_axes: tuple[int, ...] = (0, 1, 2),
) -> dict:
    """Select the best detection and report cross-view position/orientation agreement."""
    detections = [item for item in captures if item["status"] == "detected"]
    if not detections:
        raise DemoError(f"no valid {label} detection was found in any scan")
    detections.sort(key=lambda item: float(item["score"]), reverse=True)
    points = np.asarray([item["base_point_mm"] for item in detections])
    comparison_points = points[:, list(comparison_axes)]
    if len(points) > 1:
        maximum_pairwise = max(
            float(
                np.linalg.norm(
                    comparison_points[left] - comparison_points[right]
                )
            )
            for left in range(len(points))
            for right in range(left + 1, len(points))
        )
        consensus = np.median(points, axis=0)
    else:
        maximum_pairwise = 0.0
        consensus = points[0]
    orientation_yaws = [
        float(item["long_axis_base_yaw_deg"])
        for item in detections
        if "long_axis_base_yaw_deg" in item
    ]
    maximum_orientation_difference = 0.0
    if len(orientation_yaws) > 1:
        maximum_orientation_difference = max(
            abs((orientation_yaws[left] - orientation_yaws[right] + 90.0) % 180.0 - 90.0)
            for left in range(len(orientation_yaws))
            for right in range(left + 1, len(orientation_yaws))
        )
    orientation_agreement = (
        maximum_orientation_difference <= maximum_orientation_difference_deg
    )
    best = detections[0]
    return {
        **best,
        "base_point_mm": list(best["base_point_mm"]),
        "consensus_base_point_mm": consensus.tolist(),
        "detected_view_count": len(detections),
        "maximum_cross_view_difference_mm": maximum_pairwise,
        "cross_view_agreement_limit_mm": maximum_difference,
        "cross_view_comparison_axes": [
            ("x", "y", "z")[axis] for axis in comparison_axes
        ],
        "maximum_cross_view_orientation_difference_deg": (
            maximum_orientation_difference
        ),
        "cross_view_orientation_agreement_limit_deg": (
            maximum_orientation_difference_deg
        ),
        "cross_view_orientation_agreement_passed": orientation_agreement,
        "cross_view_agreement_passed": (
            maximum_pairwise <= maximum_difference and orientation_agreement
        ),
        "selected_from_slot": int(best["slot"]),
    }


def distinct_object_targets(
    captures: list[dict[str, Any]],
    maximum_difference: float,
    maximum_orientation_difference_deg: float,
) -> list[dict[str, Any]]:
    """Cluster multi-view detections into distinct candies, nearest base first."""
    detections: list[dict[str, Any]] = []
    for capture in captures:
        candidates = capture.get("object_candidates")
        if not isinstance(candidates, list):
            candidates = [capture]
        detections.extend(
            candidate
            for candidate in candidates
            if candidate.get("status") == "detected"
        )
    clusters: list[list[dict[str, Any]]] = []
    for detection in sorted(
        detections,
        key=lambda item: float(item["score"]),
        reverse=True,
    ):
        point = np.asarray(detection["base_point_mm"], dtype=np.float64)
        choices = []
        for cluster in clusters:
            if any(item["slot"] == detection["slot"] for item in cluster):
                continue
            consensus = np.median(
                np.asarray([item["base_point_mm"] for item in cluster]),
                axis=0,
            )
            distance = float(np.linalg.norm(point - consensus))
            if distance <= maximum_difference:
                choices.append((distance, cluster))
        if choices:
            min(choices, key=lambda item: item[0])[1].append(detection)
        else:
            clusters.append([detection])

    targets = []
    for cluster in clusters:
        target = choose_target(
            cluster,
            maximum_difference,
            "Starburst candy",
            maximum_orientation_difference_deg,
        )
        target["base_distance_mm"] = float(
            np.linalg.norm(
                np.asarray(
                    target["consensus_base_point_mm"][:2],
                    dtype=np.float64,
                )
            )
        )
        targets.append(target)
    return sorted(targets, key=lambda target: target["base_distance_mm"])


def select_reachable_object_target(
    targets: list[dict[str, Any]],
    start_joints: np.ndarray,
    return_joints: np.ndarray,
    nominal_grasp_rpy: np.ndarray,
    grasp_tip_z_mm: float,
    approach_height_mm: float,
    pregrasp_height_mm: float = 50.0,
    minimum_pregrasp_height_mm: float = 30.0,
    pregrasp_height_step_mm: float = 5.0,
    maximum_inward_offset_mm: float = 120.0,
    inward_step_mm: float = 20.0,
    maximum_grasp_yaw_offset_deg: float = 20.0,
    grasp_yaw_offset_step_deg: float = 5.0,
) -> tuple[
    dict[str, Any],
    np.ndarray,
    list[tuple[str, np.ndarray]],
    list[dict[str, Any]],
] | None:
    """Choose the nearest cross-view-consistent candy with a safe IK plan."""
    rejections: list[dict[str, Any]] = []
    for target in targets:
        if not target["cross_view_agreement_passed"]:
            target["planning_rejection_reason"] = (
                "cross-view position or orientation disagreement"
            )
            rejections.append(
                {
                    "base_point_mm": target["base_point_mm"],
                    "base_distance_mm": target["base_distance_mm"],
                    "reason": target["planning_rejection_reason"],
                }
            )
            continue
        try:
            orientation_candidates = grasp_rpy_candidates(
                nominal_grasp_rpy,
                np.asarray(target["long_axis_base_unit_xy"]),
                maximum_grasp_yaw_offset_deg,
                grasp_yaw_offset_step_deg,
            )
        except DemoError as exc:
            target["planning_rejection_reason"] = str(exc)
            rejections.append(
                {
                    "base_point_mm": target["base_point_mm"],
                    "base_distance_mm": target["base_distance_mm"],
                    "reason": target["planning_rejection_reason"],
                }
            )
            continue
        orientation_errors: list[str] = []
        for aligned_grasp_rpy in orientation_candidates:
            try:
                plan = plan_pickup(
                    np.asarray(target["base_point_mm"]),
                    start_joints,
                    return_joints,
                    aligned_grasp_rpy,
                    grasp_tip_z_mm,
                    approach_height_mm,
                    pregrasp_height_mm,
                    minimum_pregrasp_height_mm,
                    pregrasp_height_step_mm,
                    maximum_inward_offset_mm,
                    inward_step_mm,
                )
            except DemoError as exc:
                orientation_errors.append(
                    f"yaw {float(aligned_grasp_rpy[2]):.1f}deg: {exc}"
                )
                continue
            return target, aligned_grasp_rpy, plan, rejections
        target["planning_rejection_reason"] = (
            "all bounded candy-aligned grasp orientations failed: "
            + " | ".join(orientation_errors)
        )
        rejections.append(
            {
                "base_point_mm": target["base_point_mm"],
                "base_distance_mm": target["base_distance_mm"],
                "reason": target["planning_rejection_reason"],
            }
        )
    return None


def parse_args() -> argparse.Namespace:
    """Parse file/service locations, cycle limits, and explicit motion permissions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, default=CALIBRATION_DIR / "calibration")
    parser.add_argument("--scan-poses", type=Path, default=DEFAULT_SCAN_POSES)
    parser.add_argument("--vision-url", default=DEFAULT_VISION_URL)
    parser.add_argument("--detection-url", default=DEFAULT_DETECTION_URL)
    parser.add_argument(
        "--detector", choices=("yolo", "opencv"), default="yolo",
        help="candy detector: YOLO detection service (default) or local CPU OpenCV colour detector",
    )
    parser.add_argument("--hand-url", default="http://127.0.0.1:8095")
    parser.add_argument("--can-interface", default="can_piper")
    parser.add_argument("--check-only", action="store_true", help="preflight only; never connect to CAN")
    parser.add_argument(
        "--verbose", action="store_true",
        help="print camera, gripper, and robot-base XYZ and transformation matrices",
    )
    parser.add_argument(
        "--pause-before-grip", action="store_true",
        help="hold each grasp pose with the jaws open and ask for the measured candy offset "
        "before closing; prints the suggested candy_xy_correction_mm",
    )
    parser.add_argument(
        "--grasp-tilt-deg", type=float, default=None,
        help="test override for the grasp tilt from vertical (0-30 degrees; 0 = vertical) "
        "instead of DEMO_SETTINGS grasp_rpy_deg pitch",
    )
    parser.add_argument(
        "--allow-motion", action="store_true",
        help="connect to the arm and run the pickup and hand-delivery loop; "
        "without it only the preflight runs",
    )
    parser.add_argument("--max-cycles", type=int, default=0, help="zero loops until Ctrl+C")
    return parser.parse_args()


def check_services(
    camera: HttpClient, objects: VisionClient, hands: HandClient, camera_serial: str
) -> None:
    """Check all three services independently and report their availability.

    Require calibrated camera capabilities and the calibration's camera serial,
    the selected candy detector (loaded CUDA YOLO model or valid local OpenCV
    settings), and a hand stream sourced from the same camera URL.
    Raise after reporting failures; print overall success only if all checks pass.
    """
    failures = []
    local_opencv = isinstance(objects.detector, OpenCVCandyDetector)
    detector_label = "OpenCV candy detection" if local_opencv else "YOLO / Foundry detection"
    detector_ready = " (local CPU colour detector)" if local_opencv else " (CUDA model loaded)"
    for label, check in (
        ("OpenCV camera", camera.health),
        (detector_label, objects.detector.check),
        ("MediaPipe hand service", hands.health),
    ):
        try:
            result = check()
            if label == "OpenCV camera":
                if (
                    result.get("camera_serial") != camera_serial
                    or result.get("service") != "camera"
                    or result.get("preview_only") is not False
                    or result.get("depth_calibrated") is not True
                    or not {"localize", "deproject"}.issubset(result.get("capabilities") or [])
                ):
                    raise DemoError("requires matching D435 and calibrated depth; preview-only is not usable")
            elif label == "MediaPipe hand service":
                if result.get("source_url") != camera.base_url + "/stream.mjpg":
                    raise DemoError("hand stream must come from the configured OpenCV camera URL")
            print(f"[OK] {label}" + (f": {result}" if result else detector_ready), flush=True)
        except RuntimeError as exc:
            failures.append(f"{label}: {exc}")
            print(f"[FAILED] {label}: {exc}", flush=True)
    if failures:
        raise DemoError("required services unavailable: " + "; ".join(failures))
    print("[OK] All required services are running and available.", flush=True)


def load_hand_eye(calibration_dir: Path) -> dict:
    """Load accepted hand-eye results with a nonempty serial for camera health checks."""
    path = calibration_dir / "hand_eye.json"
    document = load_json(path)
    require_quality(document, "hand_eye.json", path)
    camera_serial = document.get("camera", {}).get("serial")
    if not isinstance(camera_serial, str) or not camera_serial.strip():
        raise DemoError(f"{path} requires a nonempty camera serial")
    return document


def load_setup(
    args: argparse.Namespace, hand_eye: dict | None = None
) -> tuple[dict, np.ndarray, list[dict]]:
    """Validate motion settings, calibration acceptance, and taught scan poses.

    Reuse the preloaded hand-eye result when supplied and require the depth
    verification serial to match. Return settings, the camera-to-gripper
    transform, and scan poses without connecting to or moving the arm.
    """
    settings = validate_demo_settings(DEMO_SETTINGS)
    settings["verbose"] = args.verbose
    settings["pause_before_grip"] = bool(getattr(args, "pause_before_grip", False))
    for key in ("grasp_rpy_deg", "hand_delivery_rpy_deg"):
        settings[key] = finite_vector(settings[key], 3, key)
    tilt_override = getattr(args, "grasp_tilt_deg", None)
    if tilt_override is not None:
        tilt_override = float(tilt_override)
        if not math.isfinite(tilt_override) or not 0.0 <= tilt_override <= MAXIMUM_GRASP_TILT_OVERRIDE_DEG:
            raise DemoError(
                f"--grasp-tilt-deg must be between 0 and {MAXIMUM_GRASP_TILT_OVERRIDE_DEG:.0f} degrees"
            )
        settings["grasp_rpy_deg"] = settings["grasp_rpy_deg"].copy()
        settings["grasp_rpy_deg"][1] = tilt_override
        print(
            f"[TEST] Grasp tilt overridden to {tilt_override:.1f}deg from vertical "
            f"(DEMO_SETTINGS uses {float(DEMO_SETTINGS['grasp_rpy_deg'][1]):.1f}deg)",
            flush=True,
        )
    for key in ("grasp_tip_z_mm", "grip_open_mm", "grip_close_mm"):
        if not math.isfinite(float(settings[key])):
            raise DemoError(f"demo.{key} must be finite")
    if not 0 <= float(settings["grip_close_mm"]) < float(settings["grip_open_mm"]) <= PIPER_C.GRIP_OPEN_MM:
        raise DemoError("gripper commands must be ordered within the physical gripper range")
    if hand_eye is None:
        hand_eye = load_hand_eye(args.calibration_dir)
    camera_serial = hand_eye["camera"]["serial"]
    for name in ("validation.json", "depth_correction.json", "depth_verification.json"):
        path = args.calibration_dir / name
        document = load_json(path)
        require_quality(document, name, path)
        if name == "depth_verification.json" and document.get("camera", {}).get("serial") != camera_serial:
            raise DemoError("depth verification belongs to a different camera")
    camera_to_gripper = rigid_transform(hand_eye, "camera_to_gripper")
    settings["camera_serial"] = camera_serial
    settings["detector"] = getattr(args, "detector", "yolo")
    settings["camera_intrinsics"] = hand_eye["camera"].get("intrinsics")
    if settings["detector"] == "opencv":
        intrinsics = settings["camera_intrinsics"]
        try:
            values = [float(intrinsics[key]) for key in ("fx", "fy", "ppx", "ppy")]
        except (KeyError, TypeError, ValueError) as exc:
            raise DemoError(f"OpenCV candy size check requires camera intrinsics in hand_eye.json: {exc}") from exc
        if not all(math.isfinite(value) for value in values) or values[0] <= 0 or values[1] <= 0:
            raise DemoError("OpenCV candy size check requires finite intrinsics with positive focal lengths")
    scans = load_scan_poses(args.scan_poses)
    return settings, camera_to_gripper, scans


def checked_move(arm: Any, name: str, target: np.ndarray, grip: float, settings: dict) -> None:
    """Sample the path from measured joints before executing a health-checked move."""
    validate_joint_path_safety(name, [measured_start_joints(arm, name), target])
    move_robot(
        arm, name, target, grip,
        float(settings["move_timeout_seconds"]), int(settings["motion_speed_percent"]),
    )


def verify_palm(palm: dict, maximum_age: float) -> np.ndarray:
    """Return finite base-frame coordinates for a fresh detected open palm.

    Handedness and temporal stability metadata are not checked by the demo.
    """
    if palm.get("status") != "detected":
        raise DemoError(f"no visible palm: {palm.get('status')}: {palm.get('error', '')}")
    if palm.get("gesture") != "open_palm":
        raise DemoError("detected hand is not an open palm")
    try:
        timestamp = datetime.fromisoformat(palm["captured_at"])
        if timestamp.tzinfo is None:
            raise ValueError("palm timestamp must include a timezone")
        age = (datetime.now(timezone.utc) - timestamp).total_seconds()
    except (KeyError, TypeError, ValueError) as exc:
        raise DemoError(f"invalid palm capture timestamp: {exc}") from exc
    if not -1.0 <= age <= maximum_age:
        raise DemoError(f"palm capture is stale or future-dated ({age:.2f}s)")
    return finite_vector(palm.get("base_point_mm"), 3, "palm center")


def plan_delivery(palm: dict, arm: Any, scan: dict, settings: dict, maximum_age: float):
    """Plan and path-check release at palm Z + 76.2 mm, retreat, and return to scan.

    Validate palm freshness and workspace bounds before IK; no motion is sent.
    """
    point = verify_palm(palm, maximum_age)
    validate_target_workspace(
        point, settings["hand_workspace_min_mm"], settings["hand_workspace_max_mm"]
    )
    release_z = float(point[2]) + RELEASE_CLEARANCE_MM
    plan = plan_hand_delivery(
        point, arm.joints_deg(), scan["joints_deg"],
        settings["hand_delivery_rpy_deg"], release_z,
        float(settings["hand_approach_height_mm"]),
    )
    validate_joint_path_safety(
        "palm delivery and retreat",
        [measured_start_joints(arm, "palm delivery and retreat"), *[joints for _, joints in plan]],
    )
    return plan


def reacquire_palm(
    arm: Any, objects: VisionClient, hands: HandClient, scan: dict,
    camera_to_gripper: np.ndarray, settings: dict,
) -> dict:
    """Capture a new palm at the current arm pose for delivery-plan validation."""
    return capture_hand_at_current_pose(
        arm, hands, objects, scan, int(scan["index"]) + 3, camera_to_gripper, settings,
    )


def wait_for_deliverable_palm(
    arm: Any, objects: VisionClient, hands: HandClient, scan: dict,
    camera_to_gripper: np.ndarray, settings: dict, maximum_age: float,
) -> tuple[dict, np.ndarray, dict]:
    """Hold the scan pose with the candy, capturing until a palm has an accepted delivery plan.

    Missing, invalid, stale, out-of-workspace, unreachable (IK), and path-rejected
    palms are retried every retry_delay_seconds without commanding motion. Arm
    health is checked before every attempt, so faults stop waiting; so does Ctrl+C.
    Return the palm, its base-frame center, and the delivery waypoints.
    """
    attempt = 0
    while True:
        require_healthy(arm, "waiting at scan pose with candy")
        attempt += 1
        try:
            palm = reacquire_palm(arm, objects, hands, scan, camera_to_gripper, settings)
            center = finite_vector(palm.get("base_point_mm"), 3, "reacquired palm center")
            plan = dict(plan_delivery(palm, arm, scan, settings, maximum_age))
            verify_palm(palm, maximum_age)
            return palm, center, plan
        except DemoError as exc:
            print(
                f"Waiting at scan pose with candy for a reachable open palm "
                f"(attempt {attempt}): {exc}",
                flush=True,
            )
        time.sleep(float(settings["retry_delay_seconds"]))


def deliver_to_palm(
    arm: Any, objects: VisionClient, hands: HandClient, scan: dict,
    camera_to_gripper: np.ndarray, settings: dict, maximum_age: float,
    carry: CandyCarry | None = None,
) -> None:
    """Wait at the scan pose for a deliverable palm, then approach and release above it.

    Palm detection and reachability failures do not stop or return the candy:
    the arm holds the scan pose and retries until a fresh palm has an accepted
    plan. The approach then uses that single detection without capturing again,
    so the palm must remain still. Reached waypoints are recorded in the carry
    tracker until the gripper starts opening at the palm, so the caller can
    retrace them after any later failure.
    """
    carry = CandyCarry() if carry is None else carry
    closed, opened = float(settings["grip_close_mm"]), float(settings["grip_open_mm"])
    checked_move(arm, "return to scan for palm reacquisition", scan["joints_deg"], closed, settings)
    carry.reached("scan pose", scan["joints_deg"])
    time.sleep(float(settings["scan_settle_seconds"]))
    palm, current, plan = wait_for_deliverable_palm(
        arm, objects, hands, scan, camera_to_gripper, settings, maximum_age,
    )
    print(f"Palm detected after pickup: {np.round(current, 3).tolist()}mm")
    checked_move(arm, "hand approach", plan["hand approach"], closed, settings)
    carry.reached("hand approach", plan["hand approach"])
    verify_palm(palm, maximum_age)
    checked_move(arm, "hand release height (palm Z + 76.2mm)", plan["hand release"], closed, settings)
    carry.reached("hand release height", plan["hand release"])
    verify_palm(palm, maximum_age)
    # Once opening starts at the palm, never carry anything back to the pickup location.
    carry.holding = False
    hold_gripper(
        arm, plan["hand release"], opened, float(settings["gripper_hold_seconds"])
    )
    checked_move(arm, "hand retreat", plan["hand retreat"], opened, settings)
    checked_move(arm, "return to scan", scan["joints_deg"], opened, settings)


def return_candy_to_pickup(
    arm: Any, pickup_plan: list[tuple[str, np.ndarray]], carry: CandyCarry, settings: dict,
) -> None:
    """Retrace the carried route to the grasp pose, release the candy, and retreat to scan.

    The carried path holds each waypoint reached since the gripper closed, so
    reversing it only crosses segments already traversed and path-checked. The
    whole route is sampled from the measured joints before the first move and
    each segment is rechecked before it is commanded. Waypoints already within
    the 2-degree arrival tolerance are skipped. The gripper stays closed until
    the grasp pose and opens only there.
    """
    if not carry.holding or not carry.path or carry.path[0][0] != "grasp":
        raise DemoError("no tracked held candy to return")
    waypoints = dict(pickup_plan)
    closed, opened = float(settings["grip_close_mm"]), float(settings["grip_open_mm"])
    timeout = float(settings["move_timeout_seconds"])
    speed = int(settings.get("pick_motion_speed_percent", settings["motion_speed_percent"]))
    grasp = carry.path[0][1]
    back = [
        (f"return candy: retrace {name}", joints) for name, joints in reversed(carry.path[1:])
    ] + [("return candy: pickup location", grasp)]
    retreat = [
        (label, waypoints[name]) for name, label in (
            ("lift", "return candy: retreat above pickup"),
            ("lift transit", "return candy: retreat transit"),
            ("scan 1", "return candy: back to scan"),
        ) if name in waypoints
    ]
    validate_joint_path_safety(
        "candy return to pickup location",
        [
            measured_start_joints(arm, "candy return to pickup location"),
            *[joints for _, joints in back + retreat],
        ],
    )
    for name, joints in back:
        current = finite_vector(arm.joints_deg(), 6, "current joints")
        if np.max(np.abs(current - joints)) < ARRIVAL_TOLERANCE_DEG:
            continue
        validate_joint_path_safety(name, [measured_start_joints(arm, name), joints])
        move_robot(arm, name, joints, closed, timeout, speed)
    print("motion: open gripper to release candy at its pickup location")
    carry.holding = False
    hold_gripper(arm, grasp, opened, float(settings["gripper_hold_seconds"]))
    for name, joints in retreat:
        validate_joint_path_safety(name, [measured_start_joints(arm, name), joints])
        move_robot(arm, name, joints, opened, timeout, speed)


def forward_lean_x(grasp_joints: np.ndarray) -> float:
    """Return the +X share of the horizontal direction the fingertips point (0 when vertical)."""
    tool_axis = fk(np.asarray(grasp_joints, dtype=np.float64))[:3, 2]
    horizontal = float(np.linalg.norm(tool_axis[:2]))
    if horizontal < math.sin(math.radians(1.0)):
        return 0.0
    return float(tool_axis[0]) / horizontal


def prefer_vertical_for_forward_lean(
    target: dict, plan: list[tuple[str, np.ndarray]], start: np.ndarray, scan: dict, settings: dict,
) -> list[tuple[str, np.ndarray]]:
    """Replace a tilted grasp whose fingertips point away from the base with a vertical one.

    Forward-leaning grasps (candies lying along X) clipped or toppled candies on
    the way down and picked 2/18 times; vertical grasps of the same candies
    picked 3/3. Keep the tilted plan when the vertical grasp is unreachable.
    """
    lean_x = forward_lean_x(dict(plan)["grasp"])
    if lean_x < float(settings["forward_lean_vertical_threshold"]):
        return plan
    vertical_rpy = np.asarray(settings["grasp_rpy_deg"], dtype=np.float64).copy()
    vertical_rpy[1] = 0.0
    vertical = select_reachable_object_target(
        [{**target, "cross_view_agreement_passed": True}], start, scan["joints_deg"], vertical_rpy,
        float(settings["grasp_tip_z_mm"]), float(settings["approach_height_mm"]),
        float(settings["pregrasp_height_mm"]), float(settings["minimum_pregrasp_height_mm"]),
        float(settings["pregrasp_height_step_mm"]), float(settings["maximum_approach_inward_offset_mm"]),
        float(settings["approach_inward_step_mm"]), float(settings["maximum_grasp_yaw_offset_deg"]),
        float(settings["grasp_yaw_offset_step_deg"]),
    )
    if vertical is None:
        if settings.get("verbose"):
            print(
                f"Grasp leans forward (lean X {lean_x:.2f}) but no vertical grasp is reachable; "
                "keeping the tilted grasp",
                flush=True,
            )
        return plan
    if settings.get("verbose"):
        print(f"Grasp leans forward (lean X {lean_x:.2f}); using a vertical grasp instead", flush=True)
    return vertical[2]


def pick_plan(arm: Any, captures: list[dict], scan: dict, settings: dict):
    """Select the nearest candy with a reachable, fully path-checked pickup plan.

    Cluster detections and try each candy nearest first, with candy-aligned
    grasp orientations and approach alternatives; a candy whose IK or route
    checks fail is skipped in favour of the next one. Raise
    NoReachableCandyError when none can be planned. No motion is sent.
    """
    targets = distinct_object_targets(
        captures, float(settings["maximum_cross_view_target_difference_mm"]),
        float(settings["maximum_cross_view_orientation_difference_deg"]),
    )
    start = measured_start_joints(arm, "pickup planning")
    selected = select_reachable_object_target(
        targets, start, scan["joints_deg"], settings["grasp_rpy_deg"],
        float(settings["grasp_tip_z_mm"]), float(settings["approach_height_mm"]),
        float(settings["pregrasp_height_mm"]), float(settings["minimum_pregrasp_height_mm"]),
        float(settings["pregrasp_height_step_mm"]), float(settings["maximum_approach_inward_offset_mm"]),
        float(settings["approach_inward_step_mm"]), float(settings["maximum_grasp_yaw_offset_deg"]),
        float(settings["grasp_yaw_offset_step_deg"]),
    )
    if selected is None:
        reasons = [
            f"target {index} at {np.round(target['base_point_mm'], 3).tolist()}mm: "
            f"{target['planning_rejection_reason']}"
            for index, target in enumerate(targets, start=1)
        ]
        detail = "; ".join(reasons) if reasons else "no detected candy targets were available"
        raise NoReachableCandyError(
            f"detected candy has no accepted reachable pickup plan: {detail}", targets,
        )
    target, _, plan, rejections = selected
    plan = prefer_vertical_for_forward_lean(target, plan, start, scan, settings)
    for rejection in rejections:
        print(
            f"Skipping unreachable candy at {np.round(rejection['base_point_mm'], 1).tolist()}mm: "
            f"{summarize_rejection(rejection['reason'])}",
            flush=True,
        )
    waypoints = dict(plan)
    ordered = [start]
    for name in ("pick transit", "pick approach", "grasp", "lift", "lift transit", "scan 1"):
        if name in waypoints:
            ordered.append(waypoints[name])
    validate_joint_path_safety("pickup and return to scan", ordered)
    return target, plan


def pick_diagnostic(capture: dict, target: dict, plan: list[tuple[str, np.ndarray]]) -> str:
    """Describe the selected pickup for log review: source frame, candy, and grasp geometry."""
    waypoints = dict(plan)
    grasp_frame = fk(waypoints["grasp"])
    rotation = grasp_frame[:3, :3]
    tilt = math.degrees(math.acos(float(np.clip(-rotation[2, 2], -1.0, 1.0))))
    axis = np.asarray(target.get("long_axis_base_unit_xy", [math.nan, math.nan]), dtype=np.float64)
    jaw = rotation[:2, 1] / max(float(np.linalg.norm(rotation[:2, 1])), 1e-9)
    jaw_angle = math.degrees(math.acos(float(np.clip(abs(jaw @ axis), 0.0, 1.0)))) if np.all(np.isfinite(axis)) else math.nan
    size = (
        f", size {target['length_mm']:.0f}x{target['width_mm']:.0f}mm" if "length_mm" in target else ""
    )
    # Horizontal direction the fingertips point; forward (+X) leans correlated with missed grasps.
    lean_xy = rotation[:2, 2]
    lean = (
        f" leaning {np.round(lean_xy / np.linalg.norm(lean_xy), 2).tolist()}"
        if tilt >= 1.0 else " (vertical)"
    )
    return (
        f"PICK PLAN: frame={capture.get('dataset_image', '?')} captured_at={capture.get('captured_at', '?')} "
        f"score={float(target.get('score', math.nan)):.2f}{size}, candy axis yaw "
        f"{float(target.get('long_axis_base_yaw_deg', math.nan)):.1f}deg, jaws at {jaw_angle:.1f}deg "
        f"to candy axis (90=across), tool tilt {tilt:.1f}deg{lean}, route={[name for name, _ in plan]}"
    )


def run_cycle(
    arm: Any, objects: VisionClient, hands: HandClient, scans: list[dict],
    camera_to_gripper: np.ndarray, settings: dict, maximum_age: float,
) -> bool:
    """Scan taught poses, pick one accepted candy, then deliver to a freshly checked palm.

    Return True after delivery, or False when no candy/palm pair is accepted,
    no detected candy can be planned (unreachable candies are skipped in favour
    of reachable ones, and the scan repeats when none are), grasp verification
    fails, or a demo error occurs while a candy may be
    gripped (the carried route is then retraced and the candy put back at its
    pickup location). A missing or unreachable palm after pickup is not an
    error: delivery waits at the scan pose until one is deliverable. Ctrl+C,
    errors with nothing gripped, and failed returns stop the outer loop.
    """
    for scan in scans:
        validate_joint_path_safety(
            "scan approach", [measured_start_joints(arm, "scan approach"), scan["joints_deg"]],
        )
        candy, palm = capture_scan_pair(
            arm, objects, hands, scan, int(scan["index"]) + 3,
            PROMPT, float(settings["detection_confidence_threshold"]), camera_to_gripper, settings,
        )
        if candy.get("status") != "detected" or palm.get("status") != "detected":
            print("No accepted candy/palm pair; continuing scan without pickup", flush=True)
            continue
        verify_palm(palm, maximum_age)
        try:
            target, plan = pick_plan(arm, [candy], scan, settings)
        except NoReachableCandyError as exc:
            print("No reachable candy in this scan; scanning again:", flush=True)
            for target in exc.targets or []:
                print(
                    f"  candy at {np.round(target['base_point_mm'], 1).tolist()}mm: "
                    f"{summarize_rejection(target.get('planning_rejection_reason', 'rejected'))}",
                    flush=True,
                )
            if settings.get("verbose"):
                print(f"  details: {exc}", flush=True)
            continue
        print(f"Picking Starburst at {np.round(target['base_point_mm'], 3).tolist()}mm", flush=True)
        if settings.get("verbose"):
            print(pick_diagnostic(candy, target, plan), flush=True)
        carry = CandyCarry()
        try:
            held_width = execute_pickup(arm, plan, settings, carry, target)
            print(f"Pickup confirmed: gripper width={held_width:.2f}mm", flush=True)
            deliver_to_palm(arm, objects, hands, scan, camera_to_gripper, settings, maximum_age, carry)
        except GraspVerificationError as exc:
            print(f"Pickup missed: {exc}; scanning again", flush=True)
            return False
        except (RuntimeError, OSError, KeyError, TypeError, ValueError) as exc:
            if not carry.holding:
                raise
            print(f"ERROR while holding candy: {exc}", flush=True)
            print("Returning candy to its pickup location", flush=True)
            try:
                return_candy_to_pickup(arm, plan, carry, settings)
            except (RuntimeError, OSError, KeyError, TypeError, ValueError) as return_exc:
                raise DemoError(f"candy return failed after '{exc}': {return_exc}") from return_exc
            print("Candy returned; continuing scan loop", flush=True)
            return False
        return True
    return False


def return_to_zero(
    arm: Piper, *, context: str = "interrupt zero return", verbose: bool = True,
    open_grip_mm: float | None = None, open_seconds: float = 1.0,
) -> None:
    """Validate and execute a slow SDK-zero return, optionally opening the gripper there.

    Reject unhealthy arms or unsafe sampled paths before sending a zero command,
    and require measured convergence. The gripper is unchanged during the move;
    when open_grip_mm is given it opens only after zero is reached and verified
    (a held candy drops at the zero pose), and the measured width must come
    within 5 mm of the command. Failures are raised without recovery motion.
    """
    require_healthy(arm, f"before {context}")
    current = finite_vector(arm.joints_deg(), 6, f"{context} joint position")
    validate_joint_path_safety(
        context, [current, np.zeros(6)],
        joint_limits=np.asarray(PIPER_C.JOINT_LIMITS_DEG, dtype=np.float64),
    )
    if verbose:
        detail = "gripper opens on arrival" if open_grip_mm is not None else "without opening the gripper"
        print(f"{context}: moving to SDK zero pose; {detail}.", flush=True)
    if not arm.move_to_zero(
        secs=PIPER_C.HOME_TIMEOUT_S, speed=PIPER_C.SPEED_REST,
    ):
        raise DemoError(f"{context} did not reach zero")
    require_healthy(arm, f"after {context}")
    measured = finite_vector(arm.joints_deg(), 6, "zero-return joint position")
    if np.any(np.abs(measured) > PIPER_C.HOME_TOL_DEG):
        raise DemoError(f"{context} feedback is outside zero-pose tolerance")
    print("[OK] Robot reached SDK zero pose.", flush=True)
    if open_grip_mm is None:
        return
    hold_gripper(arm, np.zeros(6), float(open_grip_mm), float(open_seconds))
    width = float(arm.grip_mm())
    if not math.isfinite(width) or width < float(open_grip_mm) - 5.0:
        raise DemoError(
            f"{context}: gripper did not open at zero (measured {width:.1f}mm, "
            f"commanded {float(open_grip_mm):.1f}mm)"
        )
    print(f"[OK] Gripper open at zero: {width:.1f}mm.", flush=True)


def prepare_robot_for_scan(arm: Piper, scan: dict, settings: dict) -> None:
    """Move through checked SDK zero (opening the gripper there) to the first scan pose."""
    target = finite_vector(scan["joints_deg"], 6, "first scan joints")
    factory_limits = np.asarray(PIPER_C.JOINT_LIMITS_DEG, dtype=np.float64)
    validate_joint_path_safety(
        "startup zero to first scan", [np.zeros(6), target], joint_limits=factory_limits,
    )
    return_to_zero(
        arm, context="startup zero", verbose=settings.get("verbose", False),
        open_grip_mm=float(settings["grip_open_mm"]),
        open_seconds=float(settings["gripper_hold_seconds"]),
    )
    validate_joint_path_safety(
        "startup first scan",
        [measured_start_joints(arm, "startup first scan", factory_limits), target],
        joint_limits=factory_limits,
    )
    move_robot(
        arm, "initial scan pose after zero", target, None,
        float(settings["move_timeout_seconds"]), int(settings["motion_speed_percent"]),
        verbose=settings.get("verbose", False),
    )


def main() -> int:
    """Preflight services/setup, enforce motion gates, and run the supervised loop.

    Check-only never connects to CAN. Every connected exit attempts a checked
    SDK-zero return, opens the gripper once zero is reached, then disconnects.
    """
    args = parse_args()
    if args.max_cycles < 0:
        print("ERROR: max-cycles must be nonnegative", file=sys.stderr)
        return 2
    arm = None
    exit_code = 0
    try:
        hand_eye = load_hand_eye(args.calibration_dir)
        camera_serial = hand_eye["camera"]["serial"]
        objects = VisionClient(
            args.vision_url, detection_url=args.detection_url,
            detector=OpenCVCandyDetector() if args.detector == "opencv" else None,
        )
        hands = HandClient(args.hand_url)
        check_services(HttpClient(args.vision_url), objects, hands, camera_serial)
        settings, camera_to_gripper, scans = load_setup(args, hand_eye)
        print(f"[OK] Accepted D435 calibration and {len(scans)} taught scan pose(s)", flush=True)
        if args.check_only:
            print("Preflight passed; no CAN connection or arm motion", flush=True)
            return 0
        if not args.allow_motion:
            raise DemoError("motion requires --allow-motion")
        if settings.get("pause_before_grip") and not sys.stdin.isatty():
            raise DemoError("--pause-before-grip needs an interactive terminal to type candy offsets")
        print(
            "Will begin robot movement shortly, keep emergency stop accessible. "
            "Connecting in 5 seconds.", flush=True,
        )
        for _ in range(5):
            time.sleep(1)
        arm = connect_robot(iface=args.can_interface, verbose=args.verbose)
        prepare_robot_for_scan(arm, scans[0], settings)
        cycle = 0
        while args.max_cycles == 0 or cycle < args.max_cycles:
            cycle += 1
            print(f"\nIGNITE CYCLE {cycle}", flush=True)
            run_cycle(arm, objects, hands, scans, camera_to_gripper, settings, MAXIMUM_HAND_AGE_SECONDS)
            time.sleep(float(settings["retry_delay_seconds"]))
    except KeyboardInterrupt:
        if arm is None:
            print("Interrupted before arm connection; no zero-return motion.", file=sys.stderr)
        exit_code = 130
    except (RuntimeError, CalibrationError, HandEyeError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: Ignite demo stopped: {exc}", file=sys.stderr)
        exit_code = 2
    finally:
        if arm is not None:
            try:
                return_to_zero(
                    arm, context="shutdown zero return", verbose=args.verbose,
                    open_grip_mm=float(settings["grip_open_mm"]),
                    open_seconds=float(settings["gripper_hold_seconds"]),
                )
            except KeyboardInterrupt:
                print("Zero return interrupted; disconnecting without further motion.", file=sys.stderr)
                exit_code = 130
            except (RuntimeError, OSError, KeyError, TypeError, ValueError) as exc:
                print(f"ERROR: Cannot return to zero; disconnecting without further motion: {exc}", file=sys.stderr)
                if exit_code == 0:
                    exit_code = 2
            finally:
                arm.disconnect()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
