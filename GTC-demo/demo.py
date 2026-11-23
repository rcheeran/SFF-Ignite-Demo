#!/usr/bin/env python3
"""GTC candy demo: pick a candy from a bowl and drop it on an open hand.

Each cycle, at the taught scan pose:
  1. Find the bowl and the candies in it with OpenCV (bowl_vision.py), confirm
     the rim with depth, and pick the highest reachable candy: straight down,
     or tilted up to 30 degrees for far bowls, always clear of the bowl wall.
  2. Hold the candy at the scan pose until an open palm's finger base is
     visible inside the hand workspace (hand_target.py).
  3. Release 35 mm above the palm centre, or above the finger base when the
     palm centre is not visible, then repeat.

  --check-only   preflight only; never connects to CAN
  --scan-only    move to the scan pose and report the bowl, candies, and hand; no pick
  --pick-only    pick from the bowl and drop the candy back in; no hand needed
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
import math
from pathlib import Path
import subprocess
import sys
import time
import threading
from typing import Any, Callable, Iterable

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vision_client import (  # noqa: E402
    DEFAULT_VISION_URL,
    CameraClient,
    DemoError,
    HandClient,
)
import bowl_vision  # noqa: E402
from hand_target import choose_drop_target  # noqa: E402

from piper import config as PIPER_C  # noqa: E402
from piper.kinematics import MARGINED_LIMITS, fk, ik  # noqa: E402
from piper.robot import Piper, connect_robot  # noqa: E402
from piper.transforms import euler_to_rot, fk_flange, tip_position  # noqa: E402

CALIBRATION_DIR = ROOT / "prepare-piper/hand-eye-realsense435"
K3S_NODE_IP_SYNC = ROOT / "prepare-jetson/k3s-node-ip-sync.sh"
K3S_ADDRESS_FIX_TIMEOUT_SECONDS = 600.0  # TimeoutStartSec of k3s-node-ip-sync.service
RELEASE_CLEARANCE_MM = 35.0
# A move counts as arrived when every joint is within this many degrees of its target.
ARRIVAL_TOLERANCE_DEG = 2.0
# SDK zero lies on the J2 and J3 factory limits, so an arm resting at or near zero
# (e.g. settled after a power cycle) can read a few degrees past them. A zero return
# moves back inside the limits, so its start may be this far outside them.
ZERO_RETURN_START_TOLERANCE_DEG = 5.0
MAXIMUM_HAND_AGE_SECONDS = 10.0
MAXIMUM_CANDY_XY_CORRECTION_MM = 60.0
HAND_SLOT = 4
# The finger pads reach this far below the tool tip (pads span 105..135 mm past the flange).
FINGER_EXTENSION_BELOW_TIP_MM = 15.0
LOG_DIR = Path(__file__).resolve().parent / "logs"
GRASP_OFFSET_LOG = LOG_DIR / "grasp_offsets.jsonl"
BOWL_SCAN_IMAGE = LOG_DIR / "latest_bowl_scan.jpg"
PROMPT = "christmas candy"  # label for capture_scan_images.py reference detections
DEFAULT_SCAN_POSES = ROOT / "prepare-piper/scan-poses/scan_poses.json"
WORKSPACE_CORNERS_MM = (
    (144.644, 206.317, 18.0),  # C2
    (144.0, -170.0, 18.0),  # C4
    (595.0, -170.0, 18.0),  # C6
    (595.0, 206.0, 18.0),  # C8
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
    # 40 mm clears the 25.4 mm candy and keeps the open jaws away from the bowl wall
    # (bowl_vision checks the jaw footprint against the wall at this width).
    "grip_open_mm": 40.000,
    "grip_close_mm": 1.000,
    # The candy is 19-25 mm across and an empty close reads about 1 mm.
    # Not yet measured with the new candy.
    "minimum_held_grip_mm": 12.000,
    # Vertical grasp; yaw is set per candy so the jaws close across one of its edges.
    "grasp_rpy_deg": [180.000, 0.000, 0.000],
    # Tilts tried in order when a vertical grasp cannot reach the candy (far
    # bowls). The tilt pivots about the jaws' closing axis, so they still close
    # level across the candy; the gripper descends along its own axis and must
    # stay inside the bowl opening from the fingertips up to the rim.
    "grasp_tilt_options_deg": [0.000, 10.000, 20.000, 30.000],
    # The tool tip is the middle of the finger pads, which reach ~15 mm below it.
    # 5 mm above the candy's top surface puts the fingertips ~10 mm down its
    # side, which needs less room between neighbouring candies in a full bowl.
    # Confirm with --pause-before-grip.
    "grasp_tip_above_candy_top_mm": 5.000,
    # Depth under each finger's landing spot must be at least this far below the
    # fingertips at the grasp pose, so the fingers do not come down on a neighbour.
    "finger_landing_clearance_mm": 2.000,
    # Depth reads are limited to this many of the best candidates per scan.
    "maximum_candies_depth_checked": 8,
    # Fingertips 15 mm below the tool tip stay above the bowl floor (table -3 mm,
    # floor ~6 mm thick) with margin.
    "minimum_grasp_tip_z_mm": 20.000,
    # Approach and lift this far above the rim, then move straight in/out along the tool axis.
    "rim_clearance_mm": 30.000,
    # Waypoints along the descent keep the joint-space path straight; any sampled
    # tip position below the rim (plus finger length) must stay within this
    # distance of the planned line into the candy.
    "vertical_step_mm": 15.000,
    "maximum_vertical_deviation_mm": 3.000,
    "grasp_yaw_offset_step_deg": 5.000,
    "maximum_grasp_yaw_offset_deg": 10.000,
    # Added to every candy's base XY before planning. Measure it with
    # --pause-before-grip, which prints the suggested value after each reading.
    "candy_xy_correction_mm": [0.000, 0.000],
    "motion_speed_percent": 20,
    "pick_motion_speed_percent": 20,
    "hand_approach_height_mm": 20.000,
    # Pitch raised from the taught 10.012 degrees: at 10 degrees J5 (wrist pitch,
    # +/-70 degrees) limited palm reach to X ~410-440 mm; 20 degrees reaches X 500 mm.
    "hand_delivery_rpy_deg": [178.517, 20.000, 171.532],
    "gripper_hold_seconds": 1.000,
    "maximum_scan_position_deviation_mm": 2.000,
    "maximum_scan_rotation_deviation_deg": 0.750,
    "move_timeout_seconds": 8.000,
    "retry_delay_seconds": 1.000,
    "scan_settle_seconds": 1.000,
}


class NoReachableCandyError(DemoError):
    """No pickable candy in the bowl has an accepted, path-checked vertical grasp."""

    def __init__(self, message: str, rejections: list[dict] | None = None) -> None:
        super().__init__(message)
        self.rejections = list(rejections or [])


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
    tolerance_deg: float = ARRIVAL_TOLERANCE_DEG,
) -> np.ndarray:
    """Return measured joints as a path-check start, tolerating arrival settling.

    A joint may settle up to the 2-degree arrival tolerance past a planned
    waypoint, so a waypoint planned exactly on an inward-margined limit can read
    slightly beyond it. Overshoot up to tolerance_deg is clamped back onto the
    limit for path sampling; anything further out raises. With the default
    margined limits this allows readings up to, but never past, the factory joint
    limits. Planned waypoints themselves are not relaxed.
    """
    current = finite_vector(arm.joints_deg(), 6, f"{name} measured joints")
    limits = np.asarray(joint_limits, dtype=np.float64)
    overshoot = np.maximum(limits[:, 0] - current, current - limits[:, 1])
    if np.any(overshoot > tolerance_deg):
        joint = int(np.argmax(overshoot))
        raise DemoError(
            f"{name}: measured J{joint + 1}={current[joint]:.2f}deg is more than "
            f"{tolerance_deg:.0f}deg outside "
            f"[{limits[joint, 0]:.1f}, {limits[joint, 1]:.1f}]deg"
        )
    return np.clip(current, limits[:, 0], limits[:, 1])


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


def ensure_k3s_address() -> None:
    """Check that k3s still uses this host's network address; restart k3s if not.

    k3s keeps the address it started with, so a network change while it runs
    leaves the cluster on an address the host no longer has. The restart goes
    through k3s-node-ip-sync.service; running pods keep running. An unknown
    result (k3s not reachable, or sudo needs a password) is only a warning,
    because the service checks report any outage that matters to the demo.
    """
    def check() -> tuple[int, str]:
        try:
            result = subprocess.run(
                [str(K3S_NODE_IP_SYNC), "--check"], capture_output=True, text=True, timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return 2, f"unknown: cannot run {K3S_NODE_IP_SYNC.name}: {exc}"
        return result.returncode, result.stdout.strip() or result.stderr.strip()

    code, status = check()
    if code == 1:
        print(f"[FIXING] k3s address {status}; restarting k3s (running pods keep running)", flush=True)
        fix_error = ""
        try:
            fix = subprocess.run(
                ["sudo", "-n", "systemctl", "start", "k3s-node-ip-sync.service"],
                capture_output=True, text=True, timeout=K3S_ADDRESS_FIX_TIMEOUT_SECONDS,
            )
            if fix.returncode:
                fix_error = fix.stderr.strip() or f"exit code {fix.returncode}"
        except (OSError, subprocess.TimeoutExpired) as exc:
            fix_error = str(exc)
        code, status = check()
        if code == 1 and fix_error:
            status += f" (restart failed: {fix_error})"
    if code == 0:
        print(f"[OK] k3s address {status}", flush=True)
    elif code == 1:
        print(f"[FAILED] k3s address {status}", flush=True)
        raise DemoError("k3s is not using this host's network address; see EVENT_DAY.md troubleshooting")
    else:
        print(f"[WARN] k3s address {status}", flush=True)


def load_hand_eye(calibration_dir: Path) -> dict:
    """Load accepted hand-eye results with a nonempty serial for camera health checks."""
    path = calibration_dir / "hand_eye.json"
    document = load_json(path)
    require_quality(document, "hand_eye.json", path)
    camera_serial = document.get("camera", {}).get("serial")
    if not isinstance(camera_serial, str) or not camera_serial.strip():
        raise DemoError(f"{path} requires a nonempty camera serial")
    return document


def checked_move(arm: Any, name: str, target: np.ndarray, grip: float, settings: dict) -> None:
    """Sample the path from measured joints before executing a health-checked move."""
    validate_joint_path_safety(name, [measured_start_joints(arm, name), target])
    move_robot(
        arm, name, target, grip,
        float(settings["move_timeout_seconds"]), int(settings["motion_speed_percent"]),
    )


def return_to_zero(
    arm: Piper, *, context: str = "interrupt zero return", verbose: bool = True,
    open_grip_mm: float | None = None, open_seconds: float = 1.0,
) -> None:
    """Validate and execute a slow SDK-zero return, optionally opening the gripper there.

    Reject unhealthy arms or unsafe sampled paths before sending a zero command,
    and require measured convergence. The start may read up to
    ZERO_RETURN_START_TOLERANCE_DEG past a factory limit (zero sits on the J2/J3
    limits); it is clamped onto the limit for path sampling. The gripper is unchanged during the move;
    when open_grip_mm is given it opens only after zero is reached and verified
    (a held candy drops at the zero pose), and the measured width must come
    within 5 mm of the command. Failures are raised without recovery motion.
    """
    require_healthy(arm, f"before {context}")
    factory_limits = np.asarray(PIPER_C.JOINT_LIMITS_DEG, dtype=np.float64)
    current = measured_start_joints(
        arm, context, factory_limits, ZERO_RETURN_START_TOLERANCE_DEG,
    )
    validate_joint_path_safety(context, [current, np.zeros(6)], joint_limits=factory_limits)
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


def validate_demo_settings(settings: dict) -> dict:
    """Validate code-defined runtime settings and return them with workspace bounds."""
    try:
        demo = dict(settings)
        target_min = finite_vector(WORKSPACE_MIN_MM, 3, "code-defined workspace minimum").copy()
        target_max = finite_vector(WORKSPACE_MAX_MM, 3, "code-defined workspace maximum").copy()
        for key in ("grasp_rpy_deg", "hand_delivery_rpy_deg"):
            demo[key] = finite_vector(demo[key], 3, f"demo.{key}")
        tilts = [float(value) for value in demo["grasp_tilt_options_deg"]]
        if not tilts or not all(math.isfinite(value) and 0.0 <= value <= 35.0 for value in tilts):
            raise ValueError("demo.grasp_tilt_options_deg must list tilts from 0 to 35 degrees")
        demo["grasp_tilt_options_deg"] = tilts
        correction = finite_vector(demo["candy_xy_correction_mm"], 2, "demo.candy_xy_correction_mm").copy()
        for key in ("grasp_tip_above_candy_top_mm", "minimum_grasp_tip_z_mm", "grip_close_mm", "finger_landing_clearance_mm"):
            if not math.isfinite(float(demo[key])):
                raise ValueError(f"demo.{key} must be finite")
    except (KeyError, TypeError, ValueError) as exc:
        raise DemoError(f"invalid demo configuration: {exc}") from exc
    if np.any(target_min >= target_max):
        raise DemoError("demo target workspace minimum must be below maximum")
    if np.any(np.abs(correction) > MAXIMUM_CANDY_XY_CORRECTION_MM):
        raise DemoError(
            f"demo.candy_xy_correction_mm components must be within +/-{MAXIMUM_CANDY_XY_CORRECTION_MM:.0f}mm"
        )
    demo["candy_xy_correction_mm"] = correction
    for key in (
        "grip_open_mm", "minimum_held_grip_mm", "rim_clearance_mm", "vertical_step_mm",
        "maximum_vertical_deviation_mm", "grasp_yaw_offset_step_deg", "maximum_grasp_yaw_offset_deg",
        "motion_speed_percent", "pick_motion_speed_percent", "hand_approach_height_mm",
        "gripper_hold_seconds", "maximum_scan_position_deviation_mm", "maximum_scan_rotation_deviation_deg",
        "move_timeout_seconds", "retry_delay_seconds", "scan_settle_seconds", "maximum_candies_depth_checked",
    ):
        value = float(demo[key])
        if not math.isfinite(value) or value <= 0:
            raise DemoError(f"demo.{key} must be positive")
    if not 0 <= float(demo["grip_close_mm"]) < float(demo["minimum_held_grip_mm"]) < float(demo["grip_open_mm"]) <= PIPER_C.GRIP_OPEN_MM:
        raise DemoError("gripper settings must satisfy 0 <= close < minimum held < open <= the gripper's range")
    if float(demo["grasp_yaw_offset_step_deg"]) > float(demo["maximum_grasp_yaw_offset_deg"]):
        raise DemoError("demo.grasp_yaw_offset_step_deg must not exceed demo.maximum_grasp_yaw_offset_deg")
    for key in ("motion_speed_percent", "pick_motion_speed_percent"):
        speed = int(demo[key])
        if speed != float(demo[key]) or speed > 100:
            raise DemoError(f"demo.{key} must be an integer from 1 to 100")
        demo[key] = speed
    demo["target_workspace_min_mm"], demo["target_workspace_max_mm"] = target_min, target_max
    demo["hand_workspace_min_mm"], demo["hand_workspace_max_mm"] = target_min.copy(), target_max.copy()
    # One gripper width for planning and the bowl-wall check.
    demo["bowl"] = {**bowl_vision.BOWL_SETTINGS, "grip_open_mm": float(demo["grip_open_mm"])}
    return demo


def load_setup(args: argparse.Namespace, hand_eye: dict | None = None) -> tuple[dict, np.ndarray, list[dict]]:
    """Validate settings, calibration acceptance, and taught scan poses without moving the arm.

    Return settings, the camera-to-gripper transform, and the scan poses.
    """
    settings = validate_demo_settings(DEMO_SETTINGS)
    settings["verbose"] = bool(getattr(args, "verbose", False))
    settings["pause_before_grip"] = bool(getattr(args, "pause_before_grip", False))
    if hand_eye is None:
        hand_eye = load_hand_eye(args.calibration_dir)
    camera_serial = hand_eye["camera"]["serial"]
    for name in ("validation.json", "depth_correction.json", "depth_verification.json"):
        path = args.calibration_dir / name
        document = load_json(path)
        require_quality(document, name, path)
        if name == "depth_verification.json" and document.get("camera", {}).get("serial") != camera_serial:
            raise DemoError("depth verification belongs to a different camera")
    intrinsics = hand_eye["camera"].get("intrinsics")
    try:
        bowl_vision.camera_matrix(intrinsics)
    except (KeyError, TypeError, ValueError) as exc:
        raise DemoError(f"bowl detection requires camera intrinsics in hand_eye.json: {exc}") from exc
    settings["camera_serial"] = camera_serial
    settings["camera_intrinsics"] = intrinsics
    return settings, rigid_transform(hand_eye, "camera_to_gripper"), load_scan_poses(args.scan_poses)


def check_services(camera: CameraClient, hands: HandClient | None, camera_serial: str) -> None:
    """Check the camera and, when needed, the hand service; raise after reporting all failures."""
    checks = [("OpenCV camera", camera.health)]
    if hands is not None:
        checks.append(("MediaPipe hand service", hands.health))
    failures = []
    for label, check in checks:
        try:
            result = check()
            if label == "OpenCV camera" and (
                result.get("camera_serial") != camera_serial
                or result.get("preview_only") is not False
                or result.get("depth_calibrated") is not True
            ):
                raise DemoError("requires matching D435 and calibrated depth; preview-only is not usable")
            if label == "MediaPipe hand service" and result.get("source_url") != camera.base_url + "/stream.mjpg":
                raise DemoError("hand stream must come from the configured OpenCV camera URL")
            print(f"[OK] {label}: {result}", flush=True)
        except RuntimeError as exc:
            failures.append(f"{label}: {exc}")
            print(f"[FAILED] {label}: {exc}", flush=True)
    if failures:
        raise DemoError("required services unavailable: " + "; ".join(failures))
    print("[OK] All required services are running and available.", flush=True)


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
        f"closing direction, {along:+.1f}mm along the jaws",
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
        "candy_base_point_mm": target.get("base_point_mm"),
        "uncorrected_base_point_mm": target.get("uncorrected_base_point_mm", target.get("base_point_mm")),
        "candy_edge_yaw_deg": target.get("edge_yaw_deg"),
        "jaw_point_mm": np.round(jaw_point, 3).tolist(),
        "jaw_axis_xy": np.round(jaw_axis_xy, 4).tolist(),
        "offset_mm": np.round(offset, 3).tolist(),
        "across_jaw_mm": round(across, 3),
        "along_jaw_mm": round(along, 3),
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
        f"  candy estimate {np.round(candy, 1).tolist() if candy is not None else '?'}mm, candy edge yaw "
        f"{float(target.get('edge_yaw_deg', math.nan)):.1f}deg, "
        f"jaw point {np.round(jaw_point, 1).tolist()}mm\n"
        "  Measure the candy centre relative to the midpoint between the open fingertips,\n"
        "  in robot base directions:\n"
        "    +X = straight out from the robot base (toward the far edge), -X = back toward the base\n"
        "    +Y = to your left when standing behind the robot facing +X, -Y = to your right",
        flush=True,
    )
    offset = hold_for_operator_input(
        arm, grasp_joints, open_mm,
        "  Candy offset 'dx dy' in mm (e.g. '0 -3'), or Enter to grip without recording: ",
        parse_grasp_offset,
    )
    if offset is None:
        print("  No offset recorded", flush=True)
        return None
    record_grasp_offset(offset, target, jaw_point, jaw_axis, demo)
    return offset


def stationary_camera_to_base(
    before_pose: np.ndarray, after_pose: np.ndarray, camera_to_gripper: np.ndarray, settings: dict, label: str,
) -> np.ndarray:
    """Camera-to-base transform for a capture, rejecting arm movement during it."""
    before, after = piper_pose_to_transform(before_pose), piper_pose_to_transform(after_pose)
    moved = float(np.linalg.norm(np.asarray(after_pose)[:3] - np.asarray(before_pose)[:3]))
    rotated = rotation_error_degrees(before, after)
    if moved > float(settings["maximum_scan_position_deviation_mm"]):
        raise DemoError(f"{label}: arm moved {moved:.2f}mm during capture")
    if rotated > float(settings["maximum_scan_rotation_deviation_deg"]):
        raise DemoError(f"{label}: arm rotated {rotated:.2f}deg during capture")
    return average_transforms([before, after]) @ camera_to_gripper


def to_base_point(camera_to_base: np.ndarray, camera_point: np.ndarray) -> np.ndarray:
    return (camera_to_base @ np.append(np.asarray(camera_point, dtype=np.float64), 1.0))[:3]


def base_to_pixel(camera_to_base: np.ndarray, intrinsics: dict, point_mm: Any) -> np.ndarray | None:
    """Photo pixel of a base-frame point, or None if it is behind the camera."""
    camera_point = (np.linalg.inv(camera_to_base) @ np.append(np.asarray(point_mm, dtype=np.float64), 1.0))[:3]
    if camera_point[2] <= 1.0:
        return None
    return (bowl_vision.camera_matrix(intrinsics) @ (camera_point / camera_point[2]))[:2]


def grasp_tip_z(candy_top_z: float, settings: dict) -> float:
    return max(float(candy_top_z) + float(settings["grasp_tip_above_candy_top_mm"]), float(settings["minimum_grasp_tip_z_mm"]))


def finger_landing_problem(
    candy: dict, closing_yaw_deg: float, camera: CameraClient, token: str, camera_to_base: np.ndarray, settings: dict,
) -> str | None:
    """Why the fingers would come down on something for this closing direction, or None if clear.

    Reads depth where each open finger lands (beside the candy along the
    closing direction); anything there higher than the fingertips at the grasp
    pose, less the clearance, blocks this direction. Missing depth also blocks.
    """
    top = np.asarray(candy["base_point_mm"], dtype=np.float64)
    fingertip_z = grasp_tip_z(top[2], settings) - FINGER_EXTENSION_BELOW_TIP_MM
    limit = fingertip_z - float(settings["finger_landing_clearance_mm"])
    yaw = math.radians(closing_yaw_deg)
    direction = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    offset = settings["bowl"]["grip_open_mm"] / 2.0 + settings["bowl"]["finger_thickness_mm"] / 2.0
    intrinsics = settings["camera_intrinsics"]
    for sign in (-1.0, 1.0):
        pixel = base_to_pixel(camera_to_base, intrinsics, top + sign * offset * direction)
        if pixel is None or not (0 <= pixel[0] < int(intrinsics.get("width", 640)) and 0 <= pixel[1] < int(intrinsics.get("height", 480))):
            return "a finger would land outside the photo"
        try:
            height = float(to_base_point(camera_to_base, camera.deproject(token, pixel))[2])
        except DemoError:
            return "no depth where a finger would land"
        if height > limit:
            return f"a finger would land on something {height - fingertip_z:+.0f}mm from the fingertips"
    return None


def measure_candies(
    result: dict, camera: CameraClient, token: str, camera_to_base: np.ndarray, settings: dict,
) -> list[dict]:
    """Add each candy's top-centre base point from depth and its true size at that height."""
    bowl_settings = settings["bowl"]
    view = bowl_vision.bowl_view(settings["camera_intrinsics"], camera_to_base, bowl_settings)
    measured = []
    for candy in result["candies"]:
        try:
            point = to_base_point(camera_to_base, camera.deproject(token, candy["pixel"]))
        except DemoError as exc:
            result["rejected_candies"].append({**candy, "reasons": [f"no depth: {exc}"]})
            continue
        corrected = point.copy()
        corrected[:2] += settings["candy_xy_correction_mm"]
        try:
            validate_target_workspace(
                corrected, settings["target_workspace_min_mm"], settings["target_workspace_max_mm"], check_z=False,
            )
        except DemoError as exc:
            result["rejected_candies"].append({**candy, "reasons": [str(exc)]})
            continue
        measured.append({
            **candy,
            "uncorrected_base_point_mm": point.tolist(),
            "base_point_mm": corrected.tolist(),
            "true_long_mm": bowl_vision.true_size(view, candy["long_mm"], point[2]),
            "true_short_mm": bowl_vision.true_size(view, candy["short_mm"], point[2]),
        })
    return measured


def scan_bowl(
    arm: Piper, camera: CameraClient, camera_to_gripper: np.ndarray, settings: dict,
) -> dict[str, Any]:
    """Photograph the bowl from the current (scan) pose; find, confirm, and rank its candies.

    status: "detected" (pickable candies listed best first), "no_pickable",
    "empty", "flat_circle" (a circle whose rim depth reads the table), or
    "not_found". Writes the annotated photo to logs/latest_bowl_scan.jpg.
    """
    require_healthy(arm, "before bowl scan")
    before = np.asarray(arm.pose(), dtype=np.float64)
    captured, png = camera.capture_png()
    after = np.asarray(arm.pose(), dtype=np.float64)
    camera_to_base = stationary_camera_to_base(before, after, camera_to_gripper, settings, "bowl scan")
    frame = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise DemoError("could not decode the bowl scan photo")
    bowl_settings = settings["bowl"]
    intrinsics = settings["camera_intrinsics"]
    result = bowl_vision.find_bowl(frame, intrinsics, camera_to_base, bowl_settings)
    result.update(pickable=[], unpickable=[], captured_at=captured.get("captured_at"))
    bowl = result.get("bowl")
    if bowl is not None:
        height, width = frame.shape[:2]
        rim_z = []
        for pixel in bowl["rim_depth_pixels"]:
            if not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
                continue
            try:
                rim_z.append(float(to_base_point(camera_to_base, camera.deproject(captured["token"], pixel))[2]))
            except DemoError:
                rim_z.append(None)
        raised, reason = bowl_vision.check_rim_height(rim_z, bowl_settings)
        bowl["rim_check"] = reason
        if not raised:
            result["status"] = "flat_circle"
    if result["status"] == "detected":
        candies = measure_candies(result, camera, captured["token"], camera_to_base, settings)
        rim_height = bowl_settings["table_z_mm"] + bowl_settings["bowl_height_mm"]
        pickable, result["unpickable"] = bowl_vision.grasp_options(
            candies, bowl["plane_center_mm"], rim_height, bowl_settings, bowl["radius_mm"],
        )
        for candy in pickable[: int(settings["maximum_candies_depth_checked"])]:
            problems = {}
            for closing in candy["closing_yaw_deg"]:
                problems[closing] = finger_landing_problem(candy, closing, camera, captured["token"], camera_to_base, settings)
            clear = [closing for closing, problem in problems.items() if problem is None]
            if clear:
                result["pickable"].append({**candy, "closing_yaw_deg": clear})
            else:
                result["unpickable"].append({**candy, "reasons": sorted(set(problems.values()))})
        if not result["pickable"]:
            result["status"] = "no_pickable"
    report_bowl_scan(result, settings)
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        annotated = bowl_vision.annotate(
            frame, bowl_vision.with_rim_outline(result, intrinsics, camera_to_base, bowl_settings),
            result["pickable"][0] if result["pickable"] else None,
        )
        cv2.imwrite(str(BOWL_SCAN_IMAGE), annotated)
    except (OSError, cv2.error) as exc:
        print(f"WARNING: could not save {BOWL_SCAN_IMAGE.name}: {exc}", flush=True)
    return result


def report_bowl_scan(result: dict, settings: dict) -> None:
    bowl = result.get("bowl")
    where = (
        f" bowl at {np.round(bowl['plane_center_mm'], 0).tolist()}mm r={bowl['radius_mm']:.0f}mm "
        f"({bowl.get('rim_check', '')})" if bowl else ""
    )
    counts = f", {len(result['candies'])} candies, {len(result['pickable'])} pickable" if bowl else ""
    print(f"BOWL SCAN: {result['status']}{where}{counts}", flush=True)
    if settings.get("verbose"):
        for item in result["unpickable"] + result["rejected_candies"]:
            location = item.get("base_point_mm", item.get("plane_center_mm"))
            print(f"  skipped candy at {np.round(location, 1).tolist()}mm: {'; '.join(item['reasons'])}", flush=True)


def closing_yaw_candidates(candy: dict, bowl: dict, settings: dict) -> list[float]:
    """Closing directions to try, exact first, then small yaw offsets still clear of the wall."""
    step, maximum = float(settings["grasp_yaw_offset_step_deg"]), float(settings["maximum_grasp_yaw_offset_deg"])
    offsets = [0.0] + [sign * value for value in np.arange(step, maximum + step * 0.5, step) for sign in (-1.0, 1.0)]
    point = candy["base_point_mm"]
    return [
        (yaw + offset) % 180.0
        for offset in offsets for yaw in candy["closing_yaw_deg"]
        if bowl_vision.jaw_clear_of_wall(
            point, (yaw + offset) % 180.0, bowl["plane_center_mm"], settings["bowl"], bowl.get("radius_mm"),
        )
    ]


def validate_descent_line(
    plan: list[tuple[str, np.ndarray]], grasp_tip_mm: Any, up_axis: Any, below_z_mm: float,
    maximum_deviation_mm: float, samples_per_segment: int = 20,
) -> None:
    """Require the sampled tool tip to stay on the straight line into the candy below a height.

    The line runs from the grasp tip back along the tool axis (up_axis points
    from the tip toward the flange). Checks the joint-space segments from
    "above bowl" to "grasp"; the lift retraces the same segments.
    """
    names = [name for name, _ in plan]
    joints = [value for _, value in plan]
    route = joints[names.index("above bowl"):names.index("grasp") + 1]
    origin = np.asarray(grasp_tip_mm, dtype=np.float64)
    direction = np.asarray(up_axis, dtype=np.float64) / np.linalg.norm(up_axis)
    for start, end in zip(route, route[1:]):
        for fraction in np.linspace(0.0, 1.0, samples_per_segment + 1):
            tip = tip_position((1.0 - fraction) * start + fraction * end)
            offset = float(np.linalg.norm(np.cross(tip - origin, direction)))
            if tip[2] < below_z_mm and offset > maximum_deviation_mm:
                raise DemoError(
                    f"descent into the bowl strays {offset:.1f}mm from its straight line at tip Z {tip[2]:.0f}mm"
                )


def bowl_grasp_rotations(closing_yaw_deg: float, tilt_deg: float, settings: dict) -> list[Rot]:
    """Grasp orientations whose jaws close along closing_yaw_deg, tilted about that closing axis."""
    closing = np.array([math.cos(math.radians(closing_yaw_deg)), math.sin(math.radians(closing_yaw_deg)), 0.0])
    # align_grasp_rpy_to_axis points the gripper along an axis and closes the jaws across it.
    axis_yaw = math.radians(closing_yaw_deg + 90.0)
    bases = [
        euler_to_rot(rpy)
        for rpy in equivalent_grasp_rpys(settings["grasp_rpy_deg"], np.array([math.cos(axis_yaw), math.sin(axis_yaw)]))
    ]
    signs = (1.0,) if tilt_deg == 0.0 else (1.0, -1.0)
    return [Rot.from_rotvec(math.radians(sign * tilt_deg) * closing) * base for base in bases for sign in signs]


def grasp_tilt_deg(grasp_joints: np.ndarray) -> float:
    """Angle of the tool axis from straight down."""
    return math.degrees(math.acos(float(np.clip(-fk(np.asarray(grasp_joints, dtype=np.float64))[2, 2], -1.0, 1.0))))


def plan_bowl_pick(
    candy: dict, bowl: dict, start_joints: np.ndarray, scan_joints: np.ndarray, settings: dict,
) -> list[tuple[str, np.ndarray]]:
    """Plan a bowl pick: above the candy clear of the rim, straight in along the tool axis, grip, back out.

    Waypoints: "above bowl", "descend N", "grasp", "lift N", "lift above bowl",
    "scan". Tilts from grasp_tilt_options_deg are tried smallest first; a tilt is
    used only if the open gripper, from fingertips to rim, stays inside the bowl
    opening. The whole route is path-checked and the descent must follow its
    straight line below the rim.
    """
    bowl_settings = settings["bowl"]
    point = finite_vector(candy["base_point_mm"], 3, "candy top point")
    center = bowl["plane_center_mm"]
    rim_z = bowl_settings["table_z_mm"] + bowl_settings["bowl_height_mm"]
    grasp_tip = np.array([point[0], point[1], grasp_tip_z(point[2], settings)])
    above_z = max(rim_z, grasp_tip[2]) + float(settings["rim_clearance_mm"])
    closings = closing_yaw_candidates(candy, bowl, settings)
    if not closings:
        raise DemoError("no closing direction keeps the open jaws clear of the bowl wall")
    errors = []
    for tilt in settings["grasp_tilt_options_deg"]:
        for closing in closings:
            for rotation in bowl_grasp_rotations(closing, tilt, settings):
                up = -rotation.apply([0.0, 0.0, 1.0])
                label = f"tilt {tilt:.0f}deg closing {closing:.0f}deg"
                if not bowl_vision.gripper_clear_of_bowl(
                    grasp_tip, -up, closing, center, rim_z, FINGER_EXTENSION_BELOW_TIP_MM, bowl_settings,
                    rim_radius_mm=bowl.get("radius_mm"),
                ):
                    errors.append(f"{label}: the tilted gripper would touch the bowl wall")
                    continue
                distances = np.arange((above_z - grasp_tip[2]) / up[2], 0.0, -float(settings["vertical_step_mm"]))
                tips = [grasp_tip + up * distance for distance in distances]
                try:
                    above = solve_waypoint("above bowl", flange_for_tip(tips[0], rotation), rotation, start_joints)
                    descents, seed = [], above
                    for index, tip in enumerate(tips[1:], start=1):
                        seed = solve_waypoint(f"descend {index}", flange_for_tip(tip, rotation), rotation, seed)
                        descents.append((f"descend {index}", seed))
                    grasp = solve_waypoint("grasp", flange_for_tip(grasp_tip, rotation), rotation, seed)
                    lifts = [(f"lift {index}", joints.copy()) for index, (_, joints) in enumerate(reversed(descents), start=1)]
                    plan = [
                        ("above bowl", above), *descents, ("grasp", grasp), *lifts,
                        ("lift above bowl", above.copy()), ("scan", np.asarray(scan_joints, dtype=np.float64).copy()),
                    ]
                    validate_joint_path_safety(
                        "bowl pickup and return to scan", [np.asarray(start_joints, dtype=np.float64), *[q for _, q in plan]],
                    )
                    validate_descent_line(
                        plan, grasp_tip, up, rim_z + FINGER_EXTENSION_BELOW_TIP_MM,
                        float(settings["maximum_vertical_deviation_mm"]),
                    )
                    return plan
                except DemoError as exc:
                    errors.append(f"{label}: {exc}")
    raise DemoError("no reachable bowl grasp at any allowed tilt: " + " | ".join(errors[:3]))


def pick_first_reachable(
    arm: Piper, scan_result: dict, scan: dict, settings: dict,
) -> tuple[dict, list[tuple[str, np.ndarray]]]:
    """Plan the first pickable candy (highest first) that has a reachable vertical grasp."""
    start = measured_start_joints(arm, "bowl pickup planning")
    rejections = []
    for candy in scan_result["pickable"]:
        try:
            return candy, plan_bowl_pick(candy, scan_result["bowl"], start, scan["joints_deg"], settings)
        except DemoError as exc:
            rejections.append({"base_point_mm": candy["base_point_mm"], "reason": str(exc)})
            print(
                f"Skipping unreachable candy at {np.round(candy['base_point_mm'], 1).tolist()}mm: "
                f"{summarize_rejection(str(exc))}",
                flush=True,
            )
    raise NoReachableCandyError("no candy in the bowl has a reachable grasp; move the bowl closer to the robot", rejections)


def execute_bowl_pickup(
    arm: Piper, plan: list[tuple[str, np.ndarray]], settings: dict, carry: CandyCarry, candy: dict,
) -> float:
    """Descend, grip, lift clear of the rim, verify the candy is held, and return to the scan pose.

    The carry tracker marks the candy held when the gripper closes and records
    each reached waypoint. On a missed grasp the jaws open above the bowl and
    the arm returns to the scan pose before GraspVerificationError is raised.
    """
    timeout = float(settings["move_timeout_seconds"])
    open_mm, close_mm = float(settings["grip_open_mm"]), float(settings["grip_close_mm"])
    hold_seconds = float(settings["gripper_hold_seconds"])
    speed = int(settings["pick_motion_speed_percent"])
    verbose = bool(settings.get("verbose", False))
    waypoints = dict(plan)
    names = [name for name, _ in plan]
    grasp_index = names.index("grasp")
    for name, joints in plan[:grasp_index + 1]:
        move_robot(arm, name, joints, open_mm, timeout, speed, verbose=verbose)
    if settings.get("pause_before_grip"):
        pause_for_grasp_offset(arm, waypoints["grasp"], open_mm, candy, settings)
    if verbose:
        print("motion: close gripper", flush=True)
    hold_gripper(arm, waypoints["grasp"], close_mm, hold_seconds)
    carry.gripped_at(waypoints["grasp"])
    for name, joints in plan[grasp_index + 1:names.index("lift above bowl") + 1]:
        move_robot(arm, name, joints, close_mm, timeout, speed, verbose=verbose)
        carry.reached(name, joints)
    held_width = float(arm.grip_mm())
    minimum_held = float(settings["minimum_held_grip_mm"])
    if not math.isfinite(held_width) or held_width < minimum_held:
        carry.holding = False
        print(
            f"No candy in gripper: measured width {held_width:.1f}mm is below {minimum_held:.1f}mm; "
            "opening above the bowl and returning to scan",
            flush=True,
        )
        hold_gripper(arm, waypoints["lift above bowl"], open_mm, hold_seconds)
        move_robot(arm, "return to scan pose after missed grasp", waypoints["scan"], open_mm, timeout, speed, verbose=verbose)
        raise GraspVerificationError(held_width, minimum_held)
    move_robot(arm, "scan", waypoints["scan"], close_mm, timeout, speed, verbose=verbose)
    carry.reached("scan", waypoints["scan"])
    return held_width


def return_candy_to_bowl(
    arm: Piper, pickup_plan: list[tuple[str, np.ndarray]], carry: CandyCarry, settings: dict,
) -> None:
    """Carry a held candy back above the bowl along reached waypoints, drop it, and return to scan.

    Retraces the carried path in reverse to the "lift above bowl" pose (or, if
    the error came before it was reached, finishes the planned lift up to it),
    opens the gripper there, and moves to the scan pose. The whole route is
    path-checked from the measured joints before the first move.
    """
    if not carry.holding or not carry.path or carry.path[0][0] != "grasp":
        raise DemoError("no tracked held candy to return")
    names = [name for name, _ in pickup_plan]
    waypoints = dict(pickup_plan)
    reached = [name for name, _ in carry.path]
    if "lift above bowl" in reached:
        last = len(reached) - 1 - reached[::-1].index("lift above bowl")
        back = [(f"return candy: retrace {name}", joints) for name, joints in reversed(carry.path[last + 1:])]
    else:
        remaining = names[names.index("grasp") + 1:names.index("lift above bowl")]
        back = [(f"return candy: {name}", waypoints[name]) for name in remaining if name not in reached]
    back.append(("return candy: above bowl", waypoints["lift above bowl"]))
    validate_joint_path_safety(
        "candy return to bowl",
        [measured_start_joints(arm, "candy return to bowl"), *[joints for _, joints in back], waypoints["scan"]],
    )
    closed, opened = float(settings["grip_close_mm"]), float(settings["grip_open_mm"])
    timeout, speed = float(settings["move_timeout_seconds"]), int(settings["pick_motion_speed_percent"])
    for name, joints in back:
        if np.max(np.abs(finite_vector(arm.joints_deg(), 6, "current joints") - joints)) < ARRIVAL_TOLERANCE_DEG:
            continue
        validate_joint_path_safety(name, [measured_start_joints(arm, name), joints])
        move_robot(arm, name, joints, closed, timeout, speed)
    print("motion: open gripper to drop the candy back into the bowl", flush=True)
    carry.holding = False
    hold_gripper(arm, waypoints["lift above bowl"], opened, float(settings["gripper_hold_seconds"]))
    validate_joint_path_safety("return candy: back to scan", [measured_start_joints(arm, "back to scan"), waypoints["scan"]])
    move_robot(arm, "return candy: back to scan", waypoints["scan"], opened, timeout, speed)


def capture_age_seconds(capture: dict) -> float:
    try:
        timestamp = datetime.fromisoformat(capture["captured_at"])
        if timestamp.tzinfo is None:
            raise ValueError("timestamp must include a timezone")
    except (KeyError, TypeError, ValueError) as exc:
        raise DemoError(f"invalid hand capture timestamp: {exc}") from exc
    return (datetime.now(timezone.utc) - timestamp).total_seconds()


def verify_fresh(capture: dict, maximum_age: float) -> None:
    age = capture_age_seconds(capture)
    if not -1.0 <= age <= maximum_age:
        raise DemoError(f"hand capture is stale or future-dated ({age:.2f}s)")


def capture_hand_target(
    arm: Piper, hands: HandClient, camera_to_gripper: np.ndarray, settings: dict,
) -> tuple[dict, dict]:
    """Capture the hand from the current pose and decide where to drop (hand_target.py)."""
    require_healthy(arm, "before hand scan")
    before = np.asarray(arm.pose(), dtype=np.float64)
    capture = hands.capture_hand(HAND_SLOT)
    after = np.asarray(arm.pose(), dtype=np.float64)
    camera_to_base = stationary_camera_to_base(before, after, camera_to_gripper, settings, "hand scan")
    decision = choose_drop_target(
        capture, lambda point: to_base_point(camera_to_base, point),
        settings["hand_workspace_min_mm"], settings["hand_workspace_max_mm"],
    )
    return capture, decision


def report_hand(capture: dict, decision: dict) -> str:
    if decision["status"] == "ready":
        target = "palm centre" if decision["target"] == "palm_center" else "finger base"
        note = f" ({decision['palm_note']})" if decision.get("palm_note") else ""
        return f"open palm: drop on the {target} at {np.round(decision['base_point_mm'], 1).tolist()}mm{note}"
    return f"{decision['reason']} (hand service: {capture.get('status')})"


def plan_drop(decision: dict, arm: Piper, scan: dict, settings: dict) -> dict[str, np.ndarray]:
    """Plan and path-check the approach, release 35 mm above the target, retreat, and scan."""
    point = finite_vector(decision["base_point_mm"], 3, "drop target")
    validate_target_workspace(point, settings["hand_workspace_min_mm"], settings["hand_workspace_max_mm"])
    plan = plan_hand_delivery(
        point, arm.joints_deg(), scan["joints_deg"], settings["hand_delivery_rpy_deg"],
        float(point[2]) + RELEASE_CLEARANCE_MM, float(settings["hand_approach_height_mm"]),
    )
    validate_joint_path_safety(
        "hand drop and retreat",
        [measured_start_joints(arm, "hand drop and retreat"), *[joints for _, joints in plan]],
    )
    return dict(plan)


def wait_for_hand(
    arm: Piper, hands: HandClient, scan: dict, camera_to_gripper: np.ndarray, settings: dict,
) -> tuple[dict, dict, dict[str, np.ndarray]]:
    """Hold the scan pose with the candy until an open hand has a reachable drop plan.

    Retries every retry_delay_seconds, with no timeout and no motion. Prints the
    reason only when it changes. Arm health is checked every attempt, so faults
    stop waiting; so does Ctrl+C.
    """
    last_message = None
    while True:
        require_healthy(arm, "waiting at scan pose with candy")
        try:
            capture, decision = capture_hand_target(arm, hands, camera_to_gripper, settings)
            message = report_hand(capture, decision)
            if decision["status"] == "ready":
                plan = plan_drop(decision, arm, scan, settings)
                verify_fresh(capture, MAXIMUM_HAND_AGE_SECONDS)
                print(f"HAND SCAN: {message}", flush=True)
                return capture, decision, plan
        except DemoError as exc:
            message = str(exc)
        if message != last_message:
            print(f"Waiting at scan pose with candy: {message}", flush=True)
            last_message = message
        time.sleep(float(settings["retry_delay_seconds"]))


def deliver_to_hand(
    arm: Piper, hands: HandClient, scan: dict, camera_to_gripper: np.ndarray, settings: dict, carry: CandyCarry,
) -> None:
    """Wait for an open hand, then approach and release 35 mm above the chosen point.

    The approach uses the single accepted capture, so the hand must stay still;
    the capture must still be fresh before the approach and before release.
    Reached waypoints are recorded until the gripper starts opening.
    """
    closed, opened = float(settings["grip_close_mm"]), float(settings["grip_open_mm"])
    capture, _, plan = wait_for_hand(arm, hands, scan, camera_to_gripper, settings)
    checked_move(arm, "hand approach", plan["hand approach"], closed, settings)
    carry.reached("hand approach", plan["hand approach"])
    verify_fresh(capture, MAXIMUM_HAND_AGE_SECONDS)
    checked_move(arm, f"hand release height (+{RELEASE_CLEARANCE_MM:.0f}mm)", plan["hand release"], closed, settings)
    carry.reached("hand release height", plan["hand release"])
    verify_fresh(capture, MAXIMUM_HAND_AGE_SECONDS)
    # Once opening starts above the hand, never carry anything back to the bowl.
    carry.holding = False
    hold_gripper(arm, plan["hand release"], opened, float(settings["gripper_hold_seconds"]))
    checked_move(arm, "hand retreat", plan["hand retreat"], opened, settings)
    checked_move(arm, "return to scan", scan["joints_deg"], opened, settings)


def go_to_scan(arm: Piper, scan: dict, settings: dict) -> None:
    """Move to the scan pose with the gripper open (if not already there) and let it settle."""
    current = finite_vector(arm.joints_deg(), 6, "current joints")
    if np.max(np.abs(current - scan["joints_deg"])) >= ARRIVAL_TOLERANCE_DEG:
        checked_move(arm, "scan", scan["joints_deg"], float(settings["grip_open_mm"]), settings)
    time.sleep(float(settings["scan_settle_seconds"]))


def run_cycle(
    arm: Piper, camera: CameraClient, hands: HandClient | None, scan: dict,
    camera_to_gripper: np.ndarray, settings: dict, pick_only: bool = False,
) -> bool:
    """Pick one candy from the bowl, then drop it on a hand (or back in the bowl with pick_only).

    Return True after a drop, False when nothing was picked (no bowl or
    pickable candy, no reachable grasp, missed grasp) or a held candy had to be
    returned to the bowl after an error. Errors with nothing held, failed
    returns, and Ctrl+C propagate and stop the loop.
    """
    go_to_scan(arm, scan, settings)
    result = scan_bowl(arm, camera, camera_to_gripper, settings)
    if result["status"] != "detected":
        return False
    try:
        candy, plan = pick_first_reachable(arm, result, scan, settings)
    except NoReachableCandyError as exc:
        print(f"{exc}; scanning again", flush=True)
        return False
    print(
        f"Picking candy at {np.round(candy['base_point_mm'], 1).tolist()}mm "
        f"(gripper tilt {grasp_tilt_deg(dict(plan)['grasp']):.0f}deg)",
        flush=True,
    )
    carry = CandyCarry()
    try:
        width = execute_bowl_pickup(arm, plan, settings, carry, candy)
        print(f"Pickup confirmed: gripper width={width:.1f}mm", flush=True)
        if pick_only:
            return_candy_to_bowl(arm, plan, carry, settings)
            return True
        deliver_to_hand(arm, hands, scan, camera_to_gripper, settings, carry)
        print("Candy dropped on the hand", flush=True)
    except GraspVerificationError as exc:
        print(f"Pickup missed: {exc}; scanning again", flush=True)
        return False
    except (RuntimeError, OSError, KeyError, TypeError, ValueError) as exc:
        if not carry.holding:
            raise
        print(f"ERROR while holding candy: {exc}", flush=True)
        print("Returning the candy to the bowl", flush=True)
        try:
            return_candy_to_bowl(arm, plan, carry, settings)
        except (RuntimeError, OSError, KeyError, TypeError, ValueError) as return_exc:
            raise DemoError(f"candy return failed after '{exc}': {return_exc}") from return_exc
        print("Candy returned; continuing", flush=True)
        return False
    return True


def run_scan_only(
    arm: Piper, camera: CameraClient, hands: HandClient, scan: dict,
    camera_to_gripper: np.ndarray, settings: dict,
) -> None:
    """One no-pick report at the scan pose: bowl, candies, and hand."""
    go_to_scan(arm, scan, settings)
    result = scan_bowl(arm, camera, camera_to_gripper, settings)
    if result["pickable"]:
        try:
            candy, plan = pick_first_reachable(arm, result, scan, settings)
            print(
                f"  would pick the candy at {np.round(candy['base_point_mm'], 1).tolist()}mm "
                f"(gripper tilt {grasp_tilt_deg(dict(plan)['grasp']):.0f}deg)",
                flush=True,
            )
        except NoReachableCandyError as exc:
            print(f"  {exc}", flush=True)
    print(f"  annotated photo: {BOWL_SCAN_IMAGE}", flush=True)
    try:
        capture, decision = capture_hand_target(arm, hands, camera_to_gripper, settings)
        print(f"HAND SCAN: {report_hand(capture, decision)}", flush=True)
    except DemoError as exc:
        print(f"HAND SCAN: {exc}", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse file/service locations, mode, cycle limit, and the explicit motion permission."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration-dir", type=Path, default=CALIBRATION_DIR / "calibration")
    parser.add_argument("--scan-poses", type=Path, default=DEFAULT_SCAN_POSES)
    parser.add_argument("--vision-url", default=DEFAULT_VISION_URL)
    parser.add_argument("--hand-url", default="http://127.0.0.1:8095")
    parser.add_argument("--can-interface", default="can_piper")
    parser.add_argument(
        "--check-only", action="store_true",
        help="preflight only (restarts k3s if its address is stale); never connect to CAN",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--scan-only", action="store_true",
        help="move to the scan pose and report bowl, candies, and hand each cycle; never picks",
    )
    mode.add_argument(
        "--pick-only", action="store_true",
        help="pick a candy from the bowl and drop it back in; no hand service needed",
    )
    parser.add_argument("--verbose", action="store_true", help="print each motion and skipped-candy reasons")
    parser.add_argument(
        "--pause-before-grip", action="store_true",
        help="hold each grasp pose with the jaws open and ask for the measured candy offset "
        "before closing; prints the suggested candy_xy_correction_mm",
    )
    parser.add_argument(
        "--allow-motion", action="store_true",
        help="connect to the arm and run; without it only the preflight runs",
    )
    parser.add_argument("--max-cycles", type=int, default=0, help="zero (default) loops until Ctrl+C")
    return parser.parse_args()


def main() -> int:
    """Preflight k3s, services, and setup, enforce the motion gate, and run the supervised loop.

    Check-only never connects to CAN. Every connected exit attempts a checked
    SDK-zero return, opens the gripper once zero is reached, then disconnects.
    """
    args = parse_args()
    if args.max_cycles < 0:
        print("ERROR: max-cycles must be nonnegative", file=sys.stderr)
        return 2
    arm = None
    exit_code = 0
    settings: dict = {}
    try:
        hand_eye = load_hand_eye(args.calibration_dir)
        camera = CameraClient(args.vision_url)
        hands = None if args.pick_only else HandClient(args.hand_url)
        ensure_k3s_address()
        check_services(camera, hands, hand_eye["camera"]["serial"])
        settings, camera_to_gripper, scans = load_setup(args, hand_eye)
        scan = scans[0]
        print(f"[OK] Accepted D435 calibration and scan pose {int(scan['index'])}", flush=True)
        if args.check_only:
            print("Preflight passed; no CAN connection or arm motion", flush=True)
            return 0
        if not args.allow_motion:
            raise DemoError("motion requires --allow-motion")
        if settings.get("pause_before_grip") and not sys.stdin.isatty():
            raise DemoError("--pause-before-grip needs an interactive terminal to type candy offsets")
        print("Will begin robot movement shortly, keep emergency stop accessible. Connecting in 5 seconds.", flush=True)
        for _ in range(5):
            time.sleep(1)
        arm = connect_robot(iface=args.can_interface, verbose=args.verbose)
        prepare_robot_for_scan(arm, scan, settings)
        cycle = 0
        while args.max_cycles == 0 or cycle < args.max_cycles:
            cycle += 1
            print(f"\nGTC CYCLE {cycle}", flush=True)
            if args.scan_only:
                run_scan_only(arm, camera, hands, scan, camera_to_gripper, settings)
            else:
                run_cycle(arm, camera, hands, scan, camera_to_gripper, settings, pick_only=args.pick_only)
            time.sleep(float(settings["retry_delay_seconds"]))
    except KeyboardInterrupt:
        if arm is None:
            print("Interrupted before arm connection; no zero-return motion.", file=sys.stderr)
        exit_code = 130
    except (RuntimeError, CalibrationError, HandEyeError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: GTC demo stopped: {exc}", file=sys.stderr)
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
