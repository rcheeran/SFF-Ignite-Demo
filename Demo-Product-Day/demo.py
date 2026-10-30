#!/usr/bin/env python3
"""Continuously scan for Starburst and an open palm, then pick and deliver."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    import numpy as np
    from scipy.spatial.transform import Rotation as Rot
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"ERROR: missing Python package {exc.name!r}. Run:\n"
        "  source ~/venv/bin/activate\n"
        "  python ~/productdaydemo/Demo-Product-Day/demo.py ..."
    ) from exc

PROJECT_DIR = Path(__file__).resolve().parents[1]
HAND_EYE_DIR = PROJECT_DIR / "prepare-piper" / "hand-eye"
HOMOGRAPHY_DIR = PROJECT_DIR / "prepare-piper" / "homography"
PIPER_DIR = PROJECT_DIR / "Demo-VLA-On-Piper" / "piper-openpi-V2"
sys.path.insert(0, str(HOMOGRAPHY_DIR))
sys.path.insert(0, str(HAND_EYE_DIR))
sys.path.insert(0, str(PIPER_DIR))

from common import CalibrationError, load_json, write_json_atomic  # noqa: E402
from hand_eye_common import (  # noqa: E402
    HandEyeError,
    average_transforms,
    piper_pose_to_transform,
    rotation_error_degrees,
)
from piper import config as PIPER_C  # noqa: E402
from piper.kinematics import MARGINED_LIMITS, ik  # noqa: E402
from piper.robot import Piper, connect_for_motion  # noqa: E402
from piper.transforms import euler_to_rot, fk_flange, tip_position  # noqa: E402
from verify_depth_coordinates import (  # noqa: E402
    rigid_transform,
    transform_camera_point_to_base,
)

DEFAULT_CONFIG = HAND_EYE_DIR / "config.json"
DEFAULT_HAND_EYE = HAND_EYE_DIR / "calibration" / "hand_eye.json"
DEFAULT_VALIDATION = HAND_EYE_DIR / "calibration" / "validation.json"
DEFAULT_SCAN_POSES = (
    PROJECT_DIR / "prepare-piper" / "scan-poses" / "scan_poses.json"
)
DEFAULT_OUTPUT = HAND_EYE_DIR / "calibration" / "demo_runs.json"
DEFAULT_VISION_URL = "http://127.0.0.1:8094"
DEFAULT_HAND_URL = "http://127.0.0.1:8095"
DEFAULT_OBJECT_PROMPT = "starburst candy"
EXPECTED_VISION_MODEL = "candy-yolo11s"


class DemoError(RuntimeError):
    """An invalid configuration, rejected target, or unsafe motion request."""


class GraspVerificationError(DemoError):
    """The gripper closed without retaining an object after lift."""

    def __init__(self, measured_width_mm: float) -> None:
        self.measured_width_mm = measured_width_mm
        super().__init__(
            "grasp verification failed: measured gripper width "
            f"{measured_width_mm:.1f}mm"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line configuration and explicit robot-motion safety gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "object",
        nargs="?",
        default=DEFAULT_OBJECT_PROMPT,
        help="YOLO candy class or alias (default: starburst candy)",
    )
    parser.add_argument("--vision-url", default=DEFAULT_VISION_URL)
    parser.add_argument("--hand-url", default=DEFAULT_HAND_URL)
    parser.add_argument("--scan-poses", type=Path, default=DEFAULT_SCAN_POSES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--hand-eye", type=Path, default=DEFAULT_HAND_EYE)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--can-interface", default="can_piper")
    parser.add_argument(
        "--allow-motion",
        action="store_true",
        help="required before the arm may visit taught scan poses",
    )
    parser.add_argument(
        "--allow-pick",
        action="store_true",
        help="after recognition, pick the object and replace it at the same location",
    )
    parser.add_argument(
        "--allow-hand-delivery",
        action="store_true",
        help="after pickup, scan for a stable hand and release at the taught base Z",
    )
    parser.add_argument("--grasp-tip-z-mm", type=float)
    parser.add_argument("--confidence", type=float)
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="stop after this many scan cycles; zero loops until Ctrl+C",
    )
    return parser.parse_args()


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


def validate_demo_config(config: dict) -> dict:
    """Validate demo thresholds and return normalized NumPy workspace bounds."""
    try:
        demo = dict(config["demo"])
        target_min = finite_vector(
            demo["target_workspace_min_mm"], 3, "demo.target_workspace_min_mm"
        )
        target_max = finite_vector(
            demo["target_workspace_max_mm"], 3, "demo.target_workspace_max_mm"
        )
        hand_min = finite_vector(
            demo["hand_workspace_min_mm"], 3, "demo.hand_workspace_min_mm"
        )
        hand_max = finite_vector(
            demo["hand_workspace_max_mm"], 3, "demo.hand_workspace_max_mm"
        )
        finite_vector(demo["hand_delivery_rpy_deg"], 3, "demo.hand_delivery_rpy_deg")
    except (KeyError, TypeError) as exc:
        raise DemoError(f"invalid demo configuration: {exc}") from exc
    if np.any(target_min >= target_max):
        raise DemoError("demo target workspace minimum must be below maximum")
    if np.any(hand_min >= hand_max):
        raise DemoError("demo hand workspace minimum must be below maximum")
    positive_keys = (
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
        "drop_approach_height_mm",
        "move_timeout_seconds",
        "gripper_hold_seconds",
        "retry_delay_seconds",
        "hand_release_tip_z_mm",
        "hand_approach_height_mm",
        "maximum_hand_reacquisition_difference_mm",
    )
    for key in positive_keys:
        value = float(demo[key])
        if not math.isfinite(value) or value <= 0:
            raise DemoError(f"demo.{key} must be positive")
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
    maximum_grasp_retries = int(demo["maximum_grasp_retries"])
    if (
        maximum_grasp_retries != float(demo["maximum_grasp_retries"])
        or maximum_grasp_retries < 0
    ):
        raise DemoError("demo.maximum_grasp_retries must be a non-negative integer")
    demo["maximum_grasp_retries"] = maximum_grasp_retries
    enforce_reacquisition_limit = demo.get(
        "enforce_hand_reacquisition_limit",
        True,
    )
    if not isinstance(enforce_reacquisition_limit, bool):
        raise DemoError(
            "demo.enforce_hand_reacquisition_limit must be true or false"
        )
    demo["enforce_hand_reacquisition_limit"] = enforce_reacquisition_limit
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
    demo["hand_workspace_min_mm"] = hand_min
    demo["hand_workspace_max_mm"] = hand_max
    return demo


def load_scan_poses(path: Path) -> list[dict[str, Any]]:
    """Load exactly three distinct, joint-limit-safe taught scan poses."""
    document = load_json(path)
    poses = document.get("poses")
    if document.get("pose_count") != 3 or not isinstance(poses, list) or len(poses) != 3:
        raise DemoError(f"{path} must contain exactly three taught scan poses")
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
                + ". Rerun Demo-Product-Day/Teach-scripts/teach_scan_poses.py"
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
    for left in range(3):
        for right in range(left + 1, 3):
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


class VisionClient:
    """HTTP client for the D405 OpenCV and Foundry YOLO vision service."""

    def __init__(self, base_url: str, timeout: float = 90.0) -> None:
        """Configure the service base URL and request timeout."""
        self.base_url = base_url.rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise DemoError("--vision-url must start with http:// or https://")
        self.timeout = timeout

    def _json_request(
        self, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send a JSON request and convert transport or response errors."""
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                document = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise DemoError(
                f"vision service {path} returned HTTP {exc.code}: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise DemoError(
                f"cannot reach vision service at {self.base_url}: {exc.reason}"
            ) from exc
        except (TimeoutError, json.JSONDecodeError) as exc:
            raise DemoError(f"invalid response from vision service: {exc}") from exc
        if not isinstance(document, dict):
            raise DemoError("vision service response must be a JSON object")
        return document

    def health(self) -> dict[str, Any]:
        """Return service health after requiring an explicit healthy status."""
        health = self._json_request("/healthz")
        if health.get("status") != "ok":
            raise DemoError(f"vision service is unhealthy: {health}")
        return health

    def capture(
        self, slot: int, prompt: str, confidence: float
    ) -> dict[str, Any]:
        """Capture one object scan and validate its detection status."""
        result = self._json_request(
            "/api/capture",
            {"slot": slot, "prompt": prompt, "confidence": confidence},
        )
        status = result.get("status")
        if status not in ("detected", "not_found", "rejected"):
            raise DemoError(f"vision capture {slot} was rejected: {result}")
        return result

    def set_workspace_overlay(
        self,
        base_to_camera: np.ndarray,
        minimum_mm: np.ndarray,
        maximum_mm: np.ndarray,
    ) -> None:
        """Send the current camera transform and base-frame workspace overlay."""
        self._json_request(
            "/api/workspace-overlay",
            {
                "base_to_camera": np.asarray(
                    base_to_camera,
                    dtype=np.float64,
                ).tolist(),
                "minimum_mm": np.asarray(
                    minimum_mm,
                    dtype=np.float64,
                ).tolist(),
                "maximum_mm": np.asarray(
                    maximum_mm,
                    dtype=np.float64,
                ).tolist(),
            },
        )


class HandClient(VisionClient):
    """HTTP client for stable, synchronized open-palm captures."""

    def capture_hand(self, slot: int) -> dict[str, Any]:
        """Capture an open-palm slot and validate its detection status."""
        result = self._json_request("/api/capture", {"slot": slot})
        status = result.get("status")
        if status not in ("detected", "not_found", "rejected"):
            raise DemoError(f"hand capture {slot} returned invalid status: {result}")
        return result


def validate_target_workspace(
    point_mm: np.ndarray, minimum_mm: np.ndarray, maximum_mm: np.ndarray
) -> None:
    """Require a detected base-frame point to lie inside configured bounds."""
    point = finite_vector(point_mm, 3, "detected base point")
    if np.any(point < minimum_mm) or np.any(point > maximum_mm):
        raise DemoError(
            f"detected base point {np.round(point, 1).tolist()}mm is outside "
            f"{minimum_mm.tolist()}..{maximum_mm.tolist()}mm"
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


def move_checked(
    arm: Piper,
    name: str,
    joints: np.ndarray,
    grip_mm: float,
    timeout: float,
    speed: int,
) -> None:
    """Execute a joint move with health checks before and after convergence."""
    require_healthy(arm, f"before {name}")
    print(f"motion: {name}")
    if not arm.move_joints(
        joints,
        timeout,
        grip_mm=grip_mm,
        speed=speed,
        tol=2.0,
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
) -> None:
    """Sample a joint path and enforce limits, flange bounds, and tip floor."""
    lower = MARGINED_LIMITS[:, 0]
    upper = MARGINED_LIMITS[:, 1]
    for segment_index, (start, end) in enumerate(
        zip(waypoints, waypoints[1:]),
        start=1,
    ):
        for fraction in np.linspace(0.0, 1.0, samples_per_segment + 1):
            joints = (1.0 - fraction) * start + fraction * end
            if np.any(joints < lower) or np.any(joints > upper):
                raise DemoError(
                    f"{name} segment {segment_index} crosses a joint limit"
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


def plan_pick_and_replace(
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
    """Plan a direct approach or a safe adaptive inward-transit fallback."""
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
        return [
            ("pick approach", q_approach),
            ("grasp", q_grasp),
            ("lift", q_approach.copy()),
            ("replace", q_grasp.copy()),
            ("retreat", q_approach.copy()),
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
                    ("replace", q_grasp.copy()),
                    ("retreat", q_pregrasp.copy()),
                    ("retreat transit", q_transit.copy()),
                    ("scan 1", np.asarray(return_joints, dtype=np.float64).copy()),
                ]
        raise DemoError(
            f"{direct_error}; adaptive inward/pre-grasp search also failed: "
            f"{last_error}"
        )


def serialize_pick_plan(
    target: dict[str, Any],
    aligned_grasp_rpy: np.ndarray,
    plan: list[tuple[str, np.ndarray]],
    grasp_tip_z_mm: float,
    approach_height_mm: float,
) -> dict[str, Any]:
    """Build the persisted geometry and joint record for one pickup plan."""
    base_point = target["base_point_mm"]
    waypoints = dict(plan)
    approach_tip = tip_position(waypoints["pick approach"])
    transit_tip = (
        tip_position(waypoints["pick transit"])
        if "pick transit" in waypoints
        else None
    )
    return {
        "approach_strategy": (
            "inward_high_transit" if transit_tip is not None else "direct_vertical"
        ),
        "detected_surface_point_mm": list(base_point),
        "grasp_tip_z_mm": grasp_tip_z_mm,
        "grasp_tip_pose_mm_deg": [
            float(base_point[0]),
            float(base_point[1]),
            grasp_tip_z_mm,
            *aligned_grasp_rpy.tolist(),
        ],
        "approach_tip_pose_mm_deg": [
            *approach_tip.tolist(),
            *aligned_grasp_rpy.tolist(),
        ],
        "transit_tip_pose_mm_deg": (
            [*transit_tip.tolist(), *aligned_grasp_rpy.tolist()]
            if transit_tip is not None
            else None
        ),
        "candy_long_axis_base_yaw_deg": float(
            target["long_axis_base_yaw_deg"]
        ),
        "grasp_yaw_offset_deg": (
            (
                float(aligned_grasp_rpy[2])
                - float(target["long_axis_base_yaw_deg"])
                + 90.0
            )
            % 180.0
            - 90.0
        ),
        "aligned_grasp_rpy_deg": aligned_grasp_rpy.tolist(),
        "waypoints_joints_deg": {
            name: joints.tolist() for name, joints in plan
        },
    }


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


def execute_pick_and_replace(
    arm: Piper,
    plan: list[tuple[str, np.ndarray]],
    demo: dict,
) -> dict:
    """Pick an object, replace it at the exact grasp waypoint, and retreat."""
    held_width = execute_pickup(arm, plan, demo)
    waypoints = dict(plan)
    timeout = float(demo["move_timeout_seconds"])
    open_mm = float(demo["grip_open_mm"])
    close_mm = float(demo["grip_close_mm"])
    hold_seconds = float(demo["gripper_hold_seconds"])
    speed = int(demo["motion_speed_percent"])
    move_checked(arm, "replace", waypoints["replace"], close_mm, timeout, speed)
    print("motion: open gripper")
    hold_gripper(arm, waypoints["replace"], open_mm, hold_seconds)
    move_checked(arm, "retreat", waypoints["retreat"], open_mm, timeout, speed)
    if "retreat transit" in waypoints:
        move_checked(
            arm,
            "retreat transit",
            waypoints["retreat transit"],
            open_mm,
            timeout,
            speed,
        )
    move_checked(
        arm, "return to scan 1", waypoints["scan 1"], open_mm, timeout, speed
    )
    return {"gripper_width_after_lift_mm": held_width}


def execute_pickup(
    arm: Piper,
    plan: list[tuple[str, np.ndarray]],
    demo: dict,
) -> float:
    """Execute approach and grasp, then verify an object is held after lifting."""
    waypoints = dict(plan)
    timeout = float(demo["move_timeout_seconds"])
    open_mm = float(demo["grip_open_mm"])
    close_mm = float(demo["grip_close_mm"])
    hold_seconds = float(demo["gripper_hold_seconds"])
    speed = int(
        demo.get("pick_motion_speed_percent", demo["motion_speed_percent"])
    )
    if "pick transit" in waypoints:
        move_checked(
            arm,
            "pick transit",
            waypoints["pick transit"],
            open_mm,
            timeout,
            speed,
        )
    move_checked(
        arm, "pick approach", waypoints["pick approach"], open_mm, timeout, speed
    )
    move_checked(arm, "grasp", waypoints["grasp"], open_mm, timeout, speed)
    print("motion: close gripper")
    hold_gripper(arm, waypoints["grasp"], close_mm, hold_seconds)
    move_checked(arm, "lift", waypoints["lift"], close_mm, timeout, speed)
    if "lift transit" in waypoints:
        move_checked(
            arm,
            "lift transit",
            waypoints["lift transit"],
            close_mm,
            timeout,
            speed,
        )
    held_width = float(arm.grip_mm())
    if not math.isfinite(held_width) or held_width <= 0.0:
        missed_grasp_pose = waypoints.get("lift transit", waypoints["lift"])
        hold_gripper(arm, missed_grasp_pose, open_mm, hold_seconds)
        move_checked(
            arm,
            "scan 1 after missed grasp",
            waypoints["scan 1"],
            open_mm,
            timeout,
            speed,
        )
        raise GraspVerificationError(held_width)
    return held_width


def return_held_object(
    arm: Piper,
    plan: list[tuple[str, np.ndarray]],
    demo: dict,
    reason: str,
    on_release: Callable[[], None] | None = None,
) -> None:
    """Return a held object to its planned pickup location and resume scanning."""
    waypoints = dict(plan)
    timeout = float(demo["move_timeout_seconds"])
    open_mm = float(demo["grip_open_mm"])
    close_mm = float(demo["grip_close_mm"])
    hold_seconds = float(demo["gripper_hold_seconds"])
    speed = int(demo["motion_speed_percent"])
    if "lift transit" in waypoints:
        move_checked(
            arm,
            f"{reason} transit approach",
            waypoints["lift transit"],
            close_mm,
            timeout,
            speed,
        )
    move_checked(
        arm,
        f"{reason} approach",
        waypoints["lift"],
        close_mm,
        timeout,
        speed,
    )
    move_checked(
        arm,
        f"{reason} replace",
        waypoints["replace"],
        close_mm,
        timeout,
        speed,
    )
    hold_gripper(arm, waypoints["replace"], open_mm, hold_seconds)
    if on_release is not None:
        on_release()
    move_checked(
        arm,
        f"{reason} retreat",
        waypoints["retreat"],
        open_mm,
        timeout,
        speed,
    )
    if "retreat transit" in waypoints:
        move_checked(
            arm,
            f"{reason} retreat transit",
            waypoints["retreat transit"],
            open_mm,
            timeout,
            speed,
        )
    move_checked(
        arm,
        "return to scan 1",
        waypoints["scan 1"],
        open_mm,
        timeout,
        speed,
    )


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


def validate_detected_palm_release_clearance(
    release_tip_z_mm: float,
    palm_z_mm: float,
) -> float:
    """Require the taught release height to remain above the detected palm."""
    clearance = float(release_tip_z_mm) - float(palm_z_mm)
    if not math.isfinite(clearance):
        raise DemoError("detected palm release clearance must be finite")
    if clearance <= 0:
        raise DemoError(
            f"taught release Z {release_tip_z_mm:.1f}mm is not above "
            f"detected palm Z {palm_z_mm:.1f}mm"
        )
    return clearance


def execute_hand_delivery(
    arm: Piper,
    plan: list[tuple[str, np.ndarray]],
    demo: dict,
    on_release: Callable[[], None] | None = None,
) -> None:
    """Move above the palm, release the object, retreat, and return to scan one."""
    waypoints = dict(plan)
    timeout = float(demo["move_timeout_seconds"])
    open_mm = float(demo["grip_open_mm"])
    close_mm = float(demo["grip_close_mm"])
    hold_seconds = float(demo["gripper_hold_seconds"])
    speed = int(demo["motion_speed_percent"])
    move_checked(
        arm,
        "hand approach",
        waypoints["hand approach"],
        close_mm,
        timeout,
        speed,
    )
    print("motion: automatic descent to hand release")
    move_checked(
        arm,
        "hand release",
        waypoints["hand release"],
        close_mm,
        timeout,
        speed,
    )
    print("motion: release object above palm")
    hold_gripper(arm, waypoints["hand release"], open_mm, hold_seconds)
    if on_release is not None:
        on_release()
    move_checked(
        arm,
        "hand retreat",
        waypoints["hand retreat"],
        open_mm,
        timeout,
        speed,
    )
    move_checked(
        arm, "return to scan 1", waypoints["scan 1"], open_mm, timeout, speed
    )


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
        validate_target_workspace(
            base_point,
            demo["target_workspace_min_mm"],
            demo["target_workspace_max_mm"],
        )
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


def capture_at_scan_pose(
    arm: Piper,
    client: VisionClient,
    scan: dict[str, Any],
    prompt: str,
    confidence: float,
    camera_to_gripper: np.ndarray,
    demo: dict,
) -> dict[str, Any]:
    """Capture Starburst at one pose and transform its point and axis to base."""
    index = int(scan["index"])
    move_checked(
        arm,
        f"scan {index}",
        scan["joints_deg"],
        float(demo["grip_open_mm"]),
        float(demo["move_timeout_seconds"]),
        int(demo["motion_speed_percent"]),
    )
    time.sleep(float(demo["scan_settle_seconds"]))
    before_pose = arm.pose()
    before_transform = piper_pose_to_transform(before_pose)
    client.set_workspace_overlay(
        before_transform @ camera_to_gripper,
        demo["target_workspace_min_mm"],
        demo["target_workspace_max_mm"],
    )
    service_result = client.capture(index, prompt, confidence)
    after_pose = arm.pose()
    after_transform = piper_pose_to_transform(after_pose)
    position_deviation = float(
        np.linalg.norm(after_pose[:3] - before_pose[:3])
    )
    rotation_deviation = rotation_error_degrees(
        before_transform, after_transform
    )
    if position_deviation > float(demo["maximum_scan_position_deviation_mm"]):
        raise DemoError(
            f"scan {index} moved {position_deviation:.2f}mm during inference"
        )
    if rotation_deviation > float(demo["maximum_scan_rotation_deviation_deg"]):
        raise DemoError(
            f"scan {index} rotated {rotation_deviation:.2f}deg during inference"
        )
    return process_object_capture(
        service_result,
        index,
        prompt,
        confidence,
        client,
        before_pose,
        after_pose,
        before_transform,
        after_transform,
        camera_to_gripper,
        demo,
    )


def process_object_capture(
    service_result: dict[str, Any],
    index: int,
    prompt: str,
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
        print(
            f"SCAN {index}: detected {len(valid_candidates)} candidate(s), "
            f"best score={float(result['score']):.3f}, "
            f"camera={np.round(result['camera_point_mm'], 1).tolist()}mm, "
            f"snapshot={client.base_url}{result['snapshot_url']}"
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
            f"SCAN {index}: {result['status']} at "
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
    move_checked(
        arm,
        f"scan {scan_index}",
        scan["joints_deg"],
        float(demo["grip_open_mm"]),
        float(demo["move_timeout_seconds"]),
        int(demo["motion_speed_percent"]),
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
        prompt,
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
        f"SCAN TIMING {scan_index}: move={move_seconds:.2f}s, "
        f"settle={settle_seconds:.2f}s, object={object_seconds:.2f}s, "
        f"hand={hand_seconds:.2f}s, "
        f"concurrent={concurrent_inference_seconds:.2f}s, "
        f"total={timing['total_pose_seconds']:.2f}s"
    )
    return object_result, hand_result


def capture_hand_at_scan_pose(
    arm: Piper,
    client: HandClient,
    overlay_client: VisionClient,
    scan: dict[str, Any],
    slot: int,
    camera_to_gripper: np.ndarray,
    demo: dict,
    move_to_pose: bool = True,
) -> dict[str, Any]:
    """Capture a stable open palm and transform its center into the base frame."""
    scan_index = int(scan["index"])
    if move_to_pose:
        move_checked(
            arm,
            f"hand scan {scan_index}",
            scan["joints_deg"],
            float(demo["grip_close_mm"]),
            float(demo["move_timeout_seconds"]),
            int(demo["motion_speed_percent"]),
        )
        time.sleep(float(demo["scan_settle_seconds"]))
    else:
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
                f"HAND SCAN {slot}: rejected "
                f"base={np.round(base_point, 1).tolist()}mm: {exc}"
            )
            return result
        print(
            f"OPEN PALM SCAN {slot}: detected "
            f"{service_result.get('handedness', 'unknown')} "
            f"score={float(service_result['score']):.3f}, "
            f"base={np.round(base_point, 1).tolist()}mm, "
            f"snapshot={client.base_url}{service_result['snapshot_url']}"
        )
    else:
        print(
            f"OPEN PALM SCAN {slot}: {service_result['status']}, "
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


def choose_target_if_detected(
    captures: list[dict[str, Any]],
    maximum_difference: float,
    label: str,
    maximum_orientation_difference_deg: float = 180.0,
    comparison_axes: tuple[int, ...] = (0, 1, 2),
) -> dict[str, Any] | None:
    """Return the selected target, or None when every view missed or rejected it."""
    if not any(item.get("status") == "detected" for item in captures):
        return None
    return choose_target(
        captures,
        maximum_difference,
        label,
        maximum_orientation_difference_deg,
        comparison_axes,
    )


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
                plan = plan_pick_and_replace(
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


def highest_confidence_object_capture(
    captures: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the scan capture containing the highest-scoring valid candidate."""
    valid = [
        capture
        for capture in captures
        if capture.get("status") == "detected"
        and int(capture.get("candidate_count", 0)) > 0
        and math.isfinite(float(capture.get("score", math.nan)))
    ]
    if not valid:
        raise DemoError("no valid object scan is available for grasp retry")
    return max(valid, key=lambda capture: float(capture["score"]))


def save_run(output: Path, prompt: str, cycles: list[dict[str, Any]]) -> None:
    """Atomically persist all cycle state for diagnostics and recovery review."""
    write_json_atomic(
        output,
        {
            "schema_version": 2,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "prompt": prompt,
            "cycles": cycles,
        },
    )


def main() -> int:
    """Validate services and calibration, then run the supervised scan loop."""
    args = parse_args()
    prompt = " ".join(
        args.object.strip().lower().replace("_", " ").replace("-", " ").split()
    )
    if prompt not in ("starburst", DEFAULT_OBJECT_PROMPT):
        print(
            "ERROR: this autonomous loop is configured for the YOLO "
            "'starburst' class; use 'starburst' or 'starburst candy'",
            file=sys.stderr,
        )
        return 2
    if not args.allow_motion:
        print(
            "ERROR: --allow-motion is required because the demo moves through "
            "three taught scan poses",
            file=sys.stderr,
        )
        return 2
    if not args.allow_pick:
        print(
            "ERROR: --allow-pick is required before the demo may grasp a Starburst",
            file=sys.stderr,
        )
        return 2
    if args.max_cycles < 0:
        print("ERROR: --max-cycles cannot be negative", file=sys.stderr)
        return 2
    try:
        config = load_json(args.config)
        hand_eye = load_json(args.hand_eye)
        validation = load_json(args.validation)
        demo = validate_demo_config(config)
        require_quality(hand_eye, "hand-eye calibration", args.hand_eye)
        require_quality(validation, "hand-eye validation", args.validation)
        camera_to_gripper = rigid_transform(hand_eye, "camera_to_gripper")
        scan_poses = load_scan_poses(args.scan_poses)
        confidence = float(
            args.confidence
            if args.confidence is not None
            else demo["detection_confidence_threshold"]
        )
        if not 0 < confidence < 1:
            raise DemoError("confidence threshold must be between zero and one")
        grasp_rpy = finite_vector(demo["grasp_rpy_deg"], 3, "grasp RPY")
        hand_delivery_rpy = finite_vector(
            demo["hand_delivery_rpy_deg"], 3, "hand delivery RPY"
        )
        grasp_tip_z_mm = float(
            args.grasp_tip_z_mm
            if args.grasp_tip_z_mm is not None
            else demo["grasp_tip_z_mm"]
        )
        if not math.isfinite(grasp_tip_z_mm):
            raise DemoError("grasp tip Z must be finite")
        client = VisionClient(args.vision_url)
        health = client.health()
        if health.get("model") != EXPECTED_VISION_MODEL:
            raise DemoError(
                f"vision service model must be {EXPECTED_VISION_MODEL!r}, "
                f"not {health.get('model')!r}; deploy k3s-OpenCV-service with the "
                "Foundry Local candy model"
            )
        hand_client = HandClient(args.hand_url) if args.allow_hand_delivery else None
        hand_health = hand_client.health() if hand_client is not None else None
    except (
        CalibrationError,
        HandEyeError,
        DemoError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"ERROR: cannot start demo: {exc}", file=sys.stderr)
        return 2
    print(
        f"Vision service ready: model={health.get('model')}, "
        f"camera={health.get('camera_serial')}, UI={args.vision_url}/"
    )
    if hand_health is not None:
        print(
            f"Hand service ready: model={hand_health.get('model')}, "
            f"source={hand_health.get('source_url')}, UI={args.hand_url}/"
        )
    else:
        print("Hand delivery disabled: candy will be returned to its pickup location.")

    arm: Piper | None = None
    cycles: list[dict[str, Any]] = []
    return_code = 0
    holding_object = False
    pickup_plan: list[tuple[str, np.ndarray]] | None = None
    try:
        arm = connect_for_motion(countdown=5, iface=args.can_interface)
        require_healthy(arm, "startup")
        print(
            "AUTOMATIC LOOP ENABLED. Keep the emergency "
            "stop accessible. Press Ctrl+C to stop."
        )
        while args.max_cycles == 0 or len(cycles) < args.max_cycles:
            cycle: dict[str, Any] = {
                "index": len(cycles) + 1,
                "status": "scanning",
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "captures": [],
                "hand_captures": [],
            }
            cycles.append(cycle)
            save_run(args.output, DEFAULT_OBJECT_PROMPT, cycles)
            print(f"\nCYCLE {cycle['index']}: scanning three taught poses")
            scan_started_at = time.perf_counter()
            for scan_index, scan in enumerate(scan_poses, start=1):
                if hand_client is None:
                    object_capture = capture_at_scan_pose(
                        arm,
                        client,
                        scan,
                        DEFAULT_OBJECT_PROMPT,
                        confidence,
                        camera_to_gripper,
                        demo,
                    )
                    hand_capture = None
                else:
                    object_capture, hand_capture = capture_scan_pair(
                        arm,
                        client,
                        hand_client,
                        scan,
                        scan_index + 3,
                        DEFAULT_OBJECT_PROMPT,
                        confidence,
                        camera_to_gripper,
                        demo,
                    )
                cycle["captures"].append(object_capture)
                if hand_capture is not None:
                    cycle["hand_captures"].append(hand_capture)
                save_run(args.output, DEFAULT_OBJECT_PROMPT, cycles)
            cycle["scan_duration_seconds"] = time.perf_counter() - scan_started_at
            print(f"Scan loop completed in {cycle['scan_duration_seconds']:.2f}s")

            maximum_difference = float(
                demo["maximum_cross_view_target_difference_mm"]
            )
            maximum_orientation_difference = float(
                demo["maximum_cross_view_orientation_difference_deg"]
            )
            object_targets = distinct_object_targets(
                cycle["captures"],
                maximum_difference,
                maximum_orientation_difference,
            )
            hand_target = (
                choose_target_if_detected(
                    cycle["hand_captures"],
                    maximum_difference,
                    "stable open palm",
                    comparison_axes=(0, 1),
                )
                if hand_client is not None
                else None
            )
            cycle["object_targets"] = object_targets
            cycle["hand_target"] = hand_target
            if not object_targets or (
                hand_client is not None and hand_target is None
            ):
                missing = []
                if not object_targets:
                    missing.append("Starburst")
                if hand_client is not None and hand_target is None:
                    missing.append("open palm")
                cycle["status"] = "waiting_for_" + "_and_".join(
                    value.lower().replace(" ", "_") for value in missing
                )
                print(
                    f"CYCLE {cycle['index']}: no {' and '.join(missing)}; "
                    "nothing picked, restarting the scan loop."
                )
                save_run(args.output, DEFAULT_OBJECT_PROMPT, cycles)
                time.sleep(float(demo["retry_delay_seconds"]))
                continue
            if (
                hand_target is not None
                and not hand_target["cross_view_agreement_passed"]
            ):
                cycle["status"] = "cross_view_disagreement"
                print(
                    f"CYCLE {cycle['index']}: palm coordinates disagree; "
                    "nothing picked, restarting the scan loop."
                )
                save_run(args.output, DEFAULT_OBJECT_PROMPT, cycles)
                time.sleep(float(demo["retry_delay_seconds"]))
                continue

            planned_target = select_reachable_object_target(
                object_targets,
                arm.joints_deg(),
                scan_poses[0]["joints_deg"],
                grasp_rpy,
                grasp_tip_z_mm,
                float(demo["approach_height_mm"]),
                float(demo["pregrasp_height_mm"]),
                float(demo["minimum_pregrasp_height_mm"]),
                float(demo["pregrasp_height_step_mm"]),
                float(demo["maximum_approach_inward_offset_mm"]),
                float(demo["approach_inward_step_mm"]),
                float(demo["maximum_grasp_yaw_offset_deg"]),
                float(demo["grasp_yaw_offset_step_deg"]),
            )
            if planned_target is None:
                cycle["status"] = "no_reachable_starburst"
                cycle["object_target_rejections"] = [
                    {
                        "base_point_mm": target["base_point_mm"],
                        "base_distance_mm": target["base_distance_mm"],
                        "reason": target.get(
                            "planning_rejection_reason",
                            (
                                "cross-view position or orientation disagreement"
                                if not target["cross_view_agreement_passed"]
                                else "no joint-limit-safe IK plan"
                            ),
                        ),
                    }
                    for target in object_targets
                ]
                print(
                    f"CYCLE {cycle['index']}: detected "
                    f"{len(object_targets)} distinct Starburst(s), but none "
                    "had a safe reachable grasp plan."
                )
                for index, rejection in enumerate(
                    cycle["object_target_rejections"],
                    start=1,
                ):
                    print(
                        f"  STARBURST {index} SKIPPED: "
                        f"base={np.round(rejection['base_point_mm'], 1).tolist()}mm, "
                        f"distance={float(rejection['base_distance_mm']):.1f}mm: "
                        f"{rejection['reason']}"
                    )
                save_run(args.output, DEFAULT_OBJECT_PROMPT, cycles)
                time.sleep(float(demo["retry_delay_seconds"]))
                continue
            target, aligned_grasp_rpy, plan, target_rejections = planned_target
            cycle["target"] = target
            cycle["object_target_rejections"] = target_rejections
            for index, rejection in enumerate(target_rejections, start=1):
                print(
                    f"STARBURST {index} SKIPPED: "
                    f"base={np.round(rejection['base_point_mm'], 1).tolist()}mm, "
                    f"distance={float(rejection['base_distance_mm']):.1f}mm: "
                    f"{rejection['reason']}"
                )
            print(
                f"STARBURST SELECTED: slot={target['selected_from_slot']}, "
                f"score={target['score']:.3f}, "
                f"base={np.round(target['base_point_mm'], 1).tolist()}mm, "
                f"distance={target['base_distance_mm']:.1f}mm "
                f"from {len(object_targets)} distinct candidate(s)"
            )
            if hand_target is not None:
                print(
                    f"OPEN PALM SELECTED: slot={hand_target['selected_from_slot']}, "
                    f"score={hand_target['score']:.3f}, "
                    f"base={np.round(hand_target['base_point_mm'], 1).tolist()}mm"
                )
            cycle["plan"] = serialize_pick_plan(
                target,
                aligned_grasp_rpy,
                plan,
                grasp_tip_z_mm,
                float(demo["approach_height_mm"]),
            )
            cycle["status"] = "pick_planned"
            save_run(args.output, DEFAULT_OBJECT_PROMPT, cycles)
            print(
                f"PICK PLAN: detected surface Z="
                f"{float(target['base_point_mm'][2]):.1f}mm, "
                f"configured grasp tip Z={grasp_tip_z_mm:.1f}mm"
            )
            print(
                "  grasp tip pose: "
                f"{np.round(cycle['plan']['grasp_tip_pose_mm_deg'], 1).tolist()}"
                " [X,Y,Z,RX,RY,RZ]"
            )
            for name, joints in plan:
                print(f"  {name}: {np.round(joints, 1).tolist()}deg")
            pickup_plan = plan
            cycle["grasp_attempts"] = []
            maximum_grasp_retries = int(demo["maximum_grasp_retries"])
            for attempt_index in range(maximum_grasp_retries + 1):
                try:
                    held_width = execute_pickup(arm, plan, demo)
                except GraspVerificationError as exc:
                    cycle["grasp_attempts"].append(
                        {
                            "attempt": attempt_index + 1,
                            "status": "failed",
                            "measured_width_mm": exc.measured_width_mm,
                            "scan_slot": target["selected_from_slot"],
                        }
                    )
                    save_run(args.output, DEFAULT_OBJECT_PROMPT, cycles)
                    if attempt_index >= maximum_grasp_retries:
                        raise

                    best_capture = highest_confidence_object_capture(
                        cycle["captures"]
                    )
                    retry_slot = int(best_capture["slot"])
                    print(
                        f"GRASP RETRY {attempt_index + 1}/"
                        f"{maximum_grasp_retries}: measured width "
                        f"{exc.measured_width_mm:.1f}mm; rescanning highest-"
                        f"confidence slot {retry_slot} "
                        f"(score={float(best_capture['score']):.3f})"
                    )
                    retry_capture = capture_at_scan_pose(
                        arm,
                        client,
                        scan_poses[retry_slot - 1],
                        DEFAULT_OBJECT_PROMPT,
                        confidence,
                        camera_to_gripper,
                        demo,
                    )
                    cycle.setdefault("grasp_retry_captures", []).append(
                        retry_capture
                    )
                    retry_targets = distinct_object_targets(
                        [retry_capture],
                        maximum_difference,
                        maximum_orientation_difference,
                    )
                    retry_planned = select_reachable_object_target(
                        retry_targets,
                        arm.joints_deg(),
                        scan_poses[0]["joints_deg"],
                        grasp_rpy,
                        grasp_tip_z_mm,
                        float(demo["approach_height_mm"]),
                        float(demo["pregrasp_height_mm"]),
                        float(demo["minimum_pregrasp_height_mm"]),
                        float(demo["pregrasp_height_step_mm"]),
                        float(demo["maximum_approach_inward_offset_mm"]),
                        float(demo["approach_inward_step_mm"]),
                        float(demo["maximum_grasp_yaw_offset_deg"]),
                        float(demo["grasp_yaw_offset_step_deg"]),
                    )
                    if retry_planned is None:
                        cycle["status"] = "grasp_retry_no_reachable_starburst"
                        save_run(args.output, DEFAULT_OBJECT_PROMPT, cycles)
                        raise DemoError(
                            f"grasp retry scan {retry_slot} found no safely "
                            "reachable Starburst"
                        )
                    (
                        target,
                        aligned_grasp_rpy,
                        plan,
                        target_rejections,
                    ) = retry_planned
                    pickup_plan = plan
                    cycle["target"] = target
                    cycle["object_target_rejections"] = target_rejections
                    cycle["plan"] = serialize_pick_plan(
                        target,
                        aligned_grasp_rpy,
                        plan,
                        grasp_tip_z_mm,
                        float(demo["approach_height_mm"]),
                    )
                    cycle["status"] = "grasp_retry_planned"
                    save_run(args.output, DEFAULT_OBJECT_PROMPT, cycles)
                    print(
                        f"GRASP RETRY TARGET: slot={retry_slot}, "
                        f"score={float(target['score']):.3f}, "
                        f"base={np.round(target['base_point_mm'], 1).tolist()}mm"
                    )
                    continue

                cycle["grasp_attempts"].append(
                    {
                        "attempt": attempt_index + 1,
                        "status": "succeeded",
                        "measured_width_mm": held_width,
                        "scan_slot": target["selected_from_slot"],
                    }
                )
                break
            holding_object = True
            cycle["pickup_motion"] = {
                "gripper_width_after_lift_mm": held_width
            }
            cycle["status"] = "object_held"
            save_run(args.output, DEFAULT_OBJECT_PROMPT, cycles)

            if hand_client is None:
                def mark_object_replaced() -> None:
                    """Clear recovery state immediately after replacement release."""
                    nonlocal holding_object
                    holding_object = False

                return_held_object(
                    arm,
                    plan,
                    demo,
                    "pick-and-place",
                    on_release=mark_object_replaced,
                )
                pickup_plan = None
                cycle["status"] = "picked_and_replaced"
                save_run(args.output, DEFAULT_OBJECT_PROMPT, cycles)
                print(
                    "PICK-AND-PLACE: Starburst returned to its pickup location; "
                    "restarting scan loop"
                )
                time.sleep(float(demo["retry_delay_seconds"]))
                continue

            selected_slot = int(hand_target["selected_from_slot"])
            verification = capture_hand_at_scan_pose(
                arm,
                hand_client,
                client,
                scan_poses[selected_slot - 4],
                selected_slot,
                camera_to_gripper,
                demo,
            )
            cycle["hand_verification"] = verification
            hand_unavailable_reason: str | None = None
            if verification["status"] != "detected":
                hand_unavailable_reason = (
                    "selected open palm was not detected during reacquisition"
                )
            else:
                reacquisition_difference = float(
                    np.linalg.norm(
                        np.asarray(verification["base_point_mm"])
                        - np.asarray(hand_target["base_point_mm"])
                    )
                )
                maximum_reacquisition = float(
                    demo["maximum_hand_reacquisition_difference_mm"]
                )
                if (
                    demo["enforce_hand_reacquisition_limit"]
                    and reacquisition_difference > maximum_reacquisition
                ):
                    hand_unavailable_reason = (
                        f"open palm moved {reacquisition_difference:.1f}mm; "
                        f"limit is {maximum_reacquisition:.1f}mm"
                    )
                else:
                    hand_target = {
                        **hand_target,
                        "base_point_mm": verification["base_point_mm"],
                        "reacquisition_difference_mm": reacquisition_difference,
                        "reacquisition_limit_enforced": demo[
                            "enforce_hand_reacquisition_limit"
                        ],
                        "verification": verification,
                    }
                    cycle["hand_target"] = hand_target

            if hand_unavailable_reason is not None:
                print(
                    f"HAND UNAVAILABLE: {hand_unavailable_reason}. Returning "
                    "the Starburst to its pickup location."
                )
                cycle["hand_unavailable_reason"] = hand_unavailable_reason
                return_held_object(
                    arm,
                    plan,
                    demo,
                    "no-palm return",
                )
                holding_object = False
                pickup_plan = None
                cycle["status"] = "candy_returned_hand_unavailable"
                save_run(args.output, DEFAULT_OBJECT_PROMPT, cycles)
                time.sleep(float(demo["retry_delay_seconds"]))
                continue

            release_tip_z = float(demo["hand_release_tip_z_mm"])
            release_clearance = validate_detected_palm_release_clearance(
                release_tip_z,
                float(hand_target["base_point_mm"][2]),
            )
            hand_plan = plan_hand_delivery(
                np.asarray(hand_target["base_point_mm"]),
                arm.joints_deg(),
                scan_poses[0]["joints_deg"],
                hand_delivery_rpy,
                release_tip_z,
                float(demo["hand_approach_height_mm"]),
            )
            cycle["hand_plan"] = {
                "selected_from_slot": hand_target["selected_from_slot"],
                "palm_base_point_mm": hand_target["base_point_mm"],
                "release_tip_z_mm": release_tip_z,
                "release_clearance_mm": release_clearance,
                "release_tip_point_mm": [
                    float(hand_target["base_point_mm"][0]),
                    float(hand_target["base_point_mm"][1]),
                    release_tip_z,
                ],
                "waypoints_joints_deg": {
                    name: joints.tolist() for name, joints in hand_plan
                },
            }
            cycle["status"] = "hand_delivery_planned"
            save_run(args.output, DEFAULT_OBJECT_PROMPT, cycles)
            print(
                "  release tip point: "
                f"{np.round(cycle['hand_plan']['release_tip_point_mm'], 1).tolist()}mm"
            )
            for name, joints in hand_plan:
                print(f"  {name}: {np.round(joints, 1).tolist()}deg")

            def mark_object_released() -> None:
                """Clear recovery state immediately after the gripper opens."""
                nonlocal holding_object
                holding_object = False

            execute_hand_delivery(
                arm,
                hand_plan,
                demo,
                on_release=mark_object_released,
            )
            pickup_plan = None
            cycle["status"] = "delivered_to_open_palm"
            save_run(args.output, DEFAULT_OBJECT_PROMPT, cycles)
            print("OPEN-PALM DELIVERY: completed; restarting scan loop")
            time.sleep(float(demo["retry_delay_seconds"]))
    except KeyboardInterrupt:
        print("\nCtrl+C: stopping demo.")
    except (DemoError, RuntimeError, HandEyeError) as exc:
        print(f"ERROR: demo stopped: {exc}", file=sys.stderr)
        if holding_object and arm is not None and pickup_plan is not None:
            print(
                "RECOVERY: returning the held object to its original location.",
                file=sys.stderr,
            )
            try:
                return_held_object(arm, pickup_plan, demo, "recovery")
                holding_object = False
            except (DemoError, RuntimeError) as recovery_exc:
                print(
                    f"SAFETY ERROR: automatic object recovery failed: "
                    f"{recovery_exc}. Use the emergency stop and support the "
                    "object/arm.",
                    file=sys.stderr,
                )
        if cycles:
            cycles[-1]["status"] = "failed"
            cycles[-1]["error"] = str(exc)
            save_run(args.output, DEFAULT_OBJECT_PROMPT, cycles)
        return_code = 1
    finally:
        if arm is not None:
            print("Returning Piper to rest.")
            arm.rest()
    print(f"Saved {len(cycles)} cycle(s) to {args.output}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
