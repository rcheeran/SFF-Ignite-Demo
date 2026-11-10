#!/usr/bin/env python3
"""Teach and persist one to three Piper wrist-camera scan poses."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
HOMOGRAPHY_DIR = PROJECT_DIR / "prepare-piper" / "homography"
HAND_EYE_DIR = PROJECT_DIR / "prepare-piper" / "hand-eye-realsense405"
PIPER_DIR = PROJECT_DIR / "Demo-VLA-On-Piper" / "piper-openpi-V2"
sys.path.insert(0, str(HOMOGRAPHY_DIR))
sys.path.insert(0, str(HAND_EYE_DIR))
sys.path.insert(0, str(PIPER_DIR))

from common import write_json_atomic  # noqa: E402
from piper import config as PIPER_C  # noqa: E402
from piper.robot import Piper, connect_for_teach  # noqa: E402
from verify_depth_coordinates import hold_until_power_off  # noqa: E402

DEFAULT_OUTPUT = PROJECT_DIR / "prepare-piper" / "scan-poses" / "scan_poses.json"
DEFAULT_POSE_COUNT = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--can-interface", default="can_piper")
    parser.add_argument(
        "--pose-count", type=int, choices=(1, 2, 3), default=DEFAULT_POSE_COUNT,
        help="number of scan poses to teach (default: 3)",
    )
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument("--maximum-joint-stddev-deg", type=float, default=0.25)
    parser.add_argument("--maximum-position-stddev-mm", type=float, default=0.75)
    parser.add_argument("--minimum-joint-separation-deg", type=float, default=5.0)
    parser.add_argument("--replay-timeout", type=float, default=12.0)
    return parser.parse_args()


def collect_stable_pose(
    piper: Piper,
    sample_count: int,
    interval: float,
    maximum_joint_stddev: float,
    maximum_position_stddev: float,
) -> dict:
    joints = []
    poses = []
    for _ in range(sample_count):
        if not piper.in_teach():
            raise RuntimeError("Piper left active teach mode during pose capture")
        joints.append(piper.joints_deg())
        poses.append(piper.pose())
        time.sleep(interval)
    joint_values = np.asarray(joints, dtype=np.float64)
    pose_values = np.asarray(poses, dtype=np.float64)
    joint_stddev = np.std(joint_values, axis=0)
    position_stddev = np.std(pose_values[:, :3], axis=0)
    if float(np.max(joint_stddev)) > maximum_joint_stddev:
        raise RuntimeError(
            f"arm joints moved {float(np.max(joint_stddev)):.3f}deg; "
            f"limit is {maximum_joint_stddev:.3f}deg"
        )
    if float(np.max(position_stddev)) > maximum_position_stddev:
        raise RuntimeError(
            f"end position moved {float(np.max(position_stddev)):.3f}mm; "
            f"limit is {maximum_position_stddev:.3f}mm"
        )
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "joints_deg": np.mean(joint_values, axis=0).tolist(),
        "end_pose_mm_deg": np.mean(pose_values, axis=0).tolist(),
        "joint_stddev_deg": joint_stddev.tolist(),
        "position_stddev_mm": position_stddev.tolist(),
        "sample_count": sample_count,
    }


def validate_separation(
    poses: list[dict], candidate: dict, minimum_joint_separation: float
) -> None:
    candidate_joints = np.asarray(candidate["joints_deg"], dtype=np.float64)
    for existing in poses:
        separation = float(
            np.linalg.norm(
                candidate_joints
                - np.asarray(existing["joints_deg"], dtype=np.float64)
            )
        )
        if separation < minimum_joint_separation:
            raise RuntimeError(
                f"pose is only {separation:.2f}deg from scan "
                f"{existing['index']}; need {minimum_joint_separation:.2f}deg"
            )


def validate_replayable_pose(pose: dict, margin_deg: float) -> None:
    joints = np.asarray(pose["joints_deg"], dtype=np.float64)
    limits = np.asarray(PIPER_C.JOINT_LIMITS_DEG, dtype=np.float64)
    minimum = limits[:, 0] + margin_deg
    maximum = limits[:, 1] - margin_deg
    violations = [
        f"J{index + 1}={joints[index]:.1f}deg outside "
        f"[{minimum[index]:.1f}, {maximum[index]:.1f}]deg"
        for index in range(6)
        if joints[index] < minimum[index] or joints[index] > maximum[index]
    ]
    if violations:
        raise RuntimeError(
            "pose is not safely replayable: "
            + "; ".join(violations)
            + ". Physical teach can exceed CAN replay limits; move the "
            "violating joint farther inside its range and capture again"
        )


def save_poses(path: Path, poses: list[dict]) -> None:
    write_json_atomic(
        path,
        {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pose_count": len(poses),
            "frame": "Piper base/end telemetry; joints and rotations in degrees",
            "poses": poses,
        },
    )


def move_to_sdk_zero(
    piper: Piper,
    timeout: float,
    phase: str,
) -> np.ndarray:
    zero = np.asarray(PIPER_C.HOME_JOINTS_DEG, dtype=np.float64)
    faults = piper.faults()
    if faults:
        raise RuntimeError(f"cannot move to {phase.lower()}; Piper faults: {faults}")
    if not all(piper.enabled()):
        raise RuntimeError(
            f"cannot move to {phase.lower()}; one or more joints are disabled"
        )
    print(
        f"{phase}: moving to SDK zero pose {zero.tolist()}deg at "
        f"{PIPER_C.SPEED_HOME}%"
    )
    if not piper.move_to_zero(
        timeout,
        speed=PIPER_C.SPEED_HOME,
        tol=PIPER_C.HOME_TOL_DEG,
        settle=0.5,
    ):
        measured = piper.joints_deg()
        raise RuntimeError(
            f"{phase.lower()} was not reached; target={zero.tolist()}deg, "
            f"measured={np.round(measured, 1).tolist()}deg, absolute error="
            f"{np.round(np.abs(measured - zero), 1).tolist()}deg; no further "
            "motion will be commanded"
        )
    measured = piper.joints_deg()
    if float(np.max(np.abs(measured - zero))) > PIPER_C.HOME_TOL_DEG:
        raise RuntimeError(
            f"{phase.lower()} verification failed: measured "
            f"{np.round(measured, 1).tolist()}deg"
        )
    print(f"{phase} REACHED: joints={np.round(measured, 1).tolist()}deg")
    return zero


def replay_poses_and_home(
    piper: Piper,
    poses: list[dict],
    timeout: float,
) -> np.ndarray:
    grip_mm = piper.grip_mm()
    for pose in poses:
        validate_replayable_pose(pose, PIPER_C.IK_LIMIT_MARGIN_DEG)

    move_to_sdk_zero(piper, timeout, "PRE-REPLAY ZERO")
    for pose in poses:
        index = int(pose["index"])
        target = np.asarray(pose["joints_deg"], dtype=np.float64)
        faults = piper.faults()
        if faults:
            raise RuntimeError(f"cannot replay scan {index}; Piper faults: {faults}")
        if not all(piper.enabled()):
            raise RuntimeError(
                f"cannot replay scan {index}; one or more joints are disabled"
            )
        print(
            f"REPLAY scan {index}: moving to "
            f"{np.round(target, 1).tolist()}deg at {PIPER_C.SPEED_REST}%"
        )
        if not piper.move_joints(
            target,
            timeout,
            grip_mm=grip_mm,
            speed=PIPER_C.SPEED_REST,
            tol=2.0,
            settle=0.5,
        ):
            measured = piper.joints_deg()
            raise RuntimeError(
                f"scan {index} replay target was not reached; target="
                f"{np.round(target, 1).tolist()}deg, measured="
                f"{np.round(measured, 1).tolist()}deg, absolute error="
                f"{np.round(np.abs(measured - target), 1).tolist()}deg"
            )
        print(
            f"REPLAY scan {index}: reached "
            f"{np.round(piper.joints_deg(), 1).tolist()}deg"
        )

    return move_to_sdk_zero(piper, timeout, "FINAL ZERO")


def main() -> int:
    args = parse_args()
    if (
        args.samples < 3
        or args.sample_interval <= 0
        or args.replay_timeout <= 0
    ):
        print(
            "ERROR: samples must be >= 3 and intervals/timeouts must be positive",
            file=sys.stderr,
        )
        return 2
    piper: Piper | None = None
    hold_joints: np.ndarray | None = None
    poses: list[dict] = []
    return_code = 0
    try:
        piper = connect_for_teach(iface=args.can_interface)
        print(
            "\nThe Piper SDK zero pose is "
            f"{PIPER_C.HOME_JOINTS_DEG}deg. Clear the path to the folded "
            "zero pose and keep the emergency stop accessible."
        )
        hold_joints = move_to_sdk_zero(
            piper,
            args.replay_timeout,
            "PRE-TEACH ZERO",
        )
        print("\nSupport the arm and press the PHYSICAL TEACH BUTTON.")
        if not piper.wait_teach(True):
            raise RuntimeError("Piper did not enter teach mode")
        hold_joints = None
        print("TEACH MODE detected.")
        print(
            f"Teach {args.pose_count} camera view(s). Keep each pose still while it "
            "is sampled; the vision-service URL may remain open for alignment."
        )
        while len(poses) < args.pose_count:
            index = len(poses) + 1
            input(
                f"\nMove to scan position {index}, hold it still, then press Enter: "
            )
            try:
                pose = collect_stable_pose(
                    piper,
                    args.samples,
                    args.sample_interval,
                    args.maximum_joint_stddev_deg,
                    args.maximum_position_stddev_mm,
                )
                validate_replayable_pose(
                    pose,
                    PIPER_C.IK_LIMIT_MARGIN_DEG,
                )
                validate_separation(
                    poses, pose, args.minimum_joint_separation_deg
                )
            except RuntimeError as exc:
                print(f"REJECTED scan position {index}: {exc}")
                continue
            pose["index"] = index
            poses.append(pose)
            save_poses(args.output, poses)
            print(
                f"ACCEPTED scan {index}: joints="
                f"{np.round(pose['joints_deg'], 1).tolist()}, end pose="
                f"{np.round(pose['end_pose_mm_deg'], 1).tolist()}"
            )

        print("\nKEEP HOLDING THE ARM.")
        print("Press the PHYSICAL TEACH BUTTON again; the arm should stiffen.")
        if not piper.wait_teach_stopped():
            raise RuntimeError("teach recording did not stop")
        hold_joints = piper.joints_deg()
        input("Arm stiffened. Press Enter to take CAN control back: ")
        if not piper.teach_stop(hold_joints=hold_joints):
            raise RuntimeError("Piper did not return to stable CAN control")
        save_poses(args.output, poses)
        print(f"\nSaved all {args.pose_count} scan position(s) to {args.output}")
        print(
            "\nClear the complete path between all saved scan poses and the "
            "zero position. Keep the emergency stop accessible."
        )
        input(
            "Press Enter to move to zero, replay scan(s) "
            f"{', '.join(str(pose['index']) for pose in poses)}, and return to zero: "
        )
        for seconds in range(3, 0, -1):
            print(f"Replay begins in {seconds}...", flush=True)
            time.sleep(1.0)
        hold_joints = replay_poses_and_home(
            piper,
            poses,
            args.replay_timeout,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return_code = 130
    except RuntimeError as exc:
        print(f"ERROR: scan-pose teaching failed: {exc}", file=sys.stderr)
        return_code = 1
    finally:
        if piper is not None and hold_joints is None and piper.in_teach():
            print(
                "\nKEEP SUPPORTING THE ARM. Press the PHYSICAL TEACH BUTTON "
                "again; the arm should stiffen.",
                file=sys.stderr,
            )
            if piper.wait_teach_stopped():
                hold_joints = piper.joints_deg()
                if not piper.teach_stop(hold_joints=hold_joints):
                    print(
                        "SAFETY ERROR: CAN hold could not be restored. Keep "
                        "supporting the arm and turn controller power off.",
                        file=sys.stderr,
                    )
            else:
                print(
                    "SAFETY ERROR: teach recording did not stop. Keep supporting "
                    "the arm and turn controller power off.",
                    file=sys.stderr,
                )
        if piper is not None and hold_joints is not None and not piper.in_teach():
            hold_joints = piper.joints_deg()
        if piper is not None and hold_joints is not None:
            print(
                "\nKEEP SUPPORTING THE ARM. The current pose will remain "
                "streamed until controller power is OFF."
            )
            hold_until_power_off(piper, hold_joints)
        if piper is not None:
            piper.disconnect()
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
