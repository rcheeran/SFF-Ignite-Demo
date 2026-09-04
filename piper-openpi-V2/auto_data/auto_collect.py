#!/usr/bin/env python3
"""Scripted (no-headset) collection of the block-in-bowl task, in the V2 dataset format.

Why this exists: VR teleop is hard to drive consistently, and a fumbled episode teaches the
policy the wrong thing (the first teleop set had one episode where the arm repeatedly picked
up the BOWL - every structural check passed it). A scripted routine repeats the same motion
every time, so the only variation is where the block is.

What it shares with `collect/record.py` - on purpose, so both write byte-identical datasets
that `lerobot-train --policy.path=lerobot/pi05_base` consumes unchanged:
    FEATURES, open_dataset(), Episode() (the one-frame action lag), no_interrupt()
    piper.robot.Piper (locking, clamps, faults, rest), piper.cameras, piper.kinematics

Dataset contract (docs/CONTRACT.md): fps 15, `observation.images.{top,wrist}` uint8 480x640x3,
`observation.state` float32[7] = [j1..j6 deg, gripper_measured_mm], `action` float32[7] =
[j1..j6 at the NEXT frame, gripper_commanded_mm], task = C.TASK.

Geometry is not guessed - it comes from piper.kinematics (see `--plan`):
  * a straight-down tool is UNREACHABLE (J5 is limited to +-70), so grasps use a 20 deg pitch,
    which the IK reaches over the whole sampling area (20/20 corners, >=37 deg of joint margin);
  * tip height ~= flange z - 113 at that pitch, hence the FLANGE_Z_* constants below.

Bookkeeping instead of vision: the robot PLACES the block, so it knows where the block is. The
cameras only record. Nothing depends on detecting anything.

Usage (source ~/lerobot-venv/bin/activate; cd ~/Documents/piper-openpi-V2):
    python -m auto_data.auto_collect --plan                 # no hardware: print the waypoints + reachability
    python -m auto_data.auto_collect --set-bowl 300 -90     # record where the bowl is (once per setup)
    python -m auto_data.auto_collect --probe                # move through one cycle, NO recording, gripper idle
    python -m auto_data.auto_collect --episodes 10          # record
    python -m auto_data.auto_collect --episodes 10 --push   # ... and push to the Hub
"""
import argparse
import json
import os
import random
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

import piper.config as C
from auto_data.frames import CALIB_JSON, ROI, image_angle_to_robot, load_calib, px_to_xy
from auto_data.jaw_yaw import load as load_jaw_cal, yaw_for_angle
from auto_data.vision import detect
from collect.record import Episode, FEATURES, no_interrupt, open_dataset
from piper.cameras import FakeCamera, open_cameras
from piper.kinematics import fk, ik
from piper.robot import connect_for_motion

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE_JSON = os.path.join(HERE, "scene.json")

# ---- geometry, all measured with piper.kinematics (see the module docstring) ----------------
GRASP_PITCH_DEG = 20.0          # a vertical tool is unreachable; 20 deg reaches the whole area
FLANGE_Z_GRASP = 140.0          # tool tip ~27 mm: closed on a block sitting on the table
FLANGE_Z_CARRY = 230.0          # tip ~117 mm: clear of the bowl rim while carrying
FLANGE_Z_DROP = 215.0           # tip ~102 mm: above the bowl, block falls in when released
FLANGE_Z_DIG = 155.0            # tip ~42 mm: into the bowl to retrieve the block for the next episode

# tool-tip heights above the table (plan at the TIP, never the flange - see flange_for_tip)
TIP_Z_GRASP = 20.0              # closing on a cube standing on the table
TIP_Z_CARRY = 117.0             # clear of the bowl rim while carrying
TIP_Z_DROP = 100.0              # above the bowl: opening here drops the cube in
TIP_Z_DIG = 45.0                # into the bowl, to take the cube back out between episodes

# Where the collector may SET A CUBE DOWN. This is deliberately tighter than what the arm can
# reach (measured: x 200-460, y +-160 at every cube angle, all in view) because the calibration
# was fitted on spots spanning x 280-370, y +-70 and its accuracy decays with distance from that.
# The right cure is a wider calibration, not a smaller robot - widen SPOTS in calibrate.py and
# these can follow.
DROP_X = (250.0, 420.0)
DROP_Y = (-120.0, 120.0)

# BOWLS are not held to this. A cube is grasped, so a 20 mm error closes the jaws on its edge; a
# bowl is a ~126 mm target released into from 100 mm up, where 20 mm does not matter. Bowls only
# have to be reachable.
REACH_X = (200.0, 460.0)
REACH_Y = (-160.0, 160.0)
CLEAR_MM = 100.0                # a cube must land this far from any bowl or bin centre

# ONE prompt for now: a single instruction, whichever cube and bowl happen to be on the table.
# Colour-specific prompts ("pick the RED cube...") only earn their keep once several tasks share
# one scene, because that is what forces the policy to read the sentence instead of just going for
# the only cube present. Starting single keeps the first dataset interpretable.
TASK_PROMPT = C.TASK          # must match teleop episodes byte-for-byte or the dataset splits in two
TASKS = [("red_cube", "red_bowl", TASK_PROMPT),
         ("blue_cube", "blue_bowl", TASK_PROMPT)]

GRIP_HOLD_MM = 12.0             # commanded when closing; the block stops the jaws short of this
HELD_MIN_MM, HELD_MAX_MM = 16.0, 45.0   # measured width that means "a block is really in the jaws"

SETTLE_S = 0.4
# The deploy loop clamps joint motion at 10 deg/frame, so recorded data must stay under it or we
# train toward motions inference will refuse. Teleop stays slow because a hand is slow; a scripted
# move commands a distant target and the arm runs at whatever speed it is given - hence a gentler
# speed than C.SPEED_JOINT here. tools/inspect_dataset.py reports the count; it must be 0.
SPEED_COLLECT = 12
DEFAULT_BOWL_XY = (300.0, -90.0)


# How far the jaw axis is rotated in the world when we command yaw = 0. Measured with
# tools-free empiricism (auto_data.jaw_yaw); until it is measured, 0 means "unknown, no rotation".
JAW_YAW_OFFSET_DEG = 0.0


def grasp_rot(yaw_deg=0.0):
    """Grasp orientation, optionally yawed about the WORLD z axis.

    With scipy's extrinsic 'xyz', adding to the third angle pre-multiplies by Rz, i.e. it spins the
    tool about vertical without touching the 20 deg approach pitch - exactly what is needed to line
    the jaws up with two opposite faces of a rotated cube.
    """
    return Rot.from_euler("xyz", [180.0, GRASP_PITCH_DEG, 180.0 + yaw_deg], degrees=True)


def tip_z(q_deg) -> float:
    T = fk(q_deg)
    return float((T[:3, 3] + T[:3, 2] * C.TOOL_LEN_MM)[2])


def tool_axis(yaw_deg=0.0):
    """Unit vector from the flange to the tool tip, in world coordinates."""
    return grasp_rot(yaw_deg).apply([0.0, 0.0, 1.0])


def flange_for_tip(tip_xyz, yaw_deg=0.0):
    """Flange position that puts the TOOL TIP at tip_xyz.

    The tip is not underneath the flange: with the 20 deg approach pitch it sits ~41 mm FORWARD
    and ~113 mm below. Yawing the wrist swings that forward offset around an arc - 30 mm of it at
    44 deg - so planning in flange coordinates moves the jaws off the object exactly when they are
    rotated to line up with it. Observed as the jaws landing on the cube's edge, pressing one side
    and flicking it away. Everything that grasps must be planned at the tip.
    """
    return np.asarray(tip_xyz, float) - tool_axis(yaw_deg) * C.TOOL_LEN_MM


def joints_for_tip(tip_xyz, q0=None, yaw_deg=0.0):
    """Joints that put the tool TIP at tip_xyz, with the jaws yawed as asked."""
    return joints_for(flange_for_tip(tip_xyz, yaw_deg), q0, yaw_deg)


def joints_for(xyz, q0=None, yaw_deg=0.0):
    """Joint solution for a FLANGE position at the grasp orientation. Raises if the IK refuses -
    better to abort before moving than to stream a target the firmware silently drops.
    For anything that touches an object use joints_for_tip() instead."""
    q0 = C.READY_JOINTS_DEG if q0 is None else q0
    q, ok = ik(np.asarray(xyz, float), grasp_rot(yaw_deg), q0)
    if not ok:
        raise SystemExit(f"ABORT: no IK solution for flange {np.round(xyz, 0).tolist()} "
                         f"- move the bowl/block box inside the reachable area (--plan)")
    if tip_z(q) < C.TIP_Z_FLOOR_MM:
        raise SystemExit(f"ABORT: {np.round(xyz, 0).tolist()} puts the tool tip at "
                         f"{tip_z(q):.0f} mm, below the {C.TIP_Z_FLOOR_MM} mm floor")
    return q


def load_scene():
    if os.path.exists(SCENE_JSON):
        return tuple(json.load(open(SCENE_JSON))["bowl_xy"])
    return DEFAULT_BOWL_XY


def sample_block(rng, bowl_xy):
    """A block spot inside the box and clear of the bowl."""
    for _ in range(200):
        x = rng.uniform(*BLOCK_X)
        y = rng.uniform(*BLOCK_Y)
        if np.hypot(x - bowl_xy[0], y - bowl_xy[1]) >= BOWL_CLEAR_MM:
            return x, y
    raise SystemExit("ABORT: cannot sample a block spot clear of the bowl - widen BLOCK_X/BLOCK_Y")


# ---- motion -------------------------------------------------------------------------------
def stream(robot, q, secs, grip_mm, on_tick=None, tol=C.HOME_TOL_DEG, settle=SETTLE_S):
    """Stream ONE joint target at the dataset's 15 Hz until the arm arrives (or secs elapse),
    calling on_tick() every frame. The target is not interpolated: the Piper runs its own
    trajectory per target, and re-sending a fresh waypoint each tick makes it re-plan and jerk.
    """
    q = np.asarray(q, float)
    t0, reached_at = time.time(), None
    while time.time() - t0 < secs:
        robot.send_joints(q, grip_mm, speed=SPEED_COLLECT)
        if on_tick:
            on_tick()
        time.sleep(1.0 / C.FPS)
        err = np.abs(robot.joints_deg() - q).max()
        if err < tol and reached_at is None:
            reached_at = time.time()
        elif err < tol and time.time() - reached_at >= settle:
            return True
        elif err >= tol:
            reached_at = None
    return np.abs(robot.joints_deg() - q).max() < tol


def hold(robot, q, secs, grip_mm, on_tick=None):
    t0 = time.time()
    while time.time() - t0 < secs:
        robot.send_joints(q, grip_mm, speed=SPEED_COLLECT)
        if on_tick:
            on_tick()
        time.sleep(1.0 / C.FPS)


def holding_block(robot) -> bool:
    return HELD_MIN_MM <= robot.grip_mm() <= HELD_MAX_MM


# ---- scene + episode ------------------------------------------------------------------------
def look(cam, cal, jaw_cal):
    """What is on the table right now, in robot coordinates. Arm must be parked at HOME."""
    objs = detect(cam.read(), roi=ROI)
    out = {}
    for o in objs:
        x, y = px_to_xy(cal, o["px"], o["py"])
        ang = image_angle_to_robot(cal, o["angle"])
        out[o["label"]] = {"xy": (x, y), "angle": ang,
                           "yaw": yaw_for_angle(ang, jaw_cal), "px": (o["px"], o["py"])}
    return out


def choose_task(scene, rng):
    """(cube, bowl, prompt) for a cube that is on the table and not already in a bowl.

    With one generic prompt the colours need not match: any cube may go in any bowl. The pairing
    is only restricted when the prompt names a colour, which is a later step.
    """
    cubes = [k for k in scene if k.endswith("_cube") and k in {t[0] for t in TASKS}]
    bowls = [k for k in scene if k.endswith("_bowl")]
    ready = []
    for cube in cubes:
        cx, cy = scene[cube]["xy"]
        free = [b for b in bowls if np.hypot(cx - scene[b]["xy"][0], cy - scene[b]["xy"][1]) > 70]
        if free:
            nearest = min(free, key=lambda b: np.hypot(cx - scene[b]["xy"][0], cy - scene[b]["xy"][1]))
            ready.append((cube, nearest, TASK_PROMPT))
    return rng.choice(ready) if ready else None


def free_spot(scene, rng):
    """Somewhere to put a cube down that is clear of the bowls, the bin and the other cube."""
    blockers = [v["xy"] for k, v in scene.items() if k.endswith(("_bowl", "_bin"))]
    for _ in range(200):
        x, y = rng.uniform(*DROP_X), rng.uniform(*DROP_Y)
        if all(np.hypot(x - bx, y - by) >= CLEAR_MM for bx, by in blockers):
            return x, y
    return None


def run_episode(robot, cams, ep, cube, bowl, prompt, verify_grip=True, mon=None):
    """Recorded: pick the cube and drop it in the bowl. Frames carry `prompt` as the task."""
    cx, cy = cube["xy"]
    bx, by = bowl["xy"]
    yaw = cube["yaw"]
    grip = [C.GRIP_OPEN_MM]

    def tick():
        # cameras give BGR; the dataset (and pi05) expect RGB - collect/record.py converts too
        frame = {**{f"observation.images.{n}": cv2.cvtColor(cams[n].read(), cv2.COLOR_BGR2RGB)
                    for n in C.CAMERA_NAMES},
                 "observation.state": robot.state7(), "task": prompt}
        ep.push(frame, grip[0])

    q_over = joints_for_tip([cx, cy, TIP_Z_CARRY], yaw_deg=yaw)
    q_grasp = joints_for_tip([cx, cy, TIP_Z_GRASP], q_over, yaw_deg=yaw)
    q_bowl = joints_for_tip([bx, by, TIP_Z_CARRY])
    q_drop = joints_for_tip([bx, by, TIP_Z_DROP], q_bowl)

    stream(robot, q_over, 4.0, grip[0], tick)
    stream(robot, q_grasp, 2.5, grip[0], tick)
    grip[0] = GRIP_HOLD_MM
    hold(robot, q_grasp, 1.2, grip[0], tick)
    stream(robot, q_over, 2.5, grip[0], tick)
    held = robot.grip_mm()

    stream(robot, q_bowl, 4.0, grip[0], tick)
    stream(robot, q_drop, 2.0, grip[0], tick)
    grip[0] = C.GRIP_OPEN_MM
    hold(robot, q_drop, 1.0, grip[0], tick)
    stream(robot, q_bowl, 2.0, grip[0], tick)

    ok = True if not verify_grip else (HELD_MIN_MM <= held <= HELD_MAX_MM)
    return ok, {"grip_after_lift_mm": float(held), "task": prompt,
                "cube_xy": [cx, cy], "bowl_xy": [bx, by], "yaw_deg": float(yaw)}


def take_out(robot, bowl, dest, yaw=0.0, verify_grip=True):
    """UNRECORDED: lift the cube back out of the bowl and set it down at dest."""
    bx, by = bowl["xy"]
    dx, dy = dest
    q_bowl = joints_for_tip([bx, by, TIP_Z_CARRY])
    stream(robot, q_bowl, 3.0, C.GRIP_OPEN_MM)
    stream(robot, joints_for_tip([bx, by, TIP_Z_DIG], q_bowl), 2.0, C.GRIP_OPEN_MM)
    hold(robot, joints_for_tip([bx, by, TIP_Z_DIG], q_bowl), 1.2, GRIP_HOLD_MM)
    stream(robot, q_bowl, 2.0, GRIP_HOLD_MM)
    if verify_grip and not (HELD_MIN_MM <= robot.grip_mm() <= HELD_MAX_MM):
        return False
    q_dest = joints_for_tip([dx, dy, TIP_Z_CARRY])
    stream(robot, q_dest, 4.0, GRIP_HOLD_MM)
    stream(robot, joints_for_tip([dx, dy, TIP_Z_GRASP], q_dest), 2.0, GRIP_HOLD_MM)
    hold(robot, joints_for_tip([dx, dy, TIP_Z_GRASP], q_dest), 0.8, C.GRIP_OPEN_MM)
    stream(robot, q_dest, 2.0, C.GRIP_OPEN_MM)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--repo-id", default="royc1981/piper_block_bowl")
    ap.add_argument("--root", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test", action="store_true",
                    help="validation run: write to data/test (repo id local/test) and refuse --push")
    ap.add_argument("--look", action="store_true", help="no motion: report what the cameras see")
    ap.add_argument("--watch", action="store_true", help="with --look: refresh continuously")
    ap.add_argument("--task", choices=["blue", "red", "all"], default="all",
                    help="which colour(s) to collect. Start with one; the language conditioning "
                         "only starts to matter once more than one task shares a scene.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    global TASKS
    if args.task != "all":
        TASKS = [t for t in TASKS if t[0].startswith(args.task)]
        if not TASKS:
            raise SystemExit(f"no task matches --task {args.task}")
    if not os.path.exists(CALIB_JSON):
        raise SystemExit("ABORT: run `python -m auto_data.calibrate` first")
    cal = load_calib()
    jaw_cal = load_jaw_cal()

    if args.look:
        # --watch holds the top camera until you stop it. Only one process can own a camera, so
        # leaving it running makes the next calibrate/collect run fail in a confusing way (their
        # output interleaves in the same terminal). Ctrl-C releases it.
        cams = open_cameras(names=("top",))
        cams["top"].wait(5.0)
        try:
            while True:
                time.sleep(0.6)
                scene = look(cams["top"], cal, jaw_cal)
                print("\033[2J\033[H" if args.watch else "", end="")
                wanted = {o for t in TASKS for o in t[:2]}
                for k in sorted(wanted | set(scene)):
                    if k not in scene:
                        print(f"  {k:11s} NOT SEEN" + ("   <- needed for a task" if k in wanted else ""))
                        continue
                    v = scene[k]
                    x, y = v["xy"]
                    if k.endswith("_cube"):     # grasped: needs the well-calibrated region
                        ok_here = DROP_X[0] <= x <= DROP_X[1] and DROP_Y[0] <= y <= DROP_Y[1]
                        where = "ok" if ok_here else "outside the calibrated region"
                    else:                        # released into: only has to be reachable
                        ok_here = REACH_X[0] <= x <= REACH_X[1] and REACH_Y[0] <= y <= REACH_Y[1]
                        where = "ok" if ok_here else "OUT OF REACH"
                    print(f"  {k:11s} at ({x:6.1f},{y:6.1f}) mm  angle {v['angle']:+6.1f}  "
                          f"yaw {v['yaw']:+6.1f}  {where}")
                for cube, bowl, prompt in TASKS:
                    if cube in scene and bowl in scene:
                        cx, cy = scene[cube]["xy"]; bx, by = scene[bowl]["xy"]
                        d = np.hypot(cx - bx, cy - by)
                        state = "ready" if d > 70 else f"cube is IN the bowl ({d:.0f} mm apart)"
                    else:
                        state = "missing " + ", ".join(o for o in (cube, bowl) if o not in scene)
                    print(f"    {cube} -> {bowl:11s} {state}")
                t = choose_task(scene, rng)
                print(f"\n  next: " + (f'"{t[2]}"  ({t[0]} -> {t[1]})' if t else "NOTHING READY"))
                if not args.watch:
                    break
        except KeyboardInterrupt:
            print("\nstopped watching")
        finally:
            cams["top"].close()
        return

    from pathlib import Path
    if args.test:
        if args.push:
            raise SystemExit("--test is a validation run: it never pushes.")
        args.repo_id, root = "local/test", Path(C.DATA_DIR) / "test"
    else:
        root = Path(args.root) if args.root else Path(C.DATA_DIR) / args.repo_id.split("/")[-1]

    cams = None
    robot = connect_for_motion()
    try:
        cams = open_cameras(names=tuple(C.CAMERA_NAMES))
        for n, c in cams.items():
            if not c.wait(5.0):
                raise SystemExit(f"ABORT: no frames from the {n} camera")
        ds = open_dataset(args.repo_id, root, overwrite=args.overwrite)
        done = ds.meta.total_episodes
        print(f"scripted multi-task collection -> {root}  (have {done}, target {args.episodes})")

        good = bad = 0
        while done < args.episodes and not _stop.is_set():
            if robot.arm_status() in C.FAULT_STATUS:
                print(f"ABORT: firmware reports {robot.arm_status()}")
                break
            robot.home(grip_mm=C.GRIP_OPEN_MM)     # park before looking: the arm is a huge dark blob
            time.sleep(0.5)
            scene = look(cams["top"], cal, jaw_cal)
            task = choose_task(scene, rng)
            if task is None:
                print(f"  nothing to do - seen: {sorted(scene)}. Put a cube back on the table.")
                break
            cube_l, bowl_l, prompt = task
            cube, bowl = scene[cube_l], scene[bowl_l]
            print(f"ep {done}: {prompt}  cube ({cube['xy'][0]:.0f},{cube['xy'][1]:.0f}) "
                  f"yaw {cube['yaw']:+.0f}")

            ep = Episode(ds.add_frame)
            ok, meta = run_episode(robot, cams, ep, cube, bowl, prompt)
            n = ep.close()
            if ok:
                with no_interrupt():
                    ds.save_episode()
                    ds.finalize()
                    ds = open_dataset(args.repo_id, root)
                done, good = done + 1, good + 1
                print(f"   saved {n} frames ({n / C.FPS:.1f} s) grip {meta['grip_after_lift_mm']:.0f} mm")
            else:
                ds.clear_episode_buffer()
                bad += 1
                print(f"   DISCARDED (grip {meta['grip_after_lift_mm']:.0f} mm - nothing in the jaws)")

            # put the cube back on the table for the next episode
            robot.home(grip_mm=C.GRIP_OPEN_MM)
            time.sleep(0.4)
            scene2 = look(cams["top"], cal, jaw_cal)
            dest = free_spot(scene2, rng)
            if dest is None or not take_out(robot, bowl, dest):
                print("   could not get the cube back out - put it on the table by hand, then re-run")
                break

        print(f"\n{good} saved, {bad} discarded -> {root}")
        if args.push and good:
            ds.push_to_hub(private=args.private, tags=["piper", "pi05", "scripted"])
            print(f"pushed -> https://huggingface.co/datasets/{args.repo_id}")
    finally:
        if cams:
            for c in cams.values():
                if hasattr(c, "close"):
                    c.close()
        robot.rest()


if __name__ == "__main__":
    main()
