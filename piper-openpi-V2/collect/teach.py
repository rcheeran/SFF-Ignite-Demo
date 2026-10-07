"""Drag-teach collection: hand-guide the arm through the task, then replay it while the cameras record.

    python -m collect.teach --episodes 10                                       # teach -> replay -> save, repeat
    python -m collect.teach --record pick_left                                  # save a trajectory only
    python -m collect.teach --play latest --episodes 3                          # replay a saved trajectory instead
    python -m collect.teach --play pick_left --no-dataset                       # rehearse it, write nothing
    python -m collect.teach --test --episodes 2                                 # write to data/test (unpublishable)
    python -m tools.push_dataset data/piper_block_bowl                          # publish, once you have judged them
    python -m collect.teach --fake --test --episodes 2 --overwrite              # no hardware: exercise the whole path

Why replay instead of recording the demonstration directly: during drag teach the joints are de-torqued and the
arm is being pushed by a hand that is IN SHOT, so those frames show a human doing the task, not the robot. The
replay is the robot doing it alone at exactly C.FPS, which is what the policy has to imitate. It also means one
taught trajectory can be re-recorded (different lighting, camera nudged) without teaching it again.

Dataset contract is docs/CONTRACT.md, and it is met by construction: FEATURES, open_dataset and Episode are
imported from collect.record, so teach-collected and teleop-collected episodes are interchangeable in one dataset.
Observations are always MEASURED (robot.state7()), never the commanded waypoint - the arm lags its target and the
policy will see measured state at inference, so recording the command would train on data the robot never sees.
"""
import argparse
import glob
import json
import os
import signal
import threading
import time
from pathlib import Path

import av
import cv2
import numpy as np

from collect.record import Episode, FEATURES, no_interrupt, open_dataset
from piper import config as C
from piper.cameras import FakeCamera, open_cameras
from piper.robot import JOINT_LIMITS_DEG

TRAJ_DIR = os.path.join(C.DATA_DIR, "trajectories")
SETTLE_S = 0.6           # after reaching the start pose: let the arm stop swaying before the first frame
START_MOVE_S = 8.0       # budget for the (possibly long) move from wherever the arm is to the trajectory start
MAX_STEP_DEG = 10.0      # inference clamps here; a taught trajectory with bigger steps would train for a jump
LAG_WARN_DEG = 8.0       # measured vs commanded during replay: above this the arm is not tracking
TEACH_BUTTON_S = 120.0   # how long to wait for the operator to press the arm's teach button
GRIP_SQUEEZE_MM = 8.0    # command the jaws NARROWER than taught while closing - see squeeze()
GRIP_HOLDING_BELOW_MM = 60.0   # below this the jaws are closing on something: apply the squeeze
GRIP_EMPTY_MM = 5.0            # at or below this the jaws are shut on nothing, not holding
MIN_HOLD_S = 0.5               # a grasp is HELD; the jaws sweeping through mid-range is not
RELEASE_MARGIN_MM = 4.0        # opening this far past the held width drops the object


def prompt(msg: str):
    """input() that ends the session cleanly instead of tracebacking when stdin closes (piped/nohup runs)."""
    try:
        return input(msg)
    except EOFError:
        raise KeyboardInterrupt("stdin closed") from None


def enter_pressed() -> threading.Event:
    """Event set when the operator presses Enter, without blocking the sampling loop."""
    ev = threading.Event()

    def wait():
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
        ev.set()

    threading.Thread(target=wait, daemon=True).start()
    return ev


def teach(robot, task: str) -> dict:
    """Hand-guide the arm and sample it at C.FPS. Returns the trajectory dict.

    Entering and leaving teach mode is the PHYSICAL button on the arm - there is no CAN command for it
    (see the note above Piper.in_teach). So this waits on ctrl_mode rather than commanding anything.
    """
    print("\n  hold the arm, then press the TEACH BUTTON on it - the joints go limp")
    if not robot.wait_teach(True, timeout=TEACH_BUTTON_S):
        raise SystemExit(f"ABORT: still not in teach mode after {TEACH_BUTTON_S:.0f} s "
                         "(the button is on the arm body; the arm must be powered)")
    print(f"  TEACHING. Guide the arm through: {task!r}")
    print("  (the gripper works by hand in this mode - open and close it as you go)")
    prompt("  press Enter to START sampling...")
    stop = enter_pressed()
    print(f"  RECORDING at {C.FPS} Hz - press Enter to stop")

    joints, grip, next_t = [], [], time.monotonic()
    while not stop.is_set():
        if not robot.in_teach():          # button pressed early: the arm is stiff and no longer follows the
            print("\n  left teach mode - stopping here")    # hand, so every further sample would be a lie
            break
        joints.append(robot.joints_deg().tolist())
        grip.append(float(robot.grip_mm()))
        n = len(joints)
        if n % C.FPS == 0:
            print(f"    {n / C.FPS:5.1f} s  {n:4d} samples  grip {grip[-1]:5.1f} mm", end="\r", flush=True)
        next_t += 1.0 / C.FPS
        time.sleep(max(0.0, next_t - time.monotonic()))

    print(f"\n  taught {len(joints)} samples ({len(joints) / C.FPS:.1f} s)")
    print("  KEEP HOLDING THE ARM and press the PHYSICAL TEACH BUTTON again")
    print("  the arm should stiffen where it stands; do not release it yet")
    if not robot.wait_teach_stopped(timeout=TEACH_BUTTON_S):
        raise SystemExit("ABORT: teach recording did not stop. Keep holding the arm and power-cycle it.")
    hold_joints = robot.joints_deg()
    prompt("  arm stiffened; press Enter to take CAN control back...")
    if not robot.teach_stop(hold_joints=hold_joints):
        raise SystemExit("ABORT: the arm would not leave teach mode. Keep hold of it and run "
                         "`python tools/rest.py --drop`, then power-cycle the arm.")
    print("  control taken back")
    joints, n = clip_joints(joints) if joints else (np.zeros((0, 6)), 0)
    if n:
        print(f"  clipped {n} samples into the commandable joint range (the limp arm sags past J2/J3)")
    return {"fps": C.FPS, "task": task, "joints": joints.tolist(), "grip": grip}


def squeeze(taught_mm: float, margin=GRIP_SQUEEZE_MM) -> float:
    """What to COMMAND the gripper, given what the hand achieved during teach.

    Teaching records the MEASURED opening, and while holding a block that stalls at the block's width
    (38.2 mm on the first real demo). Commanding 38.2 mm back just brings the jaws to the block without
    squeezing it, so it slips - which is exactly what happened: held at 38.2 mm, then 0.1 mm mid-carry as
    the block fell out and the jaws closed the rest of the way. Position was replayed; force was not.
    So while the jaws are closing, command `margin` narrower and let GRIP_EFFORT do the gripping.

    A HAND can also force the jaws wider than the software maximum - a real demo taught 92.1 mm against a
    70 mm limit. _grip clips that before sending, so recording the raw value as the action would teach the
    policy to ask for 22 mm of opening the arm can never deliver (159 of 500 frames on that episode).
    """
    g = float(taught_mm) if taught_mm >= GRIP_HOLDING_BELOW_MM else max(0.0, float(taught_mm) - margin)
    return float(np.clip(g, C.GRIP_CLOSED_MM, C.GRIP_OPEN_MM))


def clip_joints(joints):
    """(clipped joints, how many samples changed).

    De-torqued, the arm SAGS a couple of degrees past J2's and J3's commandable limits - it can rest there
    but can never be commanded there, because send_joints clips to the same limits. An unclipped trajectory
    therefore asks for a start pose the arm can never report reaching and move_joints calls it a failure.
    Measured on the first real demo: 189 of 542 samples out of range, J2 to -2.26 deg (limit 0), J3 to
    +2.54 deg (limit 0). Clipping costs nothing - it is exactly what the arm would do with the command."""
    j = np.asarray(joints, float)
    out = np.clip(j, JOINT_LIMITS_DEG[:, 0], JOINT_LIMITS_DEG[:, 1])
    return out, int((out != j).any(axis=1).sum())


def released(g) -> bool:
    """Did the jaws ever open enough to DROP what they were holding?

    Two ways of asking this were wrong on real data:
      * "are they open at the end?" - a good demo releases over the bowl then shuts on the way out;
      * "do they reach 60 mm after the LAST hold?" - the last hold can be the jaws idling shut afterwards
        (ep19: held 44.2, released at 69.5, then sat at 27.3), and a release only has to exceed the BLOCK,
        not open fully (ep11 dropped a 44 mm cube by opening to 45.6). Both episodes succeeded on camera
        and both were flagged.
    So: any sustained hold followed by the jaws opening past that hold's own width plus a margin."""
    g = np.asarray(g, float)
    for end in sustained_holds(g):
        seg = g[max(0, end - int(MIN_HOLD_S * C.FPS)):end]
        width = float(np.median(seg)) if len(seg) else 0.0
        if (g[end:] > width + RELEASE_MARGIN_MM).any():
            return True
    return False


def sustained_holds(g) -> list:
    """End index of each spell where the jaws sat in the mid-range long enough to be gripping something.

    The duration test is the whole point: closing from open to shut passes THROUGH the mid-range for a
    frame or two, and counting that as a grasp made the release check fire on a demo that plainly worked
    (it released at 92 mm over the bowl, then shut the jaws on the way out)."""
    holding = (np.asarray(g) > GRIP_EMPTY_MM) & (np.asarray(g) < GRIP_HOLDING_BELOW_MM)
    ends, start = [], None
    for i, h in enumerate(list(holding) + [False]):
        if h and start is None:
            start = i
        elif not h and start is not None:
            if i - start >= MIN_HOLD_S * C.FPS:
                ends.append(i)
            start = None
    return ends


def check(traj: dict) -> list:
    """Reasons this trajectory should not be replayed. Cheap, and every one of them has a real failure behind it."""
    j, g = np.asarray(traj["joints"], float), np.asarray(traj["grip"], float)
    bad = []
    if len(j) < C.FPS:
        return [f"only {len(j)} samples (< 1 s)"]      # everything below reduces over j/g: they must be non-empty
    step = np.abs(np.diff(j, axis=0)).max()
    if step > MAX_STEP_DEG:
        bad.append(f"max joint step {step:.1f} deg/frame > {MAX_STEP_DEG} (inference clamps there)")
    if g.max() - g.min() < 5.0:
        bad.append(f"gripper barely moved ({g.min():.1f}..{g.max():.1f} mm) - was the block actually grasped?")
    # A place task must OPEN the jaws after gripping. Judging this on the final sample alone was wrong: a
    # good demo releases over the bowl and then shuts the jaws again on the way out, so the last value is
    # closed even though the release happened. Look for an opening after the last holding sample instead.
    if sustained_holds(g) and not released(g):
        bad.append("the jaws never open again after gripping - the demo never releases the block. "
                   "Open the gripper over the bowl before you stop sampling.")
    return bad


def replay(robot, cams, traj: dict, ep: Episode | None, task: str, speed: int,
           squeeze_mm=GRIP_SQUEEZE_MM) -> dict:
    """Stream the trajectory at C.FPS, capturing one frame per waypoint into `ep` (None = rehearsal)."""
    joints = np.asarray(traj["joints"], float)
    grip = np.asarray(traj["grip"], float)
    robot.gripper_enable()          # --play never runs teach_stop, so nothing else clears a latched error
    print(f"  moving to the start pose {np.round(joints[0], 1).tolist()}")
    if not robot.move_joints(joints[0], START_MOVE_S, squeeze(grip[0], squeeze_mm), speed=C.SPEED_HOME):
        raise SystemExit("ABORT: could not reach the trajectory start - is something in the way?")
    time.sleep(SETTLE_S)

    lag, next_t = [], time.monotonic()
    for i, (j, g) in enumerate(zip(joints, grip)):
        # observe BEFORE acting, exactly as at inference: obs[t] -> action[t], and Episode turns the NEXT
        # observation into action[t] (joints at t+1 + the gripper commanded at t)
        g_cmd = squeeze(g, squeeze_mm)
        if ep is not None:
            frame = {**{f"observation.images.{n}": cv2.cvtColor(cams[n].read(), cv2.COLOR_BGR2RGB) for n in cams},
                     "observation.state": robot.state7(), "task": task}
            ep.push(frame, g_cmd)          # the action is what was COMMANDED, squeeze included
        robot.send_joints(j, g_cmd, speed)
        lag.append(float(np.abs(robot.joints_deg() - j).max()))
        if i and i % C.FPS == 0:
            print(f"    replay {i / C.FPS:5.1f} s  {i:4d}/{len(joints)}  lag {lag[-1]:4.1f} deg", end="\r", flush=True)
        next_t += 1.0 / C.FPS
        time.sleep(max(0.0, next_t - time.monotonic()))

    stats = {"frames": len(joints), "lag_mean": float(np.mean(lag)), "lag_max": float(np.max(lag))}
    print(f"\n  replayed {stats['frames']} frames, tracking lag mean {stats['lag_mean']:.1f} deg, "
          f"max {stats['lag_max']:.1f} deg")
    if stats["lag_max"] > LAG_WARN_DEG:
        print(f"  WARNING: the arm did not keep up (>{LAG_WARN_DEG} deg behind). Raise --speed, or teach "
              "the motion more slowly - the replay will not look like what you demonstrated.")
    return stats


def ask_keep(msg="  keep this episode? [Y/n] ") -> bool:
    return prompt(msg).strip().lower() not in ("n", "no")


def session(ds, robot, cams, args):
    """Teach -> replay -> keep/discard, until the dataset holds --episodes."""
    saved = ds.meta.total_episodes
    traj = load(args.play) if args.play else None
    while saved < args.episodes:
        print(f"\n=== episode {saved} of {args.episodes} " + "=" * 30)
        if args.play:
            print(f"  replaying {args.play}")
        else:
            print("  place the block somewhere new, then teach the motion")
            traj = teach(robot, args.task)
            if len(traj["joints"]) < args.min_frames:      # replay would index [0] of an empty trajectory
                print(f"  only {len(traj['joints'])} samples - nothing to replay, teach it again")
                continue
            for reason in check(traj):
                print(f"  WARNING: {reason}")
            if not ask_keep("  replay this demonstration? [Y/n] "):
                continue
            save(traj, args.save_as or auto_name())

        # teaching MOVED the block into the bowl. The replay is open loop - it repeats joint angles and
        # knows nothing about where the block is - so it only succeeds if the block is put back where the
        # demonstration picked it up. Missing this prompt cost a whole recorded episode.
        print("\n  PUT THE BLOCK BACK where the demonstration picked it up (as exactly as you can)")
        prompt("  press Enter when the scene is reset...")
        ep = None if args.no_dataset else Episode(ds.add_frame)
        try:
            replay(robot, cams, traj, ep, args.task, args.speed, args.squeeze)
        except KeyboardInterrupt:
            if ep:
                ds.clear_episode_buffer()
            print("\n  Ctrl-C during replay: episode dropped")
            raise
        if ep is None:
            print("  --no-dataset: nothing written")
            if not ask_keep("  rehearse again? [Y/n] "):
                break
            continue
        n = ep.close()
        if n < args.min_frames:
            ds.clear_episode_buffer()
            print(f"  too short ({n} frames) - dropped")
            continue
        if not ask_keep():
            ds.clear_episode_buffer()
            print("  discarded")
            continue
        with no_interrupt():                       # durable at once: a kill here must not corrupt the dataset
            ds.save_episode()
            ds.finalize()
            ds = open_dataset(f"local/{ds.root.name}", ds.root)
            saved += 1
        print(f"  episode {saved - 1}: saved {n} frames ({n / C.FPS:.1f} s)")
    return ds


def confirm_overwrite(root: Path, args):
    """--overwrite DELETES a recorded dataset. It must never do that silently.

    It did: a --fake smoke test run with --overwrite and a default --name destroyed two real teleop
    episodes that had never been pushed anywhere. Episodes cost an afternoon each and this repo has now
    lost recordings twice, so the prompt names what would go and refuses --fake over real frames."""
    info = root / "meta" / "info.json"
    if not (args.overwrite and info.exists()):
        return
    n = json.load(open(info)).get("total_episodes", 0)
    if not n:
        return
    if args.fake:
        raise SystemExit(f"ABORT: --fake --overwrite would delete {n} recorded episode(s) in {root}. "
                         "Synthetic runs belong in --test or their own --name.")
    print(f"\n  --overwrite will DELETE {n} episode(s) in {root}")
    if args.yes or input("  type the dataset name to confirm: ").strip() == root.name:
        return
    raise SystemExit("  not overwritten")


def auto_name() -> str:
    """A trajectory name that cannot collide with an existing one.

    The old default numbered from the episode counter, so it restarted at ep000 for every fresh dataset
    and every --test run: three real demonstrations were silently overwritten in one afternoon. A
    timestamp sorts chronologically, says when it was taught, and never repeats."""
    stamp = time.strftime("demo_%Y%m%d_%H%M%S")
    if not os.path.exists(os.path.join(TRAJ_DIR, f"{stamp}.json")):
        return stamp
    return next(f"{stamp}_{i}" for i in range(2, 100)             # same second: still never overwrite
                if not os.path.exists(os.path.join(TRAJ_DIR, f"{stamp}_{i}.json")))


def newest() -> str:
    """Name of the most recently saved trajectory - what `--play latest` resolves to."""
    paths = sorted(glob.glob(os.path.join(TRAJ_DIR, "*.json")), key=os.path.getmtime)
    if not paths:
        raise SystemExit(f"ABORT: no trajectories in {TRAJ_DIR} yet - teach one first")
    return os.path.splitext(os.path.basename(paths[-1]))[0]


def save(traj: dict, name: str) -> str:
    os.makedirs(TRAJ_DIR, exist_ok=True)
    path = os.path.join(TRAJ_DIR, f"{name}.json")
    json.dump(traj, open(path, "w"))
    print(f"  trajectory -> {path}")
    return path


def load(name: str) -> dict:
    if name == "latest":
        name = newest()
    path = name if os.path.exists(name) else os.path.join(TRAJ_DIR, f"{name}.json")
    if not os.path.exists(path):
        raise SystemExit(f"ABORT: no trajectory {path}")
    traj = json.load(open(path))
    if traj["fps"] != C.FPS:
        raise SystemExit(f"ABORT: {path} was taught at {traj['fps']} fps, this build records at {C.FPS}")
    print(f"  loaded {path}: {len(traj['joints'])} samples ({len(traj['joints']) / traj['fps']:.1f} s)")
    clipped, n = clip_joints(traj["joints"])
    if n:
        print(f"  clipped {n} samples into the commandable joint range")
        traj["joints"] = clipped.tolist()
    for reason in check(traj):
        print(f"  WARNING: {reason}")
    return traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None,
                    help="dataset directory under data/ - REQUIRED unless --test (no default: a default "
                         "pointing at a real dataset is how one got overwritten)")
    ap.add_argument("--repo-id", default=None,
                    help="deprecated: only the part after '/' is used, as --name")
    ap.add_argument("--root", default=None, help="default data/<dataset name>; resumed if it exists")
    ap.add_argument("--episodes", type=int, default=10, help="total episodes the dataset should hold")
    ap.add_argument("--record", metavar="NAME", help="teach one trajectory, save it, and exit (no dataset)")
    ap.add_argument("--play", metavar="NAME",
                    help="replay a saved trajectory instead of teaching ('latest' = the newest one)")
    ap.add_argument("--save-as", metavar="NAME",
                    help="name for the taught trajectory (default: a timestamp, never overwrites)")
    ap.add_argument("--no-dataset", action="store_true", help="rehearse the replay, write nothing")
    ap.add_argument("--test", action="store_true",
                    help="write to data/test, which tools/push_dataset.py refuses to publish")
    ap.add_argument("--fake", action="store_true", help="no arm, no cameras: exercise the dataset path")
    ap.add_argument("--speed", type=int, default=C.SPEED_JOINT, help="replay speed %% (raise it if lag is high)")
    ap.add_argument("--squeeze", type=float, default=GRIP_SQUEEZE_MM,
                    help="mm narrower than taught to command while gripping (0 = replay as taught)")
    ap.add_argument("--task", default=C.TASK)
    ap.add_argument("--min-frames", type=int, default=C.FPS)
    ap.add_argument("--overwrite", action="store_true",
                    help="delete an existing dataset first (asks before dropping recorded episodes)")
    ap.add_argument("--yes", "-y", action="store_true", help="skip the --overwrite confirmation")
    args = ap.parse_args()
    if not args.name and not args.repo_id and not args.test and not args.record:
        raise SystemExit("ABORT: say where the data goes - --name <dataset> or --test")
    if args.repo_id:                      # accept the old flag so saved commands keep working
        args.name = args.repo_id.split("/")[-1]
        print(f"note: --repo-id is deprecated here (push is a separate tool) - using --name {args.name}")
    os.environ.setdefault("SVT_LOG", "1")
    av.logging.set_libav_level(av.logging.ERROR)
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    robot = cams = ds = None
    try:
        if args.fake:
            from piper.fake_arm import FakeArm
            robot, cams = FakeArm(), {n: FakeCamera(n) for n in C.CAMERA_NAMES}
        else:
            from piper.robot import connect_for_motion
            robot = connect_for_motion()
            cams = open_cameras()

        if args.record:                              # teach only: no dataset, no cameras needed
            save(teach(robot, args.task), args.record)
            return

        root = Path(args.root or os.path.join(C.DATA_DIR, "test" if args.test else args.name))
        confirm_overwrite(root, args)
        # The repo id is not stored in the dataset and publishing moved to tools/push_dataset.py, so it is
        # only LeRobot's in-memory handle here: the directory is what identifies the dataset on disk.
        ds = open_dataset(f"local/{root.name}", root, args.overwrite)
        if ds.meta.total_episodes:
            print(f"resuming: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames")
        if ds.meta.total_episodes >= args.episodes and not args.no_dataset:
            print(f"already {ds.meta.total_episodes} episodes (--episodes is the total): nothing to record")
            return
        ds = session(ds, robot, cams, args)
    except KeyboardInterrupt:
        print("\nCtrl-C: stopping")
    finally:
        if cams:
            for cam in cams.values():
                cam.close()
        if robot is not None:
            robot.rest()
        if ds is not None:
            with no_interrupt():
                ds.finalize()
            print(f"{ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames -> {ds.root}")
            if ds.meta.total_episodes and not args.test:
                print(f"push when you are happy with them: "
                      f"python -m tools.push_dataset {ds.root}")


if __name__ == "__main__":
    main()
