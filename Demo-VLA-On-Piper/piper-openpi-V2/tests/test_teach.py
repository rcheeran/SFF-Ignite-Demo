"""collect/teach.py without hardware: trajectory validation, save/load round-trip, and the replay contract
(measured state as the observation, one-frame-lag action) against piper.fake_arm.FakeArm.

    python -m pytest tests/test_teach.py -q
"""
import numpy as np
import pytest

from collect import teach as T
from collect import record as R
from piper import config as C
from piper.cameras import FakeCamera
from piper.fake_arm import FakeArm


def traj(n=45, step=0.5, grip_span=40.0):
    """A plausible taught trajectory: joints ramping, gripper closing partway through.

    Signs matter: J2 is [0, 180] and J3 is [-170, 0], so a naive [step*i]*6 is OUT of range on J3 for every
    sample and clip_joints rewrites the whole thing - which is a broken fixture, not a broken clip."""
    j = [[step * i, step * i, -step * i, step * i, step * i * 0.5, 0.0] for i in range(n)]
    # a COMPLETE demo: approach open, hold for the middle third, release. One that only ever closes is a
    # demo that never puts the block down, and check() rightly refuses it.
    g = [C.GRIP_OPEN_MM - (grip_span if n // 3 <= i < 2 * n // 3 else 0.0) for i in range(n)]
    return {"fps": C.FPS, "task": C.TASK, "joints": j, "grip": g}


def test_check_rejects_a_trajectory_too_short_to_replay():
    """An empty trajectory used to crash check() on np.max of a zero-size array instead of being reported,
    and replay() would then index joints[0] of nothing."""
    assert "0 samples" in T.check({"joints": [], "grip": []})[0]
    assert T.check(traj(n=3))[0].startswith("only 3 samples")


def test_check_flags_a_step_inference_would_clamp():
    bad = traj(n=30)
    bad["joints"][10] = [90.0] * 6                       # a jump the deployed policy would never be allowed
    assert any("deg/frame" in r for r in T.check(bad))


def test_check_flags_a_gripper_that_never_moved():
    """The jaws not moving means no grasp happened - the single most common way a demo is silently useless."""
    assert any("gripper barely moved" in r for r in T.check(traj(grip_span=0.0)))
    assert T.check(traj()) == []


def test_save_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "TRAJ_DIR", str(tmp_path))
    T.save(traj(), "demo")
    got = T.load("demo")
    assert got["joints"] == traj()["joints"] and got["task"] == C.TASK


def test_load_refuses_a_trajectory_taught_at_another_fps(tmp_path, monkeypatch):
    """Replaying 30 Hz samples at 15 Hz would silently play the motion at half speed."""
    monkeypatch.setattr(T, "TRAJ_DIR", str(tmp_path))
    t = traj()
    t["fps"] = 30
    T.save(t, "wrong_fps")
    with pytest.raises(SystemExit, match="taught at 30 fps"):
        T.load("wrong_fps")


def test_replay_records_measured_state_and_the_one_frame_lag_action():
    """The dataset contract, end to end: observation is what the arm MEASURED (not the commanded waypoint,
    which the arm lags), and action[t] = joints measured at t+1 + the gripper commanded at t."""
    out = []
    arm, cams = FakeArm(), {n: FakeCamera(n) for n in C.CAMERA_NAMES}
    t = traj(n=20)
    ep = R.Episode(out.append)
    T.replay(arm, cams, t, ep, C.TASK, speed=C.SPEED_JOINT)
    assert ep.close() == len(t["joints"])

    states = np.stack([f["observation.state"] for f in out])
    actions = np.stack([f["action"] for f in out])
    assert np.allclose(actions[:-1, :6], states[1:, :6])          # action joints = the NEXT observation
    # and the observation is MEASURED, not the waypoint: FakeArm lags its target, so recording the command
    # instead of state7() would make these identical - the assertions above would pass either way
    assert not np.allclose(states[:, :6], np.asarray(t["joints"]))
    # action gripper = what was COMMANDED, which is the squeezed value, not the taught opening
    assert np.allclose(actions[:, 6], [T.squeeze(g) for g in t["grip"]])
    assert all(f["task"] == C.TASK for f in out)
    for n in C.CAMERA_NAMES:
        assert out[0][f"observation.images.{n}"].shape == (C.IMG_H, C.IMG_W, 3)


def test_replay_without_an_episode_writes_nothing():
    """--no-dataset rehearsal: the arm moves, the dataset does not grow."""
    out = []
    stats = T.replay(FakeArm(), {}, traj(n=10), None, C.TASK, speed=C.SPEED_JOINT)
    assert out == [] and stats["frames"] == 10


def test_clip_joints_brings_a_sagging_demo_into_the_commandable_range():
    """De-torqued the arm rests a couple of degrees PAST J2/J3's limits, but send_joints clips to them, so
    an unclipped start pose can never be 'reached' and replay aborted with 'could not reach the trajectory
    start'. Real demo: 189 of 542 samples out, J2 to -2.26 (limit 0), J3 to +2.54 (limit 0)."""
    from piper.robot import JOINT_LIMITS_DEG as L
    sag = [[0.31, -2.15, 1.62, 0.95, 26.18, 0.0]] * 3 + [[0.0, 10.0, -10.0, 0.0, 0.0, 0.0]]
    out, n = T.clip_joints(sag)
    assert n == 3                                        # only the sagging samples changed
    assert (out >= L[:, 0]).all() and (out <= L[:, 1]).all()
    assert out[0][1] == 0.0 and out[0][2] == 0.0         # J2 and J3 pulled to their limits
    assert out[3].tolist() == [0.0, 10.0, -10.0, 0.0, 0.0, 0.0]


def test_load_clips_a_trajectory_taught_before_the_fix(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "TRAJ_DIR", str(tmp_path))
    t = traj(n=20)
    t["joints"][0] = [0.31, -2.15, 1.62, 0.95, 26.18, 0.0]      # the sag, as saved by the old code
    T.save(t, "sagging")
    assert T.load("sagging")["joints"][0][1] == 0.0


def test_squeeze_commands_narrower_than_taught_so_the_block_does_not_slip():
    """Teaching records the MEASURED opening, which while holding stalls at the block's width. Commanding
    that back only kisses the block: the first real replay held at 38.2 mm then dropped to 0.1 mm mid-carry
    as it fell out. Closing commands must go NARROWER than taught; open ones must not be touched."""
    assert T.squeeze(38.2) == 38.2 - T.GRIP_SQUEEZE_MM     # holding: squeeze it
    assert T.squeeze(C.GRIP_OPEN_MM) == C.GRIP_OPEN_MM     # fully open: unchanged, do not crush on approach
    assert T.squeeze(70.0) == 70.0
    assert T.squeeze(2.0) == 0.0                           # already shut: clamp, never negative
    assert T.squeeze(38.2, margin=0.0) == 38.2             # --squeeze 0 replays exactly as taught


def test_replay_records_the_squeezed_command_as_the_action():
    """The action must be what was COMMANDED (squeeze included) - that is what the policy has to learn to
    output. Recording the taught value would teach it the grip that drops the block."""
    out = []
    t = traj(n=12, grip_span=40.0)
    ep = R.Episode(out.append)
    T.replay(FakeArm(), {n: FakeCamera(n) for n in C.CAMERA_NAMES}, t, ep, C.TASK, speed=C.SPEED_JOINT)
    ep.close()                       # the last frame is only written on close (one-frame lag)
    acted = np.stack([f["action"] for f in out])[:, 6]
    assert np.allclose(acted, [T.squeeze(g) for g in t["grip"]])
    assert acted.min() == C.GRIP_OPEN_MM - 40.0 - T.GRIP_SQUEEZE_MM   # the closing half really is narrower


def test_check_flags_a_demo_that_never_releases_the_block():
    """A place task must OPEN the jaws after gripping - a demo that only ever shuts on the block could
    never put it in the bowl."""
    t = traj(n=60)
    t["grip"] = [C.GRIP_OPEN_MM] * 15 + [45.0] * 30 + [-2.9] * 15        # grasp, carry, never let go
    assert any("never releases" in r for r in T.check(t))


def test_release_is_judged_against_the_held_width_not_a_fixed_opening():
    """ep19 (real): held 44.2, released at 69.5, then the jaws idled at 27.3 - so judging the release on
    what happens after the LAST sustained hold misses it, because that last hold is the idling."""
    n = int(T.MIN_HOLD_S * C.FPS) + 2
    assert T.released([70.0] * n + [44.2] * n + [69.5] * n + [27.3] * n)       # release, then idle mid-range
    assert T.released([70.0] * n + [43.4] * n + [70.0] * n + [0.3] * n)        # release, then shut on the way out
    assert not T.released([70.0] * n + [43.4] * n + [43.4] * n + [0.3] * n)    # never opened: no release
    assert not T.released([70.0] * 4 * n)                                       # never gripped at all


def test_check_accepts_a_demo_that_releases_then_shuts_the_jaws_on_the_way_out():
    """The real successful episode: open, hold at ~45 mm, RELEASE at 92 mm over the bowl, then shut the
    jaws and retreat. Judging the release on the final sample called this a failure; judging it on the
    jaws passing through mid-range as they close did too. Only a SUSTAINED hold counts as gripping."""
    t = traj(n=75)
    t["grip"] = ([C.GRIP_OPEN_MM] * 15 + [45.0] * 30 + [92.0] * 15       # release, wider than the software max
                 + [40.0, 20.0] + [-2.9] * 13)                            # then sweep shut on the way out
    assert not any("never releases" in r for r in T.check(t))
    assert T.sustained_holds(t["grip"]) == [45]      # the 2-frame closing sweep is not a grasp


def test_auto_name_never_overwrites_an_existing_trajectory(tmp_path, monkeypatch):
    """The old default numbered from the episode counter, so every fresh dataset and every --test run
    restarted at ep000 and silently overwrote earlier demonstrations."""
    monkeypatch.setattr(T, "TRAJ_DIR", str(tmp_path))
    first = T.auto_name()
    T.save(traj(), first)
    assert T.auto_name() != first                       # same second, still a new name
    T.save(traj(), T.auto_name())
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_play_latest_resolves_to_the_most_recent_trajectory(tmp_path, monkeypatch):
    import os, time
    monkeypatch.setattr(T, "TRAJ_DIR", str(tmp_path))
    T.save(traj(n=20), "older")
    T.save(traj(n=30), "newer")
    os.utime(tmp_path / "newer.json", (time.time() + 10, time.time() + 10))
    assert T.newest() == "newer"
    assert len(T.load("latest")["joints"]) == 30


def test_newest_says_so_when_there_are_no_trajectories(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "TRAJ_DIR", str(tmp_path))
    with pytest.raises(SystemExit, match="no trajectories"):
        T.newest()


def test_teach_no_longer_offers_push():
    """Publishing moved to tools/push_dataset.py: whether episodes are worth publishing is judged AFTER
    looking at them, not chosen before the session starts."""
    import subprocess, sys, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, "-m", "collect.teach", "--help"], capture_output=True,
                         text=True, cwd=root).stdout
    assert "--push" not in out and "--private" not in out


def test_overwrite_refuses_to_destroy_recorded_episodes_in_a_fake_run(tmp_path):
    """A --fake smoke test with --overwrite and a default --name destroyed two real teleop episodes that
    had never been pushed. Synthetic runs must never delete real recordings."""
    import argparse, json
    root = tmp_path / "piper_block_bowl"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": 2}))
    args = argparse.Namespace(overwrite=True, fake=True, yes=False)
    with pytest.raises(SystemExit, match="would delete 2 recorded episode"):
        T.confirm_overwrite(root, args)


def test_overwrite_is_silent_when_there_is_nothing_to_lose(tmp_path):
    import argparse, json
    root = tmp_path / "empty"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": 0}))
    args = argparse.Namespace(overwrite=True, fake=True, yes=False)
    T.confirm_overwrite(root, args)                       # no episodes: nothing to confirm
    T.confirm_overwrite(tmp_path / "missing", args)       # no dataset at all
