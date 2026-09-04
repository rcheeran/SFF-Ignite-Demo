"""collect/record.py without hardware: one-frame-lag bookkeeping, Ctrl-C / SIGTERM deferral, per-episode durability
(SIGKILL), resume, --push token, the --fake CLI, and tools/inspect_dataset.py."""
import importlib.util
import multiprocessing
import os
import signal
import subprocess
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from collect import record as R
from piper import config as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLANK = {f"observation.images.{c}": np.zeros((C.IMG_H, C.IMG_W, 3), np.uint8) for c in C.CAMERA_NAMES}
os.environ.setdefault("SVT_LOG", "1")


def inspect_module():
    spec = importlib.util.spec_from_file_location("inspect_dataset", os.path.join(ROOT, "tools", "inspect_dataset.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def frame(joint, grip_measured):
    return {**BLANK, "observation.state": np.array([joint] * 6 + [grip_measured], np.float32), "task": C.TASK}


def test_episode_one_frame_lag():
    out = []
    ep = R.Episode(out.append)
    ep.push(frame(1, 50), 70)
    assert (ep.n, out) == (0, [])                      # frame 0 waits for frame 1's joints
    ep.push(frame(2, 40), 30)
    ep.push(frame(3, 20), 10)
    assert ep.n == 2
    assert ep.close() == 3 and ep.close() == 3         # the last frame holds its own joints; close is idempotent
    acts = [f["action"] for f in out]
    assert all(a.dtype == np.float32 and a.shape == (7,) for a in acts)
    np.testing.assert_array_equal(acts[0], [2] * 6 + [70])   # joints at t+1, gripper commanded at t
    np.testing.assert_array_equal(acts[1], [3] * 6 + [30])
    np.testing.assert_array_equal(acts[2], [3] * 6 + [10])
    assert out[0]["observation.state"][6] == 50 and out[0]["task"] == C.TASK and "grip_cmd" not in out[0]


def test_episode_summary_flags_quality():
    ep = R.Episode(lambda f: None)
    assert ep.summary() == "0 frames (0.0 s)"
    ep.stuck, ep.no_feedback = 2, 30
    assert ep.summary() == "0 frames (0.0 s), 2 stuck, 2.0 s no arm feedback - consider B next time"


def test_no_interrupt_defers_ctrl_c_for_parent_and_forked_worker():
    assert multiprocessing.get_start_method() == "fork"   # the encoder workers inherit the deferring handler
    done, prev_term = [], signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt):
        with R.no_interrupt():
            with ProcessPoolExecutor(1) as pool:
                worker = pool.submit(os.getpid).result()
                fut = pool.submit(time.sleep, 0.5)
                time.sleep(0.1)
                os.kill(worker, signal.SIGINT)          # a terminal Ctrl-C hits the whole process group
                os.kill(os.getpid(), signal.SIGINT)
                os.kill(os.getpid(), signal.SIGTERM)    # pkill is deferred the same way (SIG_DFL would kill us here)
                assert fut.result() is None             # the worker finished its job
            done.append(1)                              # and so did the parent
    assert done and signal.getsignal(signal.SIGINT) is signal.default_int_handler
    assert signal.getsignal(signal.SIGTERM) == prev_term


def _record_episode(ds, n, joint0):
    ep = R.Episode(ds.add_frame)
    for t in range(n):
        ep.push(frame(joint0 + t, 70), 70 - t)
    ep.close()
    ds.save_episode()


def test_open_dataset_create_resume_overwrite(tmp_path):
    root = tmp_path / "ds"
    ds = R.open_dataset("local/ds", root)
    assert ds.meta.total_episodes == 0
    _record_episode(ds, 8, 0)
    ds.finalize()
    ds = R.open_dataset("local/ds", root)
    assert (ds.meta.total_episodes, ds.meta.total_frames) == (1, 8)
    _record_episode(ds, 8, 100)
    ds.finalize()
    loaded = LeRobotDataset("local/ds", root=root, video_backend="pyav")
    assert (loaded.num_episodes, loaded.num_frames, loaded.fps) == (2, 16, C.FPS)
    cols = loaded.hf_dataset.with_format("numpy", columns=["action", "observation.state", "episode_index"])[:]
    for e in (0, 1):
        m = cols["episode_index"] == e
        a, s = cols["action"][m], cols["observation.state"][m]
        np.testing.assert_array_equal(a[:-1, :6], s[1:, :6])
        np.testing.assert_array_equal(a[-1, :6], s[-1, :6])
        np.testing.assert_array_equal(a[:, 6], 70 - np.arange(8))
    assert float(loaded[8]["observation.state"][0]) == 100 and tuple(loaded[8]["observation.images.top"].shape) == (3, C.IMG_H, C.IMG_W)
    (tmp_path / "junk").mkdir()
    with pytest.raises(SystemExit):
        R.open_dataset("local/junk", tmp_path / "junk")
    ds = R.open_dataset("local/ds", root, overwrite=True)
    assert ds.meta.total_episodes == 0 and not list(root.glob("data/**/*.parquet"))
    ds.finalize()


def test_inspect_dataset_judges_and_refuses_single_frame_episodes(tmp_path, capsys):
    I = inspect_module()
    root = tmp_path / "ds"
    ds = R.open_dataset("local/ds", root)
    _record_episode(ds, 8, 0), _record_episode(ds, 8, 100)
    ds.finalize()
    I.main(str(root))
    out = capsys.readouterr().out
    assert "2 episodes, 16 frames, 15 fps" in out and "lag check: mean |action[t][:6] - state[t+1][:6]| = 0.0000" in out
    assert "frames with a joint step > 10 deg: 0 / 14" in out and (tmp_path / "ds_contact.jpg").stat().st_size > 0
    ds = R.open_dataset("local/one", tmp_path / "one")
    _record_episode(ds, 1, 0)                           # no joint step at all: say so instead of a numpy traceback
    ds.finalize()
    with pytest.raises(SystemExit, match="single frame"):
        I.main(str(tmp_path / "one"))


def _main(monkeypatch, argv):
    """record.main() in-process (no monitor port, SIGTERM handler restored afterwards)."""
    monkeypatch.setattr(R, "Monitor", lambda cams, status: types.SimpleNamespace())
    monkeypatch.setattr(sys, "argv", ["record", "--fake", *argv])
    prev = signal.getsignal(signal.SIGTERM)
    try:
        R.main()
    finally:
        signal.signal(signal.SIGTERM, prev)


def test_ctrl_c_during_save_keeps_the_episode(tmp_path, monkeypatch, capsys):
    save = LeRobotDataset.save_episode

    def save_with_ctrl_c(self, *a, **k):
        os.kill(os.getpid(), signal.SIGINT)             # lands inside no_interrupt(): deferred until the save is durable
        save(self, *a, **k)

    monkeypatch.setattr(LeRobotDataset, "save_episode", save_with_ctrl_c)
    _main(monkeypatch, ["--root", str(tmp_path / "ds"), "--repo-id", "local/ds", "--episodes", "3",
                        "--fake-episode-s", "1.0", "--min-frames", "5"])
    out = capsys.readouterr().out
    assert "episode 0: saved" in out and "Ctrl-C: stopping\n" in out and "\n1 episodes, " in out, out
    assert "dropped" not in out, out                    # the operator must not re-record it (duplicate episode)
    assert LeRobotDataset("local/ds", root=tmp_path / "ds", video_backend="pyav").num_episodes == 1


def test_push_prefers_the_env_token_over_the_token_file(tmp_path, monkeypatch):
    root = tmp_path / "ds"
    ds = R.open_dataset("local/ds", root)
    _record_episode(ds, 8, 0)
    ds.finalize()
    pushes = []
    monkeypatch.setattr(LeRobotDataset, "push_to_hub", lambda self, **kw: pushes.append(kw))
    argv = ["--root", str(root), "--repo-id", "local/ds", "--episodes", "1", "--push", "--private"]
    monkeypatch.setenv("HF_TOKEN", "from-env")
    monkeypatch.setattr(C, "HF_TOKEN_FILE", str(tmp_path / "missing-token.txt"))
    _main(monkeypatch, argv)                            # 1 episode already: nothing to record, push with the env token
    assert pushes == [{"tags": ["piper", "quest", "pi05"], "private": True}] and os.environ["HF_TOKEN"] == "from-env"
    monkeypatch.delenv("HF_TOKEN")
    (tmp_path / "token.txt").write_text("from-file\n")
    monkeypatch.setattr(C, "HF_TOKEN_FILE", str(tmp_path / "token.txt"))
    _main(monkeypatch, argv)
    assert len(pushes) == 2 and os.environ["HF_TOKEN"] == "from-file"


def _cli(root, episodes):
    return [sys.executable, "-u", "-m", "collect.record", "--fake", "--root", str(root), "--repo-id", "local/cli",
            "--episodes", str(episodes), "--fake-episode-s", "1.0", "--min-frames", "5"]


def _run_until(root, episodes, line_prefix, sig, settle_s=0.0):
    """Start the --fake CLI, send sig to its process group once a line starts with line_prefix, return (rc, text)."""
    p = subprocess.Popen(_cli(root, episodes), cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         start_new_session=True)
    out = []
    for line in p.stdout:
        out.append(line)
        if line.startswith(line_prefix):
            time.sleep(settle_s)
            os.killpg(p.pid, sig)
    return p.wait(60), "".join(out)


def test_fake_cli_survives_sigkill_and_sigterm_then_resumes(tmp_path):
    root = tmp_path / "cli"
    # power loss / SIGKILL right after a save: every saved episode is already durable (finalize + reopen per episode)
    rc, text = _run_until(root, 5, "episode 1: saved", signal.SIGKILL)
    assert rc == -signal.SIGKILL and "episode 0: saved" in text, text
    ds = LeRobotDataset("local/cli", root=root, video_backend="pyav")
    assert ds.num_episodes == 2 and ds.num_frames == sum(ds.meta.episodes["length"])
    # pkill (SIGTERM) during episode 3 takes the Ctrl-C path: episode 2 kept, 3 dropped, dataset finalized
    rc, text = _run_until(root, 5, "episode 3: REC", signal.SIGTERM, settle_s=0.3)
    assert rc == 0, text
    assert "resuming: 2 episodes" in text and "episode 2: saved" in text and "episode 3 dropped" in text, text
    assert "Ctrl-C: stopping" in text and "\n3 episodes, " in text and "Map:" not in text, text
    ds = LeRobotDataset("local/cli", root=root, video_backend="pyav")
    assert ds.num_episodes == 3 and ds.num_frames == sum(ds.meta.episodes["length"])
    cols = ds.hf_dataset.with_format("numpy", columns=["action", "observation.state", "episode_index", "index"])[:]
    np.testing.assert_array_equal(cols["index"], np.arange(ds.num_frames))
    for e in (0, 1, 2):
        m = cols["episode_index"] == e
        np.testing.assert_array_equal(cols["action"][m][:-1, :6], cols["observation.state"][m][1:, :6])
    assert tuple(ds[ds.num_frames - 1]["observation.images.wrist"].shape) == (3, C.IMG_H, C.IMG_W)


def test_fake_cli_survives_ctrl_c_then_resumes(tmp_path):
    root = tmp_path / "cli"
    # terminal-style Ctrl-C (whole process group) in the middle of episode 1: episode 0 stays, 1 is dropped
    p = subprocess.Popen(_cli(root, 3), cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         start_new_session=True)
    out = []
    for line in p.stdout:
        out.append(line)
        if line.startswith("episode 1: REC"):
            time.sleep(0.4)
            os.killpg(p.pid, signal.SIGINT)
    assert p.wait(60) == 0, "".join(out)
    text = "".join(out)
    assert "episode 0: saved" in text and "episode 1 dropped" in text and "\n1 episodes, " in text, text
    assert "Svt[" not in text and "event: engaged" not in text and "Map:" not in text, text   # calm terminal
    r = subprocess.run(_cli(root, 2), cwd=ROOT, text=True, capture_output=True, timeout=120)
    assert r.returncode == 0 and "resuming: 1 episodes" in r.stdout and "episode 1: saved" in r.stdout, r.stdout + r.stderr
    ds = LeRobotDataset("local/cli", root=root, video_backend="pyav")
    assert ds.num_episodes == 2 and ds.num_frames == sum(ds.meta.episodes["length"])
    cols = ds.hf_dataset.with_format("numpy", columns=["action", "observation.state", "episode_index"])[:]
    for e in (0, 1):
        m = cols["episode_index"] == e
        np.testing.assert_array_equal(cols["action"][m][:-1, :6], cols["observation.state"][m][1:, :6])


def test_record_closes_its_cameras(monkeypatch, tmp_path):
    """A RealSense pipeline still running at interpreter exit aborts the process (seen live 2026-08-30):
    record() must close every camera it opened, on every exit path."""
    closed = []

    class Cam(R.FakeCamera):
        def close(self):
            closed.append(self.name)
    monkeypatch.setattr(R, "FakeCamera", Cam)
    monkeypatch.setattr(R, "Monitor", lambda cams, status: types.SimpleNamespace())
    ds = R.open_dataset("local/cam_close", tmp_path / "cam_close")
    args = types.SimpleNamespace(repo_id="local/cam_close", episodes=0, fake=True, real_cams=False, sim=False,
                                 scripted=False, task=C.TASK, min_frames=1, fake_episode_s=0.5, scale=1.0,
                                 rot_scale=None, seed=0)
    R.record(ds, args)                                    # target already reached: opens and must close
    ds.finalize()
    assert sorted(closed) == sorted(C.CAMERA_NAMES)
