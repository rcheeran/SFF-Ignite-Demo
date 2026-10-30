"""inference/run_policy.py without GPU or robot: chunk prefetch (ActionStream), the arm-loop safety clamps,
pause/resume/quit, arm-fault stop, J6 pause, shadow mode, the per-tick CSV. Fake model + FakeArm + FakeCamera."""
import csv
import glob
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from inference.run_policy import ActionStream, Keys, deploy, run_episode
from piper import config as C
from piper.cameras import FakeCamera
from piper.robot import Piper
from piper.transforms import tip_position
from piper.fake_arm import FakeArm

REAL_SLEEP = time.sleep
READY = np.array(C.READY_JOINTS_DEG)
J1_FAR = READY + [90, 0, 0, 0, 0, 0]                    # far target on the vertical axis: tip height unchanged
BELOW_FLOOR = np.array([0, 120, -60, 0, 60, -119.0])    # tip z = -36 mm (through the table)
CAMS = {n: FakeCamera(n) for n in C.CAMERA_NAMES}


class Arm(FakeArm):
    """FakeArm that runs callbacks at given ticks (state7 is read exactly once per loop tick)."""
    def __init__(self, at=None, **kw):
        super().__init__(**kw)
        self.n, self.at = 0, at or {}

    def state7(self):
        self.n += 1
        if self.n in self.at:
            self.at[self.n]()
        return super().state7()


def model(target, grip=C.GRIP_OPEN_MM, delay=0.0):
    """Fake policy: every chunk is 50 copies of target (deg) + grip, delivered after `delay` s."""
    def predict(top, wrist, state):
        threading.Event().wait(delay)
        return np.tile(np.append(target, grip).astype(np.float32), (50, 1))
    return predict


def sent(arm):
    return np.array([j for j, _ in arm.sent_joints])


@pytest.fixture(autouse=True)
def fast(monkeypatch, tmp_path):
    """Ticks as fast as the loop can go; logs into tmp_path; fixed chunking."""
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setattr(C, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(C, "PREFETCH_AT", 8)


# ---- ActionStream --------------------------------------------------------------------------------------------------
def stream_ticks(delay, ticks, period, n=10):
    """Drive a stream whose fake model returns chunk[i] = i + state[0] after `delay` s, at `period` s per tick.
    With obs state[0] = tick index, a time-consistent action at tick t equals t. Returns (actions, next() seconds)."""
    def predict(top, wrist, state):
        threading.Event().wait(delay)
        return np.tile(np.arange(50, dtype=np.float32)[:, None] + state[0], (1, 7))

    stream = ActionStream(predict, n, blend_ticks=0)   # these assert exact indices: alignment, not smoothing
    actions, durs = [], []
    for k in range(ticks):
        t0 = time.perf_counter()
        a, hold = stream.next((None, None, np.array([k, 0, 0, 0, 0, 0, 0], np.float32)), float(k))
        durs.append(time.perf_counter() - t0)
        actions.append(float(a[0]))
        threading.Event().wait(period)
    return stream, actions, durs


def test_stream_prefetch_is_continuous_and_never_blocks():
    stream, actions, durs = stream_ticks(delay=0.05, ticks=40, period=0.01)   # 50 ms model, 10 ms ticks, n=10
    assert durs[0] >= 0.05                                  # the first chunk is waited for
    assert max(durs[1:]) < 0.01                             # afterwards a tick never waits for the model
    assert actions == list(range(40))                       # hand-over skips the actions executed meanwhile
    assert stream.holds == 0 and len(stream.lat) >= 4             # requests at ticks 0, 2, 12, 22, 32


def test_stream_late_model_keeps_serving_the_live_chunk():
    stream, actions, durs = stream_ticks(delay=0.2, ticks=60, period=0.01)    # 200 ms model = 20 ticks > n
    assert max(durs[1:]) < 0.01
    assert actions == list(range(60))                       # still time-consistent, from the 50-step chunk
    assert stream.late > 0 and stream.holds == 0


def test_stream_exhausted_chunk_holds_then_recovers():
    stream, actions, durs = stream_ticks(delay=0.15, ticks=200, period=0.002)  # model = 75 ticks > 50-step chunk
    assert max(durs[1:]) < 0.01
    assert stream.holds > 0
    assert actions == sorted(actions)                        # a hold repeats the last action, never jumps back
    assert actions[-1] > actions[-30]                        # streaming again after the hold


def test_stream_model_error_ends_the_run():
    def bad(top, wrist, state):
        raise ValueError("cuda exploded")
    arm = FakeArm()
    with pytest.raises(RuntimeError):
        deploy(ActionStream(bad), arm, CAMS, Keys(), 10)
    assert arm.rested


# ---- arm loop safety -----------------------------------------------------------------------------------------------
def test_step_and_gripper_clamps():
    arm = FakeArm()
    assert run_episode(ActionStream(model(J1_FAR, grip=150.0)), arm, CAMS, Keys(), 30)
    js = sent(arm)
    assert len(js) == 30
    assert np.abs(js[0] - READY).max() <= C.MAX_JOINT_STEP_DEG + 1e-9
    assert np.abs(np.diff(js, axis=0)).max() <= C.MAX_JOINT_STEP_DEG + 1e-9
    assert js[-1][0] > READY[0] + 50                        # and it does get there
    grips = [g for _, g in arm.sent_joints]
    assert max(grips) == C.GRIP_OPEN_MM
    arm = FakeArm()
    run_episode(ActionStream(model(READY, grip=-5.0)), arm, CAMS, Keys(), 3)
    assert [g for _, g in arm.sent_joints] == [C.GRIP_CLOSED_MM] * 3


def test_lead_clamp_when_the_arm_does_not_follow():
    arm = FakeArm(frozen=True)                              # blocked arm: measured joints never move
    run_episode(ActionStream(model(J1_FAR)), arm, CAMS, Keys(), 30)
    assert np.abs(sent(arm) - READY).max() == pytest.approx(C.JOINT_LEAD_MAX_DEG)


def test_send_joints_clamps_to_joint_limits():
    calls = []
    arm = Piper()
    arm._p = SimpleNamespace(MotionCtrl_2=lambda *a: None, JointCtrl=lambda *j: calls.append(j),
                             GripperCtrl=lambda *a: None)
    arm.send_joints([200, -5, 5, 0, 0, 150], 70)
    assert calls == [(150000, 0, 0, 0, 0, 120000)]           # factory limits incl. J6 +-120 deg


def test_tool_tip_floor_holds_then_recovers():
    assert tip_position(BELOW_FLOOR)[2] < 0
    arm = FakeArm()
    run_episode(ActionStream(model(BELOW_FLOOR)), arm, CAMS, Keys(), 60)
    tips = np.array([tip_position(j)[2] for j in sent(arm)])
    assert tips.min() >= C.TIP_Z_FLOOR_MM - 1e-6
    assert tips[-1] < 70                                    # it did descend to the floor band and stopped there
    assert np.allclose(sent(arm)[-1], sent(arm)[-10])       # holding, not creeping
    rows = list(csv.DictReader(open(glob.glob(f"{C.LOG_DIR}/*.csv")[0])))
    assert "floor" in rows[-1]["flags"]
    run_episode(ActionStream(model(READY)), arm, CAMS, Keys(), 40)   # the policy goes back up: followed
    assert tip_position(sent(arm)[-1])[2] > 100


def test_arm_fault_stops_streaming_and_rests():
    arm = Arm({21: lambda: setattr(arm, "status", "COLLISION_OCCURRED(0x7)")})   # real SDK string format
    deploy(ActionStream(model(J1_FAR)), arm, CAMS, Keys(), 200)
    assert len(arm.sent_joints) == 30                       # seen at the next 1 Hz check, before sending
    assert arm.rested


def test_j6_temperature_pause_and_resume():
    arm = Arm({31: lambda: setattr(arm, "temp", C.J6_TEMP_RESUME_C)}, temp=C.J6_TEMP_PAUSE_C)
    assert run_episode(ActionStream(model(J1_FAR)), arm, CAMS, Keys(), 60)
    js = sent(arm)
    assert np.allclose(js[:45], READY)                      # hot from the start: holds READY until the check at 45
    assert js[-1][0] > READY[0] + 20                        # resumed once cooled


def test_keys_pause_resume_quit_and_enter():
    keys = Keys()
    arm = Arm({11: lambda: keys.key("p"), 21: lambda: keys.key("r"), 41: lambda: keys.key("q")})
    assert not run_episode(ActionStream(model(J1_FAR)), arm, CAMS, keys, 200)
    js = sent(arm)
    assert len(js) == 41                                    # q ends the episode at the end of that tick
    assert np.allclose(js[9:20], js[9]) and js[20][0] > js[19][0]
    keys.key("")
    assert keys.enter.is_set()


def test_max_steps_episodes_and_rest():
    keys = Keys()
    keys.enter.set()                                        # a stale Enter (typed during the model load) must NOT count
    t0, ep1 = time.monotonic(), []
    arm = Arm({21: lambda: ep1.append(time.monotonic() - t0)})   # first tick of episode 2
    threading.Timer(0.5, keys.enter.set).start()            # the real Enter, typed after the prompt
    stream = ActionStream(model(J1_FAR))
    deploy(stream, arm, CAMS, keys, 20, episodes=2)
    assert ep1 and ep1[0] >= 0.5                            # episode 2 waited for the real Enter
    assert len(arm.sent_joints) == 40 and arm.rested
    assert not stream._thread.is_alive()                    # worker closed: no CUDA thread alive at exit
    assert np.allclose(arm.sent_joints[20][0], READY + [10, 0, 0, 0, 0, 0])   # episode 2 restarted from READY
    assert len(glob.glob(f"{C.LOG_DIR}/*_ep0.csv")) == 1 and len(glob.glob(f"{C.LOG_DIR}/*_ep1.csv")) == 1


def test_no_burst_after_a_stall(monkeypatch):
    """The first chunk stalls the loop ~0.3 s; the following ticks must still be one period apart, not a burst."""
    monkeypatch.setattr(time, "sleep", REAL_SLEEP)
    arm = FakeArm()
    stamps = []
    arm.send_joints = lambda j, g, speed=C.SPEED_JOINT: (stamps.append(time.monotonic()), FakeArm.send_joints(arm, j, g))
    run_episode(ActionStream(model(J1_FAR, delay=0.3)), arm, CAMS, Keys(), 8)
    periods = np.diff(stamps)
    assert periods.min() >= 0.9 / C.FPS
    assert abs(periods[1:].mean() - 1 / C.FPS) < 0.002                 # phase-locked at FPS once running


def test_shadow_never_sends_and_logs_every_tick():
    arm = FakeArm()
    deploy(ActionStream(model(J1_FAR)), arm, CAMS, Keys(), 20, shadow=True)
    assert arm.sent_joints == [] and not arm.rested and arm._p is None   # disconnected, never rested (no motion)
    rows = list(csv.DictReader(open(glob.glob(f"{C.LOG_DIR}/*.csv")[0])))
    assert len(rows) == 20
    assert {"t", "state_j1", "action_gripper_mm", "sent_j6", "sent_gripper_mm", "chunk_age_s", "model_ms",
            "work_ms", "flags"} <= set(rows[0])
    assert float(rows[7]["sent_j1"]) == pytest.approx(READY[0] + C.MAX_JOINT_STEP_DEG)   # one step from the real state
    assert float(rows[7]["action_j1"]) == pytest.approx(J1_FAR[0])


def test_tool_tip_floor_sees_the_joint_limit_clip():
    """The floor check must run on the joints send_joints actually sends (after the JOINT_LIMITS clip)."""
    from piper.transforms import tip_position
    arm = FakeArm()
    run_episode(ActionStream(model(np.array([0, 160, -45, 0, 90, -119.0]))), arm, CAMS, Keys(), 30)
    js = sent(arm)
    assert np.abs(js[:, 4]).max() <= 70 + 1e-9
    assert min(tip_position(j)[2] for j in js) >= C.TIP_Z_FLOOR_MM - 1e-6


def test_blending_removes_the_chunk_boundary_jump():
    """A fresh chunk starts where the MODEL thinks the arm is, not where the last action put it. Measured on a
    live run: action step 10.6 deg mean / 45.9 max AT a boundary against 1.08 elsewhere - that is the jitter
    felt on the arm. Blending carries the discontinuity as a decaying offset so the command stays continuous."""
    import numpy as np

    def max_step(blend):
        c = [0]

        def predict(top, wrist, st):
            REAL_SLEEP(0.02)      # an autouse fixture no-ops time.sleep here; the worker must really run
            base = c[0] * 30.0                     # every chunk starts 30 deg from the last one's end
            c[0] += 1
            return np.tile(np.arange(50, dtype=np.float32)[:, None], (1, 7)) + base

        st = ActionStream(predict, n_action_steps=10, blend_ticks=blend)
        try:
            acts = []
            for i in range(40):
                acts.append(float(st.next((None, None, np.zeros(7, np.float32)), i * 0.01)[0][0]))
                REAL_SLEEP(0.01)
            assert c[0] >= 3, f"only {c[0]} chunks: the fixture never crossed a boundary"
        finally:
            st.close()
        return float(np.abs(np.diff(acts)).max())

    unblended, blended = max_step(0), max_step(6)
    assert unblended > 10.0, f"fixture produced no boundary jump ({unblended:.1f})"
    assert blended < unblended / 2, f"blending did not smooth it: {blended:.1f} vs {unblended:.1f}"
