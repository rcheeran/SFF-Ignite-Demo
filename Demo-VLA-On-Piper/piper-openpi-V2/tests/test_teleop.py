"""Quest teleop control loop without hardware or headset: the 30 Hz tick is driven by hand with a fake clock
(deterministic, no sleeps) against piper.fake_arm.FakeArm; frozen=True mimics the firmware IK refusing targets.

    python -m pytest tests/test_teleop.py -q
"""
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from piper import config as C
from piper.fake_arm import FakeArm, fk_pose
from piper.robot import pose_to_xyz_rot
from teleop.quest import QuestTeleop, replay, status_line

DT = 1.0 / C.CONTROL_HZ
HEAD = {"p": [0.0, 1.6, 0.0], "q": [0.0, 0.0, 0.0, 1.0]}
P0 = [0.2, 1.2, -0.4]


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def packet(p=P0, q=(0.0, 0.0, 0.0, 1.0), squeeze=0.0, trigger=0.0, a=False, b=False, thumb=False, head=HEAD):
    right = {"p": None if p is None else list(p), "q": list(q), "trigger": trigger, "squeeze": squeeze,
             "a": a, "b": b, "thumb": thumb, "axes": [0.0, 0.0]}
    return {"t": 0.0, "head": head, "right": right}


def make(arm=None, **kw):
    t = QuestTeleop(arm, **kw)
    t._clock = Clock()
    t._goto_ready()
    t.feed({"event": "session_start", "mode": "test"}, "test")
    return t


def run(t, secs, pkt=None, fn=None):
    """Tick at CONTROL_HZ for secs, feeding pkt (or fn(elapsed)) before every tick; None = no packet."""
    for i in range(int(round(secs * C.CONTROL_HZ))):
        t._clock.t += DT
        p = fn((i + 1) * DT) if fn else pkt
        if p is not None:
            t.feed(p, "test")
        t._tick()
    return t.snapshot()


def events(t):
    out = []
    while not t.events.empty():
        out.append(t.events.get())
    return out


def test_hold_a_arms_short_a_only_while_armed_b_thumb_edges():
    t = make()
    s = run(t, 0.5, packet())
    assert not s["armed"] and s["status"] == "DISARMED: hold A 3 s" and s["tracked"]
    run(t, 0.3, packet(a=True)), run(t, 0.3, packet())            # short A while disarmed: nothing
    assert not [e for e in events(t) if e.startswith("a_")] and not t.armed
    s = run(t, 2.9, packet(a=True))
    assert not s["armed"] and 0.9 < s["arm_progress"] < 1.0
    s = run(t, 0.2, packet(a=True))                               # 3 s: armed, once
    assert s["armed"] and s["status"] == "ARMED: squeeze to move" and s["arm_progress"] == 1.0
    run(t, 1.0, packet(a=True))                                   # keep holding: no second toggle
    s = run(t, 0.2, packet())
    assert t.armed and events(t).count("armed") == 1 and s["arm_progress"] == 0.0
    run(t, 0.3, packet(a=True)), run(t, 0.2, packet())            # tap A while armed: discard event
    assert events(t) == ["a_tap"]
    run(t, 1.5, packet(a=True)), run(t, 0.2, packet())            # any A press under 3 s is a tap
    assert events(t) == ["a_tap"]
    run(t, 0.3, packet(b=True)), run(t, 0.1, packet())            # tap B: stop + save event
    assert events(t) == ["b_tap"]
    s = run(t, 2.0, packet(b=True))                               # holding B: progress, no event yet
    assert events(t) == [] and 0.6 < s["b_progress"] < 0.7
    run(t, 1.2, packet(b=True))                                   # 3 s: start-episode event, once
    assert events(t) == ["b_hold"]
    run(t, 1.0, packet(b=True)), run(t, 0.1, packet())            # keep holding, release: nothing more
    assert events(t) == []
    run(t, 0.1, packet(thumb=True)), run(t, 0.1, packet())        # thumbstick: READY
    assert events(t) == ["thumb", "ready"]
    run(t, 3.1, packet(a=True)), run(t, 0.1, packet())            # hold again: disarmed
    assert not t.armed and "disarmed" in events(t)
    run(t, 0.2, packet(b=True, thumb=True)), run(t, 0.1, packet())
    assert events(t) == []                                        # B / thumb do nothing while disarmed


def test_watchdog_holds_target_and_restarts_the_a_hold():
    arm = FakeArm()
    t = make(arm, armed=True)
    fwd = lambda s: packet(p=[0.2, 1.2, -0.4 - 0.02 * s], squeeze=1.0)       # 20 mm/s forward
    s = run(t, 1.0, fn=fwd)
    assert s["engaged"] and s["tracked"] and t.target_pose[0] > fk_pose(C.READY_JOINTS_DEG)[0] + 15
    x = t.target_pose[0]
    s = run(t, 0.6)                                                           # packets stop
    assert not s["engaged"] and not s["tracked"] and s["status"] == "hold: no controller packets"
    assert t.target_pose[0] == x and "released" in events(t)                  # held, not integrated
    s = run(t, 0.1, packet(p=[0.2, 1.2, -0.7], squeeze=1.0))                  # resume, squeeze still held
    assert s["engaged"] and abs(t.target_pose[0] - x) < 1e-9                  # fresh reference: no jump
    s = run(t, 0.2, packet(p=None))                                           # controller lost tracking
    assert s["status"] == "hold: controller not tracked" and not s["tracked"] and not s["engaged"]
    run(t, 2.0, packet(a=True)), run(t, 0.6)                                  # A-hold cut by a packet gap
    s = run(t, 1.5, packet(a=True))
    assert t.armed and abs(s["arm_progress"] - 0.5) < 0.05                    # started over, no toggle


def test_j6_temperature_pause_and_resume():
    arm = FakeArm(temp=32)
    t = make(arm, armed=True)
    s = run(t, 1.1, packet(squeeze=1.0))
    assert s["engaged"] and s["j6_c"] == 32
    arm.temp = C.J6_TEMP_PAUSE_C
    s = run(t, 1.1, packet(squeeze=1.0))                                      # 1 Hz poll -> pause, clutch off
    assert not s["engaged"] and s["status"] == f"PAUSED: J6 {C.J6_TEMP_PAUSE_C} C" and "j6_pause" in events(t)
    arm.temp = C.J6_TEMP_PAUSE_C - 1                                          # hysteresis band: still paused
    s = run(t, 1.1, packet(squeeze=1.0))
    assert not s["engaged"] and s["status"].startswith("PAUSED")
    arm.temp = C.J6_TEMP_RESUME_C
    s = run(t, 1.1, packet(squeeze=1.0))                                      # cooled: the held squeeze re-engages
    assert s["engaged"] and "j6_resume" in events(t)


def test_no_feedback_holds_and_streams_nothing():
    arm = FakeArm()
    t = make(arm, armed=True)
    run(t, 0.5, packet(squeeze=1.0))
    assert t.engaged
    good = arm._pose
    arm._pose = np.zeros(6)                                                   # arm off / CAN down
    n = len(arm.sent_poses)
    s = run(t, 0.5, packet(squeeze=1.0))
    assert s["no_feedback"] and not s["engaged"] and s["status"].startswith("NO ARM FEEDBACK")
    assert len(arm.sent_poses) == n
    arm._pose = good
    s = run(t, 0.2, packet(squeeze=1.0))
    assert not s["no_feedback"] and s["engaged"] and len(arm.sent_poses) > n


def test_lead_clamp_then_stuck_guard_then_resqueeze():
    """Frozen arm (IK refusing): the lead clamp caps the runaway, STUCK_S later the clutch drops and the target
    snaps back to the real pose; a held squeeze must not re-engage, a fresh one must."""
    arm = FakeArm(frozen=True)
    t = make(arm, armed=True)
    push = lambda s: packet(p=[0.2, 1.2, -0.4 - min(0.2, 0.2 * s)], squeeze=1.0)   # 0.2 m forward in 1 s
    s = run(t, 1.0, fn=push)
    assert s["engaged"] and s["lead_limited"] and not s["stuck"]
    assert s["err_mm"] <= C.LEAD_MAX_MM + 0.1 and s["err_mm"] > C.LEAD_MAX_MM - 1
    s = run(t, C.STUCK_S + 0.2, fn=lambda s: push(1.0))
    assert s["stuck"] and not s["engaged"] and s["status"] == "IK REFUSED - release and re-squeeze"
    assert s["err_mm"] == 0.0 and events(t).count("stuck") == 1                 # snapped back to the real pose
    s = run(t, 0.3, push(1.0))                                                # squeeze still held: stays off
    assert not s["engaged"] and s["stuck"]
    run(t, 0.2, packet(p=push(1.0)["right"]["p"]))                            # release
    s = run(t, 0.2, push(1.0))                                                # fresh squeeze
    assert s["engaged"] and not s["stuck"] and s["err_mm"] == 0.0
    assert s["j6_margin_deg"] == C.JOINT_LIMITS_DEG[5][1] - C.READY_JOINTS_DEG[5] and s["no_feedback"] is False
    arm2 = FakeArm()                                                          # a following arm never trips it
    t2 = make(arm2, armed=True)
    s = run(t2, 3.0, fn=lambda s: packet(p=[0.2, 1.2, -0.4 - 0.05 * s], squeeze=1.0))
    assert s["engaged"] and not s["stuck"] and not s["lead_limited"]
    assert t2.target_pose[0] > fk_pose(C.READY_JOINTS_DEG)[0] + 140


def test_yaw_rezero_and_scale():
    head_rot = Rot.from_euler("y", 30.0, degrees=True)                        # operator looking 30 deg left
    head = {"p": [0.0, 1.6, 0.0], "q": head_rot.as_quat().tolist()}
    q = head_rot.as_quat().tolist()
    t = make(scale=0.1, armed=True)
    p = lambda s: (head_rot.apply([0.0, 0.0, -0.5 * s]) + P0).tolist()       # 0.5 m/s along the head's forward
    p0 = t.target_pose.copy()
    run(t, 1.0, fn=lambda s: packet(p=p(s), q=q, squeeze=1.0, head=head))
    d = t.target_pose - p0
    assert "yaw_zeroed:30" in events(t)
    assert abs(d[0] - 0.1 * 500 * (1 - DT)) < 0.5 and abs(d[1]) < 0.5 and abs(d[2]) < 0.5   # robot +x only
    assert np.abs(d[3:]).max() < 1e-6


def test_rot_scale():
    for rot_scale, eff in ((0.0, 0.0), (None, 0.1), (1.0, 1.0)):             # default rot_scale = scale
        t = make(scale=0.1, armed=True, rot_scale=rot_scale)
        r0 = t.target_pose.copy()
        turn = lambda s: Rot.from_euler("y", 20.0 * min(1.0, s), degrees=True).as_quat().tolist()   # 20 deg about vertical
        run(t, 1.5, fn=lambda s: packet(q=turn(s), squeeze=1.0))
        pivot = np.array([0.0, 0.0, -C.TELEOP_PIVOT_MM])                         # rotation about the wrist centre:
        x0, q0 = pose_to_xyz_rot(r0)                                              # that point must not move
        x1, q1 = pose_to_xyz_rot(t.target_pose)
        d = np.concatenate([(x1 + q1.apply(pivot)) - (x0 + q0.apply(pivot)), t.target_pose[3:] - r0[3:]])
        d[3:] = (d[3:] + 180.0) % 360.0 - 180.0                                   # Euler deltas wrap at +-180 at READY
        assert abs(d[5] - eff * 20.0 * (1 - DT)) < 0.2 and np.abs(d[:5]).max() < 0.2, rot_scale


def test_session_end_disarms():
    t = make(armed=True)
    s = run(t, 0.3, packet(squeeze=1.0))
    assert s["armed"] and s["engaged"]
    t.feed({"event": "session_end"}, "test")
    s = run(t, 0.1)                                                           # the page stops sending frames
    assert not s["armed"] and not s["engaged"] and s["status"] == "DISARMED: hold A 3 s"
    ev = events(t)
    assert "disarmed" in ev and "released" in ev
    s = run(t, 0.5)                                                           # ... and the watchdog takes over
    assert s["status"] == "hold: no controller packets"
    s = run(t, 3.5)                                                           # nothing can arm without frames
    assert not s["armed"]


def test_firmware_verdict_in_ack_and_collision_declutch():
    arm = FakeArm()
    t = make(arm, armed=True)
    s = run(t, 1.1, packet(squeeze=1.0))
    assert s["arm_status"] == "NORMAL" and s["engaged"] and " fw=" not in status_line(t)
    arm.status = "NO_SOLUTION(0x2)"                                           # str(piper_sdk ArmStatus)
    s = run(t, 1.1, packet(squeeze=1.0))
    assert s["arm_status"] == "NO_SOLUTION" and s["engaged"] and " fw=NO_SOLUTION" in status_line(t)
    arm.status = "COLLISION_OCCURRED(0x7)"
    s = run(t, 1.1, packet(p=[0.2, 1.2, -0.6], squeeze=1.0))
    assert not s["engaged"] and s["stuck"] and s["status"] == "COLLISION_OCCURRED - release and re-squeeze"
    assert s["err_mm"] == 0.0 and events(t).count("stuck") == 1                 # target tracks the real pose
    arm.status = "ArmStatus.NORMAL"
    s = run(t, 1.1, packet(p=[0.2, 1.2, -0.6], squeeze=1.0))                  # cleared, squeeze held: stays off
    assert s["arm_status"] == "NORMAL" and not s["engaged"]
    run(t, 0.1, packet(p=[0.2, 1.2, -0.6]))
    s = run(t, 0.1, packet(p=[0.2, 1.2, -0.6], squeeze=1.0))
    assert s["engaged"] and not s["stuck"]


def test_malformed_packets_are_dropped_not_fatal():
    t = make(armed=True)
    bads = ({"right": {"p": [1, 2, 3]}},                                      # no quaternion
            {"right": {"p": [0.1, 0.2], "q": [0, 0, 0, 1]}},                  # wrong shape
            {"right": {"p": [float("nan"), 0, 0], "q": [0, 0, 0, 1]}},
            {"right": {"p": [0, 0, 0], "q": [0, 0, 0, 0]}},                   # zero quaternion
            {"right": "nope"}, {"head": 5, "right": None})
    for bad in bads:
        t.feed(bad, "test")
        s = run(t, 0.1)
        assert t.fault is None and not s["tracked"] and s["status"].startswith("hold:"), bad
    assert sum(e.startswith("bad packet dropped") for e in events(t)) == len(bads)
    s = run(t, 0.1, packet())
    assert s["tracked"]


def test_gripper_follows_trigger_only_while_armed():
    arm = FakeArm()
    t = make(arm, armed=True)
    s = run(t, 0.2, packet(trigger=0.8))                                      # not clutched: trigger still works
    assert abs(s["grip_cmd_mm"] - (C.GRIP_OPEN_MM - 0.8 * (C.GRIP_OPEN_MM - C.GRIP_CLOSED_MM))) < 0.1
    assert arm.sent_poses[-1][2] == t.grip_cmd_mm                             # re-sent every tick
    s = run(t, 0.2, packet(trigger=C.TRIGGER_DEADZONE / 2))
    assert s["grip_cmd_mm"] == C.GRIP_OPEN_MM
    run(t, 0.2, packet(trigger=1.0))
    assert t.grip_cmd_mm == C.GRIP_CLOSED_MM
    t.feed({"event": "session_end"}, "test")
    run(t, 0.1)
    s = run(t, 0.2, packet(trigger=0.0))                                      # disarmed: holds, does not open
    assert not s["armed"] and s["grip_cmd_mm"] == C.GRIP_CLOSED_MM and arm.sent_poses[-1][2] == C.GRIP_CLOSED_MM


def test_thumb_ready_restarts_target_with_ready_rotation():
    arm = FakeArm()
    t = make(arm, armed=True)
    ready = fk_pose(C.READY_JOINTS_DEG)
    run(t, 0.5, fn=lambda s: packet(p=[0.2, 1.2, -0.4 - 0.1 * s], squeeze=1.0))
    assert t.target_pose[0] > ready[0] + 40
    run(t, 0.1, packet(thumb=True))
    assert "ready" in events(t) and not t.engaged and np.allclose(arm.joints_deg(), C.READY_JOINTS_DEG)
    assert np.allclose(t.target_pose[:3], ready[:3]) and np.allclose(arm.sent_poses[-1][0], ready[:3])
    assert np.allclose(arm.sent_poses[-1][1], ready[3:])                      # READY's rotation, not the stale one


def test_control_loop_fault_stops_streaming_and_is_reported():
    class Broken(FakeArm):
        def send_pose(self, *a, **k):
            raise RuntimeError("CAN gone")

    t = make(Broken(), armed=True)
    t._loop()                                                                 # first tick raises, loop returns
    assert t.fault == "RuntimeError: CAN gone" and t.status == "CONTROL FAULT: RuntimeError: CAN gone"
    assert not t.engaged and any(e.startswith("fault:") for e in events(t))


def test_replay_self_test():
    assert replay(secs=6.0)


def test_start_aborts_when_ready_is_not_reached():
    """The live session streamed from [2, 49.5, -34.8, 48, 34, -4.8] after ready() timed out: never again."""
    arm = FakeArm()
    arm.ready = lambda grip=None: False
    with pytest.raises(SystemExit, match="READY not reached"):
        make(arm=arm, armed=True)


def test_a_cut_joint_declutches_and_is_named():
    arm = FakeArm()
    t = make(arm=arm, armed=True)
    s = run(t, 0.3, packet(squeeze=1.0))
    assert s["engaged"]
    arm.enabled = lambda: [True] * 5 + [False]                                 # the driver cut J6 (limp)
    s = run(t, 1.2, packet(squeeze=1.0))                                        # the 1 Hz poll sees it
    assert not s["engaged"] and s["disabled"] == ["J6"] and "J6 NOT ENABLED" in s["status"]
    assert "DISABLED=J6" in status_line(t)


def test_snapshot_is_json_serializable_even_with_numpy_values():
    """A numpy scalar in snapshot() 500s BOTH the headset feed and the recorder's /status, for good:
    the arm goes dead mid-session and the headset shows nothing. It happened - lead_limited came out
    of `np.linalg.norm(..) > x or ..` as np.bool_, which numpy 2 names 'bool', so the traceback said
    "Object of type bool is not JSON serializable" and read like a nonsense error."""
    t = make(arm=FakeArm(), armed=True)
    t.lead_limited, t.telemetry = np.bool_(True), {"n": np.float64(3.25), "v": np.array([1.5, 2.5])}
    s = t.snapshot()
    json.dumps(s)                                    # would raise TypeError before _plain()
    assert s["lead_limited"] is True and s["n"] == 3.25 and s["v"] == [1.5, 2.5]
