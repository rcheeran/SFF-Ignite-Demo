"""piper.robot / piper.transforms without hardware: clamps, FK helpers, the lock file, fault parsing, the CAN
helpers, and the exact piper_sdk call sequence of one control tick and of rest() (recording SDK stand-ins)."""
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from piper import config as C
from piper import robot
from piper.robot import Piper, acquire_lock, can_link, can_receivers, clamp_joints, clamp_pose
from piper.transforms import (clamp_rot_step, euler_to_rot, fk_flange, rot_to_6d, rot_to_euler, sixd_to_rot,
                              tip_position)

FWD = euler_to_rot([0, 85, 0])        # tool z nearly horizontal (flange at zero joints)
DOWN = euler_to_rot([180, 0, 0])      # tool z pointing straight down


# ---- clamp_pose ---------------------------------------------------------------
def test_clamp_pose_box_and_step():
    assert C.WORKSPACE_MIN_MM == [30.0, -400.0, 40.0]
    assert C.WORKSPACE_MAX_MM == [500.0, 400.0, 480.0]
    prev = np.array([200.0, 0.0, 200.0])
    xyz, _, lim = clamp_pose(prev, FWD, [300, 0, 200], FWD)
    assert np.allclose(xyz, [215, 0, 200]) and not lim                  # <= STEP_MAX_MM per tick
    xyz, _, _ = clamp_pose([40, 0, 50], FWD, [0, 0, 0], FWD)
    assert np.allclose(xyz, [C.WORKSPACE_MIN_MM[0], 0, C.WORKSPACE_MIN_MM[2]])   # workspace box


def test_clamp_pose_tip_floor_pointing_down():
    prev = np.array([200.0, 0.0, 100.0])
    xyz, r, _ = clamp_pose(prev, DOWN, prev, DOWN)
    assert np.isclose(xyz[2], C.TIP_Z_FLOOR_MM + C.TOOL_LEN_MM)          # lifted: tip sits on the floor
    assert np.isclose(xyz[2] + r.as_matrix()[2, 2] * C.TOOL_LEN_MM, C.TIP_Z_FLOOR_MM)
    horiz = euler_to_rot([0, 90, 0])
    xyz, _, _ = clamp_pose(prev, horiz, prev, horiz)
    assert np.isclose(xyz[2], 100)                                        # tool horizontal: flange may be low


def test_clamp_pose_lead_translation():
    meas = np.array([200.0, 0.0, 200.0])
    tgt = meas + [50, 0, 0]
    xyz, _, lim = clamp_pose(tgt, FWD, tgt, FWD, meas, FWD)               # prev == target: no step clamp
    assert lim and np.allclose(xyz, meas + [C.LEAD_MAX_MM, 0, 0])
    xyz, _, lim = clamp_pose(meas, FWD, meas + [10, 0, 0], FWD, meas, FWD)
    assert not lim and np.allclose(xyz, meas + [10, 0, 0])


def test_clamp_pose_lead_rotation():
    meas = np.array([200.0, 0.0, 200.0])
    tgt = Rot.from_euler("z", 35, degrees=True) * FWD
    _, r, lim = clamp_pose(meas, tgt, meas, tgt, meas, FWD)
    assert lim and np.isclose(np.degrees((FWD.inv() * r).magnitude()), C.LEAD_MAX_DEG)
    small = Rot.from_euler("z", 5, degrees=True) * FWD
    _, r, lim = clamp_pose(meas, FWD, meas, small, meas, FWD)
    assert not lim and np.isclose(np.degrees((FWD.inv() * r).magnitude()), 5)


# ---- clamp_joints (JointCtrl streaming: deploy + LeRobot plugin) ------------------
LOW = [0, 85, -30, 0, 45, -119]          # tool down, tip 19.5 mm above the table
LOWER = [0, 90, -30, 0, 45, -119]        # tip 4.8 mm: below TIP_Z_FLOOR_MM
FAR = [60.0, 120.0, -90.0, 40.0, 60.0, 0.0]


def test_clamp_joints_step_lead_and_floor():
    ready = np.array(C.READY_JOINTS_DEG)
    j, floor = clamp_joints(ready, ready, FAR)
    assert not floor and np.allclose(j, ready + [10, 10, -10, 10, 10, 0])        # <= MAX_JOINT_STEP_DEG per tick
    prev = ready
    for _ in range(10):                                                            # arm blocked: the lead saturates
        prev, _ = clamp_joints(prev, ready, FAR)
    assert np.abs(prev - ready).max() == C.JOINT_LEAD_MAX_DEG
    assert tip_position(LOW)[2] > C.TIP_Z_FLOOR_MM > tip_position(LOWER)[2]
    j, floor = clamp_joints(LOW, LOW, LOWER)
    assert floor and np.allclose(j, LOW)                                           # descending below the floor: hold
    j, floor = clamp_joints(LOWER, LOWER, LOW)
    assert not floor and np.allclose(j, LOW)                                       # climbing out of it is allowed


def test_lerobot_plugin_clamps_actions_like_deploy():
    from piper.fake_arm import FakeArm
    from piper.lerobot_robot import PiperRobot, PiperRobotConfig
    plugin = PiperRobot(PiperRobotConfig(fake_cameras=True))
    ready = np.array(C.READY_JOINTS_DEG)
    far = dict(zip(C.STATE_NAMES, [*FAR, 30.0]))
    plugin.arm = FakeArm(joints=ready)                  # parked at READY; first replay frame far away
    out = plugin.send_action(far)
    sent, grip = plugin.arm.sent_joints[-1]
    assert np.abs(sent - ready).max() <= C.MAX_JOINT_STEP_DEG and grip == 30.0
    assert out == {**dict(zip(C.STATE_NAMES[:6], sent.tolist())), "gripper_mm": 30.0}
    plugin.arm, plugin._prev = FakeArm(joints=ready, frozen=True), None   # blocked arm: never follows
    for _ in range(10):
        plugin.send_action(far)
    assert np.abs(plugin.arm.sent_joints[-1][0] - ready).max() <= C.JOINT_LEAD_MAX_DEG
    plugin.arm, plugin._prev = FakeArm(joints=LOW), None
    plugin.send_action(dict(zip(C.STATE_NAMES, [*LOWER, 30.0])))
    assert np.allclose(plugin.arm.sent_joints[-1][0], LOW)               # tool-tip floor: holds the last command


# ---- transforms ---------------------------------------------------------------
def test_euler_and_6d_round_trip():
    for rpy in ([0, 85, 0], [-143.7, -18.2, -67.0], [30, -40, 120]):
        r = euler_to_rot(rpy)
        assert np.allclose(rot_to_euler(r), rpy, atol=1e-6)
        assert np.isclose((r.inv() * sixd_to_rot(rot_to_6d(r))).magnitude(), 0, atol=1e-5)


def test_clamp_rot_step_geodesic():
    rv = np.array([10.0, 10.0, 10.0])                                      # 17.3 deg about a diagonal axis
    out = clamp_rot_step(FWD, FWD * Rot.from_rotvec(rv, degrees=True), C.STEP_MAX_DEG)
    rel = (FWD.inv() * out).as_rotvec(degrees=True)
    assert np.isclose(np.linalg.norm(rel), C.STEP_MAX_DEG)
    assert np.allclose(rel / np.linalg.norm(rel), rv / np.linalg.norm(rv))   # same axis, shortened
    small = FWD * Rot.from_rotvec([3, 0, 0], degrees=True)
    assert np.isclose((clamp_rot_step(FWD, small, C.STEP_MAX_DEG).inv() * small).magnitude(), 0, atol=1e-9)


def test_fk_and_tip_position():
    assert np.allclose(fk_flange(C.HOME_JOINTS_DEG)[:3], [56, 0, 213], atol=0.5)
    assert np.allclose(fk_flange(C.READY_JOINTS_DEG)[:3], [127.7, 0, 231.8], atol=0.5)
    down = [0, 60, -30, 0, 65, -119]                 # 85 + J2 + J3 + J5 = 180: tool z straight down
    assert np.allclose(tip_position(down), fk_flange(down)[:3] - [0, 0, C.TOOL_LEN_MM], atol=0.1)
    tip, fl = tip_position(C.READY_JOINTS_DEG), fk_flange(C.READY_JOINTS_DEG)[:3]
    assert tip[2] < fl[2] and tip[0] > fl[0] and np.isclose(np.linalg.norm(tip - fl), C.TOOL_LEN_MM)


# ---- one control tick against a recording piper_sdk stand-in --------------------
class RecSDK:
    """Records every piper_sdk call as (name, args), in order."""
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *a, **k: self.calls.append((name, a))


class FakeSDK(RecSDK):
    """RecSDK that also answers feedback: joints snap to the last JointCtrl, enable flags follow Enable/DisablePiper."""
    def __init__(self, joints=C.READY_JOINTS_DEG, enabled=True):
        super().__init__()
        self.j, self.en, self.frozen = np.array(joints, float) * 1000, [enabled] * 6, False
        self.lim = {i: (int(f[0] * 10), int(f[1] * 10)) for i, f in enumerate(C.JOINT_LIMITS_DEG, 1)}
        self.lim[6] = (-1790, -600)

    def JointCtrl(self, *a):
        self.calls.append(("JointCtrl", a))
        if not self.frozen:
            keep = ~np.array(self.en, bool)                # a de-torqued (limp) joint does not follow
            self.j = np.where(keep, self.j, np.array(a, float))

    def EnableArm(self, motor_num=7, enable_flag=0x02):
        self.calls.append(("EnableArm", (motor_num,)))
        for i in (range(6) if motor_num == 7 else [motor_num - 1]):
            self.en[i] = True

    def DisableArm(self, motor_num=7, enable_flag=0x01):
        self.calls.append(("DisableArm", (motor_num,)))
        for i in (range(6) if motor_num == 7 else [motor_num - 1]):
            self.en[i] = False

    def JointConfig(self, **kw):
        self.calls.append(("JointConfig", kw))

    def EnablePiper(self):
        self.calls.append(("EnablePiper", ()))
        self.en = [True] * 6

    def DisablePiper(self):
        self.calls.append(("DisablePiper", ()))
        self.en = [False] * 6

    def GetArmEnableStatus(self):
        return list(self.en)

    # stored motor limits: factory everywhere except the old custom J6 [-179, -60] a real arm may still hold
    def MotorAngleLimitMaxSpdSet(self, motor_num, max_angle_limit, min_angle_limit, max_joint_spd):
        self.calls.append(("MotorAngleLimitMaxSpdSet", (motor_num, max_angle_limit, min_angle_limit, max_joint_spd)))
        self.lim[motor_num] = (min_angle_limit, max_angle_limit)

    def GetAllMotorAngleLimitMaxSpd(self):
        motors = [SimpleNamespace(motor_num=0)] + [
            SimpleNamespace(motor_num=i, min_angle_limit=self.lim[i][0] & 0xFFFF, max_angle_limit=self.lim[i][1] & 0xFFFF,
                            max_joint_spd=3000) for i in range(1, 7)]          # uint16 on the wire, like the SDK
        return SimpleNamespace(all_motor_angle_limit_max_spd=SimpleNamespace(motor=motors))

    def GetArmJointMsgs(self):
        return SimpleNamespace(Hz=200.0, joint_state=SimpleNamespace(**{f"joint_{i + 1}": int(v) for i, v in enumerate(self.j)}))

    def GetArmGripperMsgs(self):
        return SimpleNamespace(gripper_state=SimpleNamespace(grippers_angle=50000))

    def GetArmEndPoseMsgs(self):
        return SimpleNamespace(end_pose=SimpleNamespace(X_axis=0, Y_axis=0, Z_axis=0, RX_axis=0, RY_axis=0, RZ_axis=0))

    def GetArmStatus(self):
        return SimpleNamespace(arm_status=SimpleNamespace(
            ctrl_mode="CtrlMode.CAN_CTRL", arm_status=SimpleNamespace(name="NORMAL"),
            err_status=SimpleNamespace(**{f"joint_{i}_angle_limit": False for i in range(1, 7)},
                                       **{f"communication_status_joint_{i}": False for i in range(1, 7)})))

    def GetArmLowSpdInfoMsgs(self):
        return SimpleNamespace(motor_6=SimpleNamespace(motor_temp=31))


def _arm(sdk=None):
    arm = Piper()
    arm._p = sdk or RecSDK()
    return arm


def test_send_joints_clamps_limits_and_orders_frames():
    arm = _arm()
    arm.send_joints([-200, 200, 10, 0, 90, 0], 80)
    assert [c[0] for c in arm._p.calls] == ["MotionCtrl_2", "JointCtrl", "GripperCtrl"]
    assert arm._p.calls[0][1] == (0x01, 0x01, C.SPEED_JOINT, 0x00)               # CAN mode, MOVE J, position
    assert arm._p.calls[1][1] == (-150000, 180000, 0, 0, 70000, 0)               # clipped to JOINT_LIMITS_DEG
    assert arm._p.calls[2][1] == (70000, C.GRIP_EFFORT, 0x01, 0)                 # gripper clipped, enabled


def test_move_to_zero_uses_mode_ctrl_and_literal_zero_joint_ctrl(monkeypatch):
    sdk = FakeSDK(joints=C.READY_JOINTS_DEG)
    arm = _arm(sdk)
    monkeypatch.setattr(robot.time, "sleep", lambda _: None)

    assert arm.move_to_zero(secs=0.1, settle=0)

    assert sdk.calls[0] == (
        "ModeCtrl",
        (0x01, 0x01, C.SPEED_HOME, 0x00),
    )
    assert sdk.calls[1] == ("JointCtrl", (0, 0, 0, 0, 0, 0))
    assert np.allclose(arm.joints_deg(), np.zeros(6))


def test_send_pose_orders_frames_and_skips_gripper_when_none():
    arm = _arm()
    arm.send_pose([200, 0, 200], [0, 85, 0], None)
    assert [c[0] for c in arm._p.calls] == ["MotionCtrl_2", "EndPoseCtrl"]
    assert arm._p.calls[0][1] == (0x01, 0x00, C.SPEED_TELEOP, 0x00)              # CAN mode, MOVE P
    assert arm._p.calls[1][1] == (200000, 0, 200000, 0, 85000, 0)
    arm.send_pose([200, 0, 200], [0, 85, 0], 35.0)
    assert arm._p.calls[-1] == ("GripperCtrl", (35000, C.GRIP_EFFORT, 0x01, 0))


def test_teach_stop_replays_probe_sequence_before_holding(monkeypatch):
    class TeachSDK(FakeSDK):
        def __init__(self):
            super().__init__()
            self.mode = Piper.TEACH_CTRL_MODE
            self.ended_recording = False
            self.primed_mode = False

        def MotionCtrl_1(self, *args):
            self.calls.append(("MotionCtrl_1", args))
            self.ended_recording = args == (0x00, 0x00, 0x02)

        def ModeCtrl(self, *args):
            self.calls.append(("ModeCtrl", args))
            self.primed_mode = self.ended_recording
            self.mode = Piper.CAN_CTRL_MODE

        def GetArmStatus(self):
            mode = self.mode
            if (
                mode == Piper.CAN_CTRL_MODE
                and not any(call[0] == "MotionCtrl_2" for call in self.calls)
            ):
                self.mode = Piper.TEACH_CTRL_MODE
            return SimpleNamespace(
                arm_status=SimpleNamespace(
                    ctrl_mode=mode,
                    teach_status=Piper.TEACH_STATUS_STOPPED,
                )
            )

        def MotionCtrl_2(self, *args):
            self.calls.append(("MotionCtrl_2", args))
            if self.ended_recording and self.primed_mode:
                self.mode = Piper.CAN_CTRL_MODE

    sdk = TeachSDK()
    arm = _arm(sdk)
    expected_hold = arm.joints_deg().copy()
    original_motion_ctrl = sdk.MotionCtrl_2

    def motion_ctrl_with_stale_feedback(*args):
        mode_before = sdk.mode
        original_motion_ctrl(*args)
        if mode_before == Piper.TEACH_CTRL_MODE and sdk.mode == Piper.CAN_CTRL_MODE:
            sdk.j = np.zeros(6)

    sdk.MotionCtrl_2 = motion_ctrl_with_stale_feedback
    monkeypatch.setattr(robot.time, "sleep", lambda _: None)
    self_advancing_time = iter(np.arange(0.0, 20.0, 0.25))
    monkeypatch.setattr(robot.time, "time", lambda: next(self_advancing_time))

    assert arm.teach_stop(timeout=8.0)
    names = [call[0] for call in sdk.calls]
    assert names.index("MotionCtrl_1") < names.index("ModeCtrl") < names.index(
        "MotionCtrl_2"
    )
    assert names[-2:] == ["MotionCtrl_2", "JointCtrl"]
    joint_commands = [call[1] for call in sdk.calls if call[0] == "JointCtrl"]
    assert joint_commands
    assert all(
        command == tuple(int(value * 1000) for value in expected_hold)
        for command in joint_commands
    )


def test_every_tick_starts_with_motion_ctrl_and_resends_gripper():
    arm = _arm()
    for _ in range(5):
        arm.send_joints(C.READY_JOINTS_DEG, 50)
        arm.send_pose([200, 0, 200], [0, 85, 0], 50)
    names = [c[0] for c in arm._p.calls]
    assert all(names[i - 1] == "MotionCtrl_2" for i, n in enumerate(names) if n in ("JointCtrl", "EndPoseCtrl"))
    assert names.count("GripperCtrl") == 10


def test_rest_homes_rests_then_de_torques(monkeypatch, capsys):
    monkeypatch.setattr(robot.time, "sleep", lambda s, _real=time.sleep: _real(min(s, 0.02)))   # fast ticks
    arm = _arm(FakeSDK())
    sdk = arm._p
    arm.rest()
    names = [c[0] for c in sdk.calls]
    joints = [c[1] for c in sdk.calls if c[0] == "JointCtrl"]
    assert joints[0] == tuple(int(v * 1000) for v in C.HOME_JOINTS_DEG)
    assert joints[-1] == tuple(int(v * 1000) for v in C.REST_JOINTS_DEG)
    assert [c for c in sdk.calls if c[0] == "GripperCtrl"][-1][1][2] == 0x00        # gripper de-torqued last
    assert names.index("DisablePiper") > len(names) - names[::-1].index("JointCtrl")  # joints disabled after REST
    assert sdk.calls[-2] == ("MotionCtrl_2", (0x00, 0x01, 0, 0x00))                   # standby, then ...
    assert sdk.calls[-1][0] == "DisconnectPort"                                          # ... unlock/disconnect
    assert sdk.en == [False] * 6 and arm._p is None
    assert "rest: DONE - de-torqued" in capsys.readouterr().out


def test_rest_refuses_to_de_torque_when_home_not_reached(monkeypatch, capsys):
    monkeypatch.setattr(robot.time, "sleep", lambda s, _real=time.sleep: _real(min(s, 0.02)))   # fast ticks
    monkeypatch.setattr(C, "HOME_TIMEOUT_S", 0.2)
    sdk = FakeSDK()
    sdk.frozen = True                                  # obstacle: the arm never reaches HOME
    arm = _arm(sdk)
    arm.rest()
    assert "DisablePiper" not in [c[0] for c in sdk.calls] and sdk.en == [True] * 6
    assert arm._p is None                              # disconnected + unlocked so tools/rest.py can take over
    assert "rest: FAILED" in capsys.readouterr().out


def test_rest_with_a_limp_wrist_homes_the_rest_and_de_torques_at_home(monkeypatch, capsys):
    """J6 cut by its driver (the live session): HOME is judged on the joints that still hold torque, the arm is
    parked and de-torqued - not de-torqued where it stands."""
    monkeypatch.setattr(robot.time, "sleep", lambda s, _real=time.sleep: _real(min(s, 0.02)))
    sdk = FakeSDK()
    sdk.en[5] = False
    arm = _arm(sdk)
    arm.rest()
    joints = [c[1] for c in sdk.calls if c[0] == "JointCtrl"]
    assert joints[0][:5] == tuple(int(v * 1000) for v in C.HOME_JOINTS_DEG[:5])    # homed first ...
    assert "DisablePiper" in [c[0] for c in sdk.calls] and sdk.en == [False] * 6   # ... then de-torqued at HOME
    out = capsys.readouterr().out
    assert "J6 not enabled" in out and "rest: DONE" in out


def test_rest_refusal_names_the_drop_option(monkeypatch, capsys):
    monkeypatch.setattr(robot.time, "sleep", lambda s, _real=time.sleep: _real(min(s, 0.02)))
    monkeypatch.setattr(C, "HOME_TIMEOUT_S", 0.2)
    sdk = FakeSDK()
    sdk.frozen = True                                  # obstacle: HOME unreachable -> stays torqued, tells how to drop
    arm = _arm(sdk)
    arm.rest()
    assert "DisablePiper" not in [c[0] for c in sdk.calls] and sdk.en == [True] * 6
    assert "--drop" in capsys.readouterr().out


def test_rest_drop_de_torques_where_it_stands(monkeypatch, capsys):
    monkeypatch.setattr(robot.time, "sleep", lambda s, _real=time.sleep: _real(min(s, 0.02)))
    sdk = FakeSDK()
    arm = _arm(sdk)
    arm.rest(drop=True)
    assert "JointCtrl" not in [c[0] for c in sdk.calls] and sdk.en == [False] * 6
    assert "--drop" in capsys.readouterr().out


def test_zero_joint_and_enable_joint_frames():
    arm = _arm(FakeSDK())
    arm.enable_joint(6, False)
    arm.zero_joint(6)
    arm.enable_joint(6, True)
    assert [c for c in arm._p.calls if c[0] != "JointCtrl"] == [
        ("DisableArm", (6,)), ("JointConfig", {"joint_num": 6, "set_zero": 0xAE}), ("EnableArm", (6,))]


# ---- connect() ------------------------------------------------------------------
def test_connect_refuses_when_can_is_down(monkeypatch):
    monkeypatch.setattr(robot, "can_link", lambda iface=None: {"up": False, "state": "STOPPED"})
    with pytest.raises(RuntimeError, match="down .STOPPED. - run `bash tools/can_up.sh`"):
        Piper().connect(enable=False)


def test_connect_times_out_without_feedback(monkeypatch):
    import piper_sdk

    class DeadSDK(RecSDK):
        def GetArmJointMsgs(self):
            return SimpleNamespace(Hz=0.0)

    monkeypatch.setattr(piper_sdk, "C_PiperInterface_V2", lambda iface: DeadSDK())
    monkeypatch.setattr(robot, "can_link", lambda iface=None: {"up": True, "state": "ERROR-PASSIVE"})
    monkeypatch.setattr(C, "FEEDBACK_WAIT_S", 0.3)
    arm, t0 = Piper(), time.time()
    with pytest.raises(RuntimeError, match="no feedback on can_piper after 0.3 s .bus ERROR-PASSIVE."):
        arm.connect(enable=False)
    assert 0.25 < time.time() - t0 < 1.5 and arm._p is None


def test_connect_resets_position_mode_before_enable(monkeypatch):
    import piper_sdk
    sdk = FakeSDK(enabled=False)
    monkeypatch.setattr(piper_sdk, "C_PiperInterface_V2", lambda iface: sdk)
    monkeypatch.setattr(robot, "can_link", lambda iface=None: {"up": True, "state": "ERROR-ACTIVE"})
    monkeypatch.setattr(robot.time, "sleep", lambda s: None)
    arm = Piper().connect(enable=True)
    names = [c[0] for c in sdk.calls]
    assert names.index("MotionCtrl_2") < names.index("EnablePiper") < names.index("MotorAngleLimitMaxSpdSet")
    assert sdk.calls[names.index("MotionCtrl_2")][1] == (0x01, 0x01, C.SPEED_HOME, 0x00)   # position mode first
    assert arm.enabled() == [True] * 6
    # the stored J6 limit was the old custom [-179, -60]: connect() writes the factory +-120 back, and only to J6
    restored = [c[1] for c in sdk.calls if c[0] == "MotorAngleLimitMaxSpdSet"]
    assert restored == [(6, 1200, -1200, 0x7FFF)]
    assert arm.joint_limits() == [list(map(float, f)) for f in C.JOINT_LIMITS_DEG]


# ---- connect_for_motion() / connect_readonly() with no bus ----------------------------
def _bus(monkeypatch, tmp_path, sdk_factory):
    """Graceful-start wrappers against an SDK stand-in: link up, no other receivers, no pkill, private lock file."""
    import piper_sdk
    monkeypatch.setattr(piper_sdk, "C_PiperInterface_V2", sdk_factory)
    monkeypatch.setattr(robot, "can_link", lambda iface=None: {"up": True, "state": "ERROR-ACTIVE"})
    monkeypatch.setattr(robot, "can_receivers", lambda iface=None: 0)
    monkeypatch.setattr(robot.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(robot, "LOCK_FILE", str(tmp_path / "piper_arm.lock"))


def test_connect_for_motion_ctrl_c_in_countdown_rests_and_unlocks(tmp_path, monkeypatch, capsys):
    sdk = FakeSDK(joints=C.HOME_JOINTS_DEG, enabled=False)        # de-torqued at HOME, like after rest.py
    _bus(monkeypatch, tmp_path, lambda iface: sdk)
    ticks = []

    def sleep(s, _real=time.sleep):
        if s == 1.0 and not ticks:                                  # first "motion in 5..." tick: operator hits Ctrl-C
            ticks.append(s)
            raise KeyboardInterrupt
        _real(min(s, 0.02))
    monkeypatch.setattr(robot.time, "sleep", sleep)
    with pytest.raises(KeyboardInterrupt):
        robot.connect_for_motion()
    names = [c[0] for c in sdk.calls]
    assert names.index("EnablePiper") < names.index("JointCtrl") < names.index("DisablePiper") < names.index("DisconnectPort")
    assert sdk.en == [False] * 6                                    # homed and de-torqued, not left in CAN-control mode
    out = capsys.readouterr().out
    assert "motion in 5" in out and "rest: DONE - de-torqued" in out
    acquire_lock().close()                                          # the lock was released on the way out


def test_connect_for_motion_ctrl_c_while_enabling_de_torques_without_moving(tmp_path, monkeypatch):
    class HalfSDK(FakeSDK):
        def EnablePiper(self):                                      # one frame enables only half the joints
            self.calls.append(("EnablePiper", ()))
            self.en[:3] = [True] * 3

    sdk = HalfSDK(joints=C.HOME_JOINTS_DEG, enabled=False)
    _bus(monkeypatch, tmp_path, lambda iface: sdk)

    def sleep(s, _real=time.sleep):
        if s == 0.3 and sdk.calls[-1][0] == "EnablePiper":          # Ctrl-C right after the first EnablePiper frame
            raise KeyboardInterrupt
        _real(min(s, 0.02))
    monkeypatch.setattr(robot.time, "sleep", sleep)
    with pytest.raises(KeyboardInterrupt):
        robot.connect_for_motion()
    names = [c[0] for c in sdk.calls]
    assert "JointCtrl" not in names and "DisablePiper" in names and names[-1] == "DisconnectPort"
    assert not any(sdk.en)
    acquire_lock().close()


def test_connect_wraps_sdk_connection_error_and_wrappers_abort(tmp_path, monkeypatch):
    def refuse(iface):                                              # piper_sdk's constructor: bitrate != 1 Mbps, iface gone
        raise ConnectionError(f"['{iface}' Interface __init__ ERROR]")
    _bus(monkeypatch, tmp_path, refuse)
    with pytest.raises(RuntimeError, match="piper_sdk refused can_piper .* bitrate must be 1000000"):
        Piper().connect(enable=False)
    with pytest.raises(SystemExit, match="ABORT: piper_sdk refused can_piper"):
        robot.connect_readonly()
    with pytest.raises(SystemExit, match="ABORT: piper_sdk refused can_piper"):
        robot.connect_for_motion()
    acquire_lock().close()                                          # both wrappers released the lock


def test_lerobot_plugin_opens_cameras_before_the_arm_and_never_rests_from_del(monkeypatch):
    from lerobot.robots import Robot
    from piper import lerobot_robot
    from piper.fake_arm import FakeArm
    monkeypatch.setattr(lerobot_robot, "open_cameras",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no frames from camera 'top'")))
    monkeypatch.setattr(lerobot_robot, "connect_for_motion",
                        lambda **k: (_ for _ in ()).throw(AssertionError("arm enabled before the cameras")))
    plugin = lerobot_robot.PiperRobot(lerobot_robot.PiperRobotConfig())
    with pytest.raises(RuntimeError, match="no frames"):
        plugin.connect()
    assert not plugin.is_connected
    plugin.arm = FakeArm()                                          # "connected" and then garbage collected
    plugin.__del__()
    assert not plugin.arm.rested                                    # no HOME move from interpreter teardown ...
    Robot.__del__(plugin)
    assert plugin.arm.rested                                        # ... which is what lerobot's base class would do


# ---- faults() on the real SDK status message -------------------------------------
def test_faults_parse_status_and_error_bits():
    from piper_sdk.piper_msgs.msg_v2.feedback.arm_feedback_status import ArmMsgFeedbackStatus

    def sdk_with(msg):
        return SimpleNamespace(GetArmStatus=lambda: SimpleNamespace(arm_status=msg))

    arm = _arm(sdk_with(ArmMsgFeedbackStatus(ctrl_mode=1, arm_status=0)))
    assert arm.faults() == [] and arm.arm_status() == "NORMAL(0x0)"
    bad = ArmMsgFeedbackStatus(ctrl_mode=1, arm_status=0x04)
    bad.err_code = (1 << 13) | (1 << 8) | (1 << 2)            # J6 + J1 angle limit, J3 comm error
    arm = _arm(sdk_with(bad))
    assert arm.faults() == ["TARGET_POS_EXCEEDS_LIMIT", "J1 angle limit", "J6 angle limit", "J3 comm error"]
    assert arm.arm_status() == "TARGET_POS_EXCEEDS_LIMIT(0x4)"


# ---- lock file ----------------------------------------------------------------
def test_lock_double_acquire_and_stale_pid(tmp_path, monkeypatch, capsys):
    lock = str(tmp_path / "piper_arm.lock")
    monkeypatch.setattr(robot, "LOCK_FILE", lock)
    fh = acquire_lock()
    assert open(lock).read() == str(os.getpid())
    with pytest.raises(SystemExit, match=f"another controller holds the arm .pid {os.getpid()}"):
        acquire_lock()
    fh.close()
    # stale: an orphan still holds the inode (flock) but the recorded controller pid can never exist
    holder = subprocess.Popen(
        [sys.executable, "-c", f"import fcntl, time; f = open({lock!r}, 'a+'); fcntl.flock(f, fcntl.LOCK_EX); "
         "f.seek(0); f.truncate(); f.write('4194304'); f.flush(); print('held', flush=True); time.sleep(30)"],
        stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        fh = acquire_lock()
        assert open(lock).read() == str(os.getpid())
        assert "pid 4194304 is dead, clearing" in capsys.readouterr().out
        fh.close()
    finally:
        holder.kill()
        holder.wait()


# ---- CAN helpers ----------------------------------------------------------------
RCVLIST = """receive list 'rx_all':
  (any: no entry)
  (can0: no entry)
  device   can_id   can_mask      function          userdata       matches  ident
   can_piper     000    00000000  0000000000000000  0000000000000000         0  raw
   can_piper     000    00000000  0000000000000001  0000000000000001      1234  raw
   can0          000    00000000  0000000000000002  0000000000000002         0  raw
"""

IP_PASSIVE = """15: can_piper: <NOARP,UP,LOWER_UP,ECHO> mtu 16 qdisc pfifo_fast state UP mode DEFAULT group default qlen 10
    link/can  promiscuity 0  allmulti 0 minmtu 0 maxmtu 0
    can state ERROR-PASSIVE (berr-counter tx 128 rx 0) restart-ms 0
\t  bitrate 1000000 sample-point 0.750
\t  re-started bus-errors arbit-lost error-warn error-pass bus-off
\t  0          0          0          4          15950454   0         numtxqueues 1 numrxqueues 1
    RX:  bytes  packets errors dropped  missed   mcast
     149728432 19280434      0     474       0       0
    TX:  bytes  packets errors dropped carrier collsns
             0   111716      3       0       0       0
"""

IP_DOWN = """16: can_piper: <NOARP,ECHO> mtu 16 qdisc pfifo_fast state DOWN mode DEFAULT group default qlen 10
    link/can  promiscuity 0  allmulti 0 minmtu 0 maxmtu 0
    can state STOPPED restart-ms 0
"""


def test_can_receivers_parses_rcvlist(tmp_path, monkeypatch):
    f = tmp_path / "rcvlist_all"
    f.write_text(RCVLIST)
    monkeypatch.setattr(robot, "RCVLIST", str(f))
    assert can_receivers("can_piper") == 2 and can_receivers("can0") == 1 and can_receivers("can1") == 0
    monkeypatch.setattr(robot, "RCVLIST", str(tmp_path / "missing"))
    assert can_receivers() == 0


def test_can_link_parses_ip_output(monkeypatch):
    out = {"text": IP_PASSIVE}
    monkeypatch.setattr(robot.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=out["text"]))
    link = can_link()
    assert link["up"] and link["state"] == "ERROR-PASSIVE" and link["berr"] == (128, 0)
    assert link["error_warn"] == 4 and link["error_pass"] == 15950454 and link["bus_off"] == 0
    assert link["rx_packets"] == 19280434 and link["tx_packets"] == 111716 and link["tx_errors"] == 3
    out["text"] = IP_DOWN
    link = can_link()
    assert not link["up"] and link["state"] == "STOPPED" and link["berr"] is None and link["error_pass"] == 0
    out["text"] = ""                                                   # interface missing
    link = can_link()
    assert not link["up"] and link["state"] is None and link["rx_packets"] == 0
