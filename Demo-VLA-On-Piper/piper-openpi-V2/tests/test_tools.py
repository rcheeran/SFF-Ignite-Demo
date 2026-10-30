"""tools/*: check_arm's bus verdict and fail-fast paths (no bus opened), log/snapshot paths under data/,
and the motion tools' argparse entry points."""
import importlib.util
import os
import subprocess
import sys

from piper import config as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tool(name):
    spec = importlib.util.spec_from_file_location(f"tools_{name}", os.path.join(ROOT, "tools", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HEALTHY = {"up": True, "state": "ERROR-ACTIVE", "rx_packets": 1000, "tx_packets": 50, "tx_errors": 0,
           "berr": None, "error_warn": 0, "error_pass": 0, "bus_off": 0}


def test_can_verdict_lines():
    ca = _tool("check_arm")
    assert ca.can_verdict(HEALTHY).startswith("CAN: OK")
    assert ca.can_verdict({**HEALTHY, "state": "ERROR-PASSIVE", "berr": (128, 0)}).startswith(
        "CAN: ERROR-PASSIVE - our frames are not ACKed")
    assert "can_up.sh" in ca.can_verdict({**HEALTHY, "state": "BUS-OFF"})
    assert ca.can_verdict({**HEALTHY, "up": False, "state": None}) == \
        "CAN: can_piper is down (missing) - run `bash ../../prepare-jetson/can_up.sh`"


def test_check_arm_fails_fast_without_opening_the_bus(monkeypatch, capsys):
    ca = _tool("check_arm")
    monkeypatch.setattr(ca, "can_link", lambda: {**HEALTHY, "up": False, "state": "STOPPED"})
    monkeypatch.setattr(ca.Piper, "connect", lambda self, **k: (_ for _ in ()).throw(AssertionError("bus opened")))
    assert ca.check_arm() == 1
    assert "!! CAN: can_piper is down (STOPPED)" in capsys.readouterr().out


def test_check_arm_reports_no_feedback_with_bus_verdict(monkeypatch, capsys):
    ca = _tool("check_arm")
    monkeypatch.setattr(ca, "can_link", lambda: {**HEALTHY, "state": "ERROR-PASSIVE"})
    monkeypatch.setattr(ca.Piper, "connect", lambda self, **k: (_ for _ in ()).throw(
        RuntimeError("no feedback on can_piper after 2.0 s (bus ERROR-PASSIVE)")))
    assert ca.check_arm() == 1
    out = capsys.readouterr().out
    assert "!! no feedback on can_piper" in out and "!! CAN: ERROR-PASSIVE" in out


def test_stream_test_ctrl_c_rests_the_arm(monkeypatch, capsys):
    from piper.fake_arm import FakeArm

    class Arm(FakeArm):
        def faults(self):
            return []

    st = _tool("stream_test")
    arm = Arm()
    monkeypatch.setattr(st, "connect_for_motion", lambda: arm)
    monkeypatch.setattr(st.time, "sleep", lambda s: (_ for _ in ()).throw(KeyboardInterrupt))   # Ctrl-C mid-wiggle
    monkeypatch.setattr(sys, "argv", ["stream_test.py"])
    st.main()
    assert arm.sent_poses and arm.rested                             # interrupted while streaming -> homed, de-torqued
    assert "Ctrl-C: stopping stream, homing, de-torquing" in capsys.readouterr().out


def test_tool_paths_live_under_data():
    assert C.SNAP_DIR == os.path.join(C.DATA_DIR, "snapshots")
    assert _tool("stream_test").LOG == os.path.join(C.DATA_DIR, "logs", "stream_test.csv")


def test_motion_tools_parse_args_before_touching_the_arm():
    for tool in ("rest.py", "stream_test.py"):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", tool), "--help"],
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 0 and "usage:" in r.stdout, (tool, r.stderr)
    assert subprocess.run(["bash", "-n", os.path.join(ROOT, "..", "..", "prepare-jetson", "can_up.sh")]).returncode == 0


def test_zero_j6_sequence(monkeypatch, capsys):
    """home -> J6 de-torqued alone -> (operator turns the wrist) -> JointConfig set_zero -> J6 re-enabled -> rest."""
    import time as _time
    from piper.robot import Piper
    from test_robot import FakeSDK
    sdk = FakeSDK(joints=C.HOME_JOINTS_DEG)
    sdk.j[5] = -165000                                                          # the old 'camera-up' reading
    arm = Piper()
    arm._p = sdk
    zj = _tool("zero_j6")
    monkeypatch.setattr(zj, "connect_for_motion", lambda: arm)
    monkeypatch.setattr(zj.time, "sleep", lambda s, _real=_time.sleep: _real(min(s, 0.02)))
    monkeypatch.setattr(zj.np, "round", zj.np.round)
    import piper.robot as robot
    monkeypatch.setattr(robot.time, "sleep", lambda s, _real=_time.sleep: _real(min(s, 0.02)))
    answers = iter(["y", "", "y"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    def set_zero(**kw):
        sdk.calls.append(("JointConfig", kw))
        if kw.get("set_zero") == 0xAE:
            sdk.j[kw["joint_num"] - 1] = 0.0                                    # the motor re-zeroes where it is
    sdk.JointConfig = set_zero
    monkeypatch.setattr(sys, "argv", ["zero_j6.py"])
    zj.main()
    names = [c[0] for c in sdk.calls if c[0] in ("DisableArm", "JointConfig", "EnableArm", "DisablePiper")]
    assert names == ["DisableArm", "JointConfig", "EnableArm", "DisablePiper"]
    assert [c[1] for c in sdk.calls if c[0] == "JointConfig"] == [{"joint_num": 6, "set_zero": 0xAE}]
    out = capsys.readouterr().out
    assert "J6 now reads 0.0 deg" in out and "rest: DONE" in out


def test_stream_test_debounces_transient_faults(monkeypatch, capsys):
    """One-status-frame EXCEEDS_LIMIT flickers at the rz wrap are counted, not printed; persistent faults print."""
    import itertools
    from piper.fake_arm import FakeArm

    class Arm(FakeArm):
        seq = itertools.count()

        def faults(self):
            n = next(self.seq)
            if n > 60:
                return ["COLLISION_OCCURRED"]                           # late, steady: must print
            return ["TARGET_POS_EXCEEDS_LIMIT"] if n % 7 == 3 else []   # one frame only: a flicker
    st = _tool("stream_test")
    monkeypatch.setattr(st, "connect_for_motion", lambda: Arm())
    monkeypatch.setattr(st.time, "sleep", lambda s: None)
    monkeypatch.setattr(sys, "argv", ["stream_test.py", "--secs", "8"])   # fake time: ~100+ rows for the stats
    fake_t = iter(x * 0.02 for x in range(10000))
    monkeypatch.setattr(st.time, "monotonic", lambda: next(fake_t))
    st.main()
    out = capsys.readouterr().out
    assert "transient status flickers:" in out
    assert out.count("TARGET_POS_EXCEEDS_LIMIT") == 1                   # only in the summary line
    assert "COLLISION_OCCURRED" in out.split("transient")[0]            # the persistent one printed live
