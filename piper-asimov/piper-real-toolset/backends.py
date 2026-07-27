"""Backends for the real AgileX Piper toolset.

Two interchangeable backends behind one interface:
  * MockPiperBackend — canned/simulated state, no hardware. Lets the whole
    Asimov stack (toolset + agent + LLM) run and be validated with no CAN bus.
  * RealPiperBackend — drives the physical Piper over CAN via piper_sdk
    (C_PiperInterface_V2). piper_sdk is imported lazily so mock mode needs no
    hardware libraries.

Tool-surface units are DEGREES (joints), METERS (Cartesian x/y/z), DEGREES
(orientation rx/ry/rz), and a gripper opening fraction 0..1. The real backend
converts to/from the SDK's raw units:
  * joints:      raw = millidegrees            (deg  = raw/1e3)
  * end pose:    raw x/y/z = micrometres        (m    = raw/1e6),
                 raw rx/ry/rz = millidegrees    (deg  = raw/1e3)
  * gripper:     raw angle = width_m * 1e6      (m    = raw/1e6)
The arm performs its OWN inverse kinematics for EndPoseCtrl, so no IK here.
"""
from __future__ import annotations

import math
import threading
import time

# Generous joint guardrails (degrees). The real arm reports a different zero
# convention than the sim URDF (e.g. joint6 rests near -174°), so these are wide
# outer safety bounds chosen to contain the real working range without clamping
# valid poses into a surprise move. The arm firmware enforces its own hard limits.
JOINT_LIMITS_DEG = [
    (-160.0, 160.0),   # joint1
    (-90.0, 200.0),    # joint2
    (-180.0, 90.0),    # joint3
    (-120.0, 120.0),   # joint4
    (-90.0, 90.0),     # joint5
    (-185.0, 185.0),   # joint6
]

# Gripper physical travel in metres (0 = closed, this = fully open).
GRIPPER_MAX_M = 0.07


class MockPiperBackend:
    """Hardware-free stand-in. Tracks a target joint/pose/gripper state and
    reports it back immediately, so the full agent stack can be exercised."""

    name = "mock"

    def __init__(self):
        self._lock = threading.RLock()
        self._joints = [0.0, 90.0, -80.0, 0.0, 30.0, 0.0]  # deg
        self._gripper = 1.0  # open fraction
        self._enabled = False
        self._estop = False

    # -- lifecycle --
    def connect(self):
        return {"ok": True, "detail": "mock backend — no CAN bus"}

    def enable(self):
        with self._lock:
            self._enabled = True
            self._estop = False
        return {"ok": True, "detail": "mock arm enabled"}

    def emergency_stop(self):
        with self._lock:
            self._estop = True
        return {"ok": True, "detail": "mock e-stop engaged"}

    # -- reads --
    def get_joint_positions_deg(self):
        with self._lock:
            return list(self._joints)

    def get_ee_pose(self):
        # Crude planar forward-kinematics stand-in — enough for mock telemetry.
        with self._lock:
            j = self._joints
        return {"x": 0.30, "y": 0.0, "z": 0.35, "rx": 0.0, "ry": 85.0, "rz": 0.0,
                "note": "mock pose (not real FK)"}

    def get_gripper_opening(self):
        with self._lock:
            return self._gripper

    def get_status(self):
        with self._lock:
            return {"mode": "mock", "enabled": self._enabled, "estop": self._estop,
                    "can_fps": 0, "telemetry_hz": 0.0,
                    "joints_deg": [round(v, 1) for v in self._joints],
                    "gripper_opening": round(self._gripper, 2)}

    # -- motion --
    def move_joints_deg(self, q_deg):
        with self._lock:
            self._joints = [float(v) for v in q_deg]
        return {"ok": True, "detail": f"mock joints set to {[round(v,1) for v in q_deg]}"}

    def move_pose(self, x, y, z, rx=0.0, ry=85.0, rz=0.0, linear=True):
        return {"ok": True, "detail": f"mock end-pose set to ({x:.3f},{y:.3f},{z:.3f}) m"}

    def go_to_zero(self):
        with self._lock:
            self._joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        return {"ok": True, "detail": "mock arm at zero position"}

    def set_gripper(self, opening):
        with self._lock:
            self._gripper = max(0.0, min(1.0, float(opening)))
        return {"ok": True, "detail": f"mock gripper -> {self._gripper:.2f}"}

    def close(self):
        pass


class RealPiperBackend:
    """Drives the physical Piper over CAN via piper_sdk.

    Safety model (mirrors the Asimov FR3 real toolset): construction only
    CONNECTS and reads; motion requires enable() first, and the toolset only
    calls enable()/motion when PIPER_ALLOW_MOTION=1. EmergencyStop is always
    available. The joint 2/3/5 gravity-fall guard from AgileX's play_pos.py is
    reproduced in _safe_release().
    """

    name = "real"
    D2MILLI = 1000.0     # deg -> millideg (JointCtrl / pose angles)
    M2MICRO = 1_000_000  # m   -> micro-m  (EndPoseCtrl x/y/z), and gripper m->raw

    def __init__(self, can_iface="can0", speed_percent=30):
        from piper_sdk import C_PiperInterface_V2  # lazy: only needed for real HW

        self.can_iface = can_iface
        self.speed = int(speed_percent)
        self._piper = C_PiperInterface_V2(can_iface)
        self._enabled = False

    # -- lifecycle --
    def connect(self):
        self._piper.ConnectPort()
        time.sleep(1.0)
        fps = self._piper.GetCanFps()
        status = self._piper.GetArmStatus()
        hz = getattr(status, "Hz", 0.0)
        warn = None
        if not hz:
            warn = ("Arm status frames (0x2A1-0x2A8) not received (Hz=0). The arm "
                    "is likely still in Master/teach mode — see SETUP.md: turn off "
                    "the Joint-5 button, set Slave mode, restart, reactivate CAN.")
        return {"ok": True, "can_fps": fps, "telemetry_hz": hz, "warning": warn}

    def enable(self):
        p = self._piper
        deadline = time.time() + 5
        while time.time() < deadline and not p.EnablePiper():
            time.sleep(0.05)
        if not all(p.GetArmEnableStatus()):
            return {"ok": False, "detail": "arm did not fully enable (check power/CAN/Slave mode)"}
        p.GripperCtrl(0, 1000, 0x01, 0x00)         # enable gripper
        p.ModeCtrl(0x01, 0x01, self.speed, 0x00)   # CAN control mode
        self._enabled = True
        return {"ok": True, "detail": "arm enabled (CAN control mode)"}

    def emergency_stop(self):
        self._piper.EmergencyStop(0x01)
        self._enabled = False
        return {"ok": True, "detail": "EMERGENCY STOP engaged — arm holding, motion disabled"}

    def _safe_release(self):
        """AgileX gravity-fall guard: only release the e-stop hold once joints
        2/3/5 are folded safely, so the arm doesn't drop when torque releases."""
        p = self._piper
        p.EmergencyStop(0x01); time.sleep(1.0)
        limit = [0.1745, 0.7854, 0.2094]  # rad
        for _ in range(1000):
            j = self.get_joint_positions_deg()
            r = [math.radians(v) for v in j]
            if abs(r[1]) < limit[0] and abs(r[2]) < limit[0] and limit[2] < r[4] < limit[1]:
                break
            time.sleep(0.01)
        p.EmergencyStop(0x02); time.sleep(1.0)

    # -- reads --
    def get_joint_positions_deg(self):
        js = self._piper.GetArmJointMsgs().joint_state
        return [getattr(js, f"joint_{i+1}") / self.D2MILLI for i in range(6)]

    def get_ee_pose(self):
        pose = self._piper.GetArmEndPoseMsgs()
        p = pose.end_pose
        return {"x": p.X_axis / self.M2MICRO, "y": p.Y_axis / self.M2MICRO,
                "z": p.Z_axis / self.M2MICRO, "rx": p.RX_axis / self.D2MILLI,
                "ry": p.RY_axis / self.D2MILLI, "rz": p.RZ_axis / self.D2MILLI,
                "telemetry_hz": getattr(pose, "Hz", 0.0)}

    def get_gripper_opening(self):
        g = self._piper.GetArmGripperMsgs().gripper_state.grippers_angle / self.M2MICRO
        return max(0.0, min(1.0, g / GRIPPER_MAX_M))

    def get_status(self):
        p = self._piper
        st = p.GetArmStatus()
        try:
            enable = list(p.GetArmEnableStatus())
        except Exception:
            enable = None
        return {"mode": "real", "enabled": self._enabled,
                "can_fps": p.GetCanFps(),
                "telemetry_hz": getattr(st, "Hz", 0.0),
                "arm_status": getattr(getattr(st, "arm_status", None), "arm_status", None),
                "ctrl_mode": getattr(getattr(st, "arm_status", None), "ctrl_mode", None),
                "enable_status": enable,
                "joints_deg": [round(v, 1) for v in self.get_joint_positions_deg()]}

    def _wait_motion_done(self, timeout=8.0):
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            st = self._piper.GetArmStatus()
            if getattr(st, "Hz", 0):
                a = st.arm_status
                if a.arm_status != 0:
                    return {"ok": False, "detail": f"arm_status={a.arm_status} (target rejected/unreachable)"}
                if a.motion_status == 0:
                    return {"ok": True}
                if a.motion_status == 1:
                    return {"ok": False, "detail": "REACH_TARGET_POS_FAILED"}
            time.sleep(0.05)
        return {"ok": False, "detail": f"timed out after {timeout}s"}

    # -- motion (only reached when PIPER_ALLOW_MOTION=1; toolset enforces) --
    def move_joints_deg(self, q_deg):
        clamped = [max(lo, min(hi, float(v))) for v, (lo, hi) in zip(q_deg, JOINT_LIMITS_DEG)]
        raw = [round(v * self.D2MILLI) for v in clamped]
        self._piper.MotionCtrl_2(0x01, 0x01, self.speed, 0x00)
        self._piper.JointCtrl(*raw)
        return self._wait_motion_done()

    def move_pose(self, x, y, z, rx=0.0, ry=85.0, rz=0.0, linear=True):
        move_mode = 0x02 if linear else 0x00  # MOVE_L vs MOVE_P
        self._piper.MotionCtrl_2(0x01, move_mode, self.speed, 0x00)
        self._piper.EndPoseCtrl(
            round(x * self.M2MICRO), round(y * self.M2MICRO), round(z * self.M2MICRO),
            round(rx * self.D2MILLI), round(ry * self.D2MILLI), round(rz * self.D2MILLI),
        )
        return self._wait_motion_done()

    def go_to_zero(self):
        """Drive all joints to the zero reference pose. Mirrors AgileX's
        go_to_zero() in draw_square.py: ModeCtrl (CAN/MOVE_J) then JointCtrl all
        zeros. NOTE: a full-range move from an arbitrary pose."""
        self._piper.ModeCtrl(0x01, 0x01, self.speed, 0x00)
        self._piper.JointCtrl(0, 0, 0, 0, 0, 0)
        return self._wait_motion_done(timeout=12.0)

    def set_gripper(self, opening):
        width_m = max(0.0, min(1.0, float(opening))) * GRIPPER_MAX_M
        self._piper.GripperCtrl(round(width_m * self.M2MICRO), 1000, 0x01, 0x00)
        time.sleep(1.0)
        return {"ok": True, "detail": f"gripper -> {opening:.2f} ({width_m*1000:.0f} mm)"}

    def close(self):
        try:
            self._piper.EmergencyStop(0x01)
        except Exception:
            pass
