"""Piper arm over CAN: feedback readers, one-tick command senders, blocking joint moves, safety clamps.

Graceful start: connect_for_motion() takes an exclusive lock (LOCK_FILE) and refuses if any other process
is on the bus. Graceful end: rest() homes slowly, de-torques every joint and releases the lock - every
entry point calls it in `finally`, so the arm is never left enabled or shared.
"""
import fcntl
import os
import re
import subprocess
import time

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from piper import config as C
from piper.transforms import clamp_rot_step, clamp_step, euler_to_rot, rot_to_euler, tip_position

JOINT_LIMITS_DEG = np.array(C.JOINT_LIMITS_DEG, np.float64)   # factory joint limits, deg (single source: config)
LOCK_FILE = "/tmp/piper_arm.lock"
RCVLIST = "/proc/net/can/rcvlist_all"


def acquire_lock():
    """One controller at a time. Returns the open lock file (keep it alive) or aborts naming the holder.
    A lock whose recorded pid is dead (an orphan of a crashed controller kept the inode open) is cleared."""
    for attempt in range(2):
        fh = open(LOCK_FILE, "a+")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fh.seek(0), fh.truncate(), fh.write(str(os.getpid())), fh.flush()
            return fh
        except BlockingIOError:
            fh.seek(0)
            pid = fh.read().strip()
            fh.close()
            if attempt == 0 and pid.isdigit() and not _alive(int(pid)):
                print(f"lock: pid {pid} is dead, clearing the stale {LOCK_FILE}")
                os.remove(LOCK_FILE)
                continue
            raise SystemExit(f"ABORT: another controller holds the arm (pid {pid or '?'}). "
                             f"Stop it (Ctrl-C there, or `python tools/rest.py --force`) first.")


def _alive(pid) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def can_receivers(iface=C.CAN_IFACE) -> int:
    """Number of raw-CAN receivers bound to the interface (other processes talking to the arm)."""
    try:
        return sum(line.split()[0] == iface for line in open(RCVLIST) if line.strip())
    except OSError:
        return 0


def can_link(iface=C.CAN_IFACE) -> dict:
    """Bus health from `ip -details -s link show`: up, CAN state (ERROR-ACTIVE = healthy, ERROR-PASSIVE =
    nobody ACKs our frames, BUS-OFF, STOPPED; None = interface missing), error-state event counters,
    rx/tx frame counts and the driver's berr-counter (tx, rx) when it reports one."""
    text = subprocess.run(["ip", "-details", "-s", "link", "show", iface], capture_output=True, text=True).stdout
    state = re.search(r"can state (\S+)", text)
    events = re.search(r"re-started\s+bus-errors\s+arbit-lost\s+error-warn\s+error-pass\s+bus-off\s+"
                       r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", text)
    rx = re.search(r"RX:.*\n\s*(\d+)\s+(\d+)\s+(\d+)", text)
    tx = re.search(r"TX:.*\n\s*(\d+)\s+(\d+)\s+(\d+)", text)
    berr = re.search(r"berr-counter tx (\d+) rx (\d+)", text)
    link = {"up": " state UP " in text, "state": state.group(1) if state else None,
            "rx_packets": int(rx.group(2)) if rx else 0, "tx_packets": int(tx.group(2)) if tx else 0,
            "tx_errors": int(tx.group(3)) if tx else 0, "berr": tuple(map(int, berr.groups())) if berr else None}
    link.update(zip(("restarts", "bus_errors", "arbit_lost", "error_warn", "error_pass", "bus_off"),
                    map(int, events.groups()) if events else (0,) * 6))
    return link


def clamp_pose(prev_xyz, prev_rot: Rot, xyz, rot: Rot, meas_xyz=None, meas_rot=None):
    """Workspace box, per-tick step limits, tool-tip floor, and (when the measured pose is given) a lead
    clamp: the target may never run further than LEAD_MAX ahead of the real arm. The firmware IK refuses
    unreachable targets SILENTLY - without this clamp the target keeps integrating away and the arm either
    freezes for good or lurches violently when the IK unsticks."""
    xyz = np.clip(np.asarray(xyz, np.float64), C.WORKSPACE_MIN_MM, C.WORKSPACE_MAX_MM)
    xyz = clamp_step(prev_xyz, xyz, C.STEP_MAX_MM)
    rot = clamp_rot_step(prev_rot, rot, C.STEP_MAX_DEG)
    lead_limited = False
    if meas_xyz is not None:
        d = xyz - np.asarray(meas_xyz, np.float64)
        n = np.linalg.norm(d)
        if n > C.LEAD_MAX_MM:
            xyz = meas_xyz + d * (C.LEAD_MAX_MM / n)
            lead_limited = True
    if meas_rot is not None:
        rv = (meas_rot.inv() * rot).as_rotvec(degrees=True)
        ang = np.linalg.norm(rv)
        if ang > C.LEAD_MAX_DEG:
            rot = meas_rot * Rot.from_rotvec(rv / ang * C.LEAD_MAX_DEG, degrees=True)
            lead_limited = True
    tip_z = xyz[2] + rot.as_matrix()[2, 2] * C.TOOL_LEN_MM
    if tip_z < C.TIP_Z_FLOOR_MM:
        xyz[2] += C.TIP_Z_FLOOR_MM - tip_z
    return xyz, rot, lead_limited


def clamp_joints(prev, meas, target):
    """JointCtrl streaming clamps (policy deploy, LeRobot plugin): <= MAX_JOINT_STEP_DEG per tick from the last
    command, <= JOINT_LEAD_MAX_DEG ahead of the measured joints, and hold the last command when the tool tip would
    descend below TIP_Z_FLOOR_MM. Returns (joints, floor_held)."""
    prev, meas = np.asarray(prev, np.float64), np.asarray(meas, np.float64)
    j = prev + np.clip(np.asarray(target, np.float64) - prev, -C.MAX_JOINT_STEP_DEG, C.MAX_JOINT_STEP_DEG)
    j = meas + np.clip(j - meas, -C.JOINT_LEAD_MAX_DEG, C.JOINT_LEAD_MAX_DEG)
    j = np.clip(j, JOINT_LIMITS_DEG[:, 0], JOINT_LIMITS_DEG[:, 1])       # what send_joints will actually send
    z = tip_position(j)[2]
    if z < C.TIP_Z_FLOOR_MM and z < tip_position(prev)[2]:   # below the floor and going down: stay put
        return prev, True
    return j, False


class Piper:
    def __init__(self, iface=C.CAN_IFACE):
        self.iface = iface
        self._p = None
        self._lock = None

    # ---- connection ----------------------------------------------------------
    def connect(self, enable=True, timeout=6.0):
        link = can_link(self.iface)
        if not link["up"]:
            raise RuntimeError(f"CAN interface {self.iface} is down ({link['state'] or 'missing'}) - "
                               "run `bash ../../prepare-jetson/can_up.sh`")
        from piper_sdk import C_PiperInterface_V2
        try:
            p = C_PiperInterface_V2(self.iface)
        except ConnectionError as e:   # the SDK wraps its JudgeCanInfo refusal (bitrate != 1 Mbps, iface dropped)
            raise RuntimeError(f"piper_sdk refused {self.iface} ({e}) - bitrate must be {C.CAN_BITRATE}: "
                               "`bash ../../prepare-jetson/can_up.sh`") from None
        p.ConnectPort()
        self._p = p
        t0 = time.time()
        while self.feedback_hz() == 0 and time.time() - t0 < C.FEEDBACK_WAIT_S:
            time.sleep(0.1)
        if self.feedback_hz() == 0:
            bus = can_link(self.iface)["state"]
            self.disconnect()
            raise RuntimeError(f"no feedback on {self.iface} after {C.FEEDBACK_WAIT_S:.1f} s (bus {bus}) - "
                               "arm powered? cable? `bash ../../prepare-jetson/can_up.sh` resets the bus")
        if enable:
            before = self.joints_deg()
            if not self._enable_motors(timeout):
                status = p.GetArmEnableStatus()
                self.disconnect()
                raise RuntimeError(f"arm did not enable {status} (teach button on? e-stop?)")
            self.restore_joint_limits()
            print(f"enable: joints before {np.round(before, 1).tolist()} -> after "
                  f"{np.round(self.joints_deg(), 1).tolist()} (twitch {np.round(self.joints_deg() - before, 1).tolist()})")
        return self

    def _enable_motors(self, timeout=6.0) -> bool:
        """Position mode BEFORE torque: a driver left latched in MIT mode (is_mit_mode 0xAD) by an
        interrupted soft_disable would otherwise fight the first JointCtrl (jam hypothesis a)."""
        self._p.MotionCtrl_2(0x01, 0x01, C.SPEED_HOME, 0x00)
        time.sleep(0.1)
        t0 = time.time()
        while not all(self._p.GetArmEnableStatus()) and time.time() - t0 < timeout:
            self._p.EnablePiper()
            time.sleep(0.3)
        return all(self._p.GetArmEnableStatus())

    # ---- drag teach ----------------------------------------------------------
    # Teach mode is entered ONLY by the physical button on the arm - there is no CAN command for it.
    # MotionCtrl_1's grag_teach_ctrl byte looks like one but does nothing on this firmware (the SDK has
    # ARM_GRAG_TEACH_CTRL commented out in can_id.py), and MasterSlaveConfig(0xFA) "teaching input arm"
    # does de-torque the arm but its own demo warns the arm must be RESTARTED to leave that mode - useless
    # for a teach/replay loop. AgileX's own recordAndPlayPos does the same as this: poll ctrl_mode.
    TEACH_CTRL_MODE = 0x02
    CAN_CTRL_MODE = 0x01
    TEACH_STATUS_STOPPED = 0x02

    def control_mode_code(self) -> int:
        return int(self._p.GetArmStatus().arm_status.ctrl_mode)

    def teach_status_code(self) -> int:
        return int(self._p.GetArmStatus().arm_status.teach_status)

    def in_teach(self) -> bool:
        return self.control_mode_code() == self.TEACH_CTRL_MODE

    def wait_teach(self, on: bool, timeout=120.0, poll=0.2) -> bool:
        """Block until the teach button has been pressed into (on) or out of (off) teach mode."""
        t0 = time.time()
        while time.time() - t0 < timeout and self.in_teach() != on:
            time.sleep(poll)
        return self.in_teach() == on

    def wait_teach_stopped(self, timeout=120.0, poll=0.2) -> bool:
        """Wait for the physical teach button to stop recording and restore stiffness."""
        t0 = time.time()
        while (
            time.time() - t0 < timeout
            and self.teach_status_code() != self.TEACH_STATUS_STOPPED
        ):
            time.sleep(poll)
        return self.teach_status_code() == self.TEACH_STATUS_STOPPED

    def gripper_enable(self, clear_errors=True):
        """Enable the gripper, clearing any latched error.

        gripper_code 0x01 enables; 0x03 enables AND clears errors (0x02 is disable+clear - not this).
        Hand-forcing the jaws, which is exactly what drag teach invites, can latch an error that 0x01
        alone will not clear: the gripper then ignores every command while the arm keeps moving normally,
        which presents as "the gripper just stays open". connect() sends no GripperCtrl at all, so on a
        --play run nothing else would ever clear it."""
        self._p.GripperCtrl(int(self.grip_mm() * 1000), C.GRIP_EFFORT, 0x03 if clear_errors else 0x01, 0)
        time.sleep(0.1)

    def teach_stop(self, timeout=8.0, hold_joints=None) -> bool:
        """Force the arm out of teach mode and take CAN control back. Does NOT need the button pressed again.

        The operator must first press the physical teach button so teach_status
        becomes STOP_RECORDING and the arm stiffens. Then reproduce the ordered
        sequence measured by tools/teach_probe.py:
          1. End teach recording with MotionCtrl_1.
          2. Stream the short ModeCtrl frame for two seconds.
          3. Stream the full MotionCtrl_2 frame until ctrl_mode becomes CAN_CTRL.
        The first two commands do not leave teach mode alone. Field runs showed
        that CAN exit is ineffective while teach_status is START_RECORDING.
        The caller must keep supporting the arm until physical stop has restored
        stiffness and this method has returned CAN control. The taught joint
        pose is captured before changing modes and streamed as soon as CAN
        control appears, preventing the controller from following a stale
        target such as zero."""
        if self.in_teach() and self.teach_status_code() != self.TEACH_STATUS_STOPPED:
            return False
        hold = np.asarray(
            self.joints_deg() if hold_joints is None else hold_joints,
            dtype=np.float64,
        )
        if hold.shape != (6,) or not np.all(np.isfinite(hold)):
            raise ValueError("hold_joints must contain six finite joint angles")
        self._p.MotionCtrl_1(0x00, 0x00, 0x02)
        prime_deadline = time.time() + min(2.0, timeout)
        while time.time() < prime_deadline:
            self._p.ModeCtrl(0x01, 0x01, C.SPEED_HOME, 0x00)
            time.sleep(0.01)

        deadline = time.time() + timeout
        stable_since = None
        while time.time() < deadline:
            self._p.MotionCtrl_2(0x01, 0x01, C.SPEED_HOME, 0x00)
            if self.control_mode_code() == self.CAN_CTRL_MODE:
                self.send_joints(hold, None, C.SPEED_REST)
                stable_since = stable_since or time.time()
                if time.time() - stable_since >= 0.5:
                    break
            else:
                stable_since = None
            time.sleep(0.01)
        if self.control_mode_code() != self.CAN_CTRL_MODE:
            return False
        ok = self._enable_motors(timeout)
        if ok:
            self.gripper_enable()          # the hand just operated the jaws: clear any error it latched
            self.send_joints(hold, None, C.SPEED_REST)
        return ok and self.control_mode_code() == self.CAN_CTRL_MODE

    def joint_limits(self):
        """[[min, max] deg] * 6 as stored in the motors (queried; None if no reply), or None if the SDK lacks it."""
        if not hasattr(self._p, "GetAllMotorAngleLimitMaxSpd"):
            return None
        self._p.SearchAllMotorMaxAngleSpd()
        time.sleep(0.3)
        motors = self._p.GetAllMotorAngleLimitMaxSpd().all_motor_angle_limit_max_spd.motor
        s16 = lambda v: v - 65536 if v > 32767 else v         # the frames are uint16, the angles are signed
        return [None if motors[i].motor_num != i else [s16(motors[i].min_angle_limit) / 10.0,
                                                       s16(motors[i].max_angle_limit) / 10.0] for i in range(1, 7)]

    def restore_joint_limits(self):
        """Write the factory limits back into any motor whose stored limits differ (the previous code set J6 to
        [-179, -60]; a limit the arm sits on makes the firmware IK refuse every target = the live freezes)."""
        lim = self.joint_limits()
        if lim is None:
            return
        for i, (stored, factory) in enumerate(zip(lim, C.JOINT_LIMITS_DEG), start=1):
            if stored is not None and (abs(stored[0] - factory[0]) > 0.5 or abs(stored[1] - factory[1]) > 0.5):
                print(f"J{i} limit stored {stored} != factory {factory} - restoring")
                self._p.MotorAngleLimitMaxSpdSet(motor_num=i, max_angle_limit=int(factory[1] * 10),
                                                 min_angle_limit=int(factory[0] * 10), max_joint_spd=0x7FFF)
                time.sleep(0.1)
        after = self.joint_limits()
        print(f"joint limits (deg): {after}" + ("" if after == lim else "  (after restoring)"))

    def disconnect(self):
        if self._p is not None:
            self._p.DisconnectPort()
            self._p = None
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def soft_disable(self, secs=C.SOFT_DISABLE_S, kp0=C.SOFT_KP, kd=C.SOFT_KD):
        """Let the joints down gently: MIT mode holding the current pose with a stiffness that ramps to zero
        under damping, so the gravity sag (shoulder/elbow ~2 deg at HOME) happens over `secs`, not as a drop
        when torque cuts. EXPERIMENTAL - a loud jam was reported around its first field use, so rest() only
        runs it when explicitly asked (tools/rest.py --soft) until it is re-validated supervised."""
        q0 = np.radians(self.joints_deg())
        t0 = time.time()
        try:
            while (t := time.time() - t0) < secs:
                kp = kp0 * max(0.0, 1.0 - t / secs)
                self._p.MotionCtrl_2(0x01, 0x04, 0, 0xAD)
                for m in range(6):
                    self._p.JointMitCtrl(m + 1, float(q0[m]), 0.0, kp, kd, 0.0)
                time.sleep(0.02)
        except Exception as e:
            print(f"soft_disable: MIT ramp failed ({e}), disabling directly")
        finally:
            # leave MIT mode cleanly: restore position mode at the current joints before torque-off,
            # so the NEXT session's enable + JointCtrl doesn't meet a driver still latched in MIT mode
            j = self.joints_deg()
            for _ in range(8):
                self.send_joints(j, None, C.SPEED_REST)
                time.sleep(0.05)

    def disable(self, timeout=5.0) -> bool:
        """De-torque every joint and drop to standby. Only safe at HOME/REST (no brakes: a raised arm would fall)."""
        self._p.GripperCtrl(int(self.grip_mm() * 1000), C.GRIP_EFFORT, 0x00, 0)   # gripper_code 0x00 = disable
        t0 = time.time()
        while any(self._p.GetArmEnableStatus()) and time.time() - t0 < timeout:
            self._p.DisablePiper()
            time.sleep(0.2)
        self._p.MotionCtrl_2(0x00, 0x01, 0, 0x00)
        return not any(self._p.GetArmEnableStatus())

    def rest(self, soft=False, drop=False):
        """Graceful end: slow move to HOME, wrist down to REST, de-torque, disconnect, unlock. Ends with one
        `rest: DONE ...` or `rest: FAILED ...` line. A joint the driver has cut (limp) is ignored when judging
        HOME; if the joints that still hold torque cannot reach HOME the arm stays torqued (no brakes: it would
        fall) - `drop=True` (tools/rest.py --drop) de-torques where it stands, for when you are holding the arm.
        soft=True adds the experimental MIT stiffness ramp before torque-off (supervised use only)."""
        if self._p is None:
            return
        try:
            j_rest = None
            limp = [f"J{i + 1}" for i, e in enumerate(self.enabled()) if not e]
            if drop:
                print(f"rest: --drop: de-torquing where it stands, joints {np.round(self.joints_deg(), 1).tolist()}")
            elif any(self.enabled()):
                if limp:
                    print(f"rest: {'+'.join(limp)} not enabled (driver cut it?) - homing with the others")
                if not self.home(self.grip_mm()):
                    print(f"rest: FAILED - home not reached, joints {np.round(self.joints_deg(), 1).tolist()}: "
                          "NOT de-torquing (the arm would fall). Clear the obstacle and run `python tools/rest.py`, "
                          "or hold the arm and run `python tools/rest.py --drop`")
                    return
                ok = self.move_joints(C.REST_JOINTS_DEG, 8.0, None, speed=C.SPEED_REST, tol=1.5, settle=0.5)
                j_rest = self.joints_deg()
                print(f"rest: at REST={ok} joints {np.round(j_rest, 1).tolist()}")
                if soft:
                    self.soft_disable()
                    print(f"rest: after stiffness ramp sag {np.round(self.joints_deg() - j_rest, 1).tolist()}")
                    j_rest = self.joints_deg()
            ok = self.disable()
            time.sleep(1.0)
            j_off = self.joints_deg()
            drop = f" drop {np.round(j_off - j_rest, 1).tolist()}" if j_rest is not None else ""
            print(f"rest: {'DONE - de-torqued' if ok else f'FAILED - still enabled {self.enabled()}'}, "
                  f"joints {np.round(j_off, 1).tolist()}{drop}")
        finally:
            self.disconnect()

    # ---- feedback ------------------------------------------------------------
    def joints_deg(self) -> np.ndarray:
        js = self._p.GetArmJointMsgs().joint_state
        return np.array([getattr(js, f"joint_{i}") for i in range(1, 7)], np.float64) / 1000.0

    def pose(self) -> np.ndarray:
        ep = self._p.GetArmEndPoseMsgs().end_pose
        return np.array([ep.X_axis, ep.Y_axis, ep.Z_axis, ep.RX_axis, ep.RY_axis, ep.RZ_axis],
                        np.float64) / 1000.0

    def grip_mm(self) -> float:
        return self._p.GetArmGripperMsgs().gripper_state.grippers_angle / 1000.0

    def j6_temp(self) -> int:
        return int(self._p.GetArmLowSpdInfoMsgs().motor_6.motor_temp)

    def ctrl_mode(self) -> str:
        return str(self._p.GetArmStatus().arm_status.ctrl_mode)

    def arm_status(self) -> str:
        return str(self._p.GetArmStatus().arm_status.arm_status)

    def enabled_mask(self) -> np.ndarray:
        return np.array(self.enabled()[:6], bool)

    def enable_joint(self, n: int, on=True):
        """Torque one motor on/off (1..6); the other joints keep holding."""
        (self._p.EnableArm if on else self._p.DisableArm)(n)

    def zero_joint(self, n: int):
        """Write the motor's zero: the CURRENT position of joint n becomes 0 deg (persistent in the driver).
        Only ever with the joint physically at its intended zero - see tools/zero_j6.py."""
        self._p.JointConfig(joint_num=n, set_zero=0xAE)

    def clear_joint_errors(self):
        self._p.JointConfig(joint_num=7, clear_err=0xAE)

    def faults(self) -> list:
        """Active faults from the 0x2A1 status frame: the arm_status name when not NORMAL (NO_SOLUTION,
        SINGULARITY_POINT, TARGET_POS_EXCEEDS_LIMIT, COLLISION_OCCURRED, ...) plus per-joint angle-limit /
        communication flags, e.g. ['TARGET_POS_EXCEEDS_LIMIT', 'J6 angle limit']. Empty = healthy."""
        st = self._p.GetArmStatus().arm_status
        es = st.err_status
        return (([] if st.arm_status.name == "NORMAL" else [st.arm_status.name])
                + [f"J{i} angle limit" for i in range(1, 7) if getattr(es, f"joint_{i}_angle_limit")]
                + [f"J{i} comm error" for i in range(1, 7) if getattr(es, f"communication_status_joint_{i}")])

    def health(self) -> dict:
        return {"enabled": self.enabled(), "ctrl_mode": self.ctrl_mode(), "arm_status": self.arm_status(),
                "faults": self.faults(), "j6_temp": self.j6_temp(), "feedback_hz": round(self.feedback_hz()),
                "can_fps": round(self.can_fps())}

    def enabled(self) -> list:
        return list(self._p.GetArmEnableStatus())

    def can_fps(self) -> float:
        return float(self._p.GetCanFps())

    def feedback_hz(self) -> float:
        return float(self._p.GetArmJointMsgs().Hz)

    def state7(self) -> np.ndarray:
        return np.append(self.joints_deg(), self.grip_mm()).astype(np.float32)

    # ---- one control tick ----------------------------------------------------
    def _grip(self, grip_mm):
        if grip_mm is not None:
            um = int(np.clip(grip_mm, C.GRIP_CLOSED_MM, C.GRIP_OPEN_MM) * 1000)
            self._p.GripperCtrl(um, C.GRIP_EFFORT, 0x01, 0)

    def send_pose(self, xyz, rpy_deg, grip_mm, speed=C.SPEED_TELEOP):
        self._p.MotionCtrl_2(0x01, 0x00, int(speed), 0x00)
        self._p.EndPoseCtrl(*[int(round(v * 1000)) for v in (*xyz, *rpy_deg)])
        self._grip(grip_mm)

    def send_joints(self, j_deg, grip_mm, speed=C.SPEED_JOINT):
        j = np.clip(np.asarray(j_deg, np.float64), JOINT_LIMITS_DEG[:, 0], JOINT_LIMITS_DEG[:, 1])
        self._p.MotionCtrl_2(0x01, 0x01, int(speed), 0x00)
        self._p.JointCtrl(*[int(round(v * 1000)) for v in j])
        self._grip(grip_mm)

    def send_gripper(self, grip_mm, speed=C.SPEED_JOINT):
        self._p.MotionCtrl_2(0x01, 0x01, int(speed), 0x00)
        self._grip(grip_mm)

    # ---- blocking moves ------------------------------------------------------
    def move_joints(self, target, secs, grip_mm=None, speed=C.SPEED_JOINT,
                    tol=C.HOME_TOL_DEG, settle=0.3) -> bool:
        """Stream one joint target until the arm is within tol for `settle` s (or secs elapse)."""
        target = np.asarray(target, np.float64)
        mask = self.enabled_mask()                       # a de-torqued (limp) joint cannot arrive: judge the others
        if not mask.any():
            return False
        t0, reached_at = time.time(), None
        while time.time() - t0 < secs:
            self.send_joints(target, grip_mm, speed)
            time.sleep(1.0 / C.CONTROL_HZ)
            close = np.abs(self.joints_deg() - target)[mask].max() < tol
            if close and reached_at is None:
                reached_at = time.time()
            elif close and time.time() - reached_at >= settle:
                return True
            elif not close:
                reached_at = None
        return np.abs(self.joints_deg() - target)[mask].max() < tol

    def move_to_zero(self, secs=C.HOME_TIMEOUT_S, speed=C.SPEED_HOME,
                     tol=C.HOME_TOL_DEG, settle=0.5) -> bool:
        """Move to the calibrated all-zero pose using the SDK's ModeCtrl/JointCtrl sequence."""
        target = np.zeros(6, dtype=np.float64)
        mask = self.enabled_mask()
        if not mask.any():
            return False
        t0, reached_at = time.time(), None
        while time.time() - t0 < secs:
            self._p.ModeCtrl(0x01, 0x01, int(speed), 0x00)
            self._p.JointCtrl(0, 0, 0, 0, 0, 0)
            time.sleep(1.0 / C.CONTROL_HZ)
            close = np.abs(self.joints_deg() - target)[mask].max() < tol
            if close and reached_at is None:
                reached_at = time.time()
            elif close and time.time() - reached_at >= settle:
                return True
            elif not close:
                reached_at = None
        return np.abs(self.joints_deg() - target)[mask].max() < tol

    def home(self, grip_mm=None) -> bool:
        print(f"home: {np.round(self.joints_deg(), 1).tolist()} -> {C.HOME_JOINTS_DEG} at {C.SPEED_HOME} %")
        ok = self.move_joints(C.HOME_JOINTS_DEG, C.HOME_TIMEOUT_S, grip_mm, speed=C.SPEED_HOME)
        print(f"home: reached={ok} joints {np.round(self.joints_deg(), 1).tolist()} pose {np.round(self.pose(), 1).tolist()}")
        return ok

    def ready(self, grip_mm=None) -> bool:
        """Cartesian-capable start pose: EndPoseCtrl is silently refused from the folded HOME."""
        return self.move_joints(C.READY_JOINTS_DEG, C.HOME_TIMEOUT_S, grip_mm, speed=C.SPEED_HOME)


def connect_for_motion(countdown=5, iface=C.CAN_IFACE) -> Piper:
    """Graceful start: take the arm lock, free the cameras, make sure CAN is ours, enable, count down.
    Exits with a one-line ABORT (no traceback) when the arm cannot be taken safely."""
    lock = acquire_lock()
    subprocess.run(["pkill", "-f", "dual_camera_stream.py"], capture_output=True)
    n = can_receivers(iface)
    if n:
        lock.close()
        raise SystemExit(f"ABORT: {n} other CAN receiver(s) on {iface} - stop them first")
    arm = Piper(iface)
    try:
        arm.connect(enable=True)
        arm._lock = lock
        print(f"enabled {arm.enabled()}  mode {arm.ctrl_mode()}  J6 {arm.j6_temp()} C  faults {arm.faults() or 'none'}  "
              f"joints {np.round(arm.joints_deg(), 1).tolist()}")
        for s in range(countdown, 0, -1):
            print(f"  motion in {s}...", flush=True)
            time.sleep(1.0)
    except RuntimeError as e:
        lock.close()
        raise SystemExit(f"ABORT: {e}")
    except BaseException:   # Ctrl-C after EnablePiper went out: home slowly + de-torque (no-op if never connected);
        arm.rest(drop=arm._p is not None and not all(arm.enabled()))   # mid-enable = was limp a moment ago: in place
        lock.close()        # idempotent when rest() -> disconnect() already closed it
        raise
    return arm


def connect_for_teach(iface=C.CAN_IFACE) -> Piper:
    """Take exclusive CAN control and enable the arm without touching camera processes.

    Camera-assisted teach workflows need the same enabled-arm lifecycle as
    connect_for_motion(), but must keep their own camera/FFmpeg processes alive.
    """
    lock = acquire_lock()
    n = can_receivers(iface)
    if n:
        lock.close()
        raise SystemExit(
            f"ABORT: {n} other CAN receiver(s) on {iface} - stop them first"
        )
    arm = Piper(iface)
    try:
        arm.connect(enable=True)
        arm._lock = lock
        if arm.in_teach():
            if arm.teach_status_code() != arm.TEACH_STATUS_STOPPED:
                raise RuntimeError(
                    "arm started in active teach recording; support it and press "
                    "the physical Teach button to stop recording before retrying"
                )
            startup_hold = arm.joints_deg()
            if not arm.teach_stop(hold_joints=startup_hold):
                raise RuntimeError(
                    "could not normalize stopped teach state to CAN control"
                )
        print(
            f"enabled {arm.enabled()}  mode {arm.ctrl_mode()}  "
            f"J6 {arm.j6_temp()} C  faults {arm.faults() or 'none'}  "
            f"joints {np.round(arm.joints_deg(), 1).tolist()}"
        )
    except RuntimeError as exc:
        arm.disconnect()
        lock.close()
        raise SystemExit(f"ABORT: {exc}") from None
    except BaseException:
        arm.disconnect()
        lock.close()
        raise
    return arm


def connect_readonly(iface=C.CAN_IFACE) -> Piper:
    """Feedback only (policy shadow mode): take the arm lock, connect WITHOUT enabling, never send a command.
    End with disconnect(), not rest()."""
    lock = acquire_lock()
    try:
        arm = Piper(iface).connect(enable=False)
    except RuntimeError as e:
        lock.close()
        raise SystemExit(f"ABORT: {e}")
    arm._lock = lock
    return arm


def pose_to_xyz_rot(pose6):
    return np.asarray(pose6[:3], np.float64), euler_to_rot(pose6[3:6])


def xyz_rot_to_pose(xyz, rot: Rot):
    return np.concatenate([xyz, rot_to_euler(rot)])
