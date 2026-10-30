"""MuJoCo digital twin of the Piper: piper.robot.Piper's interface on a simulated arm, no CAN bus.

    python -m sim.piper_sim --demo                HOME -> READY -> +-20 mm / +-10 deg Cartesian wiggle (lag and
                                                  RMS like tools/stream_test.py) -> scripted pick-and-place
                                                  -> data/sim/demo_top.mp4 + demo_wrist.mp4
    python -m sim.piper_sim --demo --episodes 10  ... and the expert success rate over 10 seeded episodes
    python -m sim.piper_sim --snap                both cameras at HOME and READY -> data/sim/snap_*.jpg (is the
                                                  wrist view upright? it must be with J6 = 0)
    MUJOCO_GL=osmesa python -m sim.piper_sim ...  software rendering if EGL misbehaves (40 ms/frame vs 2 ms)
    SIM_VIEWER=0 ...                              no window (the twin opens a live mujoco.viewer window whenever
                                                  $DISPLAY is set, i.e. on the Jetson desktop, not over plain ssh)

SimPiper has every Piper method the repo uses (feedback, send_pose / send_joints / send_gripper, move_joints,
home / ready / rest). pose() is the piper_sdk forward kinematics of the simulated joints, so the reported
convention is the real arm's by construction; send_pose runs a damped-least-squares IK on that same FK and
refuses like the firmware (the arm holds, arm_status NO_SOLUTION / TARGET_POS_EXCEEDS_LIMIT). Physics steps
in a daemon thread; joint targets slew toward the command at SIM_JOINT_VMAX_DEG_S * speed % behind a SIM_LAG_S
first-order lag. One lock serves the 30 Hz control thread, the 15 Hz recorder and the cameras.
"""
import argparse
import os
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as Rot

import sim
from piper import config as C

os.environ.setdefault("MUJOCO_GL", C.SIM_GL)   # must be set before the first mujoco import
import mujoco  # noqa: E402
from piper_sdk.kinematics import C_PiperForwardKinematics  # noqa: E402

from piper.robot import JOINT_LIMITS_DEG, Piper, clamp_pose, pose_to_xyz_rot, xyz_rot_to_pose  # noqa: E402
from piper.transforms import euler_to_rot  # noqa: E402

from piper.kinematics import fk, ik  # noqa: E402,F401  (the SDK DH chain; tests import them from here too)

OFFSET = np.array(C.SIM_JOINT_OFFSET_DEG, np.float64)   # MJCF qpos (deg) = reported joints + OFFSET
NORMAL = "NORMAL(0x0)"   # piper_sdk 0.6.2 str(ArmStatus) format, parsed by teleop.quest.arm_status_name


class SimPiper(Piper):
    """piper.robot.Piper on MuJoCo. connect() starts the physics thread at HOME; rest() parks and stops it.
    move_joints / home / ready / state7 are inherited unchanged - they only use the primitives below."""

    def __init__(self, scene=C.SIM_SCENE):
        super().__init__(iface="sim")
        self.m = mujoco.MjModel.from_xml_path(scene)
        self.d = mujoco.MjData(self.m)
        self.lock = threading.RLock()
        jid = [self._id(mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 8)]
        self._qadr = self.m.jnt_qposadr[jid]                                  # joints 1..6 + finger
        self._aid = [self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in [f"joint{i}" for i in range(1, 7)] + ["gripper"]]
        mj_lim = np.degrees(self.m.jnt_range[jid[:6]]) - OFFSET[:, None]
        self.limits = np.stack([np.maximum(JOINT_LIMITS_DEG[:, 0], mj_lim[:, 0]),
                                np.minimum(JOINT_LIMITS_DEG[:, 1], mj_lim[:, 1])], axis=1)
        self._fk = C_PiperForwardKinematics()
        bj = self._id(mujoco.mjtObj.mjOBJ_JOINT, "block")
        self._block_q, self._block_v = self.m.jnt_qposadr[bj], self.m.jnt_dofadr[bj]
        self._block_half = self.m.geom_size[self._id(mujoco.mjtObj.mjOBJ_GEOM, "block")][2]
        self.bowl_r_mm = self.m.geom_size[self._id(mujoco.mjtObj.mjOBJ_GEOM, "bowl_floor")][0] * 1000.0
        self.m.body_pos[self._id(mujoco.mjtObj.mjOBJ_BODY, "bowl")][:2] = np.array(C.SIM_BOWL_XY_MM) / 1000.0
        self._q_cmd, self._q_tgt, self._q_ref = (np.array(C.HOME_JOINTS_DEG, np.float64) for _ in range(3))
        self._grip_cmd = self._grip_tgt = self._grip_ref = C.GRIP_OPEN_MM
        self._speed, self._status, self._enabled, self._cart = C.SPEED_HOME, NORMAL, False, False
        self._stop, self._thread, self._viewer = threading.Event(), None, None

    def _id(self, kind, name):
        i = mujoco.mj_name2id(self.m, kind, name)
        if i < 0:
            raise KeyError(f"{name} not in {self.m}")
        return i

    # ---- connection ----------------------------------------------------------
    def connect(self, enable=True, timeout=6.0):
        with self.lock:
            self.d.qpos[self._qadr[:6]] = np.radians(self._q_cmd + OFFSET)
            self.d.qpos[self._qadr[6]] = self._grip_cmd / 2000.0
            self._slew(self.m.opt.timestep)
            mujoco.mj_forward(self.m, self.d)
        self.place_block(np.mean(C.SIM_BLOCK_SPAWN_MM, axis=0), 0.0)
        self._enabled = enable
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        sim.current = self
        print(f"sim: {os.path.relpath(C.SIM_SCENE, C.ROOT)} physics {1 / self.m.opt.timestep:.0f} Hz "
              f"MUJOCO_GL={os.environ['MUJOCO_GL']}  joints {np.round(self.joints_deg(), 1).tolist()}")
        if os.environ.get("DISPLAY") and os.environ.get("SIM_VIEWER", "1") != "0":
            self._open_viewer()
        return self

    def _open_viewer(self):
        """Live window of the twin (mujoco.viewer runs its own GUI thread; the physics thread syncs it at 30 Hz).
        Needs a display: the Jetson desktop, not a plain ssh terminal. SIM_VIEWER=0 keeps it headless."""
        try:
            import mujoco.viewer
            with self.lock:
                v = mujoco.viewer.launch_passive(self.m, self.d, show_left_ui=False, show_right_ui=False)
            v.cam.lookat[:] = [0.25, 0.0, 0.12]                      # table centre; drag to orbit, scroll to zoom
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.2, 160.0, -30.0
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True       # draw the top / wrist camera frusta
            self._viewer = v
            print("sim: viewer window open (drag = orbit, scroll = zoom; closing it keeps the twin running headless)")
        except Exception as e:
            print(f"sim: no viewer window ({e}) - running headless")

    def disconnect(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None

    def soft_disable(self, *a, **k):
        pass

    def disable(self, timeout=5.0) -> bool:
        self._enabled = False
        return True

    def rest(self, soft=False):
        """Graceful end like the real arm: slow move to HOME, wrist to REST, de-torque (commands ignored), stop."""
        if self._thread is None:
            return
        try:
            if self._enabled:
                if not self.home(self.grip_mm()):
                    print(f"WARNING: home not reached, joints {np.round(self.joints_deg(), 1).tolist()} - NOT de-torquing")
                    return
                ok = self.move_joints(C.REST_JOINTS_DEG, 8.0, None, speed=C.SPEED_REST, tol=1.5, settle=0.5)
                print(f"rest: at REST={ok} joints {np.round(self.joints_deg(), 1).tolist()}")
            self.disable()
            print(f"rest: de-torqued=True, enabled={self.enabled()}, joints {np.round(self.joints_deg(), 1).tolist()}")
        finally:
            self.disconnect()

    # ---- physics thread ------------------------------------------------------
    def _run(self):
        dt, n = self.m.opt.timestep, 5            # 10 ms of simulated time per wake-up
        next_t, next_sync = time.perf_counter(), 0.0
        while not self._stop.is_set():
            with self.lock:
                for _ in range(n):
                    self._slew(dt)
                    mujoco.mj_step(self.m, self.d)
                if self._viewer is not None and time.perf_counter() >= next_sync:   # window at ~30 Hz
                    next_sync = time.perf_counter() + 1.0 / 30
                    if self._viewer.is_running():
                        self._viewer.sync()
                    else:
                        self._viewer = None                            # closed by the user: headless from here
            next_t = max(next_t + n * dt, time.perf_counter() - 0.05)   # never chase more than 50 ms
            time.sleep(max(0.0, next_t - time.perf_counter()))

    def _slew(self, dt):
        """Targets slew toward the command, then a first-order lag feeds the position actuators. JointCtrl
        targets (send_joints, moves) slew at speed % of SIM_JOINT_VMAX_DEG_S - HOME -> READY at 15 % takes
        ~2 s; EndPoseCtrl streaming (send_pose) lets the joints follow the IK target at the full rate, which is
        what the real arm's 67-200 ms Cartesian lag implies (its wrist swings tens of degrees per 20 mm)."""
        v = C.SIM_JOINT_VMAX_DEG_S * (1.0 if self._cart else self._speed / 100.0) * dt
        dq = self._q_cmd - self._q_tgt
        self._q_tgt += dq * min(1.0, v / max(np.abs(dq).max(), 1e-9))      # synchronised: all joints arrive together
        self._grip_tgt += np.clip(self._grip_cmd - self._grip_tgt, -C.SIM_GRIP_VMAX_MM_S * dt, C.SIM_GRIP_VMAX_MM_S * dt)
        a = dt / (C.SIM_LAG_S + dt)
        self._q_ref += a * (self._q_tgt - self._q_ref)
        self._grip_ref += a * (self._grip_tgt - self._grip_ref)
        self.d.ctrl[self._aid[:6]] = np.radians(self._q_ref + OFFSET)
        self.d.ctrl[self._aid[6]] = self._grip_ref / 2000.0

    # ---- feedback (same names / units as the real arm) -----------------------
    def joints_deg(self) -> np.ndarray:
        with self.lock:
            return np.degrees(self.d.qpos[self._qadr[:6]]) - OFFSET

    def pose(self) -> np.ndarray:
        return np.array(self._fk.CalFK(np.radians(self.joints_deg()).tolist())[-1], np.float64)

    def grip_mm(self) -> float:
        with self.lock:
            return float(self.d.qpos[self._qadr[6]] * 2000.0)

    def j6_temp(self) -> int:
        return C.SIM_J6_TEMP_C

    def ctrl_mode(self) -> str:
        return "CtrlMode.CAN_CTRL" if self._enabled else "CtrlMode.STANDBY"

    def arm_status(self) -> str:
        return self._status

    def enabled(self) -> list:
        return [self._enabled] * 7

    def can_fps(self) -> float:
        return 1.0 / self.m.opt.timestep

    def feedback_hz(self) -> float:
        return 1.0 / self.m.opt.timestep

    def faults(self) -> list:
        return [] if self._status == NORMAL else [self._status.split("(")[0]]   # names, like Piper.faults()

    # ---- one control tick ----------------------------------------------------
    def _grip(self, grip_mm):
        if grip_mm is not None:
            self._grip_cmd = float(np.clip(grip_mm, C.GRIP_CLOSED_MM, C.GRIP_OPEN_MM))

    def send_pose(self, xyz, rpy_deg, grip_mm, speed=C.SPEED_TELEOP):
        if not self._enabled:
            return
        q0 = self.joints_deg()                       # like the firmware: solve from where the arm IS (and like
        q, ok = ik(xyz, euler_to_rot(rpy_deg), q0, self.limits)   # teleop's reachable(), so the two agree)
        with self.lock:
            if ok:
                self._q_cmd, self._status = q, NORMAL
            else:   # hold the last target, like the firmware; a limit-pinned solution reads as the limit fault
                pinned = np.any(np.isclose(q, self.limits[:, 0])) or np.any(np.isclose(q, self.limits[:, 1]))
                self._status = "TARGET_POS_EXCEEDS_LIMIT(0x4)" if pinned else "NO_SOLUTION(0x2)"
            self._speed, self._cart = speed, True
            self._grip(grip_mm)

    def send_joints(self, j_deg, grip_mm, speed=C.SPEED_JOINT):
        if not self._enabled:
            return
        with self.lock:
            self._q_cmd = np.clip(np.asarray(j_deg, np.float64), self.limits[:, 0], self.limits[:, 1])
            self._status, self._speed, self._cart = NORMAL, speed, False
            self._grip(grip_mm)

    def send_gripper(self, grip_mm, speed=C.SPEED_JOINT):
        if self._enabled:
            with self.lock:
                self._grip(grip_mm)

    # ---- privileged scene access (expert, tests) -----------------------------
    def body_xyz(self, name) -> np.ndarray:
        """World position (mm) of a body."""
        with self.lock:
            return self.d.xpos[self._id(mujoco.mjtObj.mjOBJ_BODY, name)] * 1000.0

    def block_pose(self):
        """(xyz mm, yaw deg) of the block."""
        with self.lock:
            q = self.d.qpos[self._block_q:self._block_q + 7].copy()
        return q[:3] * 1000.0, float(Rot.from_quat(q[[4, 5, 6, 3]]).as_euler("xyz", degrees=True)[2])

    def place_block(self, xy_mm, yaw_deg):
        """Drop the block at (x, y) on the table with the given yaw, at rest."""
        quat = Rot.from_euler("z", yaw_deg, degrees=True).as_quat()[[3, 0, 1, 2]]
        with self.lock:
            self.d.qpos[self._block_q:self._block_q + 7] = [xy_mm[0] / 1000.0, xy_mm[1] / 1000.0,
                                                            self._block_half + 0.001, *quat]
            self.d.qvel[self._block_v:self._block_v + 6] = 0.0
            mujoco.mj_forward(self.m, self.d)


def connect_sim() -> SimPiper:
    """The process-wide twin: the arm and its cameras share one physics thread, so entry points call this
    (in any order with open_sim_cameras) where they would call connect_for_motion()."""
    if sim.current is None or sim.current._thread is None:
        SimPiper().connect()
    return sim.current


# ---- self-test (tools/stream_test.py on the twin) ----------------------------------
def wiggle(arm, secs=12.0, period=6.0, amp_mm=20.0, amp_deg=10.0) -> dict:
    """+-amp sinusoid in x/y and +-amp_deg yaw from the current pose with the gripper opening/closing, through
    the teleop clamps. Prints and returns {axis: (cmd amp, meas amp, lag ms, rms)} like tools/stream_test.py."""
    xyz0, rot0 = pose_to_xyz_rot(arm.pose())
    cmd_xyz, cmd_rot = xyz0.copy(), rot0
    rows, refused, t0, next_t = [], 0, time.monotonic(), time.monotonic()
    while (t := time.monotonic() - t0) < secs:
        w = 2 * np.pi * t / period
        tgt_xyz = xyz0 + np.array([amp_mm * np.sin(w), amp_mm * np.cos(w) - amp_mm, 0.0])
        tgt_rot = Rot.from_euler("z", amp_deg * np.sin(w), degrees=True) * rot0
        meas_xyz, meas_rot = pose_to_xyz_rot(arm.pose())
        cmd_xyz, cmd_rot, _ = clamp_pose(cmd_xyz, cmd_rot, tgt_xyz, tgt_rot, meas_xyz, meas_rot)
        cmd = xyz_rot_to_pose(cmd_xyz, cmd_rot)
        grip = C.GRIP_OPEN_MM if int(t // (period / 2)) % 2 == 0 else C.GRIP_CLOSED_MM + 20
        arm.send_pose(cmd_xyz, cmd[3:], grip)
        refused += arm.arm_status() != NORMAL
        rows.append([t, *cmd, grip, *arm.pose(), arm.grip_mm(), *arm.joints_deg()])
        next_t += 1.0 / C.CONTROL_HZ
        time.sleep(max(0.0, next_t - time.monotonic()))
    d, out = np.array(rows), {}
    for col in (6, 13):                                       # Euler rz wraps at +-180 (where the tool sits at READY)
        d[:, col] = np.degrees(np.unwrap(np.radians(d[:, col])))
    for name, ci, mi in (("x", 1, 8), ("y", 2, 9), ("yaw", 6, 13)):
        c, m = d[:, ci] - d[:, ci].mean(), d[:, mi] - d[:, mi].mean()
        k = int(np.argmax([np.dot(c[:len(c) - k], m[k:]) for k in range(60)]))
        rms = np.sqrt(np.mean((c[:len(c) - k] - m[k:]) ** 2))
        out[name] = (np.ptp(c) / 2, np.ptp(m) / 2, k * 1000 / C.CONTROL_HZ, rms)
        print(f"{name:3s}: cmd amp {out[name][0]:5.1f}  meas amp {out[name][1]:5.1f}  lag {out[name][2]:4.0f} ms  rms err {rms:4.1f}")
    print(f"gripper: measured range {d[:, 14].min():.1f}..{d[:, 14].max():.1f} mm (cmd 70 / 20)")
    print(f"joint excursion (deg): {np.round(np.ptp(d[:, 15:21], axis=0), 1).tolist()}  J6 {d[:, 20].min():.0f}..{d[:, 20].max():.0f}  "
          f"refused ticks {refused}/{len(rows)} (firmware-limit refusals: the arm holds)")
    out["refused"] = refused / len(rows)
    return out


def snap():
    """Both cameras at HOME and READY -> data/sim/snap_<pose>_<cam>.jpg, plus where the wrist camera points:
    with J6 = 0 the wrist view must be upright (horizon at the top at HOME, fingertips at the bottom)."""
    import cv2
    from sim.cameras import open_sim_cameras
    arm, cams = connect_sim(), open_sim_cameras()
    os.makedirs(C.SIM_VIDEO_DIR, exist_ok=True)
    try:
        for pose, q in (("home", C.HOME_JOINTS_DEG), ("ready", C.READY_JOINTS_DEG)):
            arm.move_joints(q, 8.0, C.GRIP_OPEN_MM, speed=C.SPEED_JOINT)
            time.sleep(1.0)
            cid = mujoco.mj_name2id(arm.m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
            with arm.lock:
                up, fwd = arm.d.cam_xmat[cid].reshape(3, 3)[:, 1], -arm.d.cam_xmat[cid].reshape(3, 3)[:, 2]
            for name, cam in cams.items():
                path = os.path.join(C.SIM_VIDEO_DIR, f"snap_{pose}_{name}.jpg")
                cv2.imwrite(path, cam.read())
                print(f"{path}")
            print(f"{pose}: joints {np.round(arm.joints_deg(), 1).tolist()}  wrist camera up={np.round(up, 2).tolist()} "
                  f"forward={np.round(fwd, 2).tolist()} (world; up z > 0 = image upright)")
    finally:
        for cam in cams.values():
            cam.close()
        arm.disconnect()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--demo", action="store_true", help="home, ready, wiggle, expert episodes, videos")
    ap.add_argument("--snap", action="store_true", help="render both cameras at HOME and READY -> data/sim/snap_*.jpg")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    if args.snap:
        return snap()
    if not args.demo:
        ap.print_help()
        return
    from sim import expert
    from sim.cameras import VideoRecorder, open_sim_cameras
    arm, cams = connect_sim(), open_sim_cameras()
    rec = None if args.no_video else VideoRecorder(cams)
    try:
        t0 = time.monotonic()
        print("AT HOME" if arm.home() else "WARNING: home not reached", np.round(arm.joints_deg(), 1).tolist())
        t1 = time.monotonic()
        print("AT READY" if arm.ready() else "WARNING: ready not reached", np.round(arm.joints_deg(), 1).tolist(),
              f"({t1 - t0:.1f} s home, {time.monotonic() - t1:.1f} s HOME->READY at {C.SPEED_HOME} %)")
        print(f"start pose {np.round(arm.pose(), 1).tolist()}")
        wiggle(arm)
        ok = expert.run_episodes(arm, args.episodes, args.seed)
        print(f"EXPERT: {sum(ok)}/{len(ok)} episodes succeeded (seed {args.seed})")
        print(f"cameras: top {cams['top'].fps():.0f} fps, wrist {cams['wrist'].fps():.0f} fps")
    finally:
        if rec is not None:
            rec.close()
        for cam in cams.values():
            cam.close()
        arm.rest()


if __name__ == "__main__":
    main()
