#!/usr/bin/env python3
"""calibrate_camera.py - solve the wrist D405 -> Piper base transform for
detection-driven reaching, at a fixed 'observe' pose.

Method (see AgileX's cube-recognition tutorial, adapted to piper_sdk + wrist cam):
place a red can on the table, detect its 3-D position from the D405 (camera
frame) at the observe pose, jog the gripper to TOUCH that spot and record the
arm position (base frame). Repeat >=4 times, then fit a rigid transform.

Keeping a fixed top-down orientation and recording the EE position at touch means
the gripper-tip offset is absorbed, so T maps a detected can straight to its
grasp position.

PREREQ: Joint-5 TEACH button OFF (slave). Base bolted down. E-stop in hand.
Run in ~/venv (piper_sdk + pyrealsense2 + opencv + numpy).

Keys:
  jog:      w/s=+/-X  a/d=+/-Y  r/f=+/-Z   i/k=+/-RX  o/l=+/-RY  n/m=+/-RZ
  gripper:  [ close   ] open        step:  + / -
  V = save current pose as OBSERVE     G = go to OBSERVE pose
  D = detect can now (camera 3-D)      T = record touch (arm base 3-D) for last detect
  L = list pairs     S = solve + save calibration     P = print pose     Q = quit
"""
import glob
import json
import os
import signal
import sys
import termios
import threading
import time
import tty

import cv2
import numpy as np
import pyrealsense2 as rs
from piper_sdk import C_PiperInterface_V2


def free_camera():
    """Kill any same-user process holding a /dev/video* device before we open it."""
    me = os.getpid()
    devs = {os.path.realpath(p) for p in glob.glob("/dev/video*")}

    def holders():
        pids = set()
        for fd_dir in glob.glob("/proc/[0-9]*/fd"):
            pid = int(fd_dir.split("/")[2])
            if pid == me:
                continue
            try:
                for fd in os.listdir(fd_dir):
                    try:
                        if os.path.realpath(os.path.join(fd_dir, fd)) in devs:
                            pids.add(pid); break
                    except OSError:
                        continue
            except (OSError, PermissionError):
                continue
        return pids

    h = holders()
    for pid in h:
        print(f"INFO: camera held by PID {pid} - stopping it.")
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    if h:
        time.sleep(1.2)
        for pid in holders():
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(0.5)

CAN = "can_piper"
OUT = os.path.expanduser("~/Documents/piper-openpi/calib.json")
SNAP = os.path.expanduser("~/Documents/piper-openpi/calib_detect.jpg")
PREVIEW = os.path.expanduser("~/Documents/piper-openpi/calib_preview.jpg")
F = 1000.0
SPEED = 25
# red HSV (two ranges around hue wrap)
RED1 = ((0, 110, 70), (10, 255, 255))
RED2 = ((170, 110, 70), (180, 255, 255))


def detect_red(bgr):
    """Return (u, v, mask, area) of the largest red blob, or None."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(RED1[0]), np.array(RED1[1])) | \
        cv2.inRange(hsv, np.array(RED2[0]), np.array(RED2[1]))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 200:
        return None
    M = cv2.moments(c)
    u, v = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    return u, v, m, cv2.contourArea(c)


def umeyama_rigid(A, B):
    """Rigid transform R,t mapping A->B (both Nx3). Returns 4x4."""
    A, B = np.asarray(A), np.asarray(B)
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = cb - R @ ca
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


class Cal:
    def __init__(self):
        self.p = C_PiperInterface_V2(CAN); self.p.ConnectPort(); time.sleep(1.0)
        self._enable()
        # enable the gripper (separate from the arm joints: clear-error then enable)
        self.p.GripperCtrl(0, 1000, 0x02, 0)
        time.sleep(0.1)
        self.p.GripperCtrl(0, 1000, 0x01, 0)
        # camera: aligned depth+color
        free_camera()  # kill any leftover process holding the D405
        self.pipe = rs.pipeline(); cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        prof = self.pipe.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()
        for _ in range(15):
            self.pipe.wait_for_frames()
        e = self.pose()
        self.t = list(e)
        self.grip = 60.0
        self.step = 5.0
        self.observe = None
        self.pairs = []           # (P_cam[3], P_base[3])
        self.pending_cam = None
        self.latest = None          # (color_bgr, depth_uint16)
        self._lock = threading.Lock()
        self.run = True
        threading.Thread(target=self._stream, daemon=True).start()
        threading.Thread(target=self._camloop, daemon=True).start()

    def _enable(self):
        d = time.time() + 5
        while time.time() < d:
            self.p.EnablePiper()
            en = self.p.GetArmEnableStatus()
            if en and all(en):
                return
            time.sleep(0.2)
        raise SystemExit("ABORT: arm did not enable (TEACH button OFF?).")

    def pose(self):
        e = self.p.GetArmEndPoseMsgs().end_pose
        return [e.X_axis / F, e.Y_axis / F, e.Z_axis / F,
                e.RX_axis / F, e.RY_axis / F, e.RZ_axis / F]

    def _stream(self):
        while self.run:
            self.p.MotionCtrl_2(0x01, 0x00, SPEED, 0x00)
            x, y, z, rx, ry, rz = (round(v * F) for v in self.t)
            self.p.EndPoseCtrl(x, y, z, rx, ry, rz)
            self.p.GripperCtrl(round(self.grip * F), 1000, 0x01, 0)
            time.sleep(0.05)

    def _camloop(self):
        """Single camera consumer: keep latest aligned color+depth, write a
        live annotated preview so the view can be relayed while jogging."""
        i = 0
        while self.run:
            try:
                fs = self.align.process(self.pipe.wait_for_frames())
                color = np.asanyarray(fs.get_color_frame().get_data())
                depth = np.asanyarray(fs.get_depth_frame().get_data())  # uint16
            except Exception:
                continue
            with self._lock:
                self.latest = (color, depth)
            i += 1
            if i % 6 == 0:
                self._write_preview(color.copy())

    def _write_preview(self, bgr):
        det = detect_red(bgr)
        if det:
            u, v, _, area = det
            cv2.circle(bgr, (u, v), 10, (0, 255, 0), 2)
            cv2.putText(bgr, f"CAN u={u} v={v} area={int(area)}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(bgr, "no red can detected", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imwrite(PREVIEW, bgr)

    def detect(self):
        with self._lock:
            if self.latest is None:
                print("\n  no camera frame yet"); return
            bgr, depth = self.latest
            bgr, depth = bgr.copy(), depth
        det = detect_red(bgr)
        if det is None:
            print("\n  no red blob found - check lighting / can in view")
            return
        u, v, _, area = det
        # median depth (m) in a 5x5 window around the centroid
        h, w = depth.shape
        win = depth[max(0, v-2):min(h, v+3), max(0, u-2):min(w, u+3)].astype(np.float32) * self.depth_scale
        ds = win[(win > 0.02) & (win < 2.0)]
        if ds.size == 0:
            print("\n  no valid depth at the can - move closer / adjust angle")
            return
        z = float(np.median(ds))
        P = rs.rs2_deproject_pixel_to_point(self.intr, [u, v], z)  # meters, camera frame
        self.pending_cam = P
        cv2.circle(bgr, (u, v), 8, (0, 255, 0), 2)
        cv2.putText(bgr, f"({P[0]:.3f},{P[1]:.3f},{P[2]:.3f})m", (u + 10, v),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(SNAP, bgr)
        print(f"\n  DETECT: px=({u},{v}) area={area:.0f}  P_cam={[round(c,3) for c in P]} m  -> {SNAP}")
        print("  now jog the gripper to TOUCH that spot, then press T")

    def touch(self):
        if self.pending_cam is None:
            print("\n  press D to detect first"); return
        base = self.pose()[:3]  # EE x,y,z in mm
        base_m = [b / 1000.0 for b in base]  # -> meters
        self.pairs.append((list(self.pending_cam), base_m))
        print(f"\n  TOUCH: P_base={[round(b,3) for b in base_m]} m   (pair #{len(self.pairs)})")
        self.pending_cam = None

    def solve(self):
        if len(self.pairs) < 4:
            print(f"\n  need >=4 pairs, have {len(self.pairs)}"); return
        A = [c for c, _ in self.pairs]; B = [b for _, b in self.pairs]
        T = umeyama_rigid(A, B)
        res = [np.linalg.norm(T[:3, :3] @ np.array(a) + T[:3, 3] - np.array(b)) for a, b in self.pairs]
        json.dump({"T_cam_to_base": T.tolist(),
                   "observe_pose": self.observe,
                   "intrinsics": {"fx": self.intr.fx, "fy": self.intr.fy,
                                  "ppx": self.intr.ppx, "ppy": self.intr.ppy,
                                  "width": self.intr.width, "height": self.intr.height},
                   "n_pairs": len(self.pairs),
                   "residuals_m": res}, open(OUT, "w"), indent=2)
        print(f"\n  SOLVED with {len(self.pairs)} pairs. mean residual = {np.mean(res)*1000:.1f} mm")
        print(f"  saved -> {OUT}")

    def home(self):
        """Stop pose streaming and return the arm to zero (joints), then stop camera."""
        self.run = False          # stop the EndPose streaming thread
        time.sleep(0.2)
        print("\n  returning to zero...")
        for _ in range(50):       # stream MOVE_J to latch (single cmds don't)
            self.p.MotionCtrl_2(0x01, 0x01, 30, 0x00)
            self.p.JointCtrl(0, 0, 0, 0, 0, 0)
            time.sleep(0.15)
            s = self.p.GetArmJointMsgs().joint_state
            if max(abs(getattr(s, f"joint_{i+1}") / 1000.0) for i in range(6)) < 0.6:
                break
        try:
            self.pipe.stop()
        except Exception:
            pass
        print("  at zero. camera released.")


def getkey():
    fd = sys.stdin.fileno(); old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd); return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    print(__doc__)
    c = Cal()
    trans = {"w": (0, 1), "s": (0, -1), "a": (1, -1), "d": (1, 1), "r": (2, 1), "f": (2, -1)}
    rot = {"i": (3, 1), "k": (3, -1), "o": (4, 1), "l": (4, -1), "n": (5, 1), "m": (5, -1)}
    print("ready.\n")
    while True:
        try:
            k = getkey()
        except KeyboardInterrupt:
            print("\n interrupted - homing.")
            break
        if k == "Q":
            break
        elif k in trans:
            ax, s = trans[k]; c.t[ax] += s * c.step
        elif k in rot:
            ax, s = rot[k]; c.t[ax] += s * 2.0
        elif k == "[":
            c.grip = max(0.0, c.grip - 3.0)
        elif k == "]":
            c.grip = min(70.0, c.grip + 3.0)
        elif k in ("+", "="):
            c.step = min(20.0, c.step + 1.0)
        elif k == "-":
            c.step = max(1.0, c.step - 1.0)
        elif k == "V":
            c.observe = c.pose(); print(f"\n  OBSERVE pose saved: {[round(v,1) for v in c.observe]}")
        elif k == "G":
            if c.observe:
                c.t = list(c.observe); print("\n  going to observe pose...")
        elif k == "D":
            c.detect()
        elif k == "T":
            c.touch()
        elif k == "L":
            print("\n pairs:"); [print("  ", i, [round(x,3) for x in a], "->", [round(x,3) for x in b]) for i,(a,b) in enumerate(c.pairs)]
        elif k == "S":
            c.solve()
        elif k == "P":
            print(f"\n pose={[round(v,1) for v in c.pose()]} grip={c.grip:.0f}")
        print(f"  X{c.t[0]:6.0f} Y{c.t[1]:6.0f} Z{c.t[2]:6.0f} RX{c.t[3]:5.0f} RY{c.t[4]:5.0f} RZ{c.t[5]:5.0f} grip{c.grip:3.0f} step{c.step:.0f} pairs{len(c.pairs)}  ", end="\r", flush=True)
    c.home()   # return to zero + release camera, on normal quit or Ctrl-C
    print("\ndone.")


if __name__ == "__main__":
    main()
