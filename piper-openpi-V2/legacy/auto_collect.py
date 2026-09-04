#!/usr/bin/env python3
"""Self-resetting demonstration collector for the Piper pick-and-place-back task.

Each recorded EPISODE is the task: approach the can, grasp it, lift, put it back
in the same spot, retreat. Between episodes an UNRECORDED repositioning move
carries the can somewhere new, so across episodes the can appears all over the
workspace without anyone touching it.

No vision in the control loop: the robot PLACES the can, so it knows where the
can is by bookkeeping. The cameras record training data and verify.

Motion is JOINT space via joint_planner (J1 aims, J2/J3 set reach and height,
wrist frozen camera-up) and every move goes through piper_safe's table-floor
guard. Cartesian EndPoseCtrl is deliberately NOT used: it silently refuses many
targets, and it re-solves the wrist roll, which flips the wrist camera
underneath the gripper (risking the flange) and overheats J6.

Records at 15 Hz per frame:
    top_image    Realtek static  -> base_0_rgb
    wrist_image  D405            -> left_wrist_0_rgb
    side_image   HP top-down     -> right_wrist_0_rgb
    state        [x,y,z, rx,ry,rz, gripper_m]   (+ joints)

Usage (in ~/pi05-venv):
    python auto_collect.py --episodes 1
    python auto_collect.py --episodes 20 --seed 1
Watch: http://<host>:8092/
"""
import argparse
import json
import os
import random
import signal
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

import piper_safe as S
from joint_planner import JointPlanner, R_MIN, R_MAX

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw", "pick_soda_can_v3")
PORT = 8092

IMG = 224

# Workspace in POLAR form, because reach is an annulus: r is limited by the
# sampled joint table, bearing by what the static camera can see.
R_LO, R_HI = 245.0, 355.0
BEAR_LO, BEAR_HI = -33.0, 38.0        # degrees

Z_GRASP = 112.0                        # above the joint table's 108 floor
Z_APPROACH = 215.0
Z_START_LO, Z_START_HI = 200.0, 250.0
Z_RECOVER_LO, Z_RECOVER_HI = 255.0, 295.0

GRIP_OPEN_UM = 70000
GRIP_CLOSE_UM = 40000
GRIP_EFFORT = 1500
HELD_MIN_MM, HELD_MAX_MM = 52.0, 69.0
GRIP_OPEN_M, GRIP_CLOSED_M = 0.070, 0.0

CAMS = [("top", "/dev/video6", 2592, 1944),
        ("side", "/dev/video8", 1280, 720)]

_stop = threading.Event()


# ---------------------------------------------------------------- cameras
class CameraHub:
    def __init__(self):
        self.lock = threading.Lock()
        self.frames = {}
        for name, dev, w, h in CAMS:
            threading.Thread(target=self._uvc, args=(name, dev, w, h), daemon=True).start()
        threading.Thread(target=self._rs, daemon=True).start()

    def _uvc(self, name, dev, w, h):
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        while not _stop.is_set():
            ok, f = cap.read()
            if ok:
                with self.lock:
                    self.frames[name] = f
            else:
                time.sleep(0.05)
        cap.release()

    def _rs(self):
        import pyrealsense2 as rs
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipe.start(cfg)
        while not _stop.is_set():
            try:
                fr = pipe.wait_for_frames(2000)
                with self.lock:
                    self.frames["wrist"] = np.asanyarray(fr.get_color_frame().get_data())
            except Exception:
                time.sleep(0.1)
        pipe.stop()

    def get(self, name):
        with self.lock:
            f = self.frames.get(name)
        return None if f is None else f.copy()

    def ready(self, timeout=20.0):
        need = {"top", "side", "wrist"}
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                if need <= set(self.frames):
                    return True
            time.sleep(0.2)
        with self.lock:
            raise SystemExit(f"ABORT: no frames from {need - set(self.frames)}")


def resize_with_pad(bgr, size=IMG):
    h, w = bgr.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    sq = cv2.copyMakeBorder(bgr, top, s - h - top, left, s - w - left,
                            cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- monitor
class Monitor:
    def __init__(self, hub):
        self.hub = hub
        self.text = "starting"

    def panel(self):
        tiles = []
        for name in ("top", "wrist", "side"):
            f = self.hub.get(name)
            if f is None:
                t = np.zeros((360, 480, 3), np.uint8)
                cv2.putText(t, f"{name}: no frame", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            else:
                t = cv2.resize(f, (480, 360))
                cv2.rectangle(t, (0, 0), (480, 22), (0, 0, 0), -1)
                cv2.putText(t, name, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            tiles.append(t)
        strip = np.hstack(tiles)
        bar = np.zeros((46, strip.shape[1], 3), np.uint8)
        cv2.putText(bar, self.text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([strip, bar])


def start_monitor(mon):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while not _stop.is_set():
                        ok, j = cv2.imencode(".jpg", mon.panel(), [cv2.IMWRITE_JPEG_QUALITY, 75])
                        if ok:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                                             + j.tobytes() + b"\r\n")
                        time.sleep(0.1)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body style='margin:0;background:#111'>"
                             b"<img src='/stream' style='width:100%'></body></html>")

    srv = ThreadingHTTPServer(("0.0.0.0", PORT), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close()
    except Exception:
        ip = "127.0.0.1"
    print(f"\n*** watch: http://localhost:{PORT}/   http://{ip}:{PORT}/ ***\n")


# ---------------------------------------------------------------- recording
class Recorder:
    def __init__(self, hub, ep_dir):
        self.hub, self.dir = hub, ep_dir
        os.makedirs(os.path.join(ep_dir, "frames"), exist_ok=True)
        self.rows, self.n = [], 0
        self.grip_state = GRIP_OPEN_M

    def make_cb(self, p):
        def cb():
            imgs = {k: self.hub.get(k) for k in ("top", "wrist", "side")}
            if any(v is None for v in imgs.values()):
                return
            for k, v in imgs.items():
                cv2.imwrite(os.path.join(self.dir, "frames", f"{k}_{self.n:06d}.jpg"),
                            resize_with_pad(v), [cv2.IMWRITE_JPEG_QUALITY, 92])
            self.rows.append({
                "state": [float(v) for v in S.pose(p)] + [float(self.grip_state)],
                "joints": [float(v) for v in S.joints(p)],
                "grip_mm": float(S.grip_mm(p)),
            })
            self.n += 1
        return cb

    def save(self, meta):
        with open(os.path.join(self.dir, "obs.jsonl"), "w") as fh:
            for r in self.rows:
                fh.write(json.dumps(r) + "\n")
        json.dump(meta, open(os.path.join(self.dir, "meta.json"), "w"), indent=1)


# ---------------------------------------------------------------- helpers
def polar(rng):
    r = rng.uniform(R_LO, R_HI)
    b = np.radians(rng.uniform(BEAR_LO, BEAR_HI))
    return float(r * np.cos(b)), float(r * np.sin(b))


def held(p):
    return HELD_MIN_MM <= S.grip_mm(p) <= HELD_MAX_MM


# ---------------------------------------------------------------- episode
def run_episode(p, pl, mon, rec, can_xy, rng, recovery, tag):
    cx, cy = can_xy
    cb = rec.make_cb(p)
    j_above = pl.joints_for(cx, cy, Z_APPROACH)
    j_grasp = pl.joints_for(cx, cy, Z_GRASP)

    sx, sy = polar(rng)
    sz = rng.uniform(*((Z_RECOVER_LO, Z_RECOVER_HI) if recovery else (Z_START_LO, Z_START_HI)))
    j_start = pl.joints_for(sx, sy, sz)

    mon.text = f"{tag} start pose{' [recovery]' if recovery else ''}"
    if not S.move_joints(p, j_start, 3.0, GRIP_OPEN_UM):
        return False, {"abort": "floor guard during start move"}
    S.hold_joints(p, j_start, 0.4, GRIP_OPEN_UM)

    mon.text = f"{tag} approach can ({cx:.0f},{cy:.0f})"
    if not S.move_joints(p, j_above, 2.5, GRIP_OPEN_UM, on_frame=cb):
        return False, {"abort": "floor guard during approach"}

    mon.text = f"{tag} descend"
    if not S.move_joints(p, j_grasp, 2.0, GRIP_OPEN_UM, on_frame=cb):
        return False, {"abort": "floor guard during descent"}

    mon.text = f"{tag} closing"
    rec.grip_state = GRIP_CLOSED_M
    S.hold_joints(p, j_grasp, 1.3, GRIP_CLOSE_UM, on_frame=cb)
    after_close = S.grip_mm(p)

    mon.text = f"{tag} lift (grip {after_close:.0f}mm)"
    S.move_joints(p, j_above, 2.0, GRIP_CLOSE_UM, on_frame=cb)
    S.hold_joints(p, j_above, 0.6, GRIP_CLOSE_UM, on_frame=cb)
    after_lift = S.grip_mm(p)

    mon.text = f"{tag} place back"
    S.move_joints(p, j_grasp, 2.0, GRIP_CLOSE_UM, on_frame=cb)
    rec.grip_state = GRIP_OPEN_M
    S.hold_joints(p, j_grasp, 0.9, GRIP_OPEN_UM, on_frame=cb)

    mon.text = f"{tag} retreat"
    S.move_joints(p, j_above, 1.8, GRIP_OPEN_UM, on_frame=cb)

    ok = HELD_MIN_MM <= after_lift <= HELD_MAX_MM
    return ok, {"grip_after_close_mm": after_close, "grip_after_lift_mm": after_lift,
                "can_xy": [cx, cy], "recovery_start": bool(recovery),
                "start_xyz": [sx, sy, sz]}


def reposition(p, pl, mon, rng, can_xy):
    """UNRECORDED: carry the can to a new spot and retreat clear."""
    cx, cy = can_xy
    nx, ny = cx, cy
    for _ in range(40):
        nx, ny = polar(rng)
        if np.hypot(nx - cx, ny - cy) > 70:
            break
    mon.text = f"reposition ({cx:.0f},{cy:.0f}) -> ({nx:.0f},{ny:.0f})"
    S.move_joints(p, pl.joints_for(cx, cy, Z_APPROACH), 2.2, GRIP_OPEN_UM)
    S.move_joints(p, pl.joints_for(cx, cy, Z_GRASP), 1.8, GRIP_OPEN_UM)
    S.hold_joints(p, pl.joints_for(cx, cy, Z_GRASP), 1.3, GRIP_CLOSE_UM)
    if not held(p):
        return None
    S.move_joints(p, pl.joints_for(cx, cy, Z_APPROACH), 1.8, GRIP_CLOSE_UM)
    S.move_joints(p, pl.joints_for(nx, ny, Z_APPROACH), 3.0, GRIP_CLOSE_UM)
    S.move_joints(p, pl.joints_for(nx, ny, Z_GRASP), 1.8, GRIP_CLOSE_UM)
    S.hold_joints(p, pl.joints_for(nx, ny, Z_GRASP), 0.9, GRIP_OPEN_UM)
    S.move_joints(p, pl.joints_for(nx, ny, Z_APPROACH), 1.8, GRIP_OPEN_UM)
    return (nx, ny)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--recovery-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--can-xy", type=float, nargs=2, default=None)
    ap.add_argument("--load-secs", type=int, default=25)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pl = JointPlanner()
    os.makedirs(RAW, exist_ok=True)

    if args.dry_run:
        print(f"reach r {R_MIN:.0f}..{R_MAX:.0f}; sampling r {R_LO:.0f}..{R_HI:.0f}, "
              f"bearing {BEAR_LO}..{BEAR_HI} deg; z_grasp {Z_GRASP}")
        for i in range(args.episodes):
            recov = rng.random() < args.recovery_frac
            x, y = polar(rng)
            print(f"  ep {i}: can ({x:6.0f},{y:6.0f})  grasp joints "
                  f"{[round(v,1) for v in pl.joints_for(x, y, Z_GRASP)]}"
                  f"{'  [recovery]' if recov else ''}")
        return

    hub = CameraHub()
    hub.ready()
    mon = Monitor(hub)
    start_monitor(mon)

    p = S.connect()
    signal.signal(signal.SIGINT, lambda *_: _stop.set())

    mon.text = "homing"
    S.go_home(p)

    can_xy = tuple(args.can_xy) if args.can_xy else polar(rng)

    # Hold the jaws OPEN throughout the countdown. The gripper only actuates
    # while CAN-control mode is held and the command is streamed, so a plain
    # sleep here leaves it wherever the last run left it - closed, with nowhere
    # to put the can.
    hold_j = S.joints(p)
    for t in range(args.load_secs, 0, -1):
        mon.text = f"PLACE THE CAN IN THE JAWS - closing in {t}s (grip {S.grip_mm(p):.0f}mm)"
        print(f"\rPLACE THE CAN between the open jaws -- closing in {t:2d}s "
              f"(grip {S.grip_mm(p):.0f}mm) ", end="", flush=True)
        S.hold_joints(p, hold_j, 1.0, GRIP_OPEN_UM, guard=False)
    print()
    S.hold_joints(p, hold_j, 1.5, GRIP_CLOSE_UM, guard=False)
    if not held(p):
        raise SystemExit(f"ABORT: gripper reads {S.grip_mm(p):.0f}mm - can not held")

    mon.text = f"placing can at ({can_xy[0]:.0f},{can_xy[1]:.0f})"
    S.move_joints(p, pl.joints_for(*can_xy, Z_APPROACH), 3.5, GRIP_CLOSE_UM)
    S.move_joints(p, pl.joints_for(*can_xy, Z_GRASP), 2.0, GRIP_CLOSE_UM)
    S.hold_joints(p, pl.joints_for(*can_xy, Z_GRASP), 1.0, GRIP_OPEN_UM)
    S.move_joints(p, pl.joints_for(*can_xy, Z_APPROACH), 1.8, GRIP_OPEN_UM)

    existing = len([d for d in os.listdir(RAW) if d.startswith("episode_")])
    good = bad = 0
    for i in range(args.episodes):
        if _stop.is_set():
            break
        ok_j6, temp = S.j6_health(p)
        if not ok_j6 and not S.j6_recover(p):
            print("  ! J6 will not recover - stopping")
            break
        while temp >= S.J6_TEMP_PAUSE and not _stop.is_set():
            mon.text = f"cooling J6 {temp}C"
            time.sleep(5)
            _, temp = S.j6_health(p)

        idx = existing + i
        rec = Recorder(hub, os.path.join(RAW, f"episode_{idx:06d}"))
        recov = rng.random() < args.recovery_frac
        ok, meta = run_episode(p, pl, mon, rec, can_xy, rng, recov, f"ep{idx:04d}")
        meta.update(ok=bool(ok), frames=rec.n, j6_temp_c=int(S.j6_health(p)[1]))
        rec.save(meta)
        good, bad = (good + 1, bad) if ok else (good, bad + 1)
        print(f"ep {idx:04d}: {'OK ' if ok else 'BAD'} frames={rec.n} "
              f"grip_lift={meta.get('grip_after_lift_mm', -1):.0f}mm "
              f"can=({can_xy[0]:.0f},{can_xy[1]:.0f}) j6={meta['j6_temp_c']}C"
              f"{'  [recovery]' if recov else ''}"
              f"{('  ABORT: ' + meta['abort']) if 'abort' in meta else ''}")

        if i < args.episodes - 1:
            nxt = reposition(p, pl, mon, rng, can_xy)
            if nxt is None:
                print("  ! lost the can during repositioning - stopping")
                break
            can_xy = nxt

    mon.text = f"done: {good} good / {bad} bad"
    print(f"\n{good} good, {bad} bad -> {RAW}")
    S.go_home(p)
    _stop.set()
    time.sleep(0.5)


if __name__ == "__main__":
    main()
