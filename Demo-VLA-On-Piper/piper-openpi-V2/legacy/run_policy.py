#!/usr/bin/env python3
"""Run the fine-tuned pi0.5 policy on the real Piper: pick up the soda can.

Loop: D405 frame + 10-D pose state -> policy.infer -> chunk of 10 absolute
pose actions -> stream EndPoseCtrl/GripperCtrl at 15 Hz, re-infer each chunk.

Safety:
  - workspace clamped to the training data's q01/q99 box (plus margin)
  - per-step motion clamped (mm/deg per 15 Hz tick)
  - Ctrl+C stops streaming immediately, gripper opens, arm stays put
  - --dry-run: full pipeline but print actions instead of moving

PREREQS: training finished (GPU free), teach button OFF, can is in view,
arm roughly near the demo start area. Run in ~/pi05-venv:
    python run_policy.py --checkpoint ~/Documents/openpi/checkpoints/pi05_piper/piper_pick_v1/4000 [--dry-run]
"""
import argparse
import dataclasses
import glob
import http.server
import os
import signal
import socket
import socketserver
import sys
import threading
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

# ---------------- constants (match training data conventions) ----------------
CAN_IFACE = "can_piper"
PROMPT = "pick up the soda can"
IMG_W, IMG_H = 640, 480
IMG = 224
FPS = 15
DT = 1.0 / FPS
EXEC_STEPS = 8          # execute 8 of the 10-step chunk, then re-infer
GRIP_OPEN = 0.070       # meters (binary convention in the dataset)
GRIP_THRESH = 0.035
GRIP_EFFORT = 1000      # 0.001 N/m; verified sufficient (test_gripper2.py)
SPEED_PERCENT = 30
MAX_STEP_MM = 20.0      # per-tick translation clamp
MAX_STEP_DEG = 10.0     # per-tick rotation clamp
CLAMP_MARGIN_MM = 20.0  # slack beyond the data's q01/q99 box


def resize_with_pad(img, size=IMG):
    h, w = img.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    sq = cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left,
                            cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)


def euler_to_6d(rpy_deg):
    M = Rot.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    return np.concatenate([M[:, 0], M[:, 1]]).astype(np.float32)


def sixd_to_rot(r6):
    """Gram-Schmidt 6D -> Rotation."""
    a1, a2 = np.asarray(r6[:3], dtype=np.float64), np.asarray(r6[3:6], dtype=np.float64)
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return Rot.from_matrix(np.stack([b1, b2, b3], axis=1))


def sixd_to_euler(r6):
    return sixd_to_rot(r6).as_euler("xyz", degrees=True)


def clamp_rotation_step(prev_eul_deg, target_rot, max_deg):
    """Move from prev orientation toward target by at most max_deg (geodesic).

    Equivalent Euler triples encode the same rotation, so the clamp must act in
    SO(3), not on Euler components. Returns the new euler command (deg).
    """
    r_prev = Rot.from_euler("xyz", prev_eul_deg, degrees=True)
    rel = r_prev.inv() * target_rot
    rv = rel.as_rotvec(degrees=True)
    ang = np.linalg.norm(rv)
    if ang > max_deg:
        rel = Rot.from_rotvec(rv / ang * max_deg, degrees=True)
    return (r_prev * rel).as_euler("xyz", degrees=True)


# ---------------- camera (same pipeline as the recorder) ----------------
def _pids_holding_video(skip_pid):
    devs = {os.path.realpath(p) for p in glob.glob("/dev/video*")}
    pids = []
    for pdir in glob.glob("/proc/[0-9]*"):
        pid = int(os.path.basename(pdir))
        if pid == skip_pid:
            continue
        try:
            for fd in os.listdir(pdir + "/fd"):
                if os.path.realpath(f"{pdir}/fd/{fd}") in devs:
                    pids.append(pid)
                    break
        except (PermissionError, FileNotFoundError):
            continue
    return pids


def free_camera():
    me = os.getpid()
    holders = _pids_holding_video(me)
    for pid in holders:
        print(f"INFO: camera held by PID {pid} - stopping it first.")
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    if holders:
        time.sleep(1.2)
        for pid in _pids_holding_video(me):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(0.5)


def open_camera():
    import pyrealsense2 as rs
    free_camera()
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, IMG_W, IMG_H, rs.format.bgr8, 30)
    pipe.start(cfg)
    for _ in range(15):
        pipe.wait_for_frames()

    def _camloop():  # feeds both the policy and the browser stream
        while VIEW.running:
            try:
                frames = pipe.wait_for_frames()
                bgr = np.asanyarray(frames.get_color_frame().get_data())
                with VIEW.lock:
                    VIEW.frame = bgr
            except Exception:
                time.sleep(0.1)
    threading.Thread(target=_camloop, daemon=True).start()
    while VIEW.frame is None:
        time.sleep(0.05)
    return pipe


def grab_rgb224(pipe):
    with VIEW.lock:
        bgr = VIEW.frame.copy()
    return resize_with_pad(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), bgr


# ---------------- browser live view (MJPEG, like the recorder) ----------------
class _View:
    """Shared latest-frame buffer + overlay text, served as MJPEG on :8089."""
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None       # BGR 640x480
        self.text = "starting"
        self.running = True

    def update(self, bgr, text):
        with self.lock:
            self.frame = bgr.copy()
            self.text = text


VIEW = _View()


def _lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class _MJPEGHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path != "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body style='margin:0;background:#111;text-align:center'>"
                             b"<img src='/stream' style='max-width:100%'></body></html>")
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        while VIEW.running:
            with VIEW.lock:
                c = None if VIEW.frame is None else VIEW.frame.copy()
                txt = VIEW.text
            if c is None:
                time.sleep(0.05); continue
            cv2.putText(c, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            ok, jpg = cv2.imencode(".jpg", c)
            if ok:
                try:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
                except Exception:
                    break
            time.sleep(0.1)


class _StreamServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_stream_server():
    try:
        httpd = _StreamServer(("0.0.0.0", 8089), _MJPEGHandler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        print(f"\n*** LIVE VIEW: http://localhost:8089  (or http://{_lan_ip()}:8089 on your phone) ***\n")
        return httpd
    except Exception as e:
        print(f"(live view unavailable: {e})")
        return None


# ---------------- arm ----------------
def connect_arm():
    from piper_sdk import C_PiperInterface_V2
    p = C_PiperInterface_V2(CAN_IFACE)
    p.ConnectPort()
    time.sleep(1.0)
    deadline = time.time() + 5
    while time.time() < deadline:
        p.EnablePiper()
        en = p.GetArmEnableStatus()
        if en and all(en):
            break
        time.sleep(0.2)
    en = p.GetArmEnableStatus()
    if not en or not all(en):
        raise SystemExit("ERROR: could not enable all joints (teach button ON? e-stop?)")
    # J6 (wrist roll) firmware limit [-179, -60] deg: the demo wrist config
    # lives at ~-165, and an unbounded roll once wound the cable, stalled and
    # latched an overcurrent fault. Not persistent across arm power cycles,
    # so re-assert on every connect.
    p.MotorAngleLimitMaxSpdSet(motor_num=6, max_angle_limit=-600, min_angle_limit=-1790, max_joint_spd=0x7FFF)
    # Open the gripper. The command only takes effect while the arm is in
    # CAN-control mode and must be streamed - see send_pose_and_grip below.
    t0 = time.time()
    while time.time() - t0 < 2.0:
        p.MotionCtrl_2(0x01, 0x01, SPEED_PERCENT, 0x00)
        p.GripperCtrl(int(GRIP_OPEN * 1e6), GRIP_EFFORT, 0x01, 0)
        time.sleep(DT)
    return p


def read_pose(p):
    ep = p.GetArmEndPoseMsgs().end_pose
    return [ep.X_axis / 1000.0, ep.Y_axis / 1000.0, ep.Z_axis / 1000.0,
            ep.RX_axis / 1000.0, ep.RY_axis / 1000.0, ep.RZ_axis / 1000.0]


def read_grip_m(p):
    return p.GetArmGripperMsgs().gripper_state.grippers_angle / 1e6


def send_pose_and_grip(p, x, y, z, rx, ry, rz, grip_open):
    """One control tick: CAN-control mode + pose + gripper, together.

    The gripper is re-commanded EVERY tick rather than on transitions: it only
    actuates while the arm is in CAN-control mode, and single shots don't latch
    (that is why v1/v2 deployments logged gripper changes that never happened).
    """
    p.MotionCtrl_2(0x01, 0x00, SPEED_PERCENT, 0x00)
    p.EndPoseCtrl(int(x * 1000), int(y * 1000), int(z * 1000),
                  int(rx * 1000), int(ry * 1000), int(rz * 1000))
    p.GripperCtrl(int((GRIP_OPEN if grip_open else 0.0) * 1e6), GRIP_EFFORT, 0x01, 0)


def send_grip(p, open_, secs=1.5):
    """Standalone gripper command (used at shutdown), streamed in CAN mode."""
    t0 = time.time()
    while time.time() - t0 < secs:
        p.MotionCtrl_2(0x01, 0x01, SPEED_PERCENT, 0x00)
        p.GripperCtrl(int((GRIP_OPEN if open_ else 0.0) * 1e6), GRIP_EFFORT, 0x01, 0)
        time.sleep(DT)


# ---------------- policy ----------------
def load_policy(ckpt_dir):
    import openpi.training.config as _config
    from openpi.policies import policy_config
    cfg = _config.get_config("pi05_piper")
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    return policy_config.create_trained_policy(cfg, ckpt_dir, pytorch_device="cuda")


def load_clamps(ckpt_dir):
    import json
    ns = json.load(open(os.path.join(ckpt_dir, "assets/piper/pick_soda_can/norm_stats.json")))
    q01 = np.array(ns["norm_stats"]["state"]["q01"][:3]) - CLAMP_MARGIN_MM
    q99 = np.array(ns["norm_stats"]["state"]["q99"][:3]) + CLAMP_MARGIN_MM
    return q01, q99


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.expanduser(
        "~/Documents/openpi/checkpoints/pi05_piper/piper_pick_v1/4000"))
    ap.add_argument("--dry-run", action="store_true", help="infer + print, no motion")
    ap.add_argument("--max-chunks", type=int, default=60, help="stop after N chunks (~N*0.53s)")
    ap.add_argument("--save-frames", default=None, help="dir to save the camera frame per chunk")
    args = ap.parse_args()
    if args.save_frames:
        os.makedirs(args.save_frames, exist_ok=True)

    print(f"loading policy from {args.checkpoint} ...")
    policy = load_policy(args.checkpoint)
    lo, hi = load_clamps(args.checkpoint)
    print(f"workspace clamp (mm): x[{lo[0]:.0f},{hi[0]:.0f}] y[{lo[1]:.0f},{hi[1]:.0f}] z[{lo[2]:.0f},{hi[2]:.0f}]")

    pipe = open_camera()
    httpd = start_stream_server()
    piper = None if args.dry_run else connect_arm()
    reader = piper
    if args.dry_run:
        # read-only CAN connection: no EnablePiper, no commands - just live pose
        try:
            from piper_sdk import C_PiperInterface_V2
            reader = C_PiperInterface_V2(CAN_IFACE)
            reader.ConnectPort()
            time.sleep(1.0)
            if all(abs(v) < 1e-6 for v in read_pose(reader)):
                print("WARN: pose feedback all-zero (arm off?), using canned start state")
                reader = None
            else:
                print(f"dry-run: using REAL arm state {np.round(read_pose(reader),1)}")
        except Exception as e:
            print(f"WARN: no arm feedback ({e}), using canned start state")
            reader = None

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    # warm up inference (first call compiles/allocates)
    img, _ = grab_rgb224(pipe)
    state0 = ([*read_pose(reader), read_grip_m(reader)] if reader
              else [200.0, 0.0, 250.0, 100.0, 70.0, 100.0, GRIP_OPEN])
    s10 = np.concatenate([state0[:3], euler_to_6d(state0[3:6]), [state0[6]]]).astype(np.float32)
    t = time.time()
    policy.infer({"observation/wrist_image": img, "observation/state": s10, "prompt": PROMPT})
    print(f"warmup inference: {time.time()-t:.2f}s")

    grip_open = state0[6] > GRIP_THRESH
    prev_cmd = np.array(state0[:6], dtype=np.float64)

    for chunk in range(args.max_chunks):
        if stop["flag"]:
            break
        img, bgr = grab_rgb224(pipe)
        if args.save_frames:
            cv2.imwrite(os.path.join(args.save_frames, f"chunk_{chunk:03d}.jpg"), bgr)
        if piper:
            pose = read_pose(piper)
            grip = read_grip_m(piper)
        elif reader:
            # dry-run with live feedback: anchor every chunk at the REAL pose
            pose = read_pose(reader)
            grip = read_grip_m(reader)
        else:
            # dry-run without feedback: roll the predicted trajectory forward open-loop
            pose, grip = list(prev_cmd), (GRIP_OPEN if grip_open else 0.0)
        s10 = np.concatenate([pose[:3], euler_to_6d(pose[3:6]), [grip]]).astype(np.float32)

        t = time.time()
        actions = policy.infer(
            {"observation/wrist_image": img, "observation/state": s10, "prompt": PROMPT})["actions"]
        infer_s = time.time() - t

        for a in np.asarray(actions)[:EXEC_STEPS]:
            if stop["flag"]:
                break
            xyz = np.clip(a[:3], lo, hi)
            xyz = prev_cmd[:3] + np.clip(xyz - prev_cmd[:3], -MAX_STEP_MM, MAX_STEP_MM)
            eul = clamp_rotation_step(prev_cmd[3:6], sixd_to_rot(a[3:9]), MAX_STEP_DEG)
            want_open = a[9] > GRIP_THRESH
            if args.dry_run:
                print(f"  -> xyz={np.round(xyz,1)} rpy={np.round(eul,1)} grip={'OPEN' if want_open else 'CLOSE'}")
            else:
                send_pose_and_grip(piper, *xyz, *eul, want_open)
                if want_open != grip_open:
                    print(f"  gripper -> {'OPEN' if want_open else 'CLOSE'} "
                          f"(now {read_grip_m(piper)*1000:.0f}mm)")
            grip_open = want_open
            prev_cmd = np.concatenate([xyz, eul])
            time.sleep(DT)
        VIEW.text = (f"chunk {chunk+1}/{args.max_chunks}  pos={np.round(prev_cmd[:3],0)}  "
                     f"grip={'OPEN' if grip_open else 'CLOSED'}" + ("  [DRY]" if args.dry_run else ""))
        print(f"chunk {chunk+1}: infer {infer_s:.2f}s  pos={np.round(prev_cmd[:3],0)}  grip={'OPEN' if grip_open else 'CLOSED'}")

    print("stopping: opening gripper, leaving arm in place.")
    if piper:
        send_grip(piper, True)
    VIEW.running = False
    pipe.stop()


if __name__ == "__main__":
    main()
