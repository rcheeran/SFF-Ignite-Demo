#!/usr/bin/env python3
"""Drag-teach recorder for openpi pose-6D fine-tuning of the AgileX Piper.

Samples {D405 wrist RGB, end-effector 6D pose, gripper} at a fixed rate while
you physically drag the arm in TEACH mode. Writes raw episodes; the companion
build_pose_lerobot.py converts them into an openpi-format LeRobot dataset.

State recorded per frame (7-D): [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg, grip_m]
  - position from GetArmEndPoseMsgs().end_pose (0.001 mm -> mm)
  - rotation (0.001 deg -> deg)
  - gripper from GetArmGripperMsgs().gripper_state.grippers_angle (raw/1e6 -> m)

Run in ~/venv (piper_sdk + pyrealsense2 + opencv + numpy).

Workflow:
  1. Press the arm's TEACH button (Joint-5) to enter teaching mode.
  2. Per episode: position the arm at the start pose (camera seeing the object),
     press Enter to start, drag through the pick-place, press Enter to stop.
  3. Type 'q' at the prompt to finish. Press the teach button again to exit.

Gripper during teach: the arm is compliant but the gripper is motorized. Use
keys o/c at the start prompt to pre-open/close, OR operate it from the teach
pendant. Whatever the gripper does, its state is recorded. (We verify grasp
control on the first test run.)
"""
import glob
import http.server
import json
import os
import signal
import socket
import socketserver
import sys
import termios
import threading
import time
import tty

import cv2
import numpy as np
import pyrealsense2 as rs
from piper_sdk import C_PiperInterface_V2


def _pids_holding_video(exclude_pid):
    """PIDs (same user) with an open fd on any /dev/video* device."""
    devs = {os.path.realpath(p) for p in glob.glob("/dev/video*")}
    pids = set()
    for fd_dir in glob.glob("/proc/[0-9]*/fd"):
        pid = int(fd_dir.split("/")[2])
        if pid == exclude_pid:
            continue
        try:
            for fd in os.listdir(fd_dir):
                try:
                    if os.path.realpath(os.path.join(fd_dir, fd)) in devs:
                        pids.add(pid)
                        break
                except OSError:
                    continue
        except (OSError, PermissionError):
            continue
    return pids


def free_camera():
    """Kill any leftover process holding the RealSense before we open it."""
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
        for pid in _pids_holding_video(me):  # SIGKILL survivors
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(0.5)

CAN_IFACE = "can_piper"
TASK_TEXT = "pick up the soda can"
SAMPLE_HZ = 15
IMG_W, IMG_H = 640, 480
OUT_DIR = os.path.expanduser("~/Documents/piper-openpi/data/raw/pick_soda_can")
TEACH_MODE = 2  # ctrl_mode value for TEACHING
PREVIEW = os.path.expanduser("~/Documents/piper-openpi/record_preview.jpg")
RED1 = ((0, 110, 70), (10, 255, 255))
RED2 = ((170, 110, 70), (180, 255, 255))


def detect_red(bgr):
    """Return (u, v) centroid of the largest red blob, or None."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(RED1[0]), np.array(RED1[1])) | \
        cv2.inRange(hsv, np.array(RED2[0]), np.array(RED2[1]))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 200:
        return None
    M = cv2.moments(c)
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


def _lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class _MJPEGHandler(http.server.BaseHTTPRequestHandler):
    recorder = None

    def log_message(self, *a):
        pass

    def do_GET(self):
        r = _MJPEGHandler.recorder
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
        while r is not None and r._cam_running:
            with r._lock:
                c = None if r.latest_color is None else r.latest_color.copy()
            if c is None:
                time.sleep(0.05); continue
            ok, jpg = cv2.imencode(".jpg", r._annotate(c))
            if not ok:
                continue
            try:
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
            except Exception:
                break
            time.sleep(0.1)


class _StreamServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


GRIP_OPEN = 0.070    # meters (logged value; gripper is operated manually in teach mode)
GRIP_CLOSED = 0.0


def read_pose(piper):
    """Live end-effector pose (mm/deg). Works in teach mode. The gripper's own
    feedback is frozen in teach mode, so gripper state is logged by keypress."""
    ep = piper.GetArmEndPoseMsgs().end_pose
    return [ep.X_axis / 1000.0, ep.Y_axis / 1000.0, ep.Z_axis / 1000.0,   # mm
            ep.RX_axis / 1000.0, ep.RY_axis / 1000.0, ep.RZ_axis / 1000.0]  # deg


def getkey():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class Recorder:
    def __init__(self):
        self.piper = C_PiperInterface_V2(CAN_IFACE)
        self.piper.ConnectPort()
        time.sleep(1.0)
        free_camera()  # kill any leftover process holding the D405
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, IMG_W, IMG_H, rs.format.bgr8, 30)
        try:
            self.pipe.start(cfg)
        except RuntimeError as e:
            if "busy" in str(e).lower():
                raise SystemExit(
                    "ERROR: the D405 is busy - another process holds it (likely a "
                    "leftover recorder). Close it, e.g.:\n"
                    "    pkill -f record_pose_episodes.py\n"
                    "then re-run.") from None
            raise
        for _ in range(15):
            self.pipe.wait_for_frames()
        self._buf = []
        self._running = False
        self.recording = False
        self.grip_state = GRIP_OPEN
        self.latest_color = None
        self._lock = threading.Lock()
        self._cam_running = True
        threading.Thread(target=self._camloop, daemon=True).start()
        # browser live view (works on the Jetson desktop and from other devices)
        _MJPEGHandler.recorder = self
        try:
            self._httpd = _StreamServer(("0.0.0.0", 8089), _MJPEGHandler)
            threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
            print(f"\n*** LIVE VIEW: open a browser to  http://localhost:8089   "
                  f"(or http://{_lan_ip()}:8089 from another device) ***\n")
        except Exception as e:
            self._httpd = None
            print(f"(web preview unavailable: {e})")

    def _annotate(self, bgr):
        d = detect_red(bgr)
        if d:
            cv2.circle(bgr, d, 10, (0, 255, 0), 2)
            cv2.putText(bgr, "can in view", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(bgr, "no can in view", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        gopen = self.grip_state > 0
        cv2.putText(bgr, f"gripper: {'OPEN' if gopen else 'CLOSED'}", (8, IMG_H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0) if gopen else (0, 0, 255), 2)
        if self.recording:
            cv2.circle(bgr, (IMG_W - 24, 22), 9, (0, 0, 255), -1)
            cv2.putText(bgr, "REC", (IMG_W - 72, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return bgr

    def _camloop(self):
        """Single camera consumer: keep the latest frame, show a live window (if
        a display exists) and always write an annotated preview file."""
        show = True
        while self._cam_running:
            try:
                frames = self.pipe.wait_for_frames()
                color = np.asanyarray(frames.get_color_frame().get_data())
            except Exception:
                continue
            with self._lock:
                self.latest_color = color
            ann = self._annotate(color.copy())
            try:
                cv2.imwrite(PREVIEW, ann)
            except Exception:
                pass
            if show:
                try:
                    cv2.imshow("D405 live - recorder", ann)
                    cv2.waitKey(1)
                except Exception:
                    show = False  # headless: preview file only
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def _sampler(self):
        period = 1.0 / SAMPLE_HZ
        next_t = time.monotonic()
        while self._running:
            with self._lock:
                color = None if self.latest_color is None else self.latest_color.copy()
            if color is None:
                time.sleep(0.005)
                continue
            state = read_pose(self.piper) + [self.grip_state]  # pose (live) + logged gripper
            self._buf.append((time.time(), state, color))
            next_t += period
            time.sleep(max(0.0, next_t - time.monotonic()))

    def wait_for_teach(self):
        print("INFO: press the TEACH button on the arm to enter teaching mode...")
        while self.piper.GetArmStatus().arm_status.ctrl_mode != TEACH_MODE:
            time.sleep(0.05)
        print("INFO: teaching mode active - arm is now draggable.")

    def record_episode(self, ep_index):
        self._buf = []
        self._running = True
        self.recording = True
        self.grip_state = GRIP_OPEN
        th = threading.Thread(target=self._sampler, daemon=True)
        th.start()
        print(f">>> recording episode {ep_index}: drag the demo. "
              "SPACE = toggle gripper (open/closed),  s = STOP")
        while True:
            k = getkey()
            if k in ("s", "\n", "\r"):
                break
            if k == " ":
                self.grip_state = GRIP_CLOSED if self.grip_state > 0 else GRIP_OPEN
                print(f"    gripper logged -> {'CLOSED' if self.grip_state == 0 else 'OPEN'}")
        self._running = False
        self.recording = False
        th.join()
        self._flush(ep_index)

    def _flush(self, ep_index):
        ep_dir = os.path.join(OUT_DIR, f"episode_{ep_index:06d}")
        os.makedirs(os.path.join(ep_dir, "frames"), exist_ok=True)
        rows = []
        for i, (t, state, frame) in enumerate(self._buf):
            cv2.imwrite(os.path.join(ep_dir, "frames", f"frame_{i:06d}.jpg"), frame)
            rows.append({"i": i, "t": t, "state": state})
        with open(os.path.join(ep_dir, "obs.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(ep_dir, "episode.json"), "w") as f:
            json.dump({"episode_index": ep_index, "task": TASK_TEXT,
                       "length": len(rows), "fps": SAMPLE_HZ}, f, indent=2)
        print(f"    saved {len(rows)} frames -> {ep_dir}")

    def close(self):
        self._cam_running = False
        time.sleep(0.3)
        if getattr(self, "_httpd", None):
            try:
                self._httpd.shutdown()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self.pipe.stop()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rec = Recorder()
    rec.wait_for_teach()
    # continue from the highest existing episode index
    existing = [d for d in os.listdir(OUT_DIR) if d.startswith("episode_")]
    ep = max([int(d.split("_")[1]) for d in existing], default=-1) + 1
    try:
        while True:
            cmd = input(f"\nEpisode {ep}: Enter to record, 'q' to finish: ").strip().lower()
            if cmd == "q":
                break
            rec.record_episode(ep)
            ep += 1
    finally:
        rec.close()
    print(f"\nDone. {ep} episode(s) under {OUT_DIR}")
    print("Press the TEACH button again to exit teaching mode.")


if __name__ == "__main__":
    main()
