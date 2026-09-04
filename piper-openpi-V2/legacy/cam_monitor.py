#!/usr/bin/env python3
"""Live MJPEG view of all three cameras with red-can detection overlay.

  [ Realtek static (base_0_rgb) ] [ HP top-down (right_wrist_0_rgb) ]
  [ D405 wrist (left_wrist_0_rgb)] [ status panel                   ]

Each panel marks the detected can and prints its pixel centroid; the Realtek
panel additionally shows the table-XY estimate when overhead_calib.json exists.

Run in ~/pi05-venv:  python cam_monitor.py
Then open http://localhost:8091/  (or http://<lan-ip>:8091/)
"""
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

PORT = 8091
PANEL_W, PANEL_H = 640, 480
HERE = os.path.dirname(os.path.abspath(__file__))
CALIB = os.path.join(HERE, "overhead_calib.json")

CAMS = [
    ("realtek", "/dev/video6", 2592, 1944, "Realtek static  -> base_0_rgb"),
    ("hp",      "/dev/video8", 1280, 720,  "HP top-down     -> right_wrist_0_rgb"),
]

_lock = threading.Lock()
_frames = {}          # name -> annotated BGR panel
_info = {}            # name -> dict(pixel=..., xy=...)
_stop = threading.Event()
_H = None             # homography, if calibrated


def detect_red(bgr, min_area=600):
    """Union of HSV and red-dominance - neither alone survives this room's light."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = (cv2.inRange(hsv, (0, 110, 50), (12, 255, 255))
          | cv2.inRange(hsv, (166, 110, 50), (180, 255, 255)))
    f = bgr.astype(np.int16)
    B, G, R = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    m2 = (((R - np.maximum(G, B)) > 35) & (R > 70)).astype(np.uint8) * 255
    m = cv2.bitwise_or(m1, m2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
    if not cnts:
        return None, m
    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)
    (cx, cy), r = cv2.minEnclosingCircle(c)
    return (M["m10"] / M["m00"], M["m01"] / M["m00"], r, cv2.contourArea(c)), m


def annotate(bgr, det, label, scale, name):
    """Draw detection on a PANEL-sized copy. det coords are in full-res pixels."""
    panel = cv2.resize(bgr, (PANEL_W, PANEL_H))
    cv2.rectangle(panel, (0, 0), (PANEL_W, 24), (0, 0, 0), -1)
    cv2.putText(panel, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    info = {"pixel": None, "xy": None}
    if det is not None:
        cx, cy, r, area = det
        px, py = int(cx * scale[0]), int(cy * scale[1])
        pr = max(6, int(r * scale[0]))
        cv2.circle(panel, (px, py), pr, (0, 255, 0), 2)
        cv2.drawMarker(panel, (px, py), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        info["pixel"] = (round(cx), round(cy))
        txt = f"can ({round(cx)},{round(cy)}) a={int(area)}"
        if name == "realtek" and _H is not None:
            xy = cv2.perspectiveTransform(
                np.array([[[cx, cy]]], dtype=np.float64), _H).reshape(2)
            info["xy"] = (round(float(xy[0])), round(float(xy[1])))
            txt += f"  table ({info['xy'][0]},{info['xy'][1]})mm"
        cv2.rectangle(panel, (0, PANEL_H - 26), (PANEL_W, PANEL_H), (0, 0, 0), -1)
        cv2.putText(panel, txt, (6, PANEL_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(panel, (0, PANEL_H - 26), (PANEL_W, PANEL_H), (0, 0, 0), -1)
        cv2.putText(panel, "no can detected", (6, PANEL_H - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1, cv2.LINE_AA)
    with _lock:
        _frames[name] = panel
        _info[name] = info


def uvc_worker(name, dev, w, h, label):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    while not _stop.is_set():
        ok, f = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        # detect on a half-scale copy for speed, then scale coords back
        small = cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2))
        det, _ = detect_red(small, min_area=150)
        if det is not None:
            det = (det[0] * 2, det[1] * 2, det[2] * 2, det[3] * 4)
        annotate(f, det, label, (PANEL_W / f.shape[1], PANEL_H / f.shape[0]), name)
        time.sleep(0.05)
    cap.release()


def rs_worker():
    try:
        import pyrealsense2 as rs
    except ImportError:
        return
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    try:
        pipe.start(cfg)
    except Exception as e:
        with _lock:
            _frames["d405"] = _placeholder(f"D405 unavailable: {str(e)[:40]}")
        return
    while not _stop.is_set():
        try:
            fr = pipe.wait_for_frames(2000)
            bgr = np.asanyarray(fr.get_color_frame().get_data())
        except Exception:
            time.sleep(0.1)
            continue
        det, _ = detect_red(bgr, min_area=300)
        annotate(bgr, det, "D405 wrist      -> left_wrist_0_rgb",
                 (PANEL_W / bgr.shape[1], PANEL_H / bgr.shape[0]), "d405")
        time.sleep(0.05)
    pipe.stop()


def _placeholder(text):
    img = np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
    cv2.putText(img, text, (14, PANEL_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255), 1, cv2.LINE_AA)
    return img


def status_panel():
    img = np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
    y = 30
    cv2.putText(img, "DETECTION STATUS", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    y += 34
    with _lock:
        info = dict(_info)
    for name in ("realtek", "hp", "d405"):
        i = info.get(name)
        if i is None:
            line, col = f"{name:8s} : (no frame)", (120, 120, 120)
        elif i["pixel"] is None:
            line, col = f"{name:8s} : no can", (0, 0, 255)
        else:
            line = f"{name:8s} : px {i['pixel']}"
            if i.get("xy"):
                line += f"  table {i['xy']} mm"
            col = (0, 255, 0)
        cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        y += 28
    y += 12
    cv2.putText(img, "calibration: " + ("loaded" if _H is not None else "none (run calibrate_overhead.py)"),
                (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 0) if _H is not None else (0, 165, 255), 1, cv2.LINE_AA)
    y += 26
    cv2.putText(img, "detector: HSV union red-dominance", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    y += 22
    cv2.putText(img, time.strftime("%H:%M:%S"), (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def compose():
    with _lock:
        r = _frames.get("realtek")
        h = _frames.get("hp")
        d = _frames.get("d405")
    r = r if r is not None else _placeholder("Realtek: no frame")
    h = h if h is not None else _placeholder("HP: no frame")
    d = d if d is not None else _placeholder("D405: no frame")
    return np.vstack([np.hstack([r, h]), np.hstack([d, status_panel()])])


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = (b"<html><head><title>Piper cameras</title></head>"
                    b"<body style='margin:0;background:#111'>"
                    b"<img src='/stream' style='width:100%'></body></html>")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while not _stop.is_set():
                    ok, jpg = cv2.imencode(".jpg", compose(), [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                                         + jpg.tobytes() + b"\r\n")
                    time.sleep(1 / 10)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        self.send_error(404)


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def main():
    global _H
    if os.path.exists(CALIB):
        _H = np.array(json.load(open(CALIB))["H"], dtype=np.float64)
        print("loaded overhead calibration")
    for name, dev, w, h, label in CAMS:
        threading.Thread(target=uvc_worker, args=(name, dev, w, h, label), daemon=True).start()
    threading.Thread(target=rs_worker, daemon=True).start()
    time.sleep(2.5)
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"\n*** camera monitor: http://localhost:{PORT}/   or   http://{lan_ip()}:{PORT}/ ***\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _stop.set()


if __name__ == "__main__":
    main()
