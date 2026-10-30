#!/usr/bin/env python3
"""Live MJPEG preview of every attached USB camera, auto-detected.

Serves a single composited MJPEG stream (a grid) at http://localhost:8090/ .
Cameras are discovered at startup from /sys/class/video4linux, so it adapts
to whatever is plugged in and to node renumbering across reboots/replugs.
An Intel RealSense (if present) is added via its SDK as color + depth panels.

Intended as a quick sanity check on camera framing/exposure before
collecting data or fine-tuning a VLA (openpi pi0.5). No arm/CAN activity.

Run:  python3 useful-piper-scripts/dual_camera_stream.py
Then open http://localhost:8090/ in a browser.
"""
import glob
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except Exception:
    rs = None

PORT = 8090
PANEL_H = 480
PANEL_W = 640

# Shared latest frames, keyed by panel id. PANEL_SPECS is filled at startup.
_lock = threading.Lock()
_frames = {}
_stop = threading.Event()
PANEL_SPECS = []  # list of (key, label)


def _label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 1, cv2.LINE_AA)
    return img


def _placeholder(text, color=(0, 0, 255)):
    img = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    cv2.putText(img, text, (20, PANEL_H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2, cv2.LINE_AA)
    return img


def discover_uvc_cameras():
    """Return [(video_index, product_name), ...] — one capture node per
    physical USB camera. RealSense nodes are skipped (handled via the SDK)."""
    groups = {}  # usb device path -> {product, nodes:[int,...]}
    for nodedir in glob.glob("/sys/class/video4linux/video*"):
        try:
            num = int(nodedir.rsplit("video", 1)[-1])
        except ValueError:
            continue
        real = os.path.realpath(os.path.join(nodedir, "device"))
        product, usbpath, p = None, None, real
        for _ in range(6):
            if (os.path.exists(os.path.join(p, "idVendor")) and
                    os.path.exists(os.path.join(p, "product"))):
                with open(os.path.join(p, "product")) as fh:
                    product = fh.read().strip()
                usbpath = os.path.basename(p)
                break
            p = os.path.dirname(p)
        if not product or "realsense" in product.lower():
            continue
        g = groups.setdefault(usbpath, {"product": product, "nodes": []})
        g["nodes"].append(num)
    cams = [(min(g["nodes"]), g["product"]) for g in groups.values()]
    cams.sort()
    return cams


def realsense_present():
    if rs is None:
        return False
    try:
        return len(rs.context().query_devices()) > 0
    except Exception:
        return False


def uvc_worker(index, key):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while not _stop.is_set():
        ok, frame = cap.read()
        if ok:
            frame = cv2.resize(frame, (PANEL_W, PANEL_H))
            with _lock:
                _frames[key] = frame
        else:
            time.sleep(0.05)
    cap.release()


def rs_worker():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    try:
        while not _stop.is_set():
            frames = pipeline.wait_for_frames()
            c = frames.get_color_frame()
            d = frames.get_depth_frame()
            if c:
                ci = cv2.resize(np.asanyarray(c.get_data()), (PANEL_W, PANEL_H))
                with _lock:
                    _frames["rs_color"] = ci
            if d:
                di = np.asanyarray(d.get_data())
                dcm = cv2.applyColorMap(
                    cv2.convertScaleAbs(di, alpha=0.03), cv2.COLORMAP_JET)
                dcm = cv2.resize(dcm, (PANEL_W, PANEL_H))
                with _lock:
                    _frames["rs_depth"] = dcm
    finally:
        pipeline.stop()


def _panel(key, label):
    with _lock:
        frame = _frames.get(key)
    if frame is not None:
        return _label(frame.copy(), label)
    return _placeholder(f"{label}: no frame")


def compose():
    panels = [_panel(key, label) for key, label in PANEL_SPECS]
    if not panels:
        return _placeholder("no cameras detected")
    ncols = 1 if len(panels) == 1 else 2
    rows = []
    for i in range(0, len(panels), ncols):
        row = panels[i:i + ncols]
        while len(row) < ncols:
            row.append(_placeholder("(empty)", color=(80, 80, 80)))
        rows.append(np.hstack(row))
    return np.vstack(rows)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = (b"<html><body style='margin:0;background:#111'>"
                    b"<img src='/stream' style='width:100%'></body></html>")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while not _stop.is_set():
                    canvas = compose()
                    ok, jpg = cv2.imencode(".jpg", canvas,
                                           [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not ok:
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                    self.wfile.write(jpg.tobytes())
                    self.wfile.write(b"\r\n")
                    time.sleep(1 / 15)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        self.send_error(404)


def main():
    cams = discover_uvc_cameras()
    print(f"Discovered {len(cams)} USB camera(s):")
    for idx, product in cams:
        key = f"uvc{idx}"
        PANEL_SPECS.append((key, f"{product} (/dev/video{idx})"))
        print(f"  /dev/video{idx}: {product}")
        threading.Thread(target=uvc_worker, args=(idx, key),
                         daemon=True).start()

    if realsense_present():
        print("  RealSense detected -> adding color + depth panels")
        PANEL_SPECS.append(("rs_color", "RealSense D405 color"))
        PANEL_SPECS.append(("rs_depth", "RealSense D405 depth"))
        threading.Thread(target=rs_worker, daemon=True).start()

    time.sleep(2)  # let cameras warm up
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Camera stream at http://localhost:{PORT}/  ({len(PANEL_SPECS)} panels)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _stop.set()


if __name__ == "__main__":
    main()
