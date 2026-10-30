#!/usr/bin/env python3
"""Live camera view in a browser, without the recorder running.

    python -m tools.camera_server                  # both cameras -> http://<jetson>:8093
    python -m tools.camera_server --guide          # top camera with the placement overlay drawn on it
    python -m tools.camera_server --cams top       # one camera only
    python -m tools.camera_server --port 8100

No ffmpeg: it is not installed here and is not needed. The frames already arrive as numpy arrays through
piper.cameras, so this encodes them straight to MJPEG - the same thing collect/record.py's Monitor serves
on :8092 during a session, just standalone and on a different port so both can run.

It holds the cameras open, so it must be STOPPED before recording: piper.cameras cannot open a device that
is already in use, and collect.teach/record will fail to start while this is running.
"""
import argparse
import json
import os
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from piper import config as C          # noqa: E402

PORT = 8093                            # not MONITOR_PORT: leave :8092 free for the recorder's own UI
PAGE = """<!doctype html><meta charset=utf-8><title>Piper cameras</title>
<style>body{margin:0;background:#111;color:#ddd;font:15px/1.4 sans-serif;display:flex;flex-wrap:wrap;gap:10px;padding:10px}
figure{margin:0}img{background:#222;max-width:96vw}figcaption{color:#888;padding-top:4px}</style>
%s
"""


def lan_ip() -> str:
    """The address the laptop should open - not 127.0.0.1, which is useless from another machine."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 1))          # no traffic sent; just picks the outbound interface
        return s.getsockname()[0]
    except OSError:
        return "<jetson-ip>"
    finally:
        s.close()


def build_page(names):
    return PAGE % "".join(
        f'<figure><img src="/stream/{n}" width="640"><figcaption>{n}</figcaption></figure>' for n in names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", nargs="*", default=list(C.CAMERA_NAMES))
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--fps", type=float, default=10.0, help="stream rate (the cameras run at %d)" % C.CAM_FPS)
    ap.add_argument("--quality", type=int, default=70)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--guide", action="store_true",
                    help="draw the placement area and past grasp positions on the top camera")
    ap.add_argument("--dataset", default=os.path.join(C.DATA_DIR, "piper_block_bowl"))
    args = ap.parse_args()

    overlay = None
    if args.guide:
        from tools.placement_guide import CALIB, annotate, past_grasps
        Ainv = cv2.invertAffineTransform(np.array(json.load(open(CALIB))["A"], np.float64))
        grasps = past_grasps(args.dataset) if os.path.isdir(args.dataset) else []
        print(f"  overlay: placement area + {len(grasps)} past grasps")

        def overlay(name, frame):
            return annotate(frame, Ainv, grasps) if name == "top" else frame

    from piper.cameras import open_cameras
    cams = open_cameras(names=tuple(args.cams))
    page = build_page(args.cams).encode()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            name = self.path[8:] if self.path.startswith("/stream/") else None
            if name in cams:
                cam = cams[name]
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
                self.end_headers()
                try:
                    while True:
                        f = cam.read()
                        if overlay:
                            f = overlay(name, f)
                        h = int(f.shape[0] * args.width / f.shape[1])
                        ok, jpg = cv2.imencode(".jpg", cv2.resize(f, (args.width, h)),
                                               [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
                        time.sleep(1.0 / args.fps)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)

    srv = socketserver.ThreadingTCPServer(("0.0.0.0", args.port), H, bind_and_activate=False)
    srv.allow_reuse_address, srv.daemon_threads = True, True
    srv.server_bind(), srv.server_activate()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"\n  cameras {args.cams} -> http://{lan_ip()}:{args.port}"
          f"   ({args.fps:g} fps, {args.width}px, q{args.quality})")
    print("  STOP THIS before recording - it holds the cameras open.  Ctrl-C to quit\n")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("stopped")
    finally:
        srv.shutdown()
        for c in cams.values():
            c.close()


if __name__ == "__main__":
    main()
