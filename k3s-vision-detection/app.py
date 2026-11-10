#!/usr/bin/env python3
"""Internal RGB detection API; no camera, depth, or robot access."""

from __future__ import annotations

import base64
import binascii
import json
import math
import os
import struct
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import cv2
import numpy as np

from vision import FoundryYoloDetector, PROMPT_ALIASES, VisionError, normalize_prompt

MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 1920 * 1080


class RequestError(VisionError):
    """An invalid detection request."""


class BusyError(VisionError):
    """Another inference request is already in progress."""


class DetectionApplication:
    def __init__(self) -> None:
        self.model_name = os.environ.get("DETECTION_MODEL", "candy-yolo11s")
        self.detector = FoundryYoloDetector(self.model_name)
        self.inference_lock = threading.Lock()

    def health(self) -> dict[str, Any]:
        self.detector.check()
        return {"status": "ok", "model": self.model_name, "cuda": True}

    def detect(self, document: Any) -> dict[str, Any]:
        if not isinstance(document, dict):
            raise RequestError("request must be a JSON object")
        prompt = document.get("prompt")
        if not isinstance(prompt, str) or normalize_prompt(prompt) not in PROMPT_ALIASES:
            raise RequestError("prompt must name a supported candy class")
        try:
            confidence = float(document["confidence"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RequestError("confidence must be a number between zero and one") from exc
        if not math.isfinite(confidence) or not 0 < confidence < 1:
            raise RequestError("confidence must be between zero and one")
        encoded = document.get("image_png_base64")
        if not isinstance(encoded, str):
            raise RequestError("image_png_base64 must be a base64 PNG string")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RequestError("invalid base64 image") from exc
        if (
            len(data) < 24
            or not data.startswith(b"\x89PNG\r\n\x1a\n")
            or data[12:16] != b"IHDR"
        ):
            raise RequestError("image must be losslessly encoded as PNG")
        width, height = struct.unpack(">II", data[16:24])
        if not width or not height or width * height > MAX_IMAGE_PIXELS:
            raise RequestError("PNG dimensions must be positive and at most 1920x1080 pixels")
        try:
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        except cv2.error as exc:
            raise RequestError(f"could not decode PNG image: {exc}") from exc
        if frame is None:
            raise RequestError("could not decode PNG image")
        if not self.inference_lock.acquire(blocking=False):
            raise BusyError("another detection request is already running")
        try:
            detections = self.detector.detect(frame, prompt, confidence)
            return {
                "status": "ok",
                "model": self.model_name,
                "detections": detections,
            }
        finally:
            self.inference_lock.release()


def create_server(app: DetectionApplication, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            print(f"http: {format % args}", flush=True)

        def send_json(self, status: int, document: dict[str, Any]) -> None:
            data = json.dumps(document, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if urllib.parse.urlsplit(self.path).path != "/healthz":
                self.send_json(404, {"status": "error", "error": "unknown endpoint"})
                return
            try:
                self.send_json(200, app.health())
            except VisionError as exc:
                self.send_json(503, {"status": "error", "error": str(exc)})

        def do_POST(self) -> None:
            if urllib.parse.urlsplit(self.path).path != "/api/detect":
                self.send_json(404, {"status": "error", "error": "unknown endpoint"})
                return
            try:
                if self.headers.get_content_type() != "application/json":
                    raise RequestError("Content-Type must be application/json")
                try:
                    length = int(self.headers.get("Content-Length", ""))
                except ValueError as exc:
                    raise RequestError("Content-Length is required") from exc
                if not 0 < length <= MAX_REQUEST_BYTES:
                    self.send_json(413, {"status": "error", "error": "invalid request size"})
                    return
                self.connection.settimeout(10)
                try:
                    document = json.loads(self.rfile.read(length))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise RequestError(f"invalid JSON: {exc}") from exc
                result = app.detect(document)
                self.send_json(200, result)
            except TimeoutError:
                self.send_json(408, {"status": "error", "error": "request timed out"})
            except RequestError as exc:
                self.send_json(400, {"status": "error", "error": str(exc)})
            except BusyError as exc:
                self.send_json(409, {"status": "error", "error": str(exc)})
            except VisionError as exc:
                self.send_json(502, {"status": "error", "error": str(exc)})

    return ThreadingHTTPServer(("0.0.0.0", port), Handler)


def main() -> int:
    server: ThreadingHTTPServer | None = None
    try:
        port = int(os.environ.get("PORT", "8096"))
        if not 0 < port <= 65535:
            raise VisionError("PORT must be between 1 and 65535")
        app = DetectionApplication()
        server = create_server(app, port)
        print(f"Vision detection API listening on 0.0.0.0:{port}", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping vision detection service", flush=True)
    except (VisionError, ValueError, OSError) as exc:
        print(f"ERROR: vision detection stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        if server is not None:
            server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
