#!/usr/bin/env python3
"""MediaPipe Hand Landmarker service consuming the OpenCV MJPEG stream."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from hand_logic import annotate_frame, serialize_result


class HandServiceError(RuntimeError):
    """A configuration, source-stream, inference, or HTTP error."""


class VisionServiceRejected(HandServiceError):
    """The vision service safely rejected a valid request."""


def env_float(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if not np.isfinite(value):
        raise HandServiceError(f"{name} must be finite")
    return value


def env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise HandServiceError(f"{name} must be positive")
    return value


class HandLandmarkerStream:
    def __init__(self) -> None:
        self.source_url = os.environ.get(
            "SOURCE_MJPEG_URL",
            "http://127.0.0.1:8094/stream.mjpg",
        )
        self.vision_url = os.environ.get(
            "VISION_URL",
            "http://127.0.0.1:8094",
        ).rstrip("/")
        self.model_path = Path(
            os.environ.get("HAND_MODEL", "/models/hand_landmarker.task")
        )
        self.max_hands = env_int("MAX_HANDS", 2)
        self.minimum_detection_confidence = env_float(
            "MIN_HAND_DETECTION_CONFIDENCE", 0.5
        )
        self.minimum_presence_confidence = env_float(
            "MIN_HAND_PRESENCE_CONFIDENCE", 0.5
        )
        self.minimum_tracking_confidence = env_float(
            "MIN_TRACKING_CONFIDENCE", 0.5
        )
        self.inference_fps = env_float("INFERENCE_FPS", 10.0)
        self.stability_samples = env_int("STABILITY_SAMPLES", 5)
        self.minimum_stable_samples = env_int("MINIMUM_STABLE_SAMPLES", 3)
        self.maximum_palm_spread_mm = env_float(
            "MAXIMUM_PALM_SPREAD_MM", 50.0
        )
        if self.inference_fps <= 0:
            raise HandServiceError("INFERENCE_FPS must be positive")
        for name, value in (
            ("MIN_HAND_DETECTION_CONFIDENCE", self.minimum_detection_confidence),
            ("MIN_HAND_PRESENCE_CONFIDENCE", self.minimum_presence_confidence),
            ("MIN_TRACKING_CONFIDENCE", self.minimum_tracking_confidence),
        ):
            if not 0 <= value <= 1:
                raise HandServiceError(f"{name} must be between zero and one")
        if not self.model_path.is_file():
            raise HandServiceError(f"hand model does not exist: {self.model_path}")

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(self.model_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.minimum_detection_confidence,
            min_hand_presence_confidence=self.minimum_presence_confidence,
            min_tracking_confidence=self.minimum_tracking_confidence,
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.inference_lock = threading.Lock()
        self.last_timestamp_ms = 0
        self.stop_event = threading.Event()
        self.condition = threading.Condition()
        self.thread: threading.Thread | None = None
        self.latest_jpeg: bytes | None = None
        self.latest_result: dict[str, Any] = {
            "status": "starting",
            "hands": [],
            "hand_count": 0,
        }
        self.sequence = 0
        self.last_frame_monotonic: float | None = None
        self.error: str | None = None
        self.snapshots: dict[int, dict[str, Any]] = {}
        self.snapshot_jpegs: dict[int, bytes] = {}

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 30.0
        with self.condition:
            while (
                self.latest_jpeg is None
                and self.error is None
                and time.monotonic() < deadline
            ):
                self.condition.wait(timeout=0.2)
        self.check()
        if self.latest_jpeg is None:
            raise HandServiceError(
                f"no frame received from {self.source_url} within 30 seconds"
            )

    def check(self) -> None:
        if self.error is not None:
            raise HandServiceError(self.error)
        if (
            self.last_frame_monotonic is None
            or time.monotonic() - self.last_frame_monotonic > 10.0
        ):
            raise HandServiceError("source MJPEG stream has no recent frame")

    def _run(self) -> None:
        retry_delay = 1.0
        next_inference = 0.0
        try:
            while not self.stop_event.is_set():
                capture = cv2.VideoCapture(self.source_url)
                if not capture.isOpened():
                    capture.release()
                    time.sleep(retry_delay)
                    continue
                try:
                    while not self.stop_event.is_set():
                        ok, frame = capture.read()
                        if not ok or frame is None:
                            break
                        now = time.monotonic()
                        if now < next_inference:
                            continue
                        next_inference = now + 1.0 / self.inference_fps
                        hands = self._detect_frame(frame)
                        annotated = annotate_frame(frame, hands)
                        encoded_ok, encoded = cv2.imencode(".jpg", annotated)
                        if not encoded_ok:
                            raise HandServiceError(
                                "could not encode annotated hand frame"
                            )
                        with self.condition:
                            self.sequence += 1
                            self.last_frame_monotonic = now
                            self.latest_jpeg = encoded.tobytes()
                            self.latest_result = {
                                "status": "ok",
                                "captured_at": datetime.now(
                                    timezone.utc
                                ).isoformat(),
                                "frame_sequence": self.sequence,
                                "source_url": self.source_url,
                                "hand_count": len(hands),
                                "hands": hands,
                            }
                            self.condition.notify_all()
                finally:
                    capture.release()
                if not self.stop_event.is_set():
                    time.sleep(retry_delay)
        except Exception as exc:
            if not self.stop_event.is_set():
                self.error = str(exc)
                with self.condition:
                    self.condition.notify_all()

    def _detect_frame(self, frame: np.ndarray) -> list[dict[str, Any]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        with self.inference_lock:
            timestamp_ms = max(
                int(time.monotonic() * 1000),
                self.last_timestamp_ms + 1,
            )
            self.last_timestamp_ms = timestamp_ms
            result = self.landmarker.detect_for_video(image, timestamp_ms)
        return serialize_result(result, frame.shape[1], frame.shape[0])

    def _json_request(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            f"{self.vision_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if data is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                value = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            try:
                rejected = json.loads(exc.read())
                detail = str(rejected.get("error", rejected))
            except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                detail = str(exc)
            raise VisionServiceRejected(
                f"vision service request {path} rejected: {detail}"
            ) from exc
        except Exception as exc:
            raise HandServiceError(
                f"vision service request {path} failed: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise HandServiceError("vision service response was not an object")
        return value

    def _synchronized_frame(self) -> tuple[np.ndarray, dict[str, Any]]:
        capture = self._json_request("/api/rgbd-capture", {})
        image_url = str(capture.get("image_url", ""))
        if not image_url.startswith("/"):
            raise HandServiceError("vision service returned invalid image URL")
        try:
            with urllib.request.urlopen(
                f"{self.vision_url}{image_url}",
                timeout=10,
            ) as response:
                encoded = np.frombuffer(response.read(), dtype=np.uint8)
        except Exception as exc:
            raise HandServiceError(
                f"could not fetch synchronized RGB frame: {exc}"
            ) from exc
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise HandServiceError("could not decode synchronized RGB frame")
        return frame, capture

    def capture_slot(self, slot: int) -> dict[str, Any]:
        if slot not in (4, 5, 6):
            raise HandServiceError("hand slot must be 4, 5, or 6")
        samples = []
        detected_sample_count = 0
        visible_hand_sample_count = 0
        depth_rejections: list[str] = []
        best_frame = None
        best_hands = []
        best_score = -1.0
        for _ in range(self.stability_samples):
            frame, synchronized = self._synchronized_frame()
            hands = self._detect_frame(frame)
            if hands:
                visible_hand_sample_count += 1
            open_hands = [hand for hand in hands if hand.get("is_open_palm")]
            if open_hands:
                detected_sample_count += 1
                hand = max(
                    open_hands,
                    key=lambda value: float(value["handedness_score"]),
                )
                score = float(hand["handedness_score"])
                if score > best_score:
                    best_score = score
                    best_frame = frame
                    best_hands = hands
                try:
                    depth = self._json_request(
                        "/api/deproject",
                        {
                            "token": synchronized["token"],
                            "pixel_uv": hand["palm_center_pixel"],
                        },
                    )
                except VisionServiceRejected as exc:
                    depth_rejections.append(str(exc))
                    time.sleep(0.08)
                    continue
                sample = {
                    "score": score,
                    "hand": hand,
                    "camera_point_mm": depth["camera_point_mm"],
                    "depth": depth,
                    "frame_sequence": synchronized["frame_sequence"],
                }
                samples.append(sample)
            elif best_frame is None:
                best_frame = frame
            time.sleep(0.08)

        captured_at = datetime.now(timezone.utc).isoformat()
        result: dict[str, Any] = {
            "status": "not_found",
            "slot": slot,
            "captured_at": captured_at,
            "model": self.model_path.name,
            "gesture": "open_palm",
            "sample_count": self.stability_samples,
            "visible_hand_sample_count": visible_hand_sample_count,
            "detected_sample_count": detected_sample_count,
            "depth_sample_count": len(samples),
            "depth_rejection_count": len(depth_rejections),
        }
        if depth_rejections:
            result["depth_rejection_errors"] = list(dict.fromkeys(depth_rejections))
        if len(samples) >= self.minimum_stable_samples:
            points = np.asarray(
                [sample["camera_point_mm"] for sample in samples],
                dtype=np.float64,
            )
            initial_consensus = np.median(points, axis=0)
            distances = np.linalg.norm(points - initial_consensus, axis=1)
            inlier_mask = distances <= self.maximum_palm_spread_mm
            stable_samples = [
                sample
                for sample, is_inlier in zip(samples, inlier_mask)
                if is_inlier
            ]
            stable_points = points[inlier_mask]
            stable_sample_count = len(stable_samples)
            consensus = (
                np.median(stable_points, axis=0)
                if stable_sample_count
                else initial_consensus
            )
            stable_spread = (
                float(
                    np.max(
                        np.linalg.norm(stable_points - consensus, axis=1)
                    )
                )
                if stable_sample_count
                else math.inf
            )
            observed_spread = float(np.max(distances))
            accepted = stable_sample_count >= self.minimum_stable_samples
            representative_samples = stable_samples if stable_samples else samples
            best = max(representative_samples, key=lambda value: value["score"])
            result.update(
                {
                    "status": "detected" if accepted else "rejected",
                    "score": best["score"],
                    "handedness": best["hand"]["handedness"],
                    "palm_center_pixel": best["hand"]["palm_center_pixel"],
                    "camera_point_mm": consensus.tolist(),
                    "stable_sample_count": stable_sample_count,
                    "temporal_outlier_count": len(samples) - stable_sample_count,
                    "maximum_temporal_spread_mm": stable_spread,
                    "maximum_observed_spread_mm": observed_spread,
                    "maximum_allowed_spread_mm": self.maximum_palm_spread_mm,
                    "hand": best["hand"],
                }
            )
            if result["status"] == "rejected":
                result["error"] = (
                    f"only {stable_sample_count}/{len(samples)} palm samples "
                    "formed a stable cluster; need "
                    f"{self.minimum_stable_samples} within "
                    f"{self.maximum_palm_spread_mm:.1f}mm"
                )
        elif detected_sample_count:
            result.update(
                {
                    "status": "rejected",
                    "score": best_score,
                    "error": (
                        f"hand detected in {detected_sample_count}/"
                        f"{self.stability_samples} samples, but only "
                        f"{len(samples)} had usable palm depth; need "
                        f"{self.minimum_stable_samples}"
                    ),
                }
            )
        annotated = annotate_frame(best_frame, best_hands)
        ok, encoded = cv2.imencode(".jpg", annotated)
        if not ok:
            raise HandServiceError("could not encode hand snapshot")
        result["snapshot_url"] = f"/snapshots/{slot}.jpg"
        self.snapshots[slot] = result
        self.snapshot_jpegs[slot] = encoded.tobytes()
        return result

    def close(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
        self.landmarker.close()


def page_html() -> bytes:
    return b"""<!doctype html>
<meta charset="utf-8">
<title>Piper Hand Landmarker</title>
<style>
body{margin:0;background:#111;color:#eee;font:15px sans-serif}
main{max-width:1000px;margin:auto;padding:12px}
img{width:100%;border:1px solid #555} pre{white-space:pre-wrap;background:#222;padding:10px}
</style>
<main><h2>MediaPipe Hand Landmarker</h2>
<img src="/stream.mjpg" alt="Annotated hand stream">
<h3>Latest result</h3><pre id="result">Waiting for hand service...</pre></main>
<script>
async function refresh(){
  const response=await fetch('/api/hands',{cache:'no-store'});
  document.getElementById('result').textContent=JSON.stringify(await response.json(),null,2);
}
setInterval(refresh,500); refresh();
</script>"""


def create_server(
    application: HandLandmarkerStream,
    port: int,
) -> ThreadingHTTPServer:
    page = page_html()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            print(f"http: {format % args}", flush=True)

        def send_bytes(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status: int, value: dict[str, Any]) -> None:
            self.send_bytes(
                status,
                "application/json",
                json.dumps(value).encode("utf-8"),
            )

        def do_GET(self) -> None:
            path = urllib.parse.urlsplit(self.path).path
            if path in ("/", "/index.html"):
                self.send_bytes(200, "text/html; charset=utf-8", page)
                return
            if path == "/healthz":
                try:
                    application.check()
                    self.send_json(
                        200,
                        {
                            "status": "ok",
                            "source_url": application.source_url,
                            "model": application.model_path.name,
                            "frame_sequence": application.sequence,
                        },
                    )
                except HandServiceError as exc:
                    self.send_json(
                        503,
                        {"status": "error", "error": str(exc)},
                    )
                return
            if path == "/api/hands":
                self.send_json(200, application.latest_result)
                return
            if path == "/api/snapshots":
                self.send_json(
                    200,
                    {
                        "snapshots": {
                            str(slot): value
                            for slot, value in application.snapshots.items()
                        }
                    },
                )
                return
            if path.startswith("/snapshots/") and path.endswith(".jpg"):
                try:
                    slot = int(
                        path.removeprefix("/snapshots/").removesuffix(".jpg")
                    )
                    jpeg = application.snapshot_jpegs[slot]
                except (ValueError, KeyError):
                    self.send_error(404)
                    return
                self.send_bytes(200, "image/jpeg", jpeg)
                return
            if path == "/frame.jpg":
                if application.latest_jpeg is None:
                    self.send_error(404)
                else:
                    self.send_bytes(
                        200,
                        "image/jpeg",
                        application.latest_jpeg,
                    )
                return
            if path != "/stream.mjpg":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame",
            )
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                sequence = -1
                while not application.stop_event.is_set():
                    with application.condition:
                        application.condition.wait_for(
                            lambda: application.sequence > sequence
                            or application.stop_event.is_set(),
                            timeout=1,
                        )
                        jpeg = application.latest_jpeg
                        sequence = application.sequence
                    if jpeg is None:
                        continue
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                    )
                    self.wfile.write(
                        f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self) -> None:
            if urllib.parse.urlsplit(self.path).path != "/api/capture":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 4096:
                    raise HandServiceError(
                        "request body must be 1..4096 bytes"
                    )
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise HandServiceError("request must be a JSON object")
                self.send_json(
                    200,
                    application.capture_slot(int(body.get("slot"))),
                )
            except (
                HandServiceError,
                ValueError,
                TypeError,
                json.JSONDecodeError,
            ) as exc:
                self.send_json(
                    400,
                    {"status": "rejected", "error": str(exc)},
                )

    class Server(ThreadingHTTPServer):
        allow_reuse_address = True
        daemon_threads = True

    return Server(("0.0.0.0", port), Handler)


def main() -> int:
    port = env_int("PORT", 8095)
    application = HandLandmarkerStream()
    try:
        application.start()
        server = create_server(application, port)
        print(f"Hand service listening on 0.0.0.0:{port}", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping hand service", flush=True)
    finally:
        application.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
