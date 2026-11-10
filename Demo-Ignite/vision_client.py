"""Application-side orchestration of independent camera and detection services."""

from __future__ import annotations

import http.client
import json
import math
from pathlib import Path
import sys
import urllib.error
import urllib.request
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "k3s-vision-detection"))
from detection_client import DetectionClient, DetectionError  # noqa: E402

DEFAULT_VISION_URL = "http://127.0.0.1:8094"
DEFAULT_DETECTION_URL = "http://127.0.0.1:8096"
EXPECTED_VISION_MODEL = "candy-yolo11s"


class DemoError(RuntimeError):
    """An invalid configuration, rejected target, or unsafe motion request."""


class HttpClient:
    def __init__(self, base_url: str, timeout: float = 90.0) -> None:
        self.base_url = base_url.rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise DemoError("service URL must start with http:// or https://")
        self.timeout = timeout

    def _request_bytes(self, path: str, payload: dict[str, Any] | None = None) -> bytes:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise DemoError(f"service {path} returned HTTP {exc.code}: {body}") from exc
        except (OSError, http.client.HTTPException) as exc:
            raise DemoError(f"cannot reach service at {self.base_url}{path}: {exc}") from exc

    def _json_request(
        self, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        try:
            document = json.loads(self._request_bytes(path, payload))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise DemoError(f"invalid JSON from service {path}: {exc}") from exc
        if not isinstance(document, dict):
            raise DemoError("service response must be a JSON object")
        return document

    def health(self) -> dict[str, Any]:
        health = self._json_request("/healthz")
        if health.get("status") != "ok":
            raise DemoError(f"service is unhealthy: {health}")
        return health


class VisionClient(HttpClient):
    def __init__(
        self, base_url: str, timeout: float = 90.0, *,
        detection_url: str = DEFAULT_DETECTION_URL,
        detector: Any = None,
    ) -> None:
        """Use the YOLO detection service unless a local detector is supplied."""
        super().__init__(base_url, timeout)
        if detector is not None:
            self.detector = detector
            return
        try:
            self.detector = DetectionClient(EXPECTED_VISION_MODEL, detection_url, timeout)
        except DetectionError as exc:
            raise DemoError(str(exc)) from exc

    def health(self) -> dict[str, Any]:
        health = super().health()
        capabilities = health.get("capabilities")
        if (
            health.get("service") != "camera" or not isinstance(capabilities, list)
            or "localize" not in capabilities
        ):
            raise DemoError("demo requires the independent camera service with token localization")
        try:
            self.detector.check()
        except DetectionError as exc:
            raise DemoError(str(exc)) from exc
        return {
            **health, "model": self.detector.model_name,
            "cuda": getattr(self.detector, "uses_cuda", True),
            "detection_url": self.detector.url,
        }

    def capture_png(self) -> tuple[dict[str, Any], bytes]:
        """Capture one token-bound RGB-D frame and download its lossless PNG."""
        captured = self._json_request("/api/rgbd-capture", {})
        token = captured.get("token")
        if (
            captured.get("status") != "ok" or not isinstance(token, str)
            or not token or not token.isalnum()
            or not isinstance(captured.get("frame_sequence"), int)
            or captured.get("image_png_url") != f"/rgbd/{token}.png"
        ):
            raise DemoError("camera did not return a valid lossless RGB-D frame token")
        image = self._request_bytes(captured["image_png_url"])
        if not image.startswith(b"\x89PNG\r\n\x1a\n"):
            raise DemoError("camera frame download is not a PNG image")
        return captured, image

    def capture(self, slot: int, prompt: str, confidence: float) -> dict[str, Any]:
        if slot not in (1, 2, 3) or not prompt.strip() or not 0 < confidence < 1:
            raise DemoError("capture requires slot 1..3, prompt, and confidence between 0 and 1")
        if not math.isfinite(confidence):
            raise DemoError("confidence must be finite")
        captured, image = self.capture_png()
        token = captured["token"]
        try:
            detections = self.detector.detect_png(image, prompt, confidence)
        except DetectionError as exc:
            raise DemoError(str(exc)) from exc
        result = self._json_request("/api/localize", {
            "token": token, "slot": slot, "prompt": prompt,
            "model": self.detector.model_name,
            "detections": [
                {**detection, "pickup_point": "long_axis_midpoint"} for detection in detections
            ],
        })
        if (
            result.get("token") != token
            or result.get("frame_sequence") != captured["frame_sequence"]
            or result.get("slot") != slot
        ):
            raise DemoError("localization response does not match the captured RGB-D frame")
        if result.get("status") not in ("detected", "not_found", "rejected"):
            raise DemoError(f"vision capture {slot} was rejected: {result}")
        candidates = result.get("candidates", [result])
        if not isinstance(candidates, list):
            raise DemoError("camera localization candidates must be a list")
        for candidate in [result, *candidates]:
            if candidate.get("status") == "detected" and candidate.get("pickup_point") != "long_axis_midpoint":
                raise DemoError(
                    "camera did not localize the candy long-axis midpoint; "
                    "deploy the updated OpenCV camera service before running Ignite"
                )
        return result

    def set_workspace_overlay(
        self, base_to_camera: np.ndarray, minimum_mm: np.ndarray, maximum_mm: np.ndarray
    ) -> None:
        self._json_request("/api/workspace-overlay", {
            "base_to_camera": np.asarray(base_to_camera, dtype=np.float64).tolist(),
            "minimum_mm": np.asarray(minimum_mm, dtype=np.float64).tolist(),
            "maximum_mm": np.asarray(maximum_mm, dtype=np.float64).tolist(),
        })


class HandClient(HttpClient):
    def capture_hand(self, slot: int) -> dict[str, Any]:
        result = self._json_request("/api/capture", {"slot": slot})
        if result.get("status") not in ("detected", "not_found", "rejected"):
            raise DemoError(f"hand capture {slot} returned invalid status: {result}")
        return result
