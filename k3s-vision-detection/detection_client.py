"""HTTP detection client used by applications, never by the camera service."""

from __future__ import annotations

import base64
import http.client
import json
import math
import urllib.error
import urllib.request
from typing import Any

import numpy as np


class DetectionError(RuntimeError):
    """A detection service communication or response failure."""


class DetectionClient:
    def __init__(
        self, model_name: str, base_url: str, timeout: float = 60.0
    ) -> None:
        self.model_name = model_name
        self.url = base_url.rstrip("/")
        if not self.url.startswith(("http://", "https://")):
            raise DetectionError("detection URL must start with http:// or https://")
        self.timeout = timeout
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise DetectionError("detection timeout must be positive and finite")

    def _request(
        self, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=min(self.timeout, 3.0) if payload is None else self.timeout,
            ) as response:
                document = json.load(response)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise DetectionError(
                f"detection service {path} returned HTTP {exc.code}: {body}"
            ) from exc
        except (
            OSError, http.client.HTTPException, json.JSONDecodeError, UnicodeDecodeError
        ) as exc:
            raise DetectionError(f"detection service request {path} failed: {exc}") from exc
        if not isinstance(document, dict) or document.get("status") != "ok":
            raise DetectionError(f"invalid detection service response: {document}")
        if document.get("model") != self.model_name:
            raise DetectionError(
                f"detection service model must be {self.model_name!r}, "
                f"not {document.get('model')!r}"
            )
        return document

    def check(self) -> None:
        document = self._request("/healthz")
        if document.get("cuda") is not True:
            raise DetectionError("detection service must use a CUDA Foundry model")

    def detect_png(
        self, image_png: bytes, prompt: str, threshold: float
    ) -> list[dict[str, Any]]:
        if not image_png.startswith(b"\x89PNG\r\n\x1a\n"):
            raise DetectionError("detection requires a lossless PNG image")
        document = self._request(
            "/api/detect",
            {
                "prompt": prompt,
                "confidence": threshold,
                "image_png_base64": base64.b64encode(image_png).decode("ascii"),
            },
        )
        detections = document.get("detections")
        if not isinstance(detections, list):
            raise DetectionError("detection response must contain a detections list")
        for detection in detections:
            if not isinstance(detection, dict):
                raise DetectionError("each detection must be a JSON object")
            try:
                score = float(detection["score"])
                box = np.asarray(detection["box_xyxy"], dtype=np.float64)
            except (KeyError, TypeError, ValueError) as exc:
                raise DetectionError(f"invalid detection geometry: {exc}") from exc
            if (
                not math.isfinite(score)
                or not 0 <= score <= 1
                or box.shape != (4,)
                or not np.all(np.isfinite(box))
                or box[2] <= box[0]
                or box[3] <= box[1]
            ):
                raise DetectionError("detection has invalid score or bounding box")
            if "long_axis_pixels" in detection:
                try:
                    axis = np.asarray(detection["long_axis_pixels"], dtype=np.float64)
                except (TypeError, ValueError) as exc:
                    raise DetectionError(f"invalid detection orientation: {exc}") from exc
                if axis.shape != (2, 2) or not np.all(np.isfinite(axis)):
                    raise DetectionError("detection orientation must contain two finite pixels")
        return detections
