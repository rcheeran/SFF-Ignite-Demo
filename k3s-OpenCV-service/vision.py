"""Foundry YOLO detection plus shared D405 depth helpers."""

from __future__ import annotations

import base64
import json
import os
import ssl
import urllib.error
import urllib.request
from typing import Any

import cv2
import numpy as np

from base_vision import (  # noqa: F401
    VisionError,
    annotate_capture,
    camera_point_from_pixel,
    robust_box_depth,
    robust_pixel_depth,
)


CLASS_NAMES = (
    "MMs_peanut",
    "MMs_regular",
    "airheads",
    "gummy_worms",
    "milky_way",
    "nerds",
    "skittles",
    "snickers",
    "starburst",
    "three_musketeers",
    "twizzlers",
)
PROMPT_ALIASES = {
    "mms peanut": "MMs_peanut",
    "peanut mms": "MMs_peanut",
    "mms regular": "MMs_regular",
    "regular mms": "MMs_regular",
    "airheads": "airheads",
    "airhead": "airheads",
    "gummy worms": "gummy_worms",
    "gummy worm": "gummy_worms",
    "milky way": "milky_way",
    "nerds": "nerds",
    "skittles": "skittles",
    "snickers": "snickers",
    "starburst": "starburst",
    "starburst candy": "starburst",
    "three musketeers": "three_musketeers",
    "3 musketeers": "three_musketeers",
    "twizzlers": "twizzlers",
}


def normalize_prompt(prompt: str) -> str:
    return " ".join(
        prompt.strip().lower().replace("_", " ").replace("-", " ").split()
    )


def letterbox_tensor(
    frame_bgr: np.ndarray, size: int = 640
) -> tuple[np.ndarray, float, float, float]:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise VisionError("camera frame must be an HWC BGR image")
    height, width = frame_bgr.shape[:2]
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(
        frame_bgr,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    pad_x = size - resized_width
    pad_y = size - resized_height
    left = pad_x // 2
    top = pad_y // 2
    letterboxed = cv2.copyMakeBorder(
        resized,
        top,
        pad_y - top,
        left,
        pad_x - left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    tensor = np.ascontiguousarray(
        letterboxed[:, :, ::-1].transpose(2, 0, 1),
        dtype=np.float32,
    )[None]
    tensor /= 255.0
    return tensor, scale, float(left), float(top)


def decode_yolo_output(
    output: np.ndarray,
    class_id: int,
    confidence: float,
    iou_threshold: float,
    scale: float,
    pad_x: float,
    pad_y: float,
    frame_shape: tuple[int, ...],
) -> list[dict[str, Any]]:
    array = np.asarray(output, dtype=np.float32)
    if array.shape != (1, 4 + len(CLASS_NAMES), 8400):
        raise VisionError(f"unexpected Foundry YOLO output shape: {array.shape}")
    predictions = array[0].T
    scores = predictions[:, 4 + class_id]
    candidate_indices = np.flatnonzero(scores >= confidence)
    if candidate_indices.size == 0:
        return []

    boxes_xywh: list[list[float]] = []
    candidate_scores: list[float] = []
    height, width = frame_shape[:2]
    for index in candidate_indices:
        center_x, center_y, box_width, box_height = [
            float(value) for value in predictions[index, :4]
        ]
        left = (center_x - box_width / 2.0 - pad_x) / scale
        top = (center_y - box_height / 2.0 - pad_y) / scale
        right = (center_x + box_width / 2.0 - pad_x) / scale
        bottom = (center_y + box_height / 2.0 - pad_y) / scale
        left = float(np.clip(left, 0.0, width - 1.0))
        top = float(np.clip(top, 0.0, height - 1.0))
        right = float(np.clip(right, 0.0, width - 1.0))
        bottom = float(np.clip(bottom, 0.0, height - 1.0))
        if right <= left or bottom <= top:
            continue
        boxes_xywh.append([left, top, right - left, bottom - top])
        candidate_scores.append(float(scores[index]))
    if not boxes_xywh:
        return []

    selected = cv2.dnn.NMSBoxes(
        boxes_xywh,
        candidate_scores,
        score_threshold=confidence,
        nms_threshold=iou_threshold,
    )
    detections = []
    for selected_index in np.asarray(selected).reshape(-1):
        left, top, box_width, box_height = boxes_xywh[int(selected_index)]
        detections.append(
            {
                "score": candidate_scores[int(selected_index)],
                "box_xyxy": [
                    left,
                    top,
                    left + box_width,
                    top + box_height,
                ],
            }
        )
    return sorted(detections, key=lambda item: item["score"], reverse=True)


def estimate_long_axis(
    frame_bgr: np.ndarray,
    box_xyxy: list[float],
    minimum_aspect_ratio: float = 1.35,
) -> dict[str, Any]:
    height, width = frame_bgr.shape[:2]
    left, top, right, bottom = np.rint(box_xyxy).astype(int)
    left = int(np.clip(left, 0, width - 1))
    top = int(np.clip(top, 0, height - 1))
    right = int(np.clip(right, left + 1, width))
    bottom = int(np.clip(bottom, top + 1, height))
    roi = frame_bgr[top:bottom, left:right]
    if roi.shape[0] < 12 or roi.shape[1] < 12:
        raise VisionError("Starburst box is too small to estimate orientation")

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 70, 35), (179, 255, 255))
    kernel_size = max(3, int(round(min(roi.shape[:2]) * 0.05)) | 1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    minimum_area = max(80.0, roi.shape[0] * roi.shape[1] * 0.08)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= minimum_area]
    if not contours:
        raise VisionError("could not isolate the Starburst wrapper orientation")
    contour = max(contours, key=cv2.contourArea)
    center, dimensions, angle = cv2.minAreaRect(contour)
    short_side = float(min(dimensions))
    long_side = float(max(dimensions))
    if short_side <= 1.0:
        raise VisionError("Starburst orientation contour is degenerate")
    aspect_ratio = long_side / short_side
    if aspect_ratio < minimum_aspect_ratio:
        raise VisionError(
            f"Starburst orientation is ambiguous: aspect ratio "
            f"{aspect_ratio:.2f} is below {minimum_aspect_ratio:.2f}"
        )

    theta = np.deg2rad(angle + (90.0 if dimensions[1] > dimensions[0] else 0.0))
    direction = np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float64)
    half_length = long_side * 0.4
    center_global = np.asarray(
        [center[0] + left, center[1] + top],
        dtype=np.float64,
    )
    endpoints = np.stack(
        [
            center_global - direction * half_length,
            center_global + direction * half_length,
        ]
    )
    endpoints[:, 0] = np.clip(endpoints[:, 0], 0, width - 1)
    endpoints[:, 1] = np.clip(endpoints[:, 1], 0, height - 1)
    return {
        "long_axis_pixels": endpoints.tolist(),
        "orientation_aspect_ratio": aspect_ratio,
        "orientation_mask_fraction": (
            float(cv2.contourArea(contour)) / float(roi.shape[0] * roi.shape[1])
        ),
    }


class OwlVitDetector:
    """Drop-in detector used by the shared vision application."""

    def __init__(self, model_name: str, device_name: str = "cuda") -> None:
        del device_name
        self.model_name = model_name
        self.url = os.environ.get(
            "FOUNDRY_URL",
            "https://candy-yolo11s.foundry-local-operator.svc.cluster.local:5000",
        ).rstrip("/")
        self.api_key = os.environ.get("FOUNDRY_API_KEY", "")
        if not self.api_key:
            raise VisionError("FOUNDRY_API_KEY is required")
        self.timeout = float(os.environ.get("FOUNDRY_TIMEOUT_SECONDS", "30"))
        self.iou_threshold = float(os.environ.get("NMS_IOU_THRESHOLD", "0.45"))
        if not 0.0 < self.iou_threshold < 1.0:
            raise VisionError("NMS_IOU_THRESHOLD must be between zero and one")
        ca_path = os.environ.get("FOUNDRY_CA_CERT", "")
        try:
            self.ssl_context = (
                ssl.create_default_context(cafile=ca_path)
                if ca_path
                else ssl.create_default_context()
            )
        except OSError as exc:
            raise VisionError(f"could not load Foundry CA certificate: {exc}") from exc
        self.metadata = self._request("/v1/model")
        if self.metadata.get("status") != "loaded":
            raise VisionError(f"Foundry model is not loaded: {self.metadata}")
        if self.metadata.get("input_shape") != [1, 3, 640, 640]:
            raise VisionError(
                f"unexpected Foundry input shape: {self.metadata.get('input_shape')}"
            )
        if self.metadata.get("execution_provider") != "CUDAExecutionProvider":
            raise VisionError(
                "Foundry candy model is not using CUDAExecutionProvider"
            )

    def _request(
        self, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.url}{path}",
            data=data,
            headers={
                "api-key": self.api_key,
                "Content-Type": "application/json",
            },
            method="GET" if payload is None else "POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                context=self.ssl_context,
                timeout=self.timeout,
            ) as response:
                document = json.load(response)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise VisionError(
                f"Foundry {path} returned HTTP {exc.code}: {body}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise VisionError(f"Foundry request {path} failed: {exc}") from exc
        if not isinstance(document, dict):
            raise VisionError("Foundry response must be a JSON object")
        return document

    def check(self) -> None:
        metadata = self._request("/v1/model")
        if metadata.get("status") != "loaded":
            raise VisionError(f"Foundry model is not loaded: {metadata}")

    def detect(
        self, frame_bgr: np.ndarray, prompt: str, threshold: float
    ) -> list[dict[str, Any]]:
        normalized = normalize_prompt(prompt)
        class_name = PROMPT_ALIASES.get(normalized)
        if class_name is None:
            supported = ", ".join(sorted(PROMPT_ALIASES))
            raise VisionError(
                f"unsupported candy prompt {prompt!r}; supported prompts: {supported}"
            )
        class_id = CLASS_NAMES.index(class_name)
        tensor, scale, pad_x, pad_y = letterbox_tensor(frame_bgr)
        encoded = base64.b64encode(
            json.dumps(tensor.tolist(), separators=(",", ":")).encode("utf-8")
        ).decode("ascii")
        response = self._request(
            "/v1/predict",
            {
                "items": [
                    {
                        "content_type": "application/json",
                        "encoder": "base64",
                        "data": encoded,
                    }
                ]
            },
        )
        try:
            item = response["items"][0]
            outputs = json.loads(base64.b64decode(item["data"]))
            output = next(iter(outputs.values()))
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise VisionError(f"invalid Foundry prediction response: {exc}") from exc
        detections = decode_yolo_output(
            np.asarray(output, dtype=np.float32),
            class_id,
            threshold,
            self.iou_threshold,
            scale,
            pad_x,
            pad_y,
            frame_bgr.shape,
        )
        for detection in detections:
            try:
                detection.update(
                    estimate_long_axis(frame_bgr, detection["box_xyxy"])
                )
            except VisionError as exc:
                detection["orientation_error"] = str(exc)
        return detections
