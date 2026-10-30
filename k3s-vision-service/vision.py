"""Pure OWL-ViT detection and depth helpers for the vision service."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import cv2
import numpy as np


class VisionError(RuntimeError):
    """A malformed request, rejected depth sample, or model failure."""


class OwlVitDetector:
    def __init__(self, model_name: str, device_name: str = "cuda") -> None:
        import torch
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        if device_name == "cuda" and not torch.cuda.is_available():
            raise VisionError("CUDA is required but unavailable")
        self.torch = torch
        self.device = torch.device(device_name)
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def detect(
        self, frame_bgr: np.ndarray, prompt: str, threshold: float
    ) -> list[dict[str, Any]]:
        from PIL import Image

        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(
            text=[[prompt]], images=image, return_tensors="pt"
        ).to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        target_sizes = self.torch.tensor(
            [[frame_bgr.shape[0], frame_bgr.shape[1]]], device=self.device
        )
        if hasattr(self.processor, "post_process_grounded_object_detection"):
            processed = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=threshold,
            )[0]
        elif hasattr(self.processor, "post_process_object_detection"):
            processed = self.processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=threshold,
            )[0]
        else:
            raise VisionError("OwlViTProcessor has no detection post-processor")
        detections = [
            {
                "score": float(score),
                "box_xyxy": [float(value) for value in box],
            }
            for score, box in zip(
                processed["scores"].detach().cpu().tolist(),
                processed["boxes"].detach().cpu().tolist(),
                strict=True,
            )
        ]
        return sorted(detections, key=lambda item: item["score"], reverse=True)


def robust_box_depth(
    depth_raw: np.ndarray,
    box_xyxy: list[float],
    depth_scale: float,
    inset_fraction: float,
    minimum_pixels: int,
    minimum_depth_m: float,
    maximum_depth_m: float,
    cluster_bin_mm: float,
    cluster_tolerance_mm: float,
    minimum_cluster_fraction: float,
) -> tuple[tuple[int, int], float, dict[str, float | int]]:
    if depth_raw.ndim != 2:
        raise VisionError("aligned depth frame must be two-dimensional")
    height, width = depth_raw.shape
    left, top, right, bottom = [float(value) for value in box_xyxy]
    left, right = sorted((max(0.0, left), min(float(width), right)))
    top, bottom = sorted((max(0.0, top), min(float(height), bottom)))
    if right - left < 4 or bottom - top < 4:
        raise VisionError("detected box is too small for depth sampling")
    inset_x = (right - left) * inset_fraction
    inset_y = (bottom - top) * inset_fraction
    x0, x1 = int(math.floor(left + inset_x)), int(math.ceil(right - inset_x))
    y0, y1 = int(math.floor(top + inset_y)), int(math.ceil(bottom - inset_y))
    if x1 <= x0 or y1 <= y0:
        raise VisionError("detected box has no interior depth region")
    region = depth_raw[y0:y1, x0:x1].astype(np.float64) * depth_scale * 1000.0
    valid_mask = (
        np.isfinite(region)
        & (region >= minimum_depth_m * 1000.0)
        & (region <= maximum_depth_m * 1000.0)
    )
    valid = region[valid_mask]
    if valid.size < minimum_pixels:
        raise VisionError(
            f"only {valid.size} valid depth pixels inside detection; "
            f"need {minimum_pixels}"
        )
    bin_ids = np.floor(valid / cluster_bin_mm).astype(np.int64)
    dominant_bin, _ = Counter(bin_ids.tolist()).most_common(1)[0]
    center_mm = (dominant_bin + 0.5) * cluster_bin_mm
    cluster_mask = valid_mask & (np.abs(region - center_mm) <= cluster_tolerance_mm)
    cluster = region[cluster_mask]
    fraction = float(cluster.size / valid.size)
    if cluster.size < minimum_pixels or fraction < minimum_cluster_fraction:
        raise VisionError(
            f"dominant depth cluster has {cluster.size} pixels "
            f"({fraction:.2f}); need {minimum_pixels} and "
            f"{minimum_cluster_fraction:.2f}"
        )
    rows, columns = np.nonzero(cluster_mask)
    pixel = (
        int(round(x0 + float(np.median(columns)))),
        int(round(y0 + float(np.median(rows)))),
    )
    median_mm = float(np.median(cluster))
    mad_mm = float(np.median(np.abs(cluster - median_mm)))
    return pixel, median_mm, {
        "valid_box_depth_pixels": int(valid.size),
        "depth_cluster_pixels": int(cluster.size),
        "depth_cluster_fraction": fraction,
        "spatial_depth_mad_mm": mad_mm,
    }


def camera_point_from_pixel(
    intrinsics: Any, pixel: tuple[int, int], corrected_depth_mm: float
) -> np.ndarray:
    import pyrealsense2 as rs

    point = rs.rs2_deproject_pixel_to_point(
        intrinsics,
        [float(pixel[0]), float(pixel[1])],
        corrected_depth_mm / 1000.0,
    )
    return np.asarray(point, dtype=np.float64) * 1000.0


def robust_pixel_depth(
    depth_raw: np.ndarray,
    pixel: list[int] | tuple[int, int],
    depth_scale: float,
    radius: int,
    minimum_pixels: int,
    minimum_depth_m: float,
    maximum_depth_m: float,
) -> tuple[float, dict[str, float | int]]:
    if depth_raw.ndim != 2:
        raise VisionError("aligned depth frame must be two-dimensional")
    if radius <= 0 or minimum_pixels <= 0:
        raise VisionError("depth radius and minimum pixels must be positive")
    x, y = int(pixel[0]), int(pixel[1])
    height, width = depth_raw.shape
    if not 0 <= x < width or not 0 <= y < height:
        raise VisionError(f"pixel {[x, y]} is outside {width}x{height}")
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    region = depth_raw[y0:y1, x0:x1].astype(np.float64) * depth_scale * 1000.0
    valid = region[
        np.isfinite(region)
        & (region >= minimum_depth_m * 1000.0)
        & (region <= maximum_depth_m * 1000.0)
    ]
    if valid.size < minimum_pixels:
        raise VisionError(
            f"only {valid.size} valid depth pixels around palm; "
            f"need {minimum_pixels}"
        )
    median_mm = float(np.median(valid))
    mad_mm = float(np.median(np.abs(valid - median_mm)))
    tolerance_mm = max(10.0, 3.0 * mad_mm)
    inliers = valid[np.abs(valid - median_mm) <= tolerance_mm]
    if inliers.size < minimum_pixels:
        raise VisionError(
            f"only {inliers.size} stable depth pixels around palm; "
            f"need {minimum_pixels}"
        )
    return float(np.median(inliers)), {
        "valid_palm_depth_pixels": int(valid.size),
        "palm_depth_inlier_pixels": int(inliers.size),
        "palm_depth_mad_mm": mad_mm,
    }


def annotate_capture(
    frame: np.ndarray,
    slot: int,
    prompt: str,
    result: dict[str, Any],
) -> np.ndarray:
    annotated = frame.copy()
    status = str(result["status"])
    color = (0, 190, 0) if status == "detected" else (0, 0, 255)
    candidates = [
        candidate
        for candidate in result.get("candidates", [result])
        if candidate.get("status") == "detected"
    ]
    for index, candidate in enumerate(candidates, start=1):
        x0, y0, x1, y1 = np.rint(candidate["box_xyxy"]).astype(int)
        cv2.rectangle(annotated, (x0, y0), (x1, y1), color, 2)
        cv2.drawMarker(
            annotated,
            tuple(candidate["pixel_uv"]),
            (0, 255, 255),
            cv2.MARKER_CROSS,
            24,
            2,
        )
        if "long_axis_pixels" in candidate:
            axis = np.rint(candidate["long_axis_pixels"]).astype(int)
            cv2.line(
                annotated,
                tuple(axis[0]),
                tuple(axis[1]),
                (255, 0, 255),
                3,
            )
        cv2.putText(
            annotated,
            f"#{index} {candidate['score']:.2f}",
            (x0, max(45, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    if candidates:
        text = (
            f"scan {slot}: {prompt} - {len(candidates)} candidate(s)"
        )
    else:
        text = f"scan {slot}: {prompt} - {status}"
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        text,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        1,
        cv2.LINE_AA,
    )
    return annotated
