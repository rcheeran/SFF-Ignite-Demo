#!/usr/bin/env python3
"""CPU OpenCV candy detector: warm-colour wrapper blobs with long-axis geometry.

Returns the same detection fields as the YOLO detection client, so camera
localization, long-axis midpoint pickup, workspace checks, and planning are
unchanged. Thresholds were tuned on scan-pose captures of Starburst wrappers;
they have not been validated against hands or other warm-coloured objects.

Run directly to evaluate saved captures (PNG + JSON with YOLO references)
without any robot or service connection:

    python Demo-Ignite/opencv_detector.py Demo-Ignite/captures
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "k3s-vision-detection"))
from detection_client import DetectionError  # noqa: E402

OPENCV_DETECTOR_NAME = "opencv-hsv-candy"
OPENCV_DETECTOR_SETTINGS = {
    # OpenCV hue is 0..179: red/orange/yellow wrap-around plus pink-red.
    "hue_low_max": 30,
    "hue_high_min": 165,
    "saturation_min": 100,
    "value_min": 100,
    "close_kernel_px": 7,
    "open_kernel_px": 5,
    "minimum_area_px": 1500.0,
    "maximum_area_px": 9000.0,
    # Wrappers measured aspect 2.22-2.63 / fill 0.70-0.78 at the scan pose; warm
    # patterned fabric reached aspect 1.75 at fill >= 0.6, so 2.0 separates them.
    "minimum_aspect_ratio": 2.0,
    "maximum_aspect_ratio": 3.5,
    "minimum_fill_ratio": 0.6,
    # Matches the YOLO path's long-axis endpoint convention (40% of length each side).
    "axis_half_length_fraction": 0.4,
    # Physical limits measured on the table plane: a wrapper must be shorter than
    # 4 inches and narrower than 2 inches, so colour alone cannot select a target.
    "maximum_length_mm": 101.6,
    "maximum_width_mm": 50.8,
}


def warm_mask(frame_bgr: np.ndarray, settings: dict = OPENCV_DETECTOR_SETTINGS) -> np.ndarray:
    """Return the cleaned binary mask of saturated, bright warm-colour pixels."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    floor_s, floor_v = settings["saturation_min"], settings["value_min"]
    mask = cv2.inRange(hsv, (0, floor_s, floor_v), (settings["hue_low_max"], 255, 255))
    mask |= cv2.inRange(hsv, (settings["hue_high_min"], floor_s, floor_v), (179, 255, 255))
    for operation, size in (
        (cv2.MORPH_CLOSE, settings["close_kernel_px"]), (cv2.MORPH_OPEN, settings["open_kernel_px"]),
    ):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.morphologyEx(mask, operation, kernel)
    return mask


def detect_candies(
    frame_bgr: np.ndarray, settings: dict = OPENCV_DETECTOR_SETTINGS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (accepted detections sorted by score, rejected blobs with reasons).

    The score is the blob's fill ratio within its rotated rectangle (a shape
    score from 0 to 1), not a learned confidence.
    """
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise DetectionError("OpenCV candy detection requires a 3-channel BGR image")
    height, width = frame_bgr.shape[:2]
    contours, _ = cv2.findContours(warm_mask(frame_bgr, settings), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    accepted, rejected = [], []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 50.0:
            continue
        (center_x, center_y), (side_a, side_b), angle = cv2.minAreaRect(contour)
        long_side, short_side = max(side_a, side_b), min(side_a, side_b)
        aspect = long_side / short_side if short_side > 1.0 else math.inf
        fill = area / (side_a * side_b) if side_a * side_b > 0 else 0.0
        x, y, box_width, box_height = cv2.boundingRect(contour)
        candidate = {
            "box_xyxy": [float(x), float(y), float(x + box_width), float(y + box_height)],
            "area_px": area, "aspect_ratio": aspect, "fill_ratio": fill,
        }
        reasons = [
            reason for reason, failed in (
                ("area", not settings["minimum_area_px"] <= area <= settings["maximum_area_px"]),
                ("aspect", not settings["minimum_aspect_ratio"] <= aspect <= settings["maximum_aspect_ratio"]),
                ("fill", fill < settings["minimum_fill_ratio"]),
            ) if failed
        ]
        if reasons:
            rejected.append({**candidate, "reasons": reasons})
            continue
        theta = math.radians(angle + (90.0 if side_b > side_a else 0.0))
        direction = np.array([math.cos(theta), math.sin(theta)])
        half_length = long_side * settings["axis_half_length_fraction"]
        center = np.array([center_x, center_y])
        endpoints = np.stack([center - direction * half_length, center + direction * half_length])
        endpoints[:, 0] = np.clip(endpoints[:, 0], 0, width - 1)
        endpoints[:, 1] = np.clip(endpoints[:, 1], 0, height - 1)
        accepted.append({
            "score": float(min(1.0, fill)),
            "box_xyxy": candidate["box_xyxy"],
            "long_axis_pixels": endpoints.tolist(),
            "orientation_aspect_ratio": float(aspect),
            "orientation_mask_fraction": float(area / (box_width * box_height)),
        })
    return sorted(accepted, key=lambda item: item["score"], reverse=True), rejected


def table_point_from_pixel(
    pixel: np.ndarray, intrinsics: dict, camera_to_base: np.ndarray, plane_z_mm: float,
) -> np.ndarray:
    """Intersect a pixel's camera ray with the horizontal base-frame plane Z = plane_z_mm.

    Uses pinhole intrinsics (the accepted D435 calibration has zero distortion).
    """
    fx, fy, ppx, ppy = (float(intrinsics[key]) for key in ("fx", "fy", "ppx", "ppy"))
    if not all(math.isfinite(value) for value in (fx, fy, ppx, ppy)) or fx <= 0 or fy <= 0:
        raise ValueError("camera intrinsics must be finite with positive focal lengths")
    ray = np.asarray(camera_to_base, dtype=np.float64)[:3, :3] @ np.array(
        [(float(pixel[0]) - ppx) / fx, (float(pixel[1]) - ppy) / fy, 1.0]
    )
    origin = np.asarray(camera_to_base, dtype=np.float64)[:3, 3]
    if ray[2] >= -1e-9:
        raise ValueError("camera ray does not point down toward the table plane")
    distance = (float(plane_z_mm) - origin[2]) / ray[2]
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError("table plane is not in front of the camera")
    return origin + distance * ray


def candy_size_mm(
    long_axis_pixels: Any, aspect_ratio: float, intrinsics: dict,
    camera_to_base: np.ndarray, plane_z_mm: float,
    settings: dict = OPENCV_DETECTOR_SETTINGS,
) -> tuple[float, float]:
    """Return the wrapper's (length, width) in mm, measured on the table plane.

    The full pixel length is recovered from the long-axis endpoints (which span
    2 x axis_half_length_fraction of it) and the width from the rotated-rectangle
    aspect ratio. Length is the distance between the projected axis ends. Width
    is the projected image-normal segment's component perpendicular to that
    table axis, which removes perspective skew from a tilted camera. At the
    current ~27-degree scan tilt, modelled errors are overestimates only: for an
    86 x 33 mm wrapper, length 0-9 mm (largest near image corners) and width
    under 2 mm; errors scale with object size.
    """
    endpoints = np.asarray(long_axis_pixels, dtype=np.float64)
    if endpoints.shape != (2, 2) or not np.all(np.isfinite(endpoints)):
        raise ValueError("long-axis pixels must contain two finite image points")
    aspect_ratio = float(aspect_ratio)
    axis = endpoints[1] - endpoints[0]
    axis_length_px = float(np.linalg.norm(axis))
    if axis_length_px <= 1e-6 or not math.isfinite(aspect_ratio) or aspect_ratio < 1.0:
        raise ValueError("candy long axis or aspect ratio is degenerate")
    unit = axis / axis_length_px
    normal = np.array([-unit[1], unit[0]])
    length_px = axis_length_px / (2.0 * settings["axis_half_length_fraction"])
    width_px = length_px / aspect_ratio
    center = endpoints.mean(axis=0)

    def on_table(pixel: np.ndarray) -> np.ndarray:
        return table_point_from_pixel(pixel, intrinsics, camera_to_base, plane_z_mm)

    length_vector = on_table(center + unit * length_px / 2.0) - on_table(center - unit * length_px / 2.0)
    length_mm = float(np.linalg.norm(length_vector))
    table_axis = length_vector / length_mm
    width_vector = on_table(center + normal * width_px / 2.0) - on_table(center - normal * width_px / 2.0)
    width_mm = float(np.linalg.norm(width_vector - (width_vector @ table_axis) * table_axis))
    return length_mm, width_mm


def check_candy_size(length_mm: float, width_mm: float, settings: dict = OPENCV_DETECTOR_SETTINGS) -> None:
    """Reject wrappers not shorter than the length limit and narrower than the width limit."""
    if length_mm >= settings["maximum_length_mm"] or width_mm >= settings["maximum_width_mm"]:
        raise ValueError(
            f"candidate size {length_mm:.1f} x {width_mm:.1f} mm is not within the "
            f"{settings['maximum_length_mm']:.1f} x {settings['maximum_width_mm']:.1f} mm candy limit"
        )


class OpenCVCandyDetector:
    """Drop-in replacement for the YOLO DetectionClient used by VisionClient."""

    model_name = OPENCV_DETECTOR_NAME
    uses_cuda = False
    url = "local-cpu"

    def __init__(self, settings: dict | None = None) -> None:
        self.settings = dict(OPENCV_DETECTOR_SETTINGS if settings is None else settings)

    def check(self) -> None:
        """No service to contact; validate settings so misconfiguration fails at startup."""
        values = self.settings
        if not (
            0 <= values["hue_low_max"] < values["hue_high_min"] <= 179
            and 0 <= values["saturation_min"] <= 255 and 0 <= values["value_min"] <= 255
            and 0 < values["minimum_area_px"] < values["maximum_area_px"]
            and 1 <= values["minimum_aspect_ratio"] < values["maximum_aspect_ratio"]
            and 0 <= values["minimum_fill_ratio"] <= 1
            and 0 < values["maximum_width_mm"] <= values["maximum_length_mm"]
        ):
            raise DetectionError("invalid OpenCV candy detector settings")

    def detect_png(self, image_png: bytes, prompt: str, threshold: float) -> list[dict[str, Any]]:
        """Detect candies in a lossless PNG; the prompt is ignored."""
        if not image_png.startswith(b"\x89PNG\r\n\x1a\n"):
            raise DetectionError("detection requires a lossless PNG image")
        frame = cv2.imdecode(np.frombuffer(image_png, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise DetectionError("could not decode PNG for OpenCV candy detection")
        detections, _ = detect_candies(frame, self.settings)
        return [detection for detection in detections if detection["score"] >= threshold]


def box_iou(left: list[float], right: list[float]) -> float:
    """Intersection-over-union of two xyxy boxes."""
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = lambda box: (box[2] - box[0]) * (box[3] - box[1])  # noqa: E731
    union = area(left) + area(right) - intersection
    return intersection / union if union > 0 else 0.0


def evaluate_captures(capture_dir: Path, output_dir: Path, plane_z_mm: float = 12.0) -> int:
    """Compare OpenCV detections with saved YOLO references and write annotated images.

    When capture metadata includes camera_to_base and intrinsics, the runtime
    physical-size limit is applied using a table plane at plane_z_mm.
    """
    images = sorted(capture_dir.glob("*.png"))
    if not images:
        raise DetectionError(f"no PNG captures in {capture_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    disagreements = 0
    for path in images:
        frame = cv2.imread(str(path))
        metadata_path = path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        reference = metadata.get("yolo_reference", {}).get("detections") if metadata else None
        detections, rejected = detect_candies(frame)
        sizes: list[str] = []
        if metadata.get("camera_to_base") is not None and metadata.get("camera_intrinsics"):
            kept = []
            for detection in detections:
                length, width = candy_size_mm(
                    detection["long_axis_pixels"], detection["orientation_aspect_ratio"],
                    metadata["camera_intrinsics"], np.asarray(metadata["camera_to_base"]), plane_z_mm,
                )
                try:
                    check_candy_size(length, width)
                except ValueError:
                    rejected.append({
                        "box_xyxy": detection["box_xyxy"], "area_px": 0.0,
                        "reasons": [f"size {length:.0f}x{width:.0f}mm"],
                    })
                    continue
                sizes.append(f"{length:.0f}x{width:.0f}mm")
                kept.append(detection)
            detections = kept
        annotated = frame.copy()
        unmatched = set(range(len(detections)))
        notes = []
        for yolo in reference or []:
            x0, y0, x1, y1 = np.rint(yolo["box_xyxy"]).astype(int)
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (255, 128, 0), 2)
            best = max(unmatched, key=lambda index: box_iou(yolo["box_xyxy"], detections[index]["box_xyxy"]), default=None)
            if best is None or box_iou(yolo["box_xyxy"], detections[best]["box_xyxy"]) < 0.3:
                notes.append(f"MISSED YOLO box {np.round(yolo['box_xyxy']).astype(int).tolist()}")
            else:
                unmatched.discard(best)
        for index, detection in enumerate(detections):
            x0, y0, x1, y1 = np.rint(detection["box_xyxy"]).astype(int)
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 220, 0), 2)
            ends = np.rint(detection["long_axis_pixels"]).astype(int)
            cv2.line(annotated, tuple(ends[0]), tuple(ends[1]), (255, 0, 255), 2)
            middle = tuple(np.rint(np.mean(detection["long_axis_pixels"], axis=0)).astype(int))
            cv2.drawMarker(annotated, middle, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
            if reference is not None and index in unmatched:
                notes.append(f"EXTRA OpenCV box {np.round(detection['box_xyxy']).astype(int).tolist()}")
        for blob in rejected:
            x0, y0, x1, y1 = np.rint(blob["box_xyxy"]).astype(int)
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 0, 255), 1)
        disagreements += len(notes)
        summary = f"OpenCV={len(detections)} YOLO={'n/a' if reference is None else len(reference)}"
        if sizes:
            summary += f" sizes={', '.join(sizes)}"
        print(f"{path.name}: {summary}" + ("" if not notes else "  " + "; ".join(notes)))
        if rejected:
            print("   rejected blobs: " + ", ".join(
                (f"{round(blob['area_px'])}px " if blob["area_px"] else "")
                + "+".join(blob["reasons"]) for blob in rejected
            ))
        legend = "blue=YOLO green=OpenCV red=rejected magenta=axis yellow=pickup midpoint"
        for color, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(annotated, legend, (6, frame.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, thickness)
        cv2.imwrite(str(output_dir / f"opencv_{path.name}"), annotated)
    print(f"{len(images)} image(s); {disagreements} disagreement(s) with YOLO references; annotated: {output_dir}")
    return disagreements


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("captures", type=Path, help="folder of capture PNG/JSON files")
    parser.add_argument("--output-dir", type=Path, help="annotated image folder (default: <captures>/opencv-eval)")
    parser.add_argument(
        "--plane-z-mm", type=float, default=12.0,
        help="table-surface base Z used for physical candy size (default 12 mm)",
    )
    args = parser.parse_args()
    try:
        disagreements = evaluate_captures(
            args.captures, args.output_dir or args.captures / "opencv-eval", args.plane_z_mm,
        )
    except (DetectionError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 1 if disagreements else 0


if __name__ == "__main__":
    raise SystemExit(main())
