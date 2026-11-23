#!/usr/bin/env python3
"""OpenCV bowl and candy detection for the GTC demo (no robot, no network).

The scan-pose camera looks at the table at an angle, so the bowl's round rim
is an ellipse in the photo. PlaneView warps the photo onto the horizontal
plane at rim height using the hand-eye calibration; there the rim is a circle
of known size, found with a Hough transform and confirmed by how much of the
circle lies on image edges. Colour is not used to find the bowl. Candies are
the pixels inside the rim whose colour differs from the bowl's own colour
(learned from a ring just inside the rim), kept only when they form
candy-sized rectangles; touching candies are split.

Projecting from the camera between two horizontal planes is a uniform scaling
about the point below the camera, so shapes and edge angles seen on the rim
plane are exact, and a candy's true position and size follow from its height
(point_at_height, true_size). demo.py measures those heights live from depth.

Evaluate saved captures (PNG + JSON from capture_scan_images.py) without any
robot or service connection; annotated images go to <captures>/bowl-eval:

    python GTC-demo/bowl_vision.py GTC-demo/captures
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

BOWL_SETTINGS = {
    # Table plane in the camera-derived base frame: depth readings of the bare
    # table measured Z -3.5..-2.1 mm (hand-eye-realsense435 click_coordinates.json).
    "table_z_mm": -3.0,
    "bowl_diameter_mm": 152.4,      # 6 inches across the rim
    "bowl_height_mm": 76.2,         # 3 inches, table to rim
    # Wide enough for the inner edge of the bowl's red rim band (r ~66 mm in the
    # 2026-11-23 photos) as well as the outer rim (r 76 mm).
    "bowl_radius_tolerance": 0.15,
    # Top-down view covers the demo workspace plus room for a rim past its edge.
    "view_x_mm": [50.0, 690.0],
    "view_y_mm": [-260.0, 300.0],
    "mm_per_px": 1.0,
    "blur_px": 5,
    "canny_low": 40,
    "canny_high": 120,
    # Low accumulator threshold: Hough only proposes circles; edge support decides.
    "hough_accumulator": 15,
    "maximum_circle_candidates": 25,
    "edge_tolerance_px": 2,
    # Real rims had edge support 0.84-0.99; rings printed on the hand board 0.57 or less.
    "minimum_rim_support": 0.75,
    # Fraction of the rim inside the photo; bowls at the workspace edge were 46-83% visible.
    "minimum_rim_visible_fraction": 0.45,
    # Candy search skips this ring inside the detected rim (rim band, highlights).
    "rim_margin_mm": 12.0,
    # Candies are the foil colours (OpenCV HSV, hue 0-179): red, green, and gold.
    # Each colour is segmented on its own so touching candies of different
    # colours separate; ribbons and prints are filled by the closing kernel.
    "candy_colours_hsv": {
        "red": [[[0, 110, 60], [10, 255, 255]], [[165, 110, 60], [179, 255, 255]]],
        "green": [[[40, 70, 40], [90, 255, 255]]],
        "gold": [[[12, 90, 80], [35, 255, 255]]],
    },
    "candy_close_kernel_px": 5,
    "mask_kernel_px": 3,
    "candy_size_mm": [25.4, 25.4, 19.05],  # 1 x 1 x 0.75 inch cuboid
    "candy_size_tolerance": 0.3,
    "minimum_candy_fill": 0.65,
    "maximum_candy_aspect": 1.8,
    "split_peak_fraction": 0.6,     # distance-transform peaks used as watershed seeds
    "maximum_chain_candies": 4,     # split straight rows of up to this many candies
    # Merged same-colour blobs in a packed bowl: candy-sized squares are fitted
    # at distance-transform peaks at least this fraction of the smallest half side.
    "heap_peak_fraction": 0.8,
    "maximum_heap_candidates": 12,
    "minimum_candies": 1,
    # Live checks (demo.py): depth just inside the rim on the side away from the
    # camera reads the top of the bowl's inner wall; a flat circle reads the table.
    "rim_depth_samples": 8,
    "rim_depth_inset_mm": 4.0,
    "rim_depth_arc_deg": 120.0,
    "minimum_rim_depth_samples": 4,
    "minimum_rim_height_fraction": 0.5,
    # Grasp selection inside the bowl.
    "grip_open_mm": 40.0,
    "finger_thickness_mm": 10.0,
    "finger_width_mm": 20.0,
    "bowl_wall_mm": 5.0,
    "wall_margin_mm": 5.0,
    # Deepest candy top below the rim the gripper may reach without the wrist or
    # camera touching the rim. Placeholder until measured on the arm.
    "maximum_descent_below_rim_mm": 40.0,
}


class BowlVisionError(ValueError):
    """Invalid calibration, settings, or image for bowl detection."""


def camera_matrix(intrinsics: dict) -> np.ndarray:
    fx, fy, ppx, ppy = (float(intrinsics[key]) for key in ("fx", "fy", "ppx", "ppy"))
    if not all(math.isfinite(value) for value in (fx, fy, ppx, ppy)) or fx <= 0 or fy <= 0:
        raise BowlVisionError("camera intrinsics must be finite with positive focal lengths")
    return np.array([[fx, 0.0, ppx], [0.0, fy, ppy], [0.0, 0.0, 1.0]])


def apply_homography(matrix: np.ndarray, points: Any) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    mapped = np.column_stack([points, np.ones(len(points))]) @ np.asarray(matrix).T
    return mapped[:, :2] / mapped[:, 2:3]


class PlaneView:
    """Mapping between photo pixels, a horizontal base-frame plane, and a top-down raster.

    The raster has +X (away from the robot) up and +Y to the left, at mm_per_px.
    """

    def __init__(
        self, intrinsics: dict, camera_to_base: Any, plane_z_mm: float,
        x_range_mm: Any, y_range_mm: Any, mm_per_px: float,
    ) -> None:
        transform = np.asarray(camera_to_base, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise BowlVisionError("camera_to_base must be a finite 4x4 transform")
        (x_min, x_max), (y_min, y_max) = (map(float, x_range_mm), map(float, y_range_mm))
        if not (x_min < x_max and y_min < y_max and mm_per_px > 0):
            raise BowlVisionError("view ranges must be increasing and mm_per_px positive")
        rotation_t = transform[:3, :3].T
        origin = transform[:3, 3]
        self.plane_z_mm = float(plane_z_mm)
        self.camera_position = origin
        if origin[2] <= self.plane_z_mm:
            raise BowlVisionError("camera must be above the plane")
        # camera point = R^T (P - t) for P = (X, Y, plane_z): linear in (X, Y, 1).
        to_camera = np.column_stack([
            rotation_t[:, 0], rotation_t[:, 1], rotation_t[:, 2] * self.plane_z_mm - rotation_t @ origin,
        ])
        self.mm_per_px = float(mm_per_px)
        self.width = int(math.ceil((y_max - y_min) / mm_per_px))
        self.height = int(math.ceil((x_max - x_min) / mm_per_px))
        self.raster_to_plane_h = np.array([
            [0.0, -mm_per_px, x_max], [-mm_per_px, 0.0, y_max], [0.0, 0.0, 1.0],
        ])
        self.plane_to_raster_h = np.linalg.inv(self.raster_to_plane_h)
        self.plane_to_image_h = camera_matrix(intrinsics) @ to_camera
        self.image_to_plane_h = np.linalg.inv(self.plane_to_image_h)
        self.raster_to_image_h = self.plane_to_image_h @ self.raster_to_plane_h
        columns, rows = np.meshgrid(np.arange(self.width), np.arange(self.height))
        depth = (to_camera @ self.raster_to_plane_h)[2]
        self._raster_depth_mm = depth[0] * columns + depth[1] * rows + depth[2]

    def plane_to_image(self, points_mm: Any) -> np.ndarray:
        return apply_homography(self.plane_to_image_h, points_mm)

    def image_to_plane(self, pixels: Any) -> np.ndarray:
        return apply_homography(self.image_to_plane_h, pixels)

    def plane_to_raster(self, points_mm: Any) -> np.ndarray:
        return apply_homography(self.plane_to_raster_h, points_mm)

    def raster_to_plane(self, raster_px: Any) -> np.ndarray:
        return apply_homography(self.raster_to_plane_h, raster_px)

    def raster_direction_to_base(self, direction: Any) -> np.ndarray:
        column, row = (float(value) for value in direction)
        return np.array([-row, -column]) * self.mm_per_px

    def warp(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the top-down raster and a mask of raster pixels the photo actually covers."""
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        size = (self.width, self.height)
        top = cv2.warpPerspective(frame_bgr, self.raster_to_image_h, size, flags=flags)
        coverage = cv2.warpPerspective(
            np.full(frame_bgr.shape[:2], 255, np.uint8), self.raster_to_image_h, size,
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        )
        valid = (coverage > 0) & (self._raster_depth_mm > 50.0)
        return top, valid.astype(np.uint8) * 255


def point_at_height(view: PlaneView, plane_xy: Any, z_mm: float) -> np.ndarray:
    """Return the base-frame point at height z on the camera ray through a plane point."""
    camera = view.camera_position
    plane_point = np.array([*np.asarray(plane_xy, dtype=np.float64)[:2], view.plane_z_mm])
    scale = (float(z_mm) - camera[2]) / (view.plane_z_mm - camera[2])
    return camera + (plane_point - camera) * scale


def true_size(view: PlaneView, apparent_mm: float, z_mm: float) -> float:
    """Convert a length seen on the view plane to its true length at height z."""
    camera_z = view.camera_position[2]
    return float(apparent_mm) * (camera_z - float(z_mm)) / (camera_z - view.plane_z_mm)


def bowl_view(intrinsics: dict, camera_to_base: Any, settings: dict = BOWL_SETTINGS) -> PlaneView:
    """PlaneView at the bowl's rim height over the configured top-down area."""
    return PlaneView(
        intrinsics, camera_to_base, settings["table_z_mm"] + settings["bowl_height_mm"],
        settings["view_x_mm"], settings["view_y_mm"], settings["mm_per_px"],
    )


def circle_points(center: Any, radius: float, count: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    return np.asarray(center, dtype=np.float64) + radius * np.column_stack([np.cos(angles), np.sin(angles)])


def rim_support(edges: np.ndarray, valid: np.ndarray, center: Any, radius_px: float) -> tuple[float, float]:
    """Return (fraction of the visible circle on edges, fraction of the circle visible)."""
    samples = np.rint(circle_points(center, radius_px, 180)).astype(int)
    height, width = edges.shape
    inside = (samples[:, 0] >= 0) & (samples[:, 0] < width) & (samples[:, 1] >= 0) & (samples[:, 1] < height)
    samples = samples[inside]
    visible = samples[valid[samples[:, 1], samples[:, 0]] > 0]
    if not len(visible):
        return 0.0, 0.0
    hits = edges[visible[:, 1], visible[:, 0]] > 0
    return float(hits.mean()), len(visible) / 180.0


def find_rim_circles(top: np.ndarray, valid: np.ndarray, settings: dict) -> list[dict[str, Any]]:
    """Return bowl-sized circles on the top-down view, best edge support first."""
    gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
    if np.any(valid):
        # Fill uncovered raster with the covered mean so the photo border adds no circle edges.
        gray = np.where(valid > 0, gray, int(gray[valid > 0].mean())).astype(np.uint8)
    blur = int(settings["blur_px"]) | 1
    gray = cv2.GaussianBlur(gray, (blur, blur), 0)
    edges = cv2.Canny(gray, settings["canny_low"], settings["canny_high"])
    edges[cv2.erode(valid, np.ones((5, 5), np.uint8)) == 0] = 0
    tolerance = int(settings["edge_tolerance_px"])
    edges = cv2.dilate(edges, np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8))
    radius_px = settings["bowl_diameter_mm"] / 2.0 / settings["mm_per_px"]
    spread = settings["bowl_radius_tolerance"]
    found = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
        param1=settings["canny_high"], param2=settings["hough_accumulator"],
        minRadius=int(radius_px * (1 - spread)), maxRadius=int(math.ceil(radius_px * (1 + spread))),
    )
    circles = []
    for column, row, radius in ([] if found is None else found[0][: settings["maximum_circle_candidates"]]):
        support, visible = rim_support(edges, valid, (column, row), radius)
        circles.append({
            "raster_center": [float(column), float(row)], "radius_px": float(radius),
            "rim_support": support, "visible_fraction": visible,
            "accepted": support >= settings["minimum_rim_support"]
            and visible >= settings["minimum_rim_visible_fraction"],
        })
    return sorted(circles, key=lambda circle: circle["rim_support"], reverse=True)


def candy_side_limits(view: PlaneView, settings: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    """Apparent (long, short) side ranges on the rim plane for a candy anywhere in the bowl.

    A candy on the bowl floor looks smaller on the rim plane than one at the rim.
    """
    sides = sorted(settings["candy_size_mm"], reverse=True)
    camera_z = view.camera_position[2]
    smallest = (camera_z - view.plane_z_mm) / (camera_z - settings["table_z_mm"])
    tolerance = settings["candy_size_tolerance"]
    long_range = (sides[1] * smallest * (1 - tolerance), sides[0] * (1 + tolerance))
    short_range = (sides[2] * smallest * (1 - tolerance), sides[1] * (1 + tolerance))
    return long_range, short_range


def describe_region(region: np.ndarray, view: PlaneView, settings: dict) -> dict[str, Any] | None:
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    rect = cv2.minAreaRect(contour)
    (column, row), (side_a, side_b), _ = rect
    corners = cv2.boxPoints(rect)
    long_edge = max((corners[1] - corners[0], corners[2] - corners[1]), key=np.linalg.norm)
    if np.linalg.norm(long_edge) < 1e-6:
        return None
    direction = view.raster_direction_to_base(long_edge / np.linalg.norm(long_edge))
    scale = view.mm_per_px
    long_mm, short_mm = max(side_a, side_b) * scale, min(side_a, side_b) * scale
    return {
        "raster_center": [float(column), float(row)],
        "raster_corners": corners.tolist(),
        "long_mm": long_mm, "short_mm": short_mm,
        "area_mm2": area * scale * scale,
        "fill_ratio": area / (side_a * side_b) if side_a * side_b > 0 else 0.0,
        "edge_yaw_deg": math.degrees(math.atan2(direction[1], direction[0])) % 180.0,
    }


def split_chain(shape: dict, count: int) -> list[dict[str, Any]]:
    """Split a straight row of `count` touching candies into equal parts along its length."""
    corners = np.asarray(shape["raster_corners"])
    edges = [corners[1] - corners[0], corners[2] - corners[1]]
    long_edge = max(edges, key=np.linalg.norm)
    short_edge = min(edges, key=np.linalg.norm)
    center = np.asarray(shape["raster_center"])
    part_length = shape["long_mm"] / count
    # A part shorter than the row's width has its long edge across the row.
    swapped = part_length < shape["short_mm"]
    parts = []
    for index in range(count):
        offset = (index - (count - 1) / 2.0) / count
        part_center = center + long_edge * offset
        half_long, half_short = long_edge / (2 * count), short_edge / 2.0
        part_corners = [part_center + sx * half_long + sy * half_short for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        parts.append({
            **shape, "raster_center": part_center.tolist(), "raster_corners": np.asarray(part_corners).tolist(),
            "long_mm": max(part_length, shape["short_mm"]), "short_mm": min(part_length, shape["short_mm"]),
            "area_mm2": shape["area_mm2"] / count, "split": f"row of {count}",
            "edge_yaw_deg": (shape["edge_yaw_deg"] + (90.0 if swapped else 0.0)) % 180.0,
        })
    return parts


def classify_shape(shape: dict, limits: tuple, settings: dict) -> tuple[list[dict], list[str]]:
    """Return (candies, rejection reasons) for one segmented region."""
    (long_min, long_max), (short_min, short_max) = limits
    aspect = shape["long_mm"] / max(shape["short_mm"], 1e-6)
    shape_ok = shape["fill_ratio"] >= settings["minimum_candy_fill"]
    if (
        shape_ok and long_min <= shape["long_mm"] <= long_max and short_min <= shape["short_mm"] <= short_max
        and aspect <= settings["maximum_candy_aspect"]
    ):
        return [shape], []
    if shape_ok and short_min <= shape["short_mm"] <= short_max:
        for count in range(2, int(settings["maximum_chain_candies"]) + 1):
            part = shape["long_mm"] / count
            if long_min <= max(part, shape["short_mm"]) <= long_max and short_min <= min(part, shape["short_mm"]):
                if max(part, shape["short_mm"]) / min(part, shape["short_mm"]) <= settings["maximum_candy_aspect"]:
                    return split_chain(shape, count), []
    reasons = []
    if not shape_ok:
        reasons.append("fill")
    if not long_min <= shape["long_mm"] <= long_max:
        reasons.append("length")
    if not short_min <= shape["short_mm"] <= short_max:
        reasons.append("width")
    if aspect > settings["maximum_candy_aspect"]:
        reasons.append("aspect")
    return [], reasons


def candy_colour_masks(top: np.ndarray, inner: np.ndarray, settings: dict) -> dict[str, np.ndarray]:
    """One cleaned mask per candy colour inside the bowl interior."""
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (settings["candy_close_kernel_px"],) * 2)
    opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (settings["mask_kernel_px"],) * 2)
    masks = {}
    for name, ranges in settings["candy_colours_hsv"].items():
        mask = np.zeros(inner.shape, np.uint8)
        for low, high in ranges:
            mask |= cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))
        mask = cv2.morphologyEx(cv2.morphologyEx(mask & inner, cv2.MORPH_CLOSE, close), cv2.MORPH_OPEN, opening)
        masks[name] = mask & inner
    return masks


def dominant_edge_angle(gray: np.ndarray, region: np.ndarray, center: Any, radius: float) -> float:
    """Dominant edge direction (radians, modulo 90 degrees) of a region around a raster point.

    Gradient orientations are folded by 4 (square symmetry) and averaged,
    weighted by magnitude; candy edges and ribbons run along the same two directions.
    """
    height, width = gray.shape
    x0, y0 = max(0, int(center[0] - radius)), max(0, int(center[1] - radius))
    x1, y1 = min(width, int(center[0] + radius) + 1), min(height, int(center[1] + radius) + 1)
    patch = gray[y0:y1, x0:x1].astype(np.float32)
    support = cv2.dilate(region[y0:y1, x0:x1], np.ones((3, 3), np.uint8)) > 0
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)[support]
    angle = np.arctan2(gy, gx)[support]
    if magnitude.sum() <= 1e-6:
        return 0.0
    folded = np.sum(magnitude * np.exp(4j * angle))
    return float(np.angle(folded) / 4.0)


def heap_candidates(region: np.ndarray, gray: np.ndarray, view: PlaneView, limits: tuple, settings: dict) -> list[dict[str, Any]]:
    """Candy-sized squares inside a merged same-colour blob (a packed bowl).

    Centres are distance-transform peaks where a candy-sized square fits; each
    square is oriented by the local edge direction and kept if the colour
    fills at least minimum_candy_fill of it.
    """
    scale = view.mm_per_px
    (long_min, long_max), _ = limits
    distance = cv2.distanceTransform(region, cv2.DIST_L2, 5)
    threshold = settings["heap_peak_fraction"] * long_min / 2.0 / scale
    spacing = 0.7 * long_min / scale
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(spacing) | 1,) * 2)
    peaks = np.argwhere((distance >= threshold) & (distance >= cv2.dilate(distance, kernel)))
    order = sorted(peaks.tolist(), key=lambda rc: -distance[rc[0], rc[1]])
    chosen: list[tuple[int, int]] = []
    for row, column in order:
        if all(math.hypot(row - r, column - c) >= spacing for r, c in chosen):
            chosen.append((row, column))
        if len(chosen) >= settings["maximum_heap_candidates"]:
            break
    squares = []
    for row, column in chosen:
        if 2.0 * distance[row, column] > long_max / scale:
            continue  # wider than one candy in every direction: cannot tell which candy this is
        side = float(np.clip(2.0 * distance[row, column], long_min / scale, long_max / scale))
        theta = dominant_edge_angle(gray, region, (column, row), side)
        along = np.array([math.cos(theta), math.sin(theta)]) * side / 2.0
        across = np.array([-along[1], along[0]])
        center = np.array([column, row], dtype=np.float64)
        corners = np.array([center + sa * along + sc * across for sa, sc in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        square = np.zeros(region.shape, np.uint8)
        cv2.fillConvexPoly(square, np.int32(np.round(corners)), 1)
        fill = float(region[square > 0].mean() / 255.0) if np.any(square) else 0.0
        if fill < settings["minimum_candy_fill"]:
            continue
        direction = view.raster_direction_to_base(along / np.linalg.norm(along))
        squares.append({
            "raster_center": center.tolist(), "raster_corners": corners.tolist(),
            "long_mm": side * scale, "short_mm": side * scale, "area_mm2": side * side * fill * scale * scale,
            "fill_ratio": fill, "edge_yaw_deg": math.degrees(math.atan2(direction[1], direction[0])) % 180.0,
            "split": "heap",
        })
    return squares


def segment_candies(
    top: np.ndarray, valid: np.ndarray, view: PlaneView, circle: dict, settings: dict,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    """Return (candies, rejected regions, candy mask) inside one rim circle on the top-down view.

    Each candy colour is segmented separately. A region is a candy if it is a
    candy-sized rectangle (touching candies are split at distance-transform
    peaks or as straight rows); larger merged regions are searched for
    candy-sized squares (heap_candidates).
    """
    height, width = valid.shape
    center = tuple(int(round(value)) for value in circle["raster_center"])
    scale = view.mm_per_px
    inner_px = circle["radius_px"] - settings["rim_margin_mm"] / scale
    if inner_px <= 0:
        raise BowlVisionError("rim margin leaves no bowl interior")
    inner = np.zeros((height, width), np.uint8)
    cv2.circle(inner, center, int(inner_px), 255, -1)
    inner &= valid
    gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
    limits = candy_side_limits(view, settings)
    minimum_area_px = (limits[1][0] / scale) ** 2 * 0.5
    single_area_px = (limits[0][0] / scale) ** 2
    candies, rejected = [], []
    combined = np.zeros_like(inner)
    for colour, mask in candy_colour_masks(top, inner, settings).items():
        combined |= mask
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < minimum_area_px:
                continue
            component = (labels == label).astype(np.uint8) * 255
            for region in split_touching(component, top, settings):
                shape = describe_region(region, view, settings)
                if shape is None:
                    continue
                accepted, reasons = classify_shape(shape, limits, settings)
                if not accepted and shape["area_mm2"] >= single_area_px * scale * scale:
                    accepted = heap_candidates(region, gray, view, limits, settings)
                if accepted:
                    candies.extend({**item, "colour": colour} for item in accepted)
                elif shape["area_mm2"] >= minimum_area_px * scale * scale:
                    rejected.append({**shape, "colour": colour, "reasons": reasons})
    return candies, rejected, combined


def split_touching(component: np.ndarray, top: np.ndarray, settings: dict) -> list[np.ndarray]:
    """Split one blob at distinct distance-transform peaks with a watershed."""
    distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
    peaks = (distance >= settings["split_peak_fraction"] * distance.max()).astype(np.uint8)
    seed_count, seeds = cv2.connectedComponents(peaks)
    if seed_count <= 2:
        return [component]
    markers = seeds.astype(np.int32) + 1
    markers[(component > 0) & (peaks == 0)] = 0
    cv2.watershed(np.ascontiguousarray(top), markers)
    return [
        ((markers == label) & (component > 0)).astype(np.uint8) * 255
        for label in range(2, seed_count + 1)
        if np.any((markers == label) & (component > 0))
    ]


def to_base(view: PlaneView, item: dict[str, Any]) -> dict[str, Any]:
    """Add rim-plane base coordinates and photo pixels to a raster-space record."""
    center = view.raster_to_plane([item["raster_center"]])[0]
    out = {**item, "plane_center_mm": center.tolist(), "pixel": view.plane_to_image([center])[0].round(1).tolist()}
    if "raster_corners" in item:
        out["pixel_corners"] = view.plane_to_image(view.raster_to_plane(item["raster_corners"])).round(1).tolist()
    return out


def find_bowl(
    frame_bgr: np.ndarray, intrinsics: dict, camera_to_base: Any, settings: dict = BOWL_SETTINGS,
) -> dict[str, Any]:
    """Find the bowl and the candies in it.

    status is "detected" (bowl with candies), "empty" (bowl-like circle without
    candies), or "not_found". Positions are on the rim plane; demo.py corrects
    them with live depth.
    """
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise BowlVisionError("bowl detection requires a 3-channel BGR image")
    view = bowl_view(intrinsics, camera_to_base, settings)
    top, valid = view.warp(frame_bgr)
    circles = find_rim_circles(top, valid, settings)
    result: dict[str, Any] = {"status": "not_found", "circles_considered": len(circles), "candies": [], "rejected_candies": []}
    empty = None
    for circle in (circle for circle in circles if circle["accepted"]):
        candies, rejected, _ = segment_candies(top, valid, view, circle, settings)
        bowl = {
            **to_base(view, circle),
            "radius_mm": circle["radius_px"] * view.mm_per_px,
            "plane_z_mm": view.plane_z_mm,
        }
        bowl["rim_depth_pixels"] = rim_depth_pixels(view, bowl["plane_center_mm"], bowl["radius_mm"], settings)
        if len(candies) >= settings["minimum_candies"]:
            result.update(
                status="detected", bowl=bowl,
                candies=[to_base(view, candy) for candy in candies],
                rejected_candies=[to_base(view, item) for item in rejected],
            )
            return result
        if empty is None:
            empty = (bowl, rejected)
    if empty is not None:
        result.update(status="empty", bowl=empty[0], rejected_candies=[to_base(view, item) for item in empty[1]])
    return result


def rim_depth_pixels(view: PlaneView, center_xy: Any, radius_mm: float, settings: dict) -> list[list[int]]:
    """Photo pixels just inside the rim on the far side, where depth reads the inner wall top."""
    center = np.asarray(center_xy, dtype=np.float64)[:2]
    away = center - view.camera_position[:2]
    heading = math.atan2(away[1], away[0])
    half_arc = math.radians(settings["rim_depth_arc_deg"]) / 2.0
    angles = np.linspace(heading - half_arc, heading + half_arc, int(settings["rim_depth_samples"]))
    radius = radius_mm - settings["rim_depth_inset_mm"]
    points = center + radius * np.column_stack([np.cos(angles), np.sin(angles)])
    return view.plane_to_image(points).round().astype(int).tolist()


def check_rim_height(rim_z_mm: list[float | None], settings: dict = BOWL_SETTINGS) -> tuple[bool, str]:
    """Require depth at the rim to show a raised bowl, which rejects flat circles."""
    values = [float(value) for value in rim_z_mm if value is not None and math.isfinite(float(value))]
    if len(values) < settings["minimum_rim_depth_samples"]:
        return False, f"only {len(values)} rim depth readings (need {settings['minimum_rim_depth_samples']})"
    height = float(np.median(values)) - settings["table_z_mm"]
    needed = settings["minimum_rim_height_fraction"] * settings["bowl_height_mm"]
    if height < needed:
        return False, f"rim is {height:.0f} mm above the table (need {needed:.0f} mm): flat circle, not a bowl"
    return True, f"rim {height:.0f} mm above the table"


def jaw_footprint(center_xy: Any, closing_yaw_deg: float, settings: dict) -> np.ndarray:
    """Corners (4x2, mm) of the area the open jaws occupy around a candy centre."""
    yaw = math.radians(closing_yaw_deg)
    along = np.array([math.cos(yaw), math.sin(yaw)])
    across = np.array([-along[1], along[0]])
    half_along = settings["grip_open_mm"] / 2.0 + settings["finger_thickness_mm"]
    half_across = settings["finger_width_mm"] / 2.0
    center = np.asarray(center_xy, dtype=np.float64)[:2]
    return np.array([center + sa * half_along * along + sc * half_across * across for sa, sc in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def wall_clear_radius(settings: dict, rim_radius_mm: float | None = None) -> float:
    """Radius around the bowl centre the open jaws must stay inside.

    Uses the detected rim radius when given (it may be the rim band's inner
    edge), but never more than the nominal inner wall.
    """
    nominal = settings["bowl_diameter_mm"] / 2.0 - settings["bowl_wall_mm"]
    inner = nominal if rim_radius_mm is None else min(float(rim_radius_mm), nominal)
    return inner - settings["wall_margin_mm"]


def jaw_clear_of_wall(
    center_xy: Any, closing_yaw_deg: float, bowl_center_xy: Any, settings: dict, rim_radius_mm: float | None = None,
) -> bool:
    """True if the open jaws around a candy centre stay inside the bowl with the wall margin."""
    footprint = jaw_footprint(center_xy, closing_yaw_deg, settings)
    center = np.asarray(bowl_center_xy, dtype=np.float64)[:2]
    return bool(np.all(np.linalg.norm(footprint - center, axis=1) <= wall_clear_radius(settings, rim_radius_mm)))


def gripper_clear_of_bowl(
    grasp_tip_mm: Any, tool_axis: Any, closing_yaw_deg: float, bowl_center_xy: Any,
    rim_z_mm: float, finger_extension_mm: float, settings: dict, step_mm: float = 2.0,
    rim_radius_mm: float | None = None,
) -> bool:
    """True if a (possibly tilted) open gripper at the grasp pose stays inside the bowl opening.

    The gripper is modelled as the open-jaw footprint swept along the tool axis
    from the fingertips (finger_extension_mm past the tool tip) up to the rim.
    A descent along the tool axis only occupies part of that same sweep.
    """
    tip = np.asarray(grasp_tip_mm, dtype=np.float64)
    axis = np.asarray(tool_axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    if axis[2] >= -1e-6:
        return False
    fingertip_z = float((tip + axis * finger_extension_mm)[2])
    for height in np.arange(fingertip_z, float(rim_z_mm) + step_mm, step_mm):
        point = tip + axis * ((height - tip[2]) / axis[2])
        if not jaw_clear_of_wall(point[:2], closing_yaw_deg, bowl_center_xy, settings, rim_radius_mm):
            return False
    return True


def grasp_options(
    candies: list[dict[str, Any]], bowl_center_xy: Any, rim_z_mm: float, settings: dict = BOWL_SETTINGS,
    rim_radius_mm: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Order pickable candies (highest first) and give each its closing directions.

    Each candy needs live "base_point_mm" (its top centre from depth) and
    "true_long_mm"/"true_short_mm". The jaws close across one candy edge, so
    there are two closing directions; one is kept if the open jaws stay clear
    of the bowl wall. Returns (pickable, rejected) with reasons.
    """
    sides = sorted(settings["candy_size_mm"], reverse=True)
    tolerance = settings["candy_size_tolerance"]
    center = np.asarray(bowl_center_xy, dtype=np.float64)[:2]
    pickable, rejected = [], []
    for candy in candies:
        point = np.asarray(candy["base_point_mm"], dtype=np.float64)
        reasons = []
        if not (sides[1] * (1 - tolerance) <= candy["true_long_mm"] <= sides[0] * (1 + tolerance)
                and sides[2] * (1 - tolerance) <= candy["true_short_mm"] <= sides[1] * (1 + tolerance)):
            reasons.append(f"size {candy['true_long_mm']:.0f} x {candy['true_short_mm']:.0f} mm is not a candy")
        if point[2] < rim_z_mm - settings["maximum_descent_below_rim_mm"]:
            reasons.append(f"too deep ({rim_z_mm - point[2]:.0f} mm below the rim)")
        closing = [
            yaw % 180.0 for yaw in (candy["edge_yaw_deg"] + 90.0, candy["edge_yaw_deg"])
            if jaw_clear_of_wall(point, yaw % 180.0, center, settings, rim_radius_mm)
        ]
        if not closing:
            reasons.append("too close to the bowl wall for the open jaws")
        if reasons:
            rejected.append({**candy, "reasons": reasons})
        else:
            pickable.append({**candy, "closing_yaw_deg": closing})
    pickable.sort(key=lambda candy: (-candy["base_point_mm"][2], float(np.linalg.norm(np.asarray(candy["base_point_mm"][:2]) - center))))
    return pickable, rejected


def annotate(frame_bgr: np.ndarray, result: dict[str, Any], chosen: dict[str, Any] | None = None) -> np.ndarray:
    """Draw the rim, candies (green), rejected regions (red), and the chosen candy (yellow)."""
    image = frame_bgr.copy()
    bowl = result.get("bowl")
    if bowl:
        rim = bowl.get("rim_outline_pixels") or bowl.get("rim_depth_pixels")
        cv2.polylines(image, [np.asarray(rim, np.int32)], "rim_outline_pixels" in bowl, (255, 200, 0), 2)
    for item, colour in [(c, (0, 200, 0)) for c in result.get("candies", [])] + [
        (r, (0, 0, 255)) for r in result.get("rejected_candies", [])
    ]:
        if "pixel_corners" in item:
            cv2.polylines(image, [np.asarray(item["pixel_corners"], np.int32)], True, colour, 2)
    if chosen and "pixel_corners" in chosen:
        cv2.polylines(image, [np.asarray(chosen["pixel_corners"], np.int32)], True, (0, 255, 255), 3)
    label = f"{result['status']}: {len(result.get('candies', []))} candies"
    if bowl:
        label += f", rim r={bowl['radius_mm']:.0f}mm support={bowl['rim_support']:.2f}"
    cv2.putText(image, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(image, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return image


def with_rim_outline(result: dict[str, Any], intrinsics: dict, camera_to_base: Any, settings: dict) -> dict[str, Any]:
    """Add a dense rim outline in photo pixels for drawing."""
    bowl = result.get("bowl")
    if bowl:
        view = bowl_view(intrinsics, camera_to_base, settings)
        bowl["rim_outline_pixels"] = view.plane_to_image(
            circle_points(bowl["plane_center_mm"], bowl["radius_mm"], 90)
        ).round().astype(int).tolist()
    return result


def evaluate_captures(capture_dir: Path, output_dir: Path, settings: dict = BOWL_SETTINGS) -> int:
    """Run find_bowl on each capture PNG with JSON metadata; write annotated images.

    Returns the number of images processed.
    """
    images = sorted(capture_dir.glob("*.png"))
    if not images:
        raise BowlVisionError(f"no PNG captures in {capture_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    for image_path in images:
        metadata_path = image_path.with_suffix(".json")
        if not metadata_path.exists():
            print(f"{image_path.name}: skipped (no {metadata_path.name})")
            continue
        metadata = json.loads(metadata_path.read_text())
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        intrinsics, camera_to_base = metadata["camera_intrinsics"], metadata["camera_to_base"]
        result = with_rim_outline(find_bowl(frame, intrinsics, camera_to_base, settings), intrinsics, camera_to_base, settings)
        cv2.imwrite(str(output_dir / f"bowl_{image_path.name}"), annotate(frame, result))
        line = f"{image_path.name}: {result['status']}"
        if result.get("bowl"):
            bowl = result["bowl"]
            line += (f", bowl at {np.round(bowl['plane_center_mm'], 0).tolist()} mm r={bowl['radius_mm']:.0f} "
                     f"support={bowl['rim_support']:.2f} visible={bowl['visible_fraction']:.2f}")
        line += f", candies={len(result['candies'])}, rejected={len(result['rejected_candies'])}"
        print(line)
        processed += 1
    return processed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("captures", type=Path, help="folder of capture PNG/JSON files")
    parser.add_argument("--output-dir", type=Path, help="annotated image folder (default: <captures>/bowl-eval)")
    args = parser.parse_args()
    try:
        count = evaluate_captures(args.captures, args.output_dir or args.captures / "bowl-eval")
    except (BowlVisionError, OSError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Processed {count} capture(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
