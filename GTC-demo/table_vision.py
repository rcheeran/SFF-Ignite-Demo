#!/usr/bin/env python3
"""OpenCV candy detection on the table for the GTC demo (no robot, no network).

The scan-pose photo is warped onto the horizontal plane through the tops of
candies lying on the table (PlaneView from bowl_vision.py), so a candy's top
face appears at its true size and position. A candy is any blob of the foil
colours (bowl_vision.candy_colour_masks); the colours are merged before
closing, so a ribbon in another colour does not split a candy.

The camera looks down at an angle and also sees each candy's near side faces,
which on the top plane stretch the blob toward the point below the camera.
Seen from the camera, a cuboid's outline is its top face swept toward that
point, so the top face is what remains of the blob after intersecting it with
itself moved back by the same homothety (trim_side_faces). Blobs cut off by the
photo edge, the gripper, or the edge of the workspace are rejected because
their centre is unknown. demo.py then measures each candy's top from depth and
keeps only real candies lying on the table.

Evaluate saved captures (PNG + JSON from capture_scan_images.py) without any
robot or service connection; annotated images go to <captures>/table-eval:

    python GTC-demo/table_vision.py GTC-demo/captures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

import bowl_vision

SHARED_KEYS = (
    "table_z_mm", "mm_per_px", "candy_colours_hsv", "candy_close_kernel_px", "mask_kernel_px",
    "candy_size_mm", "candy_size_tolerance", "minimum_candy_fill", "maximum_candy_aspect",
    "split_peak_fraction", "maximum_chain_candies", "gripper_max_value", "gripper_min_area_px",
    "gripper_dilate_px",
)
TABLE_SETTINGS = {
    **{key: bowl_vision.BOWL_SETTINGS[key] for key in SHARED_KEYS},
    # Candies lie on a 1 x 1 inch face, so their tops are 0.75 inch above the table.
    "candy_top_above_table_mm": 19.05,
    # Closes ribbons, prints, and highlights inside a candy once its colours are merged.
    "candy_merge_close_px": 9,
    # Pick area; demo.py replaces it with its workspace bounds.
    "view_x_mm": [144.0, 595.0],
    "view_y_mm": [-170.0, 206.317],
    # Blobs within this many raster pixels of the photo edge, the gripper, or the
    # pick-area edge are cut off, so their centre is unknown.
    "edge_margin_px": 3,
    # Depth-measured top above the table: 19 mm lying flat, 25 mm on its side.
    # Lower is a print or the table itself; higher is not a lone candy (a hand, a stack).
    "candy_top_height_mm": [10.0, 32.0],
}


class TableVisionError(ValueError):
    """Invalid calibration, settings, or image for table candy detection."""


def table_view(intrinsics: dict, camera_to_base: Any, settings: dict = TABLE_SETTINGS) -> bowl_vision.PlaneView:
    """PlaneView on the candy-top plane over the pick area."""
    return bowl_vision.PlaneView(
        intrinsics, camera_to_base, settings["table_z_mm"] + settings["candy_top_above_table_mm"],
        settings["view_x_mm"], settings["view_y_mm"], settings["mm_per_px"],
    )


def edge_band(valid: np.ndarray, margin_px: int) -> np.ndarray:
    """Raster pixels near anything the photo does not show, including the raster border."""
    hidden = np.where(valid > 0, 0, 255).astype(np.uint8)
    hidden[[0, -1], :] = 255
    hidden[:, [0, -1]] = 255
    size = 2 * int(margin_px) + 1
    return cv2.dilate(hidden, np.ones((size, size), np.uint8))


def candy_mask(top: np.ndarray, valid: np.ndarray, settings: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Merged, closed, hole-filled mask of all candy colours, and the per-colour masks."""
    colours = bowl_vision.candy_colour_masks(top, valid, settings)
    merged = np.zeros(valid.shape, np.uint8)
    for mask in colours.values():
        merged |= mask
    size = int(settings["candy_merge_close_px"]) | 1
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(merged)
    cv2.drawContours(filled, contours, -1, 255, -1)
    return filled & valid, colours


def trim_side_faces(blob: np.ndarray, view: bowl_vision.PlaneView, settings: dict) -> np.ndarray:
    """Top face of a candy blob on the candy-top plane, without its visible side faces.

    The candy's bottom outline maps onto the top plane by a homothety about the
    point below the camera (ratio < 1), and the blob is the top face swept
    toward it. Pixels that stay in the blob when moved back out by the same
    homothety form the top face.
    """
    camera = view.camera_position
    ratio = (camera[2] - view.plane_z_mm) / (camera[2] - settings["table_z_mm"])
    nadir = view.plane_to_raster([camera[:2]])[0]
    homothety = np.float32([[ratio, 0.0, (1 - ratio) * nadir[0]], [0.0, ratio, (1 - ratio) * nadir[1]]])
    moved_out = cv2.warpAffine(
        blob, homothety, (blob.shape[1], blob.shape[0]), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
    )
    return cv2.morphologyEx(blob & moved_out, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def find_table_candies(
    frame_bgr: np.ndarray, intrinsics: dict, camera_to_base: Any, settings: dict = TABLE_SETTINGS,
) -> dict[str, Any]:
    """Find candies lying on the table in a scan-pose photo.

    status is "detected" or "not_found". Positions are on the candy-top plane;
    demo.py corrects them with live depth.
    """
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise TableVisionError("table candy detection requires a 3-channel BGR image")
    view = table_view(intrinsics, camera_to_base, settings)
    top, valid = view.warp(frame_bgr, bowl_vision.gripper_mask(frame_bgr, settings))
    band = edge_band(valid, settings["edge_margin_px"])
    limits = bowl_vision.candy_side_limits(view, settings)
    scale = view.mm_per_px
    minimum_area_px = (limits[1][0] / scale) ** 2 * 0.5
    merged, colours = candy_mask(top, valid, settings)
    candies, rejected = [], []
    count, labels, stats, _ = cv2.connectedComponentsWithStats(merged)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < minimum_area_px:
            continue
        blob = (labels == label).astype(np.uint8) * 255
        cut_off = bool(np.any(blob & band))
        face = trim_side_faces(blob, view, settings)
        for region in bowl_vision.split_touching(face, top, settings) if np.any(face) else []:
            shape = bowl_vision.describe_region(region, view, settings)
            if shape is None:
                continue
            colour = max(colours, key=lambda name: int(np.count_nonzero(colours[name] & region)))
            accepted, reasons = ([], ["cut off by the photo edge, gripper, or workspace edge"]) if cut_off \
                else bowl_vision.classify_shape(shape, limits, settings)
            if accepted:
                candies.extend({**item, "colour": colour} for item in accepted)
            elif shape["area_mm2"] >= minimum_area_px * scale * scale:
                rejected.append({**shape, "colour": colour, "reasons": reasons})
    return {
        "status": "detected" if candies else "not_found",
        "plane_z_mm": view.plane_z_mm,
        "candies": [bowl_vision.to_base(view, candy) for candy in candies],
        "rejected_candies": [bowl_vision.to_base(view, item) for item in rejected],
    }


def grasp_options(candies: list[dict[str, Any]], settings: dict = TABLE_SETTINGS) -> tuple[list[dict], list[dict]]:
    """Keep real candies lying on the table, nearest the robot first, with both closing directions.

    Each candy needs live "base_point_mm" (its top centre from depth) and
    "true_long_mm"/"true_short_mm". The jaws close across one candy edge, so
    there are two closing directions. Returns (pickable, rejected) with reasons.
    """
    sides = sorted(settings["candy_size_mm"], reverse=True)
    tolerance = settings["candy_size_tolerance"]
    low, high = (float(value) for value in settings["candy_top_height_mm"])
    pickable, rejected = [], []
    for candy in candies:
        reasons = []
        if not (sides[1] * (1 - tolerance) <= candy["true_long_mm"] <= sides[0] * (1 + tolerance)
                and sides[2] * (1 - tolerance) <= candy["true_short_mm"] <= sides[1] * (1 + tolerance)):
            reasons.append(f"size {candy['true_long_mm']:.0f} x {candy['true_short_mm']:.0f} mm is not a candy")
        height = float(candy["base_point_mm"][2]) - float(settings["table_z_mm"])
        if not low <= height <= high:
            reasons.append(f"top is {height:.0f} mm above the table (a candy is {low:.0f}-{high:.0f} mm)")
        if reasons:
            rejected.append({**candy, "reasons": reasons})
        else:
            edge = float(candy["edge_yaw_deg"])
            pickable.append({**candy, "closing_yaw_deg": [(edge + 90.0) % 180.0, edge % 180.0]})
    pickable.sort(key=lambda candy: float(np.linalg.norm(np.asarray(candy["base_point_mm"][:2], dtype=np.float64))))
    return pickable, rejected


def evaluate_captures(capture_dir: Path, output_dir: Path, settings: dict = TABLE_SETTINGS) -> int:
    """Run find_table_candies on each capture PNG with JSON metadata; write annotated images."""
    images = sorted(capture_dir.glob("*.png"))
    if not images:
        raise TableVisionError(f"no PNG captures in {capture_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    for image_path in images:
        metadata_path = image_path.with_suffix(".json")
        if not metadata_path.exists():
            print(f"{image_path.name}: skipped (no {metadata_path.name})")
            continue
        metadata = json.loads(metadata_path.read_text())
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        result = find_table_candies(frame, metadata["camera_intrinsics"], metadata["camera_to_base"], settings)
        cv2.imwrite(str(output_dir / f"table_{image_path.name}"), bowl_vision.annotate(frame, result))
        located = [np.round(candy["plane_center_mm"], 0).tolist() for candy in result["candies"]]
        print(f"{image_path.name}: {result['status']}, candies at {located}, rejected={len(result['rejected_candies'])}")
        processed += 1
    return processed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("captures", type=Path, help="folder of capture PNG/JSON files")
    parser.add_argument("--output-dir", type=Path, help="annotated image folder (default: <captures>/table-eval)")
    args = parser.parse_args()
    try:
        count = evaluate_captures(args.captures, args.output_dir or args.captures / "table-eval")
    except (TableVisionError, bowl_vision.BowlVisionError, OSError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Processed {count} capture(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
