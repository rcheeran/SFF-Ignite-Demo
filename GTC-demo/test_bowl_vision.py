import json
import math
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

import bowl_vision as bv

# Real D435 intrinsics and a camera placed like the scan pose: 418 mm up,
# tilted 26.5 degrees from vertical toward +X.
INTRINSICS = {"width": 640, "height": 480, "fx": 605.99, "fy": 605.55, "ppx": 314.85, "ppy": 255.70}
RED, GREEN, GOLD = (40, 40, 200), (50, 150, 40), (40, 180, 220)
BOWL_CENTER = (360.0, 30.0)


def ray_plane_point(pixel, c2b, plane_z):
    """Independent reference: intersect the pixel's camera ray with Z = plane_z."""
    ray = c2b[:3, :3] @ np.array([
        (pixel[0] - INTRINSICS["ppx"]) / INTRINSICS["fx"], (pixel[1] - INTRINSICS["ppy"]) / INTRINSICS["fy"], 1.0,
    ])
    origin = c2b[:3, 3]
    return origin + ray * (plane_z - origin[2]) / ray[2]


def camera_to_base(position=(154.0, 41.0, 418.0), tilt_deg=26.5):
    tilt = math.radians(tilt_deg)
    optical = np.array([math.sin(tilt), 0.0, -math.cos(tilt)])
    right = np.array([0.0, -1.0, 0.0])
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack([right, np.cross(optical, right), optical])
    transform[:3, 3] = position
    return transform


def render(bowl_center=BOWL_CENTER, bowl_radius=76.2, candies=(), seed=0):
    """Draw a top-down scene on the rim plane and project it into the camera photo.

    candies: (x, y, yaw_deg, colour[, (length, width)]) on the rim plane.
    """
    c2b = camera_to_base()
    view = bv.bowl_view(INTRINSICS, c2b)
    table = (150, 160, 170)
    canvas = np.empty((view.height, view.width, 3), np.uint8)
    canvas[:] = table
    noise = np.random.default_rng(seed).normal(0, 4, canvas.shape)
    canvas = np.clip(canvas + noise, 0, 255).astype(np.uint8)
    if bowl_center is not None:
        center = tuple(int(round(v)) for v in view.plane_to_raster([bowl_center])[0])
        radius = int(bowl_radius / view.mm_per_px)
        cv2.circle(canvas, center, radius, (170, 110, 40), -1)
        cv2.circle(canvas, center, radius, (230, 170, 100), 4)
        cv2.circle(canvas, center, radius + 2, (60, 60, 60), 2)
    for x, y, yaw, colour, *size in candies:
        length, width = size[0] if size else (25.4, 25.4)
        raster_center = view.plane_to_raster([(x, y)])[0]
        direction = view.plane_to_raster([(x + math.cos(math.radians(yaw)), y + math.sin(math.radians(yaw)))])[0] - raster_center
        angle = math.degrees(math.atan2(direction[1], direction[0]))
        box = np.int32(np.round(cv2.boxPoints((tuple(raster_center), (length, width), angle))))
        cv2.fillPoly(canvas, [box], colour)
        cv2.polylines(canvas, [box], True, tuple(int(v * 0.6) for v in colour), 1)
    return cv2.warpPerspective(canvas, view.raster_to_image_h, (640, 480), borderValue=table), c2b


def yaw_error_mod(left, right, period):
    return abs((left - right + period / 2) % period - period / 2)


class BowlDetectionTests(unittest.TestCase):
    def test_bowl_and_separate_candies_are_located_on_the_rim_plane(self):
        candies = [(350, 10, 0, RED), (380, 50, 30, GREEN), (330, 55, 60, GOLD)]
        frame, c2b = render(candies=candies)
        result = bv.find_bowl(frame, INTRINSICS, c2b)
        self.assertEqual(result["status"], "detected")
        bowl = result["bowl"]
        self.assertLess(np.linalg.norm(np.subtract(bowl["plane_center_mm"], BOWL_CENTER)), 3.0)
        self.assertAlmostEqual(bowl["radius_mm"], 76.2, delta=3.0)
        self.assertEqual(len(bowl["rim_depth_pixels"]), bv.BOWL_SETTINGS["rim_depth_samples"])
        # Depth samples sit just inside the rim, on the side away from the camera.
        view = bv.bowl_view(INTRINSICS, c2b)
        center = np.asarray(bowl["plane_center_mm"])
        for point in view.image_to_plane(bowl["rim_depth_pixels"]):
            self.assertAlmostEqual(np.linalg.norm(point - center), bowl["radius_mm"] - 4.0, delta=1.5)
            self.assertGreater(np.dot(point - center, center - c2b[:2, 3]), 0.0)
        self.assertEqual(len(result["candies"]), 3)
        for x, y, yaw, _ in candies:
            found = min(result["candies"], key=lambda c: np.linalg.norm(np.subtract(c["plane_center_mm"], (x, y))))
            self.assertLess(np.linalg.norm(np.subtract(found["plane_center_mm"], (x, y))), 2.0)
            self.assertAlmostEqual(found["long_mm"], 25.4, delta=2.0)
            self.assertLess(yaw_error_mod(found["edge_yaw_deg"], yaw, 90.0), 4.0)  # square: any edge

    def test_rectangular_candy_reports_its_long_edge_direction(self):
        frame, c2b = render(candies=[(355, 25, 40, RED, (25.4, 19.05))])
        (candy,) = bv.find_bowl(frame, INTRINSICS, c2b)["candies"]
        self.assertAlmostEqual(candy["long_mm"], 25.4, delta=2.0)
        self.assertAlmostEqual(candy["short_mm"], 19.05, delta=2.0)
        self.assertLess(yaw_error_mod(candy["edge_yaw_deg"], 40.0, 180.0), 4.0)

    def test_touching_candies_are_split(self):
        for name, candies in (
            ("offset pair", [(350, 20, 0, RED), (372, 38, 0, RED)]),
            ("straight row", [(350, 20, 0, RED), (375.4, 20, 0, RED)]),
        ):
            with self.subTest(name):
                frame, c2b = render(candies=candies)
                found = bv.find_bowl(frame, INTRINSICS, c2b)["candies"]
                self.assertEqual(len(found), 2)
                for x, y, *_ in candies:
                    self.assertLess(min(np.linalg.norm(np.subtract(c["plane_center_mm"], (x, y))) for c in found), 3.0)

    def test_empty_bowl_missing_bowl_and_wrong_size_circle(self):
        for name, kwargs, status in (
            ("empty bowl", {}, "empty"),
            ("no bowl", {"bowl_center": None}, "not_found"),
            ("too small", {"bowl_radius": 45.0, "candies": [(360, 30, 0, RED)]}, "not_found"),
        ):
            with self.subTest(name):
                frame, c2b = render(**kwargs)
                self.assertEqual(bv.find_bowl(frame, INTRINSICS, c2b)["status"], status)

    def test_non_candy_regions_inside_the_bowl_are_rejected(self):
        frame, c2b = render(candies=[(360, 30, 0, RED, (70.0, 8.0)), (340, 60, 0, GOLD)])
        result = bv.find_bowl(frame, INTRINSICS, c2b)
        self.assertEqual(len(result["candies"]), 1)
        self.assertLess(np.linalg.norm(np.subtract(result["candies"][0]["plane_center_mm"], (340, 60))), 2.0)
        self.assertTrue(any("width" in item["reasons"] for item in result["rejected_candies"]))


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.c2b = camera_to_base()
        self.view = bv.bowl_view(INTRINSICS, self.c2b)

    def test_plane_mapping_round_trips_and_matches_ray_intersection(self):
        pixels = np.array([[100.0, 80.0], [320.0, 240.0], [600.0, 400.0]])
        plane = self.view.image_to_plane(pixels)
        np.testing.assert_allclose(self.view.plane_to_image(plane), pixels, atol=1e-6)
        for pixel, plane_xy in zip(pixels, plane):
            expected = ray_plane_point(pixel, self.c2b, -3.0)
            np.testing.assert_allclose(bv.point_at_height(self.view, plane_xy, -3.0), expected, atol=1e-6)

    def test_true_size_inverts_perspective_shrink(self):
        camera_z = self.c2b[2, 3]
        apparent = 25.4 * (camera_z - self.view.plane_z_mm) / (camera_z - 20.0)
        self.assertAlmostEqual(bv.true_size(self.view, apparent, 20.0), 25.4, places=9)

    def test_rim_height_check_rejects_flat_circles_and_missing_depth(self):
        table = bv.BOWL_SETTINGS["table_z_mm"]
        self.assertTrue(bv.check_rim_height([table + 70.0] * 6)[0])
        self.assertTrue(bv.check_rim_height([table + 70.0] * 4 + [table] * 2 + [None, None])[0])
        ok, reason = bv.check_rim_height([table + 2.0] * 8)
        self.assertFalse(ok)
        self.assertIn("flat circle", reason)
        self.assertFalse(bv.check_rim_height([table + 70.0] * 3 + [None] * 5)[0])


class GraspSelectionTests(unittest.TestCase):
    RIM_Z = bv.BOWL_SETTINGS["table_z_mm"] + bv.BOWL_SETTINGS["bowl_height_mm"]

    def candy(self, x, y, z, long_mm=25.4, short_mm=25.4, yaw=0.0):
        return {"base_point_mm": [x, y, z], "true_long_mm": long_mm, "true_short_mm": short_mm, "edge_yaw_deg": yaw}

    def test_highest_pickable_first_with_closing_directions_clear_of_the_wall(self):
        cx, cy = BOWL_CENTER
        candies = [
            self.candy(cx, cy, 50.0),                      # centre: both directions clear
            self.candy(cx + 40, cy, 60.0),                 # near the +X wall: only closing along Y
            self.candy(cx, cy, self.RIM_Z - 50.0),         # deeper than the gripper can reach
            self.candy(cx, cy - 20, 55.0, long_mm=50.0),   # not candy-sized
            self.candy(cx + 60, cy, 65.0),                 # jaws would hit the wall either way
        ]
        pickable, rejected = bv.grasp_options(candies, BOWL_CENTER, self.RIM_Z)
        self.assertEqual([c["base_point_mm"][2] for c in pickable], [60.0, 50.0])
        self.assertEqual(pickable[0]["closing_yaw_deg"], [90.0])
        self.assertEqual(sorted(pickable[1]["closing_yaw_deg"]), [0.0, 90.0])
        reasons = [" ".join(item["reasons"]) for item in rejected]
        self.assertTrue(any("too deep" in reason for reason in reasons))
        self.assertTrue(any("not a candy" in reason for reason in reasons))
        self.assertTrue(any("bowl wall" in reason for reason in reasons))

    def test_narrow_side_is_preferred_for_rectangular_candy(self):
        (candy,), _ = bv.grasp_options(
            [self.candy(*BOWL_CENTER, 50.0, long_mm=25.4, short_mm=19.05, yaw=30.0)], BOWL_CENTER, self.RIM_Z,
        )
        self.assertEqual(candy["closing_yaw_deg"][0], 120.0)  # across the long edge: squeezes the 19 mm side


class EvaluateCapturesTests(unittest.TestCase):
    def test_writes_annotated_images_for_captures_with_metadata(self):
        frame, c2b = render(candies=[(350, 10, 0, RED)])
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            cv2.imwrite(str(folder / "scan1_a.png"), frame)
            (folder / "scan1_a.json").write_text(json.dumps({
                "camera_intrinsics": INTRINSICS, "camera_to_base": c2b.tolist(),
            }))
            cv2.imwrite(str(folder / "no_metadata.png"), frame)
            self.assertEqual(bv.evaluate_captures(folder, folder / "eval"), 1)
            self.assertTrue((folder / "eval" / "bowl_scan1_a.png").exists())


if __name__ == "__main__":
    unittest.main()
