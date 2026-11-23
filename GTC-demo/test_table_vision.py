"""Tests for table_vision.py on rendered scan-pose photos of cuboid candies (no camera, no robot)."""

import math
import unittest
from unittest import mock

import cv2
import numpy as np

import table_vision as tv
from test_bowl_vision import GOLD, GREEN, INTRINSICS, RED, camera_to_base

WHITE = (230, 230, 230)
SETTINGS = tv.TABLE_SETTINGS
SIDE = 25.4


def render_table(candies=(), seed=0, finger=None):
    """Photo of candies lying on the table, drawn as cuboids seen from the scan-pose camera.

    candies: (x, y, yaw_deg, body colour, ribbon colour) for each top-face centre.
    Each candy's visible side faces are its top face swept toward the point below
    the camera, as a cuboid appears on the candy-top plane. finger: photo polygon
    filled black, like the gripper at the bottom of the photo.
    """
    c2b = camera_to_base()
    view = tv.table_view(INTRINSICS, c2b, SETTINGS)
    table = (150, 120, 40)  # blue board
    canvas = np.empty((view.height, view.width, 3), np.uint8)
    canvas[:] = table
    canvas = np.clip(canvas + np.random.default_rng(seed).normal(0, 4, canvas.shape), 0, 255).astype(np.uint8)
    camera = view.camera_position
    ratio = (camera[2] - view.plane_z_mm) / (camera[2] - SETTINGS["table_z_mm"])
    for x, y, yaw, body, ribbon in candies:
        along = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw))]) * SIDE / 2
        across = np.array([-along[1], along[0]])
        top = np.array([(x, y) + sa * along + sc * across for sa, sc in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        bottom = camera[:2] + ratio * (top - camera[:2])
        hull = cv2.convexHull(np.float32(view.plane_to_raster(np.vstack([top, bottom]))))
        cv2.fillConvexPoly(canvas, np.int32(np.round(hull)), tuple(int(v * 0.7) for v in body))
        cv2.fillConvexPoly(canvas, np.int32(np.round(view.plane_to_raster(top))), body)
        for direction in (along, across):
            ends = view.plane_to_raster([(x, y) - direction, (x, y) + direction])
            cv2.line(canvas, tuple(np.int32(np.round(ends[0]))), tuple(np.int32(np.round(ends[1]))), ribbon, 4)
    photo = cv2.warpPerspective(canvas, view.raster_to_image_h, (640, 480), borderValue=table)
    if finger is not None:
        cv2.fillPoly(photo, [np.int32(finger)], (20, 20, 20))
    return photo, c2b


def yaw_error_mod(left, right, period=90.0):
    return abs((left - right + period / 2) % period - period / 2)


class TableDetectionTests(unittest.TestCase):
    def test_candies_with_ribbons_and_visible_sides_are_located_at_their_top_face(self):
        candies = [(300, 0, 0, RED, GOLD), (430, -80, 30, GREEN, WHITE), (380, 120, 60, GOLD, RED)]
        frame, c2b = render_table(candies)
        result = tv.find_table_candies(frame, INTRINSICS, c2b)
        self.assertEqual(result["status"], "detected")
        self.assertEqual(len(result["candies"]), 3)
        for x, y, yaw, body, _ in candies:
            found = min(result["candies"], key=lambda c: np.linalg.norm(np.subtract(c["plane_center_mm"], (x, y))))
            self.assertLess(np.linalg.norm(np.subtract(found["plane_center_mm"], (x, y))), 2.0)
            self.assertAlmostEqual(found["long_mm"], SIDE, delta=3.0)
            self.assertLess(yaw_error_mod(found["edge_yaw_deg"], yaw), 5.0)
            self.assertEqual(found["colour"], {RED: "red", GREEN: "green", GOLD: "gold"}[body])
        # Without the side-face trim, the stretched blobs are not located at their top faces.
        with mock.patch.object(tv, "trim_side_faces", side_effect=lambda blob, *_: blob):
            untrimmed = tv.find_table_candies(frame, INTRINSICS, c2b)["candies"]
        located = [
            any(np.linalg.norm(np.subtract(c["plane_center_mm"], (x, y))) < 2.0 for c in untrimmed)
            for x, y, *_ in candies
        ]
        self.assertFalse(all(located))

    def test_candies_cut_off_by_the_gripper_or_the_workspace_edge_are_rejected(self):
        view = tv.table_view(INTRINSICS, camera_to_base(), SETTINGS)
        near = view.plane_to_image([(300, 40)])[0]
        finger = [[near[0] - 20, 479], [near[0] - 20, near[1]], [near[0] + 60, near[1]], [near[0] + 60, 479]]
        candies = [(300, 40, 0, RED, GOLD), (330, -165, 0, GREEN, WHITE), (330, -60, 20, GOLD, RED)]
        frame, c2b = render_table(candies, finger=finger)
        result = tv.find_table_candies(frame, INTRINSICS, c2b)
        self.assertEqual(len(result["candies"]), 1)
        self.assertLess(np.linalg.norm(np.subtract(result["candies"][0]["plane_center_mm"], (330, -60))), 2.0)
        cut = [item for item in result["rejected_candies"] if any("cut off" in reason for reason in item["reasons"])]
        self.assertGreaterEqual(len(cut), 1)
        self.assertTrue(any(np.linalg.norm(np.subtract(item["plane_center_mm"], (330, -165))) < 15 for item in cut))

    def test_bare_board_has_no_candies(self):
        frame, c2b = render_table()
        self.assertEqual(tv.find_table_candies(frame, INTRINSICS, c2b)["status"], "not_found")


class GraspOptionTests(unittest.TestCase):
    def candy(self, xy=(300.0, 0.0), top_above_table=19.05, size=(25.0, 24.0), edge=10.0):
        return {
            "base_point_mm": [*xy, SETTINGS["table_z_mm"] + top_above_table],
            "true_long_mm": size[0], "true_short_mm": size[1], "edge_yaw_deg": edge,
        }

    def test_candies_lying_on_the_table_are_pickable_nearest_first_with_both_closing_directions(self):
        pickable, rejected = tv.grasp_options([self.candy(xy=(400.0, 50.0)), self.candy(xy=(250.0, 0.0), top_above_table=25.0)])
        self.assertEqual(rejected, [])
        self.assertEqual([candy["base_point_mm"][0] for candy in pickable], [250.0, 400.0])
        self.assertEqual(pickable[0]["closing_yaw_deg"], [100.0, 10.0])

    def test_prints_tall_objects_and_wrong_sizes_are_rejected(self):
        _, rejected = tv.grasp_options([
            self.candy(top_above_table=1.0), self.candy(top_above_table=40.0), self.candy(size=(45.0, 24.0)),
        ])
        reasons = [" ".join(item["reasons"]) for item in rejected]
        self.assertEqual(len(rejected), 3)
        self.assertIn("1 mm above the table", reasons[0])
        self.assertIn("40 mm above the table", reasons[1])
        self.assertIn("not a candy", reasons[2])


if __name__ == "__main__":
    unittest.main()
