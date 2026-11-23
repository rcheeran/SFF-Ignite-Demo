"""Hardware-free tests for the OpenCV candy detector using synthetic images."""

import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

import opencv_detector as detector
from detection_client import DetectionError

WRAPPER_BGR = (30, 80, 230)  # saturated red-orange
BLUE_BGR = (200, 120, 20)


def scene(*shapes, background=(60, 60, 60)):
    """Draw rotated filled rectangles: (center, (long, short), angle_deg, bgr)."""
    image = np.full((480, 640, 3), background, dtype=np.uint8)
    for center, size, angle, color in shapes:
        box = cv2.boxPoints((center, size, angle)).astype(np.int32)
        cv2.fillPoly(image, [box], color)
    return image


def png(image):
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def axis_angle(endpoints):
    vector = np.subtract(endpoints[1], endpoints[0])
    return math.degrees(math.atan2(vector[1], vector[0])) % 180


class OpenCVDetectorTests(unittest.TestCase):
    def test_detects_rotated_wrapper_with_axis_midpoint_and_yolo_fields(self):
        for angle in (0, 30, 75, 120):
            with self.subTest(angle=angle):
                found, _ = detector.detect_candies(scene(((320, 240), (110, 45), angle, WRAPPER_BGR)))
                self.assertEqual(len(found), 1)
                item = found[0]
                self.assertEqual(
                    set(item), {"score", "box_xyxy", "long_axis_pixels",
                                "orientation_aspect_ratio", "orientation_mask_fraction"},
                )
                np.testing.assert_allclose(np.mean(item["long_axis_pixels"], axis=0), [320, 240], atol=2)
                difference = abs((axis_angle(item["long_axis_pixels"]) - angle + 90) % 180 - 90)
                self.assertLess(difference, 3)
                length = np.linalg.norm(np.subtract(*item["long_axis_pixels"]))
                self.assertAlmostEqual(length, 0.8 * 110, delta=6)
                self.assertGreater(item["score"], 0.9)
                self.assertLessEqual(item["score"], 1.0)

    def test_multiple_wrappers_are_separate_detections(self):
        found, _ = detector.detect_candies(scene(
            ((150, 240), (110, 45), 90, WRAPPER_BGR),
            ((320, 240), (110, 45), 90, (40, 200, 240)),  # yellow
            ((490, 240), (110, 45), 60, (130, 50, 230)),  # pink-red (hue wraps near 167)
        ))
        self.assertEqual(len(found), 3)

    def test_non_candy_colours_shapes_and_sizes_are_rejected(self):
        cases = {
            "blue": scene(((320, 240), (110, 45), 0, BLUE_BGR)),
            "dark warm": scene(((320, 240), (110, 45), 0, (10, 30, 70))),
            "pale warm": scene(((320, 240), (110, 45), 0, (190, 200, 230)), background=(40, 40, 40)),
            "tiny speck": scene(((320, 240), (20, 8), 0, WRAPPER_BGR)),
            "square": scene(((320, 240), (70, 70), 0, WRAPPER_BGR)),
            "stubby (aspect 1.6, like patterned fabric)": scene(((320, 240), (80, 50), 0, WRAPPER_BGR)),
            "too elongated": scene(((320, 240), (250, 40), 0, WRAPPER_BGR)),
            "too large": scene(((320, 240), (300, 120), 0, WRAPPER_BGR)),
        }
        for name, image in cases.items():
            with self.subTest(name=name):
                found, _ = detector.detect_candies(image)
                self.assertEqual(found, [])

    def test_touching_wrappers_are_rejected_not_merged_into_a_false_target(self):
        found, rejected = detector.detect_candies(scene(
            ((300, 240), (110, 45), 90, WRAPPER_BGR), ((345, 240), (110, 45), 90, WRAPPER_BGR),
        ))
        self.assertEqual(found, [])
        self.assertIn("aspect", rejected[0]["reasons"])

    def test_png_interface_matches_detection_client_and_threshold(self):
        local = detector.OpenCVCandyDetector()
        local.check()
        self.assertFalse(local.uses_cuda)
        image = png(scene(((320, 240), (110, 45), 30, WRAPPER_BGR)))
        self.assertEqual(len(local.detect_png(image, "ignored prompt", 0.05)), 1)
        self.assertEqual(local.detect_png(image, "ignored prompt", 0.9999), [])
        for invalid in (b"not a png", b"\x89PNG\r\n\x1a\ntruncated"):
            with self.subTest(invalid=invalid), self.assertRaises(DetectionError):
                local.detect_png(invalid, "prompt", 0.05)

    def test_invalid_settings_fail_check(self):
        for change in (
            {"hue_low_max": 170}, {"saturation_min": 300}, {"minimum_area_px": 20000},
            {"minimum_aspect_ratio": 5}, {"minimum_fill_ratio": 1.5},
            {"maximum_width_mm": 0}, {"maximum_width_mm": 120},
        ):
            with self.subTest(change=change), self.assertRaises(DetectionError):
                detector.OpenCVCandyDetector({**detector.OPENCV_DETECTOR_SETTINGS, **change}).check()

    def test_size_limits_are_four_by_two_inches_and_exclusive(self):
        self.assertAlmostEqual(detector.OPENCV_DETECTOR_SETTINGS["maximum_length_mm"], 4 * 25.4)
        self.assertAlmostEqual(detector.OPENCV_DETECTOR_SETTINGS["maximum_width_mm"], 2 * 25.4)
        detector.check_candy_size(101.5, 50.7)
        for length, width in ((101.6, 30), (60, 50.8), (150, 80)):
            with self.subTest(length=length, width=width), self.assertRaisesRegex(ValueError, "candy limit"):
                detector.check_candy_size(length, width)

    def test_size_from_straight_down_camera(self):
        intrinsics = {"fx": 500.0, "fy": 500.0, "ppx": 320.0, "ppy": 240.0}
        camera_to_base = np.array([[1, 0, 0, 300], [0, -1, 0, 0], [0, 0, -1, 500], [0, 0, 0, 1]], float)
        np.testing.assert_allclose(
            detector.table_point_from_pixel([320, 240], intrinsics, camera_to_base, 0), [300, 0, 0],
        )
        # 1 px = 1 mm at 500 mm; endpoints span 80% of a 100 px length; aspect 2.5 -> 40 mm wide.
        length, width = detector.candy_size_mm([[280, 240], [360, 240]], 2.5, intrinsics, camera_to_base, 0)
        self.assertAlmostEqual(length, 100.0, places=6)
        self.assertAlmostEqual(width, 40.0, places=6)

    def test_size_recovers_true_dimensions_through_tilted_camera(self):
        """Model what the detector measures: minAreaRect of the projected wrapper."""
        intrinsics = {"fx": 606.0, "fy": 605.5, "ppx": 314.8, "ppy": 255.7}
        tilt = math.radians(26.6)  # measured optical-axis tilt at the taught scan pose
        rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ np.array([
            [1, 0, 0], [0, math.cos(tilt), -math.sin(tilt)], [0, math.sin(tilt), math.cos(tilt)],
        ])
        camera_to_base = np.eye(4)
        camera_to_base[:3, :3], camera_to_base[:3, 3] = rotation, [300, -150, 418]
        base_to_camera = np.linalg.inv(camera_to_base)

        def project(point):
            camera = base_to_camera @ np.append(point, 1.0)
            return [
                intrinsics["fx"] * camera[0] / camera[2] + intrinsics["ppx"],
                intrinsics["fy"] * camera[1] / camera[2] + intrinsics["ppy"],
            ]

        for pixel in ([320, 250], [120, 120], [520, 400]):
            center = detector.table_point_from_pixel(pixel, intrinsics, camera_to_base, 12)
            for yaw in range(0, 180, 15):
                for true_length, true_width in ((86, 33), (120, 45)):
                    with self.subTest(pixel=pixel, yaw=yaw, length=true_length):
                        along = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0])
                        across = np.array([-along[1], along[0], 0])
                        corners = np.array([
                            project(center + a * along * true_length / 2 + b * across * true_width / 2)
                            for a in (-1, 1) for b in (-1, 1)
                        ], dtype=np.float32)
                        (cx, cy), (side_a, side_b), angle = cv2.minAreaRect(corners)
                        long_side, short_side = max(side_a, side_b), min(side_a, side_b)
                        theta = math.radians(angle + (90.0 if side_b > side_a else 0.0))
                        direction = np.array([math.cos(theta), math.sin(theta)])
                        ends = [np.array([cx, cy]) + sign * direction * long_side * 0.4 for sign in (-1, 1)]
                        length, width = detector.candy_size_mm(
                            ends, long_side / short_side, intrinsics, camera_to_base, 12,
                        )
                        # Errors are conservative: never meaningfully underestimated.
                        self.assertGreaterEqual(length - true_length, -0.5)
                        self.assertLessEqual(length - true_length, 0.12 * true_length)
                        self.assertGreaterEqual(width - true_width, -0.5)
                        self.assertLessEqual(width - true_width, 0.08 * true_width)

    def test_size_rejects_degenerate_geometry_and_rays_that_miss_the_table(self):
        intrinsics = {"fx": 500.0, "fy": 500.0, "ppx": 320.0, "ppy": 240.0}
        looking_up = np.eye(4)
        looking_up[2, 3] = 500
        down = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 500], [0, 0, 0, 1]], float)
        for args in (
            ([[300, 240], [340, 240]], 2.5, intrinsics, looking_up, 0),
            ([[300, 240], [300, 240]], 2.5, intrinsics, down, 0),
            ([[300, 240], [340, 240]], 0.5, intrinsics, down, 0),
            ([[300, 240], [340, 240]], math.nan, intrinsics, down, 0),
            ([[300, 240], [340, 240]], 2.5, {**intrinsics, "fx": 0}, down, 0),
            ([[300, 240]], 2.5, intrinsics, down, 0),
        ):
            with self.subTest(args=args[:2]), self.assertRaises(ValueError):
                detector.candy_size_mm(*args)

    def test_capture_evaluation_reports_agreement_misses_and_extras(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            image = scene(((200, 240), (110, 45), 90, WRAPPER_BGR), ((450, 240), (110, 45), 90, WRAPPER_BGR))
            cv2.imwrite(str(folder / "a.png"), image)
            truth = [
                {"box_xyxy": [177, 185, 223, 295]},   # matches left wrapper
                {"box_xyxy": [20, 20, 60, 60]},       # nothing there: OpenCV miss
            ]
            (folder / "a.json").write_text(json.dumps({"yolo_reference": {"detections": truth}}))
            with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
                disagreements = detector.evaluate_captures(folder, folder / "eval")
            self.assertEqual(disagreements, 2)
            self.assertIn("MISSED", output.getvalue())
            self.assertIn("EXTRA", output.getvalue())
            self.assertTrue((folder / "eval" / "opencv_a.png").exists())


if __name__ == "__main__":
    unittest.main()
