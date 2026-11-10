from __future__ import annotations

import unittest

import cv2
import numpy as np

from vision import (
    CLASS_NAMES,
    VisionError,
    decode_yolo_output,
    estimate_long_axis,
    letterbox_tensor,
    normalize_prompt,
)


class FoundryYoloTests(unittest.TestCase):
    def test_decode_rejects_non_finite_model_output(self) -> None:
        output = np.zeros((1, 4 + len(CLASS_NAMES), 8400), dtype=np.float32)
        output[0, 0, 0] = np.nan
        with self.assertRaisesRegex(VisionError, "non-finite"):
            decode_yolo_output(
                output, 8, 0.25, 0.45, 1.0, 0.0, 80.0, (480, 640, 3)
            )

    def test_prompt_alias(self) -> None:
        self.assertEqual(normalize_prompt(" Starburst-Candy "), "starburst candy")

    def test_letterbox_shape_and_scale(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor, scale, pad_x, pad_y = letterbox_tensor(frame)
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(scale, 1.0)
        self.assertEqual(pad_x, 0.0)
        self.assertEqual(pad_y, 80.0)

    def test_decode_and_nms(self) -> None:
        output = np.zeros((1, 4 + len(CLASS_NAMES), 8400), dtype=np.float32)
        class_id = CLASS_NAMES.index("starburst")
        output[0, :4, 0] = [320.0, 320.0, 100.0, 80.0]
        output[0, 4 + class_id, 0] = 0.9
        output[0, :4, 1] = [322.0, 322.0, 100.0, 80.0]
        output[0, 4 + class_id, 1] = 0.8

        detections = decode_yolo_output(
            output,
            class_id=class_id,
            confidence=0.25,
            iou_threshold=0.45,
            scale=1.0,
            pad_x=0.0,
            pad_y=80.0,
            frame_shape=(480, 640, 3),
        )

        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0]["score"], 0.9)
        self.assertEqual(detections[0]["box_xyxy"], [270.0, 200.0, 370.0, 280.0])

    def test_decode_returns_distinct_starbursts(self) -> None:
        output = np.zeros((1, 4 + len(CLASS_NAMES), 8400), dtype=np.float32)
        class_id = CLASS_NAMES.index("starburst")
        output[0, :4, 0] = [150.0, 300.0, 100.0, 60.0]
        output[0, 4 + class_id, 0] = 0.9
        output[0, :4, 1] = [450.0, 300.0, 100.0, 60.0]
        output[0, 4 + class_id, 1] = 0.8

        detections = decode_yolo_output(
            output,
            class_id=class_id,
            confidence=0.25,
            iou_threshold=0.45,
            scale=1.0,
            pad_x=0.0,
            pad_y=80.0,
            frame_shape=(480, 640, 3),
        )

        self.assertEqual(len(detections), 2)
        np.testing.assert_allclose(
            [item["score"] for item in detections],
            [0.9, 0.8],
        )

    def test_estimates_rotated_wrapper_long_axis(self) -> None:
        frame = np.full((300, 400, 3), 60, dtype=np.uint8)
        rectangle = ((200.0, 150.0), (180.0, 55.0), 32.0)
        corners = np.rint(cv2.boxPoints(rectangle)).astype(np.int32)
        cv2.fillConvexPoly(frame, corners, (0, 0, 255))

        result = estimate_long_axis(frame, [90.0, 70.0, 310.0, 230.0])

        start, end = np.asarray(result["long_axis_pixels"])
        measured = np.degrees(np.arctan2(end[1] - start[1], end[0] - start[0]))
        difference = abs((measured - 32.0 + 90.0) % 180.0 - 90.0)
        self.assertLess(difference, 3.0)
        self.assertGreater(result["orientation_aspect_ratio"], 2.5)

    def test_rejects_non_elongated_orientation(self) -> None:
        frame = np.full((200, 200, 3), 60, dtype=np.uint8)
        cv2.circle(frame, (100, 100), 45, (0, 0, 255), -1)

        with self.assertRaisesRegex(Exception, "ambiguous"):
            estimate_long_axis(frame, [45.0, 45.0, 155.0, 155.0])


if __name__ == "__main__":
    unittest.main()
