"""Hardware-free tests for Hand Landmarker serialization and annotation."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from hand_logic import annotate_frame, extended_finger_count, serialize_result


class HandLogicTests(unittest.TestCase):
    def test_serializes_landmarks_handedness_and_palm_center(self) -> None:
        landmarks = [
            SimpleNamespace(x=index / 20, y=index / 40, z=-index / 100)
            for index in range(21)
        ]
        result = SimpleNamespace(
            hand_landmarks=[landmarks],
            hand_world_landmarks=[landmarks],
            handedness=[
                [SimpleNamespace(category_name="Right", score=0.91)]
            ],
        )

        hands = serialize_result(result, 640, 480)

        self.assertEqual(len(hands), 1)
        self.assertEqual(hands[0]["handedness"], "Right")
        self.assertAlmostEqual(hands[0]["handedness_score"], 0.91)
        self.assertEqual(len(hands[0]["landmarks"]), 21)
        self.assertEqual(hands[0]["extended_finger_count"], 4)
        self.assertTrue(hands[0]["is_open_palm"])
        expected_x = (0 + np.mean([5, 9, 13, 17])) / 2 / 20
        expected_y = (0 + np.mean([5, 9, 13, 17])) / 2 / 40
        self.assertEqual(
            hands[0]["palm_center_pixel"],
            [round(expected_x * 640), round(expected_y * 480)],
        )

    def test_annotation_preserves_shape(self) -> None:
        points = [
            {"x": 0.25 + index * 0.01, "y": 0.25 + index * 0.01, "z": 0.0}
            for index in range(21)
        ]
        hand = {
            "landmarks": points,
            "palm_center_pixel": [200, 150],
            "handedness": "Left",
            "handedness_score": 0.88,
            "is_open_palm": True,
        }
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        annotated = annotate_frame(frame, [hand])

        self.assertEqual(annotated.shape, frame.shape)
        self.assertGreater(int(np.count_nonzero(annotated)), 0)
        self.assertEqual(int(np.count_nonzero(frame)), 0)

    def test_closed_fingers_are_not_an_open_palm(self) -> None:
        points = [{"x": 0.5, "y": 0.8, "z": 0.0} for _ in range(21)]
        for mcp, pip, tip in ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)):
            points[mcp] = {"x": 0.5, "y": 0.6, "z": 0.0}
            points[pip] = {"x": 0.5, "y": 0.5, "z": 0.0}
            points[tip] = {"x": 0.5, "y": 0.7, "z": 0.0}

        self.assertEqual(extended_finger_count(points), 0)


if __name__ == "__main__":
    unittest.main()
