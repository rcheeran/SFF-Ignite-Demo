#!/usr/bin/env python3
"""Synthetic eye-in-hand test; expected output is an all-tests-OK report."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from hand_eye_common import (
    rotation_error_degrees,
    solve_hand_eye_candidates,
    transform,
)


def rotation_from_vector(vector: list[float]) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.radians(np.asarray(vector, dtype=np.float64)))
    return rotation


class HandEyeSolverTests(unittest.TestCase):
    def test_recovers_known_camera_to_gripper_transform(self) -> None:
        expected_camera_to_gripper = transform(
            rotation_from_vector([12, -8, 20]),
            [35, -18, 72],
        )
        fixed_target_to_base = transform(
            rotation_from_vector([0, 0, 15]),
            [420, 30, 20],
        )
        gripper_to_base = []
        target_to_camera = []
        pose_specs = [
            ([250, -120, 300], [5, 20, -30]),
            ([300, -80, 340], [-15, 35, -5]),
            ([350, -20, 380], [25, 10, 20]),
            ([280, 60, 330], [-20, -15, 45]),
            ([380, 100, 360], [30, -25, 10]),
            ([420, 20, 420], [10, 40, 35]),
            ([330, -150, 440], [-30, 15, -40]),
            ([460, -60, 350], [35, 5, 55]),
            ([390, 140, 400], [-10, -35, 25]),
            ([270, 130, 370], [20, 30, -55]),
            ([440, -140, 390], [-25, 45, 15]),
            ([310, 30, 460], [40, -10, -20]),
            ([370, -100, 320], [-35, -20, 40]),
            ([410, 80, 450], [15, 25, 60]),
            ([290, -10, 410], [30, -40, -10]),
        ]
        for translation, rotation_vector in pose_specs:
            base_gripper = transform(
                rotation_from_vector(rotation_vector), translation
            )
            camera_target = (
                np.linalg.inv(expected_camera_to_gripper)
                @ np.linalg.inv(base_gripper)
                @ fixed_target_to_base
            )
            gripper_to_base.append(base_gripper)
            target_to_camera.append(camera_target)

        candidates = solve_hand_eye_candidates(gripper_to_base, target_to_camera)
        recovered = candidates[0]["camera_to_gripper"]
        translation_error = np.linalg.norm(
            recovered[:3, 3] - expected_camera_to_gripper[:3, 3]
        )
        rotation_error = rotation_error_degrees(
            recovered, expected_camera_to_gripper
        )
        self.assertLess(translation_error, 1e-5)
        self.assertLess(rotation_error, 1e-5)
        self.assertLess(
            candidates[0]["metrics"]["target_translation_rmse_mm"], 1e-5
        )


if __name__ == "__main__":
    unittest.main()
