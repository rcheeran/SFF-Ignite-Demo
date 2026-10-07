#!/usr/bin/env python3
"""Synthetic tests for depth filtering and camera-to-base point transforms."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from hand_eye_common import transform
from verify_depth_coordinates import (
    load_depth_correction,
    robust_center_depth_m,
    transform_camera_point_to_base,
)


class DepthCoordinateTests(unittest.TestCase):
    def test_missing_depth_correction_uses_identity(self) -> None:
        scale, offset, document = load_depth_correction(
            Path("/definitely/not/present/depth-correction.json"),
            disabled=False,
        )
        self.assertEqual((scale, offset, document), (1.0, 0.0, None))

    def test_robust_depth_rejects_zeros_and_outlier(self) -> None:
        depth = np.full((11, 11), 400, dtype=np.uint16)
        depth[4:7, 4:7] = np.array(
            [[0, 401, 399], [400, 900, 400], [399, 401, 0]],
            dtype=np.uint16,
        )
        median, count, mad = robust_center_depth_m(
            depth,
            np.array([5.0, 5.0]),
            depth_scale=0.001,
            radius_px=2,
            minimum_pixels=20,
            minimum_depth_m=0.05,
            maximum_depth_m=1.0,
        )
        self.assertAlmostEqual(median, 0.4)
        self.assertEqual(count, 23)
        self.assertLessEqual(mad, 0.001)

    def test_camera_point_transform_composition(self) -> None:
        camera_to_gripper = transform(np.eye(3), np.array([10, 20, 30]))
        gripper_to_base = transform(np.eye(3), np.array([100, 200, 300]))
        result = transform_camera_point_to_base(
            np.array([1, 2, 3]), gripper_to_base, camera_to_gripper
        )
        np.testing.assert_allclose(result, [111, 222, 333])


if __name__ == "__main__":
    unittest.main()
