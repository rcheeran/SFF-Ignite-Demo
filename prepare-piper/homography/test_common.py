#!/usr/bin/env python3
"""Synthetic tests for homography math; expected output is an all-tests-OK report."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from common import CalibrationError, error_metrics, solve_homography, transform_pixels


class HomographyMathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pixels = np.array(
            [[80, 60], [560, 70], [550, 420], [90, 410], [320, 240], [180, 300]],
            dtype=np.float64,
        )
        self.expected_homography = np.array(
            [[0.52, 0.03, 110.0], [-0.02, -0.61, 170.0], [0.0001, -0.0002, 1.0]],
            dtype=np.float64,
        )
        self.robot = cv2.perspectiveTransform(
            self.pixels.reshape(-1, 1, 2), self.expected_homography
        ).reshape(-1, 2)

    def test_solves_known_projective_mapping(self) -> None:
        solved, inliers = solve_homography(self.pixels, self.robot, 1.0)
        projected = transform_pixels(solved, self.pixels)
        metrics = error_metrics(projected, self.robot)
        self.assertTrue(np.all(inliers))
        self.assertLess(metrics["rmse_mm"], 1e-4)
        self.assertTrue(
            np.allclose(
                solved / solved[2, 2],
                self.expected_homography / self.expected_homography[2, 2],
                atol=1e-6,
            )
        )

    def test_rejects_collinear_points(self) -> None:
        collinear = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.float64)
        with self.assertRaisesRegex(CalibrationError, "collinear"):
            solve_homography(collinear, collinear, 1.0)

    def test_error_metrics_are_in_millimetres(self) -> None:
        predicted = np.array([[0, 0], [3, 4]], dtype=np.float64)
        expected = np.zeros((2, 2), dtype=np.float64)
        metrics = error_metrics(predicted, expected)
        self.assertAlmostEqual(metrics["mean_error_mm"], 2.5)
        self.assertAlmostEqual(metrics["rmse_mm"], np.sqrt(12.5))
        self.assertAlmostEqual(metrics["max_error_mm"], 5.0)


if __name__ == "__main__":
    unittest.main()
