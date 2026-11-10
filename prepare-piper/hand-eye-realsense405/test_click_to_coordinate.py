#!/usr/bin/env python3
"""Tests for browser click validation and temporal depth consensus."""

from __future__ import annotations

import unittest

import numpy as np

from click_to_coordinate import temporal_inlier_mask, validate_pixel


class ClickCoordinateTests(unittest.TestCase):
    def test_validate_pixel_rounds_and_bounds(self) -> None:
        self.assertEqual(validate_pixel(12.4, 20.6, 640, 480), (12, 21))
        with self.assertRaises(ValueError):
            validate_pixel(640, 20, 640, 480)
        with self.assertRaises(ValueError):
            validate_pixel(True, 20, 640, 480)

    def test_temporal_consensus_rejects_disparity_spikes(self) -> None:
        depths = np.array([300, 301, 299, 302, 298, 360, 250], dtype=float)
        inliers, median, mad, threshold = temporal_inlier_mask(depths, 3.0, 10.0)
        self.assertEqual(int(np.sum(inliers)), 5)
        self.assertEqual(median, 300.0)
        self.assertEqual(mad, 2.0)
        self.assertEqual(threshold, 10.0)


if __name__ == "__main__":
    unittest.main()
