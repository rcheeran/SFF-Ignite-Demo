#!/usr/bin/env python3
"""Synthetic tests for multi-distance D405 depth correction."""

from __future__ import annotations

import unittest

import numpy as np

from calibrate_depth import fit_depth_correction, leave_one_out_errors


class DepthCalibrationTests(unittest.TestCase):
    def test_recovers_scale_and_offset(self) -> None:
        raw = np.array([180, 220, 260, 300, 340, 380, 420], dtype=np.float64)
        expected = 1.025 * raw + 11.5
        scale, offset, residuals = fit_depth_correction(raw, expected)
        self.assertAlmostEqual(scale, 1.025)
        self.assertAlmostEqual(offset, 11.5)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-10)
        np.testing.assert_allclose(
            leave_one_out_errors(raw, expected), 0.0, atol=1e-10
        )


if __name__ == "__main__":
    unittest.main()
