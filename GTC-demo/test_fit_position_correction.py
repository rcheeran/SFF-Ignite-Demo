"""Tests for fit_position_correction.py (no robot)."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import fit_position_correction as fit


def reading(camera_xy, true_xy, jaw_xy=(300.0, 0.0), when="2026-11-23T23:45:00+00:00"):
    offset = np.subtract(true_xy, jaw_xy)
    return {
        "recorded_at": when, "uncorrected_base_point_mm": [*camera_xy, 12.0],
        "jaw_point_mm": [*jaw_xy, 21.0], "offset_mm": offset.tolist(),
    }


class FitTests(unittest.TestCase):
    def test_four_readings_recover_the_matrix_and_offset_and_old_readings_are_skipped(self):
        matrix, offset = np.array([[0.85, 0.02], [-0.01, 0.98]]), np.array([25.0, 3.0])
        camera = np.array([[280.0, 90.0], [300.0, -100.0], [420.0, 10.0], [380.0, 150.0], [460.0, -60.0]])
        true = camera @ matrix.T + offset
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "offsets.jsonl"
            lines = [reading((300.3, 88.3), (290.0, 88.0), when="2026-11-23T23:30:00+00:00")]
            lines += [reading(c, t, jaw_xy=c + 5.0) for c, t in zip(camera, true)]
            path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
            camera_xy, true_xy = fit.load_readings(path, since=fit.datetime.fromisoformat("2026-11-23T23:40:00+00:00"))
        self.assertEqual(len(camera_xy), 5)
        fitted_matrix, fitted_offset = fit.fit_correction(camera_xy, true_xy)
        np.testing.assert_allclose(fitted_matrix, matrix, atol=1e-6)
        np.testing.assert_allclose(fitted_offset, offset, atol=1e-4)

    def test_fewer_than_four_readings_fit_an_offset_only(self):
        matrix, offset = fit.fit_correction(np.array([[300.0, 88.0], [400.0, 0.0]]), np.array([[280.0, 88.0], [370.0, 2.0]]))
        np.testing.assert_array_equal(matrix, np.eye(2))
        np.testing.assert_allclose(offset, [-25.0, 1.0])


if __name__ == "__main__":
    unittest.main()
