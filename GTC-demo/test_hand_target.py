import unittest

import numpy as np

from hand_target import choose_drop_target

MINIMUM, MAXIMUM = [200.0, -150.0, 0.0], [500.0, 150.0, 200.0]


def to_base(point):
    """Test transform: camera-frame point is the base point shifted by +300 mm in X."""
    return np.asarray(point) + [300.0, 0.0, 0.0]


def capture(palm_status="visible", palm=(0.0, 0.0, 50.0), finger_status="visible", finger=(40.0, 0.0, 50.0)):
    return {"points": {
        "palm_center": {"status": palm_status, "camera_point_mm": list(palm)},
        "finger_base": {"status": finger_status, "camera_point_mm": list(finger)},
    }}


class DropTargetTests(unittest.TestCase):
    def choose(self, value):
        return choose_drop_target(value, to_base, MINIMUM, MAXIMUM)

    def test_palm_centre_preferred_when_visible(self):
        result = self.choose(capture())
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["target"], "palm_center")
        self.assertEqual(result["base_point_mm"], [300.0, 0.0, 50.0])
        self.assertEqual(result["finger_base_mm"], [340.0, 0.0, 50.0])

    def test_finger_base_used_when_palm_hidden_or_outside(self):
        for name, value, note in (
            ("palm without depth", capture(palm_status="no_depth"), "not visible (no_depth)"),
            ("palm outside workspace", capture(palm=(-150.0, 0.0, 50.0)), "outside the hand workspace"),
        ):
            with self.subTest(name):
                result = self.choose(value)
                self.assertEqual(result["status"], "ready")
                self.assertEqual(result["target"], "finger_base")
                self.assertEqual(result["base_point_mm"], [340.0, 0.0, 50.0])
                self.assertIn(note, result["palm_note"])

    def test_waits_without_a_visible_finger_base_inside_the_workspace(self):
        for name, value, reason in (
            ("no open palm", {"status": "not_found"}, "no open palm"),
            ("finger base occluded", capture(finger_status="inconsistent"), "not visible (inconsistent)"),
            ("finger base outside", capture(finger=(250.0, 0.0, 50.0)), "outside the hand workspace"),
        ):
            with self.subTest(name):
                result = self.choose(value)
                self.assertEqual(result["status"], "waiting")
                self.assertIn(reason, result["reason"])


if __name__ == "__main__":
    unittest.main()
