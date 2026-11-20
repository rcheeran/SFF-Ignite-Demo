import contextlib
from collections import Counter
import io
import unittest

from check_joint_feedback import report


class JointFeedbackTests(unittest.TestCase):
    def output(self, counts, modes=None):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            ok = report(Counter(counts), 10, modes)
        return ok, stream.getvalue()

    def test_motor_telemetry_does_not_replace_position_feedback(self):
        counts = {(base + joint, 8): 20 for base in (0x250, 0x260)
                  for joint in range(1, 7)}
        counts[(0x3A5, 8)] = 20
        ok, output = self.output(counts)
        self.assertFalse(ok)
        self.assertIn("ALL joint pairs lack", output)
        self.assertIn("Motor telemetry is present for ALL six", output)
        self.assertIn("Controller mode: UNKNOWN", output)
        self.assertNotIn("STANDBY", output)

    def test_controller_mode_reports_observed_modes_and_unrecognized_codes(self):
        _, output = self.output({(0x2A1, 8): 6}, {0x01: 3, 0x02: 2, 0x81: 1})
        self.assertIn("CAN_CTRL (0x01): 3 frames", output)
        self.assertIn("TEACHING_MODE (0x02): 2 frames", output)
        self.assertIn("UNRECOGNIZED (0x81): 1 frames", output)

    def test_missing_pair_and_short_payload_are_not_valid_position_feedback(self):
        ok, output = self.output({
            (0x2A1, 8): 10, (0x2A5, 8): 10, (0x2A6, 4): 10, (0x2A7, 8): 10,
        })
        self.assertFalse(ok)
        self.assertIn("Missing position feedback for: J3/J4", output)

    def test_complete_feedback_and_no_traffic(self):
        ok, output = self.output({(can_id, 8): 10 for can_id in (0x2A1, 0x2A5, 0x2A6, 0x2A7)})
        self.assertTrue(ok)
        self.assertIn("Position feedback is present for ALL", output)
        ok, output = self.output({})
        self.assertFalse(ok)
        self.assertIn("NO TRAFFIC", output)


if __name__ == "__main__":
    unittest.main()
