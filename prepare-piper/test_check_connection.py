import contextlib
from enum import Enum
import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


spec = importlib.util.spec_from_file_location(
    "check_connection", Path(__file__).with_name("check_connection.py")
)
check_connection = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"piper_sdk": SimpleNamespace(__all__=[])}):
    spec.loader.exec_module(check_connection)


class ConnectionWarningTests(unittest.TestCase):
    def joints(self, hz=100, **overrides):
        values = {f"joint_{i}": 0 for i in range(1, 7)}
        values.update(overrides)
        return SimpleNamespace(Hz=hz, joint_state=SimpleNamespace(**values))

    def test_missing_and_partial_joint_states_warn(self):
        status = SimpleNamespace(Hz=100)
        for angles in (None, SimpleNamespace(Hz=100, joint_state=None),
                       self.joints(joint_4=None)):
            with self.subTest(angles=angles):
                warnings = check_connection.feedback_warnings(status, angles)
                self.assertTrue(any("None or incomplete" in warning for warning in warnings))

    def test_zero_feedback_warns_but_live_zero_positions_are_valid(self):
        warnings = check_connection.feedback_warnings(
            SimpleNamespace(Hz=0), self.joints(hz=0)
        )
        self.assertTrue(any("SDK defaults" in warning for warning in warnings))
        self.assertTrue(any("not verified positions" in warning for warning in warnings))
        self.assertEqual(
            check_connection.feedback_warnings(SimpleNamespace(Hz=100), self.joints()), []
        )

    def test_main_prints_recovery_without_changing_arm_configuration(self):
        piper = Mock()
        piper.GetArmStatus.return_value = None
        piper.GetArmJointMsgs.return_value = None
        output, errors = io.StringIO(), io.StringIO()
        with patch.object(check_connection, "C_PiperInterface_V2", return_value=piper, create=True), \
                patch.object(check_connection.time, "sleep"), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            check_connection.main()
        self.assertIn("WARNING:", errors.getvalue())
        self.assertIn("MasterSlaveConfig(0xFC, 0, 0, 0)", output.getvalue())
        self.assertIn("sudo systemctl restart piper-can.service", output.getvalue())
        self.assertIn("piper.EnablePiper()", output.getvalue())
        piper.MasterSlaveConfig.assert_not_called()
        piper.EnablePiper.assert_not_called()
        piper.ModeCtrl.assert_not_called()

    def test_standby_is_highlighted_with_live_or_missing_status(self):
        class Mode(Enum):
            STANDBY = 0

        for mode in (0, Mode.STANDBY):
            for hz in (0, 100):
                with self.subTest(mode=mode, hz=hz):
                    errors = io.StringIO()
                    status = SimpleNamespace(Hz=hz, arm_status=SimpleNamespace(ctrl_mode=mode))
                    with contextlib.redirect_stderr(errors):
                        check_connection.print_standby_warning(status)
                    text = errors.getvalue()
                    self.assertIn("WARNING: CONTROL MODE: STANDBY", text)
                    if hz:
                        self.assertIn("not a connection failure", text)
                        self.assertIn("piper.ModeCtrl", text)
                    else:
                        self.assertIn("UNVERIFIED", text)
                        self.assertNotIn("piper.EnablePiper()", text)

    def test_other_or_absent_modes_do_not_produce_standby_warning(self):
        for status in (None, SimpleNamespace(Hz=100),
                       SimpleNamespace(Hz=100, arm_status=SimpleNamespace(ctrl_mode=1))):
            with self.subTest(status=status):
                errors = io.StringIO()
                with contextlib.redirect_stderr(errors):
                    check_connection.print_standby_warning(status)
                self.assertEqual(errors.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
