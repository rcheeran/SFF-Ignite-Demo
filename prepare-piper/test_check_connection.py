import contextlib
from enum import Enum
import importlib.util
import io
import itertools
import json
from pathlib import Path
import subprocess
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
                patch.object(check_connection, "can_link_problem", return_value=None), \
                patch.object(check_connection.time, "sleep"), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            self.assertEqual(check_connection.main(), 0)
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

    @staticmethod
    def ip_link(returncode=0, flags=("NOARP", "UP", "LOWER_UP"), state="ERROR-ACTIVE", bitrate=1000000):
        link = [{"flags": list(flags), "linkinfo": {"info_kind": "can", "info_data": {
            "state": state, "bittiming": {"bitrate": bitrate}}}}]
        return subprocess.CompletedProcess([], returncode, stdout=json.dumps(link), stderr="")

    def test_can_link_problem_reports_missing_down_passive_or_wrong_bitrate(self):
        cases = (
            (self.ip_link(), None),
            (self.ip_link(returncode=1), "does not exist"),
            (self.ip_link(flags=("NOARP",), state="STOPPED"), "link is DOWN"),
            (self.ip_link(state="ERROR-PASSIVE"), "can state ERROR-PASSIVE"),
            (self.ip_link(bitrate=500000), "bitrate 500000"),
        )
        for result, expected in cases:
            with self.subTest(expected=expected), \
                    patch.object(check_connection.subprocess, "run", return_value=result):
                problem = check_connection.can_link_problem("can_piper")
            if expected is None:
                self.assertIsNone(problem)
            else:
                self.assertIn(expected, problem)

    def test_unusable_can_link_stops_before_contacting_the_arm(self):
        output, errors = io.StringIO(), io.StringIO()
        with patch.object(check_connection, "can_link_problem", return_value="CAN interface can_piper does not exist"), \
                patch.object(check_connection, "C_PiperInterface_V2", create=True) as interface, \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            self.assertEqual(check_connection.main(), 1)
        interface.assert_not_called()
        self.assertIn("does not exist", errors.getvalue())
        self.assertIn("sudo systemctl restart piper-can.service", output.getvalue())

    def arm(self, mode, enabled=False, switch_works=True):
        """Mock arm whose ModeCtrl/EnablePiper take effect only if switch_works."""
        state = {"mode": mode, "enabled": enabled}
        piper = Mock()
        piper.GetArmStatus.side_effect = lambda: SimpleNamespace(
            Hz=200, arm_status=SimpleNamespace(ctrl_mode=state["mode"]))
        piper.GetArmJointMsgs.return_value = self.joints()
        piper.GetArmEnableStatus.side_effect = lambda: [state["enabled"]] * 6
        if switch_works:
            piper.ModeCtrl.side_effect = lambda *args: state.update(mode=check_connection.CAN_CTRL)
            piper.EnablePiper.side_effect = lambda: state.update(enabled=True)
        return piper

    def run_main(self, piper):
        output, errors = io.StringIO(), io.StringIO()
        with patch.object(check_connection, "C_PiperInterface_V2", return_value=piper, create=True), \
                patch.object(check_connection, "can_link_problem", return_value=None), \
                patch.object(check_connection.time, "sleep"), \
                patch.object(check_connection.time, "monotonic", side_effect=itertools.count(0, 1)), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = check_connection.main()
        return code, output.getvalue(), errors.getvalue()

    def test_live_standby_enables_then_switches_to_can_ctrl_and_prints_new_status(self):
        piper = self.arm(check_connection.STANDBY)
        code, output, _ = self.run_main(piper)
        self.assertEqual(code, 0)
        piper.ModeCtrl.assert_called_with(0x01, 0x01, 30, 0x00)
        piper.EnablePiper.assert_called()
        calls = [name for name, *_ in piper.mock_calls]
        self.assertLess(calls.index("ModeCtrl"), calls.index("EnablePiper"))
        after = output.split("Arm status after switching (piper.GetArmStatus()):", 1)[1]
        self.assertIn("ctrl_mode=1", after)
        self.assertIn("EnableStatus: [True, True, True, True, True, True]", after)
        self.assertIn("OK: Control Mode CAN_CTRL", after)

    def test_failed_switch_reports_and_returns_error(self):
        piper = self.arm(check_connection.STANDBY, switch_works=False)
        code, output, errors = self.run_main(piper)
        self.assertEqual(code, 1)
        self.assertIn("FAILED: not in CAN_CTRL", errors)
        self.assertIn("Arm status after switching", output)

    def test_can_ctrl_teaching_and_other_modes_are_left_unchanged(self):
        for mode, message in ((check_connection.CAN_CTRL, "already CAN_CTRL"),
                              (check_connection.TEACHING_MODE, "TEACHING_MODE"),
                              (0x07, "only STANDBY is switched")):
            with self.subTest(mode=mode):
                piper = self.arm(mode, enabled=True)
                code, output, _ = self.run_main(piper)
                self.assertEqual(code, 0)
                self.assertIn(message, output)
                piper.ModeCtrl.assert_not_called()
                piper.EnablePiper.assert_not_called()


if __name__ == "__main__":
    unittest.main()
