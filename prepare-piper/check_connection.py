#!/usr/bin/env python3
"""Connectivity check for the Piper arm over CAN, using piper_sdk.

First checks that the CAN interface is up, ERROR-ACTIVE, and at 1 Mbps. Then
reports CAN fps, arm status, joint angles, and enable state. If live status
reports Control Mode STANDBY, enables all six motors and switches to CAN_CTRL
(prepare-jetson/SETUP.md step 5), then prints piper.GetArmStatus() again. The
joints stiffen where they are; no motion command is sent. Other modes are left
unchanged. Missing feedback prints manual recovery steps.

Before running this, the arm must be in Slave mode - see prepare-jetson/SETUP.md. If it's
still in Master mode (or was switched without a restart), ArmStatus.Hz will
read 0.0 and Control Mode will show TEACHING_MODE instead of STANDBY/CAN_CTRL,
even though the CAN link itself looks healthy.

Usage: python3 prepare-piper/check_connection.py [can_iface]
Exit status: 0 on success, 1 if the CAN interface is unusable or the switch to
CAN_CTRL fails.
"""
import json
import subprocess
import sys
import time
from piper_sdk import *

CAN_IFACE = sys.argv[1] if len(sys.argv) > 1 else "can_piper"
CAN_BITRATE = 1000000
STANDBY, CAN_CTRL, TEACHING_MODE = 0x00, 0x01, 0x02
MODE_CTRL_ARGS = (CAN_CTRL, 0x01, 30, 0x00)   # CAN_CTRL, MOVE J, speed 30%, position-speed (not MIT)
SWITCH_TIMEOUT_S = 6.0


def can_link_problem(iface):
    """Return why iface is unusable for the arm, or None if it is UP, ERROR-ACTIVE, 1 Mbps."""
    try:
        result = subprocess.run(
            ["ip", "-details", "-json", "link", "show", "dev", iface],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"cannot read CAN interface {iface}: {exc}"
    if result.returncode:
        return (f"CAN interface {iface} does not exist "
                "(USB-CAN adapter unplugged or not recognised by USB)")
    try:
        link = json.loads(result.stdout)[0]
        can = link["linkinfo"]["info_data"]
    except (ValueError, LookupError, TypeError):
        return f"{iface} is not a CAN interface"
    problems = []
    if "UP" not in link.get("flags", []):
        problems.append("link is DOWN")
    if can.get("state") != "ERROR-ACTIVE":
        problems.append(f"can state {can.get('state') or 'unknown'} (expected ERROR-ACTIVE)")
    bitrate = (can.get("bittiming") or {}).get("bitrate")
    if bitrate != CAN_BITRATE:
        problems.append(f"bitrate {bitrate or 'not set'} (expected {CAN_BITRATE})")
    return f"CAN interface {iface}: " + ", ".join(problems) if problems else None


def print_can_link_fix(iface, problem):
    print(f"ERROR: {problem}", file=sys.stderr)
    print(f"""
The arm was not contacted. Fix the CAN link first (prepare-jetson/SETUP.md step 3):

1. If the interface does not exist, unplug the USB-CAN adapter, wait 5 seconds,
   and plug it back in (directly into the Jetson if possible).
2. Power on the arm and release the emergency stop. ERROR-PASSIVE or BUS-OFF
   means nothing is answering on the bus.
3. Bring the interface up again:

   sudo systemctl restart piper-can.service
   ip -details link show {iface}   # expect: state UP, can state ERROR-ACTIVE, bitrate {CAN_BITRATE}

Then run this check again.
""")


def ctrl_mode_code(status):
    mode = getattr(getattr(status, "arm_status", None), "ctrl_mode", None)
    return getattr(mode, "value", mode)


def has_live_status(status):
    return bool(getattr(status, "Hz", 0))


def switch_to_can_ctrl(piper, timeout=SWITCH_TIMEOUT_S):
    """Enable all six motors and select CAN_CTRL; return True once both are reported.

    As in Demo-Ignite/piper/robot.py, position mode is selected before torque, so
    a driver left in MIT mode does not fight the first command. EnablePiper and
    ModeCtrl are repeated until all joints report enabled and the arm reports
    CAN_CTRL, because a single command does not always take effect.
    """
    piper.ModeCtrl(*MODE_CTRL_ARGS)
    time.sleep(0.1)
    deadline = time.monotonic() + timeout
    while True:
        enabled = all(piper.GetArmEnableStatus())
        can_ctrl = ctrl_mode_code(piper.GetArmStatus()) == CAN_CTRL
        if enabled and can_ctrl:
            return True
        if time.monotonic() >= deadline:
            return False
        if not enabled:
            piper.EnablePiper()
        if not can_ctrl:
            piper.ModeCtrl(*MODE_CTRL_ARGS)
        time.sleep(0.3)


def feedback_warnings(status, angles):
    warnings = []
    if status is None or not getattr(status, "Hz", 0):
        warnings.append("No live arm-status feedback; displayed status/mode may be SDK defaults.")
    joints = getattr(angles, "joint_state", None)
    if joints is None or any(getattr(joints, f"joint_{i}", None) is None for i in range(1, 7)):
        warnings.append("Joint states/positions are None or incomplete; current joint positions are unavailable.")
    if angles is None or not getattr(angles, "Hz", 0):
        warnings.append("No live joint-position feedback; displayed zero angles are not verified positions.")
    return warnings


def print_standby_warning(status):
    mode = getattr(getattr(status, "arm_status", None), "ctrl_mode", None)
    if getattr(mode, "value", mode) != 0:
        return
    print("\n" + "!" * 72, file=sys.stderr)
    print("WARNING: CONTROL MODE: STANDBY (0x0) -- NOT CAN_CTRL", file=sys.stderr)
    if not getattr(status, "Hz", 0):
        print("UNVERIFIED: no live status feedback; STANDBY may be an SDK default.",
              file=sys.stderr)
        print("Restore live status and joint feedback before enabling or changing control mode.",
              file=sys.stderr)
    else:
        print("Live status reports STANDBY; the arm is not in CAN control mode.",
              file=sys.stderr)
        print("STANDBY alone is not a connection failure or proof of Master mode.",
              file=sys.stderr)
        print("Enabling all six motors and switching to CAN_CTRL (SETUP.md step 5):",
              file=sys.stderr)
        print("  piper.EnablePiper()", file=sys.stderr)
        print("  piper.ModeCtrl(0x01, 0x01, 30, 0x00)  # CAN_CTRL, MOVE_J, speed=30%",
              file=sys.stderr)
        print("The joints stiffen where they are; no motion command is sent.",
              file=sys.stderr)
    print("!" * 72 + "\n", file=sys.stderr)


def print_recovery_steps(iface):
    print(f"""
Connection is NOT verified. Do not start the demo yet.
CAN traffic alone does not establish valid arm-status or position feedback.

1. Keep the demo stopped and the arm safely supported, with the emergency stop
   accessible. Check that the Joint 5 teaching button is off/disengaged.
   To configure Slave mode, run this separately in Python:

   from piper_sdk import *
   piper = C_PiperInterface_V2({iface!r})
   piper.ConnectPort()
   piper.MasterSlaveConfig(0xFC, 0, 0, 0)   # 0xFA = Master, 0xFC = Slave

2. Exit that Python session. If previously in Master mode, safely power-cycle
   the arm as required by prepare-jetson/SETUP.md for the change to take effect.
   Then run:

   sudo systemctl restart piper-can.service

3. Verify the interface:

   ip -details link show {iface}

   Expect: state UP, can state ERROR-ACTIVE, bitrate 1000000.

4. Verify the connection again (use the Python environment with piper_sdk):

   /home/clouduser/venv/bin/python prepare-piper/check_connection.py {iface}

   Expect:
   - fps typically around 3000-3100 on the documented setup.
     Around 1400-1500 can indicate only low-level driver frames; revisit step 1.
     Rates vary by firmware/configuration: fps alone is not a pass/fail test.
   - ArmStatus.Hz and JointMsgs.Hz both non-zero, with real values.
   - Live STANDBY(0x0) is switched to CAN_CTRL(0x1) by this check, which
     enables all six motors; the status is then printed again.
   - EnableStatus all True once enabled.

5. This check enables the motors and selects CAN control itself when live
   status reports STANDBY. To do it by hand, with the workspace safe:

   from piper_sdk import *
   piper = C_PiperInterface_V2({iface!r})
   piper.ConnectPort()
   piper.EnablePiper()
   piper.ModeCtrl(0x01, 0x01, 30, 0x00)   # CAN_CTRL, MOVE_J, speed=30%

   Re-check live status and EnableStatus; verify all six joints are enabled.
""")


def report_control_mode(piper, status):
    """Switch live STANDBY to CAN_CTRL and print the new status; return the exit code."""
    if not has_live_status(status):
        return 0
    mode = ctrl_mode_code(status)
    if mode == CAN_CTRL:
        print("\nControl Mode is already CAN_CTRL; nothing changed.")
        return 0
    if mode == TEACHING_MODE:
        print("\nControl Mode is TEACHING_MODE: the arm is in Master mode or teach recording. "
              "Not changed; put it in Slave mode and power-cycle it (SETUP.md steps 1-2).")
        return 0
    if mode != STANDBY:
        print(f"\nControl Mode is {status.arm_status.ctrl_mode}; only STANDBY is switched "
              "to CAN_CTRL, so nothing changed.")
        return 0
    switched = switch_to_can_ctrl(piper)
    status = piper.GetArmStatus()
    print("\nArm status after switching (piper.GetArmStatus()):")
    print("ArmStatus.Hz:", getattr(status, "Hz", "n/a"))
    print(status)
    print("EnableStatus:", piper.GetArmEnableStatus())
    if switched:
        print("OK: Control Mode CAN_CTRL with all six motors enabled; "
              "the arm is ready for scripted motion commands.")
        return 0
    print(f"FAILED: not in CAN_CTRL with all six motors enabled after {SWITCH_TIMEOUT_S:g} s. "
          "Check the emergency stop, arm power, and the Joint 5 teach button "
          "(prepare-jetson/SETUP.md).", file=sys.stderr)
    return 1


def main():
    problem = can_link_problem(CAN_IFACE)
    if problem:
        print_can_link_fix(CAN_IFACE, problem)
        return 1

    piper = C_PiperInterface_V2(CAN_IFACE)
    piper.ConnectPort()
    time.sleep(1.0)

    print("fps:", piper.GetCanFps())

    status = piper.GetArmStatus()
    print("\nArmStatus.Hz:", getattr(status, "Hz", "n/a"),
          "(0.0 means no real status frame received - check Slave mode / restart, see prepare-jetson/SETUP.md)")
    print(status)
    print_standby_warning(status)

    angles = piper.GetArmJointMsgs()
    print("JointMsgs.Hz:", getattr(angles, "Hz", "n/a"))
    print(angles)

    print("EnableStatus:", piper.GetArmEnableStatus())
    exit_code = report_control_mode(piper, status)
    warnings = feedback_warnings(status, angles)
    if warnings:
        for warning in warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
        print_recovery_steps(CAN_IFACE)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
