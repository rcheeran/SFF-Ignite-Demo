#!/usr/bin/env python3
"""Read-only connectivity check for the Piper arm over CAN, using piper_sdk.

No motion or motor-enable commands are sent. Reports CAN fps, arm status,
joint angles, and enable state. Missing feedback prints manual recovery steps.
STANDBY is highlighted separately from missing-feedback failures.

Before running this, the arm must be in Slave mode - see prepare-jetson/SETUP.md. If it's
still in Master mode (or was switched without a restart), ArmStatus.Hz will
read 0.0 and Control Mode will show TEACHING_MODE instead of STANDBY/CAN_CTRL,
even though the CAN link itself looks healthy.

Usage: python3 prepare-piper/check_connection.py [can_iface]
"""
import sys
import time
from piper_sdk import *

CAN_IFACE = sys.argv[1] if len(sys.argv) > 1 else "can_piper"


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
        print("Only after live joint feedback is verified and the workspace is safe:",
              file=sys.stderr)
        print("  piper.EnablePiper()", file=sys.stderr)
        print("  piper.ModeCtrl(0x01, 0x01, 30, 0x00)  # CAN_CTRL, MOVE_J, speed=30%",
              file=sys.stderr)
        print("Manual commands only; this diagnostic does not execute them.",
              file=sys.stderr)
    print("!" * 72 + "\n", file=sys.stderr)


def print_recovery_steps(iface):
    print(f"""
Connection is NOT verified. Do not start the demo or enable motors yet.
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
   - Live STANDBY(0x0) is flagged: connected but not yet in CAN control.
     CAN_CTRL(0x1) is expected after manually selecting CAN control.
   - EnableStatus may be False before enabling; all True is expected once enabled.

5. ONLY after live feedback is verified and the workspace is safe:
   The following manual commands enable motor torque and select CAN control;
   this diagnostic does NOT execute them.

   from piper_sdk import *
   piper = C_PiperInterface_V2({iface!r})
   piper.ConnectPort()
   piper.EnablePiper()
   piper.ModeCtrl(0x01, 0x01, 30, 0x00)   # CAN_CTRL, MOVE_J, speed=30%

   Re-check live status and EnableStatus; verify all six joints are enabled.
""")


def main():
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
    warnings = feedback_warnings(status, angles)
    if warnings:
        for warning in warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
        print_recovery_steps(CAN_IFACE)


if __name__ == "__main__":
    main()
