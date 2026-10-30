#!/usr/bin/env python3
"""Diagnose gripper control over CAN (slave mode).

Tries each plausible enable/command sequence and VERIFIES by reading the
gripper angle back, so we learn what actually actuates rather than what the
SDK silently accepts.

gripper_code: 0x00 disable | 0x01 enable | 0x02 disable+clear_err | 0x03 enable+clear_err
NOTE: our old scripts used 0x02 as the "enable handshake" - that DISABLES it.

PREREQ: TEACH BUTTON OFF (slave mode). Run in ~/pi05-venv.
"""
import time

from piper_sdk import C_PiperInterface_V2

IFACE = "can_piper"
OPEN_UM = 70000     # 70 mm  (units are 0.001 mm)
CLOSE_UM = 20000    # 20 mm
EFFORT = 2000       # 0.001 N/m, range 0-5000
HZ = 15


def angle_mm(p):
    return p.GetArmGripperMsgs().gripper_state.grippers_angle / 1000.0


def gstat(p):
    f = p.GetArmGripperMsgs().gripper_state.foc_status
    return (f"enabled={f.driver_enable_status} err={f.driver_error_status} "
            f"overcurrent={f.driver_overcurrent} homing={f.homing_status}")


def stream_grip(p, um, code, secs=3.0, effort=EFFORT):
    """Stream the gripper command - single shots often don't latch on the Piper."""
    t0 = time.time()
    while time.time() - t0 < secs:
        p.GripperCtrl(um, effort, code, 0)
        time.sleep(1.0 / HZ)


def trial(p, label, setup, target_um):
    print(f"\n--- {label} ---")
    before = angle_mm(p)
    setup()
    stream_grip(p, target_um, 0x01)
    time.sleep(0.5)
    after = angle_mm(p)
    moved = abs(after - before)
    print(f"  {before:.1f}mm -> {after:.1f}mm  (moved {moved:.1f}mm)  {gstat(p)}")
    return moved > 2.0


def main():
    p = C_PiperInterface_V2(IFACE)
    p.ConnectPort()
    time.sleep(1.0)

    st = p.GetArmStatus().arm_status
    print(f"ctrl_mode: {st.ctrl_mode}")
    if "TEACH" in str(st.ctrl_mode).upper():
        raise SystemExit("ABORT: arm is in TEACHING mode - turn the teach button OFF first.")

    for i in range(6):
        p.EnablePiper()
        time.sleep(0.4)
        if all(p.GetArmEnableStatus()):
            break
    print("arm enabled:", p.GetArmEnableStatus())
    print("gripper    :", gstat(p), f"angle={angle_mm(p):.1f}mm")

    results = {}
    # A: plain enable (0x01), streamed
    results["A enable 0x01 -> close"] = trial(
        p, "A: stream code 0x01 (enable), close", lambda: None, CLOSE_UM)
    results["A enable 0x01 -> open"] = trial(
        p, "A: stream code 0x01 (enable), open", lambda: None, OPEN_UM)

    # B: enable+clear_err (0x03) first, then stream 0x01
    def setup_b():
        p.GripperCtrl(0, EFFORT, 0x03, 0)   # enable + clear error
        time.sleep(0.4)
    results["B 0x03 then 0x01 -> close"] = trial(p, "B: 0x03 enable+clear, then close", setup_b, CLOSE_UM)
    results["B 0x03 then 0x01 -> open"] = trial(p, "B: 0x03 enable+clear, then open", setup_b, OPEN_UM)

    # C: the OLD (buggy) sequence, for comparison: 0x02 = disable+clear
    def setup_c():
        p.GripperCtrl(0, EFFORT, 0x02, 0)   # disable + clear  <-- what we used to send
        time.sleep(0.4)
    results["C old 0x02 then 0x01 -> close"] = trial(p, "C: OLD 0x02 first, then close", setup_c, CLOSE_UM)

    print("\n================ SUMMARY ================")
    for k, v in results.items():
        print(f"  {'MOVED' if v else 'no move'}  {k}")
    print("\nfinal angle: %.1f mm  %s" % (angle_mm(p), gstat(p)))
    # leave it open
    p.GripperCtrl(0, EFFORT, 0x03, 0)
    time.sleep(0.3)
    stream_grip(p, OPEN_UM, 0x01, secs=2.0)
    print("left open at %.1f mm" % angle_mm(p))


if __name__ == "__main__":
    main()
