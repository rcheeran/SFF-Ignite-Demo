#!/usr/bin/env python3
"""joint_range_demo.py - sweep each Piper joint one at a time and report the
range of joint states (from encoder feedback).

*** THIS MOVES THE ARM. Clear the workspace; keep the E-stop in hand. ***

For each selected joint (the other joints are held at 0 deg), the arm sweeps
    0 -> lower -> upper -> 0
at low speed while continuously sampling that joint's encoder, then prints the
observed min / max / range. Between joints it returns to the all-zero home pose.

Ranges: by default a conservative SAFE range is used per joint - J2/J3 are
capped so the arm does not fold down into the surface it is mounted on. Pass
--full to sweep the full mechanical range from the Piper URDF (larger, riskier
motions - only with plenty of clearance below and around the arm).

Mechanical limits (Piper URDF), degrees:
    J1 +/-150    J2 0..180    J3 -170..0    J4 +/-100    J5 +/-70    J6 +/-120

Prereqs (see prepare-jetson/SETUP.md): arm in Slave mode, CAN up, arm reachable via piper_sdk.
The arm is on interface `can_piper` on this machine (see README/memory).

Usage:
    source ~/venv/bin/activate
    python3 useful-piper-scripts/joint_range_demo.py                 # all joints, safe range
    python3 useful-piper-scripts/joint_range_demo.py --joint 5       # only J5
    python3 useful-piper-scripts/joint_range_demo.py --full          # full mechanical range
    python3 useful-piper-scripts/joint_range_demo.py --iface can0    # different CAN interface
"""
import argparse
import time

import numpy as np
from piper_sdk import C_PiperInterface_V2

# Mechanical limits from the Piper URDF (degrees).
MECH_DEG = {1: (-150, 150), 2: (0, 180), 3: (-170, 0),
            4: (-100, 100), 5: (-70, 70), 6: (-120, 120)}
# Conservative default sweep range. J2/J3 capped to avoid folding into the
# mounting surface; wrist joints kept a touch inside their stops.
SAFE_DEG = {1: (-120, 120), 2: (0, 90), 3: (-90, 0),
            4: (-90, 90), 5: (-65, 65), 6: (-110, 110)}

MARGIN_DEG = 3.0       # stay this far inside whichever range is chosen
SPEED = 25             # percent (low, for a controlled sweep)
WP_TIMEOUT = 14.0      # max seconds to reach each waypoint (wrist joints are slow)
SAMPLE_HZ = 20
JOINT_NAMES = {1: "base yaw", 2: "shoulder", 3: "elbow",
               4: "forearm roll", 5: "wrist pitch", 6: "wrist roll"}


def read_joints_deg(p):
    js = p.GetArmJointMsgs().joint_state
    vals = [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]
    return np.array(vals) / 1000.0  # Piper units are 0.001 deg


def move_to(p, targets_deg, speed=SPEED):
    tgt = [int(round(v * 1000)) for v in targets_deg]
    p.ModeCtrl(0x01, 0x01, speed, 0x00)  # CAN_CTRL, MOVE_J
    p.JointCtrl(*tgt)


def goto_zero(p):
    move_to(p, [0.0] * 6)
    time.sleep(3.0)


def apply_margin(lo, hi):
    lo = lo + MARGIN_DEG if lo < -1e-6 else lo
    hi = hi - MARGIN_DEG if hi > 1e-6 else hi
    return lo, hi


def sweep_joint(p, j, lo, hi):
    """Sweep joint j (1-indexed) through 0 -> lo -> hi -> 0, others held at 0.
    Returns (observed_min, observed_max) from encoder feedback."""
    observed = []
    # Each leg starts from 0 so no single move exceeds a half-range (keeps the
    # slower wrist joints from being cut off by WP_TIMEOUT mid-traverse).
    for wp in (lo, 0.0, hi, 0.0):
        targets = [0.0] * 6
        targets[j - 1] = wp
        move_to(p, targets)
        t0 = time.time()
        while time.time() - t0 < WP_TIMEOUT:
            cur = read_joints_deg(p)[j - 1]
            observed.append(cur)
            if abs(cur - wp) < 0.5:
                break
            time.sleep(1.0 / SAMPLE_HZ)
    obs = np.array(observed)
    return float(obs.min()), float(obs.max())


def main():
    ap = argparse.ArgumentParser(description="Sweep each Piper joint and report its range.")
    ap.add_argument("--iface", default="can_piper", help="CAN interface (default: can_piper)")
    ap.add_argument("--joint", type=int, choices=[1, 2, 3, 4, 5, 6], help="sweep only this joint")
    ap.add_argument("--full", action="store_true", help="use full mechanical range (riskier)")
    args = ap.parse_args()

    ranges = MECH_DEG if args.full else SAFE_DEG
    label = "FULL mechanical" if args.full else "safe"
    joints = [args.joint] if args.joint else [1, 2, 3, 4, 5, 6]

    print("=" * 64)
    print(f"Piper joint range demo  ({label} ranges, speed {SPEED}%)")
    print("*** THE ARM WILL MOVE. Keep the E-stop in hand. ***")
    print("=" * 64)

    p = C_PiperInterface_V2(args.iface)
    p.ConnectPort()
    time.sleep(1.0)

    deadline = time.time() + 6
    while time.time() < deadline:
        p.EnablePiper()
        en = p.GetArmEnableStatus()
        if en and all(en):
            break
        time.sleep(0.3)
    en = p.GetArmEnableStatus()
    if not en or not all(en):
        raise SystemExit(f"ABORT: not all joints enabled: {en} (power-cycle the arm if a joint is stuck)")
    print("enabled:", en)

    print("homing before start...")
    goto_zero(p)

    results = {}
    for j in joints:
        lo, hi = apply_margin(*ranges[j])
        print(f"\n--- J{j} ({JOINT_NAMES[j]}): sweeping {lo:.0f} -> {hi:.0f} deg "
              f"(mech {MECH_DEG[j][0]}..{MECH_DEG[j][1]}) ---")
        omin, omax = sweep_joint(p, j, lo, hi)
        results[j] = (lo, hi, omin, omax)
        print(f"    observed: min={omin:7.2f}  max={omax:7.2f}  range={omax - omin:7.2f} deg")

    print("\nhoming...")
    goto_zero(p)

    print("\n" + "=" * 64)
    print(f"{'joint':<18}{'commanded':>18}{'observed':>18}{'range':>10}")
    print("-" * 64)
    for j in joints:
        lo, hi, omin, omax = results[j]
        print(f"J{j} {JOINT_NAMES[j]:<14}{f'{lo:.0f}..{hi:.0f}':>18}"
              f"{f'{omin:.1f}..{omax:.1f}':>18}{f'{omax - omin:.1f}':>10}")
    print("=" * 64)
    print("done. arm homed and left enabled/holding.")


if __name__ == "__main__":
    main()
