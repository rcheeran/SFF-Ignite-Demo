#!/usr/bin/env python3
"""Set joint 6's zero to the physical centre of the wrist (camera on TOP), supervised.

Why: the firmware's J6 zero is whatever the motor was taught. On this arm it sat ~180 deg away from the physical
centre - at a mechanical stop, camera down - so every J6 = 0 target (HOME / REST / READY, the centre of the factory
+-120 range) drove the wrist into the stop: the driver cut the motor (limp J6), the firmware refused every target
(freeze), and the old team's -165 "camera-up" roll + custom limit was the workaround. The fix is the motor's own
zero command: with the wrist physically at its centre, `JointConfig(joint 6, set_zero)`.

    python tools/zero_j6.py            # enables, homes (shoulder/elbow folded), de-torques J6 ONLY, you turn the
                                       # wrist by hand to camera-up centre, Enter, confirm -> zero written ->
                                       # read back ~0 -> J6 re-enabled -> rest
    python tools/zero_j6.py --stops    # also measures the two mechanical stops (turn to each, Enter) afterwards

User present, hand near the e-stop. Afterwards power-cycle and run `python tools/check_arm.py`: J6 must read ~0
with the camera on top (the zero is stored in the driver).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from piper import config as C
from piper.robot import connect_for_motion


def ask(prompt) -> str:
    return input(prompt).strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stops", action="store_true", help="measure the mechanical stops after zeroing")
    args = ap.parse_args()
    arm = connect_for_motion()
    try:
        print("J6 reads", round(arm.joints_deg()[5], 1), "deg now (camera on top? if yes there is nothing to fix)")
        if ask("continue and re-zero J6? [y/N] ") != "y":
            return
        if not arm.home():
            raise SystemExit("ABORT: home not reached - fix that first (tools/rest.py)")
        arm.enable_joint(6, False)                       # wrist limp, everything else keeps holding HOME
        time.sleep(0.5)
        print(f"J6 de-torqued (enabled {arm.enabled()}). Turn the wrist by hand so the CAMERA IS ON TOP, centred "
              "between the two hard stops.")
        ask("press Enter when it is there ")
        before = arm.joints_deg()[5]
        print(f"J6 reads {before:.1f} deg at the position you chose")
        if ask(f"write this position as J6 = 0 into the motor? [y/N] ") != "y":
            return
        arm.zero_joint(6)
        time.sleep(1.0)
        after = arm.joints_deg()[5]
        print(f"J6 now reads {after:.1f} deg (expect ~0){'' if abs(after) < 2 else '  !! not zero - the motor did not take it'}")
        if args.stops:
            rng = []
            for side in ("one", "the other"):
                ask(f"turn the wrist gently to {side} hard stop, then Enter ")
                rng.append(arm.joints_deg()[5])
                print(f"  stop at {rng[-1]:.1f} deg")
            print(f"mechanical range {min(rng):.0f}..{max(rng):.0f} deg; factory software limit {C.JOINT_LIMITS_DEG[5]}"
                  f" - the centre should be ~0 now")
            ask("turn the wrist back to camera-up centre, then Enter ")
        arm.enable_joint(6, True)
        time.sleep(0.5)
        arm.send_joints(arm.joints_deg(), None, C.SPEED_HOME)   # hold where it is before resting
        print(f"J6 re-enabled {arm.enabled()}; joints {np.round(arm.joints_deg(), 1).tolist()}")
    finally:
        arm.rest()


if __name__ == "__main__":
    main()
