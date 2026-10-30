#!/usr/bin/env python3
"""Park the arm: slow move to HOME, wrist to REST, then de-torque every joint. Run this after any session,
and any time the arm is left enabled.

    python tools/rest.py             # home + rest + de-torque (5 s countdown)
    python tools/rest.py --force     # override a LIVE controller's lock (a dead controller's lock clears itself)
    python tools/rest.py --measure   # also log where the joints settle over 20 s after de-torque
    python tools/rest.py --soft      # experimental MIT stiffness ramp before torque-off - supervised only
    python tools/rest.py --drop      # de-torque WHERE IT STANDS (no homing): only while you hold the arm
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from piper.robot import LOCK_FILE, Piper, connect_for_motion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--soft", action="store_true", help="experimental MIT stiffness ramp before torque-off")
    ap.add_argument("--drop", action="store_true", help="de-torque without homing - hold the arm")
    args = ap.parse_args()
    if args.force and os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)
    arm = connect_for_motion()
    arm.rest(soft=args.soft, drop=args.drop)
    if args.measure:
        ro = Piper().connect(enable=False)
        t0 = time.time()
        for t in (1, 2, 3, 5, 8, 12, 20):
            time.sleep(max(0.0, t0 + t - time.time()))
            print(f"settle +{t:2d}s  joints {np.round(ro.joints_deg(), 1).tolist()}")
        ro.disconnect()


if __name__ == "__main__":
    main()
