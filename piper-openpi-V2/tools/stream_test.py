#!/usr/bin/env python3
"""Cartesian streaming self-test: +-20 mm sinusoid in x/y and +-10 deg yaw at constant height with the
gripper opening/closing, through the same clamps as teleop. Logs commanded vs measured, reports lag.

    python tools/stream_test.py              # home, then READY (joint-space), then stream from READY
    python tools/stream_test.py --from-home  # stream from the folded HOME (firmware IK refuses: no motion)
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from piper import config as C
from piper.robot import clamp_pose, connect_for_motion, pose_to_xyz_rot, xyz_rot_to_pose

SECS, PERIOD, AMP_MM, AMP_DEG = 24.0, 6.0, 20.0, 10.0
LOG = os.path.join(C.LOG_DIR, "stream_test.csv")


def main():
    global SECS, AMP_MM, AMP_DEG
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-home", action="store_true")
    ap.add_argument("--joints", type=float, nargs=6, default=None, help="start pose instead of READY (deg)")
    ap.add_argument("--secs", type=float, default=SECS)
    ap.add_argument("--amp-mm", type=float, default=AMP_MM)
    ap.add_argument("--amp-deg", type=float, default=AMP_DEG)
    args = ap.parse_args()
    SECS, AMP_MM, AMP_DEG = args.secs, args.amp_mm, args.amp_deg
    arm = connect_for_motion()
    try:
        print("AT HOME" if arm.home() else "WARNING: home not reached", np.round(arm.joints_deg(), 1).tolist())
        if args.joints:
            ok = arm.move_joints(args.joints, C.HOME_TIMEOUT_S, None, speed=C.SPEED_HOME)
            print("AT START" if ok else "WARNING: start not reached", np.round(arm.joints_deg(), 1).tolist())
        elif not args.from_home:
            print("AT READY" if arm.ready() else "WARNING: ready not reached", np.round(arm.joints_deg(), 1).tolist())
        xyz0, rot0 = pose_to_xyz_rot(arm.pose())
        cmd_xyz, cmd_rot = xyz0.copy(), rot0
        print(f"start pose xyz={np.round(xyz0, 1)} rpy={np.round(arm.pose()[3:], 1)}  J6 {arm.j6_temp()} C  "
              f"faults {arm.faults() or 'none'}")
        rows, faults, t0, next_t = [], arm.faults(), time.monotonic(), time.monotonic()
        pend, flickers = None, 0                       # debounce: the firmware flags EXCEEDS_LIMIT for one status
                                                       # frame at the rz +-180 wrap (cosmetic; motion unaffected)
        while (t := time.monotonic() - t0) < SECS:
            w = 2 * np.pi * t / PERIOD
            tgt_xyz = xyz0 + np.array([AMP_MM * np.sin(w), AMP_MM * np.cos(w) - AMP_MM, 0.0])
            tgt_rot = Rot.from_euler("z", AMP_DEG * np.sin(w), degrees=True) * rot0
            meas_xyz, meas_rot = pose_to_xyz_rot(arm.pose())
            cmd_xyz, cmd_rot, _ = clamp_pose(cmd_xyz, cmd_rot, tgt_xyz, tgt_rot, meas_xyz, meas_rot)
            cmd = xyz_rot_to_pose(cmd_xyz, cmd_rot)
            grip = C.GRIP_OPEN_MM if int(t // (PERIOD / 2)) % 2 == 0 else C.GRIP_CLOSED_MM + 20
            arm.send_pose(cmd_xyz, cmd[3:], grip)
            rows.append([t, *cmd, grip, *arm.pose(), arm.grip_mm(), *arm.joints_deg()])
            f = arm.faults()
            if f != faults:
                pend = (f, t) if pend is None or pend[0] != f else pend
                if t - pend[1] >= 0.3:                 # persisted: report it
                    faults, pend = f, None
                    print(f"t={t:4.1f} arm status -> {arm.arm_status()}  faults {faults or 'none'}")
            else:
                flickers += pend is not None           # bounced back within 0.3 s: count, don't print
                pend = None
            if len(rows) % 30 == 0:
                m = rows[-1]
                print(f"t={t:4.1f} cmd xyz={np.round(m[1:4], 1)} rz={m[6]:.1f} | meas xyz={np.round(m[8:11], 1)} "
                      f"rz={m[13]:.1f} | grip cmd {m[7]:.0f} meas {m[14]:.1f}")
            next_t += 1.0 / C.CONTROL_HZ
            time.sleep(max(0.0, next_t - time.monotonic()))
        d = np.array(rows)
        os.makedirs(C.LOG_DIR, exist_ok=True)
        np.savetxt(LOG, d, delimiter=",", comments="",
                   header="t,cx,cy,cz,crx,cry,crz,cgrip,mx,my,mz,mrx,mry,mrz,mgrip,j1,j2,j3,j4,j5,j6")
        print(f"joint excursion (deg) during the wiggle: {np.round(np.ptp(d[:, 15:21], axis=0), 1).tolist()}")
        print(f"\nlog -> {LOG}")
        moved = np.ptp(d[:, 8:11], axis=0).max() > 5
        for col in (6, 13):                                   # Euler rz wraps at +-180 (where the tool sits at READY)
            d[:, col] = np.degrees(np.unwrap(np.radians(d[:, col])))
        for name, ci, mi in (("x", 1, 8), ("y", 2, 9), ("yaw", 6, 13)):
            c, m = d[:, ci] - d[:, ci].mean(), d[:, mi] - d[:, mi].mean()
            corr = [np.dot(c[:len(c) - k], m[k:]) for k in range(60)]
            k = int(np.argmax(corr))
            rms = np.sqrt(np.mean((c[:len(c) - k] - m[k:]) ** 2))
            print(f"{name:3s}: cmd amp {np.ptp(c) / 2:5.1f}  meas amp {np.ptp(m) / 2:5.1f}  "
                  f"lag {k * 1000 / C.CONTROL_HZ:4.0f} ms  rms err {rms:4.1f}")
        print(f"gripper: measured range {d[:, 14].min():.1f}..{d[:, 14].max():.1f} mm (cmd 70 / 20)")
        if flickers:
            print(f"transient status flickers: {flickers} (single status frames, mostly TARGET_POS_EXCEEDS_LIMIT at the "
                  "rz +-180 wrap - cosmetic, see the rms/lag lines for whether motion was affected)")
        print("CARTESIAN STREAMING: TRACKS" if moved else "CARTESIAN STREAMING: ARM DID NOT MOVE (firmware IK refused every target)")
        print(f"J6 {arm.j6_temp()} C; homing back and de-torquing")
    except KeyboardInterrupt:
        print("\nCtrl-C: stopping stream, homing, de-torquing")
    finally:
        arm.rest()


if __name__ == "__main__":
    main()
