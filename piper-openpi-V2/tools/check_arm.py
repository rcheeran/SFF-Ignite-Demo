#!/usr/bin/env python3
"""Read-only arm + camera check: CAN bus health, feedback rate, joints/pose/gripper, faults, FK sanity, camera
snapshots. Never enables or moves the arm. Exit code 1 with a one-line reason when the arm is not usable.
Restart order: bash tools/can_up.sh -> python tools/check_arm.py -> python tools/stream_test.py -> teleop -> rest.py

    python tools/check_arm.py             # arm + camera snapshots -> data/snapshots/
    python tools/check_arm.py --no-cams   # arm only (leaves the cameras to whoever holds them)
"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from piper import config as C
from piper.cameras import RealSenseCamera, UsbCamera
from piper.robot import Piper, can_link, can_receivers
from piper.transforms import euler_to_rot, fk_flange


def can_verdict(link) -> str:
    """One line on the bus. ERROR-PASSIVE after traffic = nothing ACKs our frames (arm off, cable); the jam
    session left the bus there for hours and nobody noticed."""
    if not link["up"]:
        return f"CAN: {C.CAN_IFACE} is down ({link['state'] or 'missing'}) - run `bash tools/can_up.sh`"
    counts = (f"rx {link['rx_packets']} / tx {link['tx_packets']} frames, {link['error_warn']} warn / "
              f"{link['error_pass']} passive / {link['bus_off']} bus-off events"
              + (f", berr tx {link['berr'][0]} rx {link['berr'][1]}" if link["berr"] else ""))
    if link["state"] == "ERROR-ACTIVE":
        return f"CAN: OK ({counts})"
    if link["state"] == "ERROR-PASSIVE":
        return (f"CAN: ERROR-PASSIVE - our frames are not ACKed: arm powered? cable? ({counts}) - "
                "`bash tools/can_up.sh` resets it")
    return f"CAN: {link['state']} - bus errors (wiring? bitrate?) ({counts}) - `bash tools/can_up.sh` restarts it"


def check_arm() -> int:
    link = can_link()
    print(f"CAN link     : {'UP' if link['up'] else 'DOWN'}  state {link['state']}  "
          f"rx {link['rx_packets']}  tx {link['tx_packets']}")
    if not link["up"]:
        print("!! " + can_verdict(link))
        return 1
    print(f"CAN receivers: {can_receivers()} other process(es) on {C.CAN_IFACE} before connect")
    try:
        arm = Piper().connect(enable=False)
    except RuntimeError as e:
        print(f"!! {e}")
        print("!! " + can_verdict(can_link()))
        return 1
    time.sleep(0.5)
    j, p, h = arm.joints_deg(), arm.pose(), arm.health()
    print(f"joints (deg) : {np.round(j, 2).tolist()}")
    print(f"pose         : xyz={np.round(p[:3], 1).tolist()} mm  rpy={np.round(p[3:], 1).tolist()} deg")
    print(f"gripper      : {arm.grip_mm():.1f} mm")
    print(f"ctrl mode    : {h['ctrl_mode']}   arm status: {h['arm_status']}")
    print(f"enabled      : {h['enabled']}")
    print(f"faults       : {h['faults'] or 'none'}")
    print(f"J6 temp      : {h['j6_temp']} C")
    lim = arm.joint_limits()
    off = [f"J{i} {l}" for i, (l, f) in enumerate(zip(lim or [], C.JOINT_LIMITS_DEG), 1)
           if l is not None and (abs(l[0] - f[0]) > 0.5 or abs(l[1] - f[1]) > 0.5)]
    print(f"joint limits : {lim}" + (f"  !! not factory: {off} - the next enable restores {C.JOINT_LIMITS_DEG}" if off else "  (factory)"))
    print(f"J6 zero      : reads {j[5]:.1f} deg. With the wrist camera on TOP of the gripper this must read ~0 "
          f"(the motor's zero is the centre of the factory +-120 range and of every pose); if the camera is not on "
          f"top at ~0, run `python tools/zero_j6.py` BEFORE anything moves")
    print(f"CAN fps      : {h['can_fps']}   joint feedback Hz: {h['feedback_hz']}")
    fk = fk_flange(j)
    ang = np.degrees((euler_to_rot(p[3:]).inv() * euler_to_rot(fk[3:])).magnitude())
    dxyz = np.linalg.norm(fk[:3] - p[:3])
    print(f"SDK FK       : xyz={np.round(fk[:3], 1).tolist()} mm  rpy={np.round(fk[3:], 1).tolist()} deg")
    print(f"FK vs report : |dxyz|={dxyz:.1f} mm  rotation diff={ang:.1f} deg "
          f"({'OK - same frame & Euler convention' if ang < 3 and dxyz < 10 else 'MISMATCH'})")
    arm.disconnect()
    verdict = can_verdict(can_link())
    print(verdict)
    ok = not h["faults"] and verdict.startswith("CAN: OK")
    print("ARM: OK - ready for `python tools/stream_test.py`" if ok else "ARM: CHECK the lines above before any motion")
    return 0 if ok else 1


def check_cameras():
    os.makedirs(C.SNAP_DIR, exist_ok=True)
    cams = {"wrist_d405": RealSenseCamera,
            "top_hp": lambda: UsbCamera(C.TOP_CAM_DEV),
            "top_realtek": lambda: UsbCamera(C.ALT_TOP_CAM_DEV)}
    for name, ctor in cams.items():
        try:
            cam = ctor()
            ok = cam.wait()
            time.sleep(2.0)
            f = cam.read()
            path = os.path.join(C.SNAP_DIR, f"cam_{name}.jpg")
            if f is not None:
                cv2.imwrite(path, f)
            print(f"camera {name:12s}: {'ok' if ok else 'NO FRAMES'}  {None if f is None else f.shape}  ~{cam.fps():.1f} fps  -> {path}")
            if name == "wrist_d405" and f is not None:
                print("                      (the D405 sits on top of the gripper at J6 = 0: upright picture <=> J6 near 0 - see the joints line)")
            cam.close()
        except Exception as e:
            print(f"camera {name:12s}: FAILED ({e})")


if __name__ == "__main__":
    rc = check_arm()
    if "--no-cams" not in sys.argv:
        check_cameras()
    sys.exit(rc)
