#!/usr/bin/env python3
"""Receive-only Piper feedback diagnostic; never enables motors or sends frames.

Stop other controllers before use:
    python prepare-piper/check_joint_feedback.py can_piper --seconds 10

Position feedback is shared by joint pairs. Frame presence does not establish
motor health; absent frames can indicate mode/protocol issues, not broken joints.
Uses Linux SocketCAN and the standard library only (no piper_sdk required).
Controller mode is read from broadcast status 0x2A1; if absent, mode is UNKNOWN.
No configuration/query command is transmitted to force a response.
"""

import argparse
from collections import Counter
import math
import socket
import struct
import sys
import time


CONTROL_MODES = {
    0x00: "STANDBY",
    0x01: "CAN_CTRL",
    0x02: "TEACHING_MODE",
    0x03: "ETHERNET_CONTROL_MODE",
    0x04: "WIFI_CONTROL_MODE",
    0x05: "REMOTE_CONTROL_MODE",
    0x06: "LINKAGE_TEACHING_INPUT_MODE",
    0x07: "OFFLINE_TRAJECTORY_MODE",
}


def collect(iface, seconds):
    counts = Counter()
    modes = Counter()
    with socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW) as bus:
        bus.bind((iface,))
        deadline = time.monotonic() + seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            bus.settimeout(remaining)
            try:
                frame = bus.recv(16)
            except socket.timeout:
                break
            if len(frame) != 16:
                raise ValueError(f"Unexpected CAN frame length: {len(frame)}")
            can_id, length, data = struct.unpack("=IB3x8s", frame)
            # Ignore extended, remote-request, and error frames.
            if can_id & 0xE0000000:
                continue
            counts[(can_id, length)] += 1
            if can_id == 0x2A1 and length == 8:
                modes[data[0]] += 1
    return counts, modes


def report(counts, seconds, modes=None):
    def frames(can_id):
        return counts[(can_id, 8)]

    print(f"\nReceived {sum(counts.values())} standard data frames in {seconds:g}s.")
    print("Counts below require 8-byte frames; position frames cover joint PAIRS.")
    print("\nJoint | Position ID/count | Motor high-speed ID/count | Motor low-speed ID/count")
    print("------|-------------------|---------------------------|-------------------------")
    for joint in range(1, 7):
        position = 0x2A5 + (joint - 1) // 2
        high, low = 0x250 + joint, 0x260 + joint
        print(
            f"J{joint}    | 0x{position:03X}: {frames(position):7d} | "
            f"0x{high:03X}: {frames(high):15d} | 0x{low:03X}: {frames(low):14d}"
        )
    print(f"\nArm status 0x2A1: {frames(0x2A1)}")
    if modes:
        print("Controller mode(s) observed from 0x2A1:")
        for code, count in sorted(modes.items()):
            print(f"  {CONTROL_MODES.get(code, 'UNRECOGNIZED')} (0x{code:02X}): {count} frames")
    else:
        print("Controller mode: UNKNOWN (no valid 0x2A1 status payload captured).")
        print("Cannot determine Master/Slave mode from missing feedback or default SDK values.")
    missing = [f"J{start}/J{start + 1}" for start in (1, 3, 5)
               if not frames(0x2A5 + (start - 1) // 2)]
    if not counts:
        print("NO TRAFFIC: check interface state, USB adapter, arm power, and CAN wiring.")
    elif len(missing) == 3:
        print("ALL joint pairs lack expected position feedback.")
        if all(frames(0x250 + j) and frames(0x260 + j) for j in range(1, 7)):
            print("Motor telemetry is present for ALL six joints: investigate arm mode/protocol.")
    elif missing:
        print("Missing position feedback for: " + ", ".join(missing))
        print("A missing pair frame cannot identify which individual joint is responsible.")
    else:
        print("Position feedback is present for ALL six joints.")
        print("If the demo still reports zero Hz, investigate SDK decoding/receive handling.")
    print("Frame presence alone does not prove motor health or safe motion.")
    print("\nObserved standard CAN IDs (all payload lengths):")
    totals = Counter()
    for (can_id, _), count in counts.items():
        totals[can_id] += count
    print(", ".join(f"0x{can_id:03X}={count}" for can_id, count in sorted(totals.items())) or "(none)")
    return not missing and bool(frames(0x2A1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("iface", nargs="?", default="can_piper")
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and greater than zero")
    if not hasattr(socket, "PF_CAN"):
        parser.error("Linux SocketCAN is required; run this script on the Jetson")
    print(f"Listening on {args.iface} for {args.seconds:g}s; no CAN commands will be sent.")
    try:
        counts, modes = collect(args.iface, args.seconds)
    except (OSError, ValueError) as exc:
        print(f"ERROR: CAN capture failed: {exc}", file=sys.stderr)
        return 2
    return 0 if report(counts, args.seconds, modes) else 1


if __name__ == "__main__":
    sys.exit(main())
