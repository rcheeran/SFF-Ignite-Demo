# SFF AgileX Piper Setup

Setup and diagnostic scripts for controlling the AgileX Piper arm over CAN
from an Ubuntu machine.

Tested on Ubuntu 24.04.4 LTS / Python 3.12.3 with a single AgileX USB-CAN
adapter connected.

## Setup

```bash
bash setup.sh
```

This installs `can-utils`/`ethtool`/`python3-venv`, creates a venv at
`~/venv`, installs `pyAgxArm` (from GitHub source - it is not published on
PyPI, so `pip install pyAgxArm` alone will fail) and `piper_sdk` (from
PyPI), and brings up the CAN interface at 1 Mbps.

Requires `sudo` (interactive password prompt for `apt install` and the CAN
interface commands).

If more than one CAN adapter is plugged in, don't rely on the single-adapter
shortcut in `setup.sh` - run `scripts/can_activate.sh <name> <bitrate>
"<usb-bus-address>"` manually per adapter instead (see `scripts/can_activate.sh`
usage output, or `scripts/can_muti_activate.sh` upstream in the pyAgxArm repo
for activating several at once).

## Verifying the connection

```bash
source ~/venv/bin/activate
python3 scripts/check_connection.py
```

Read-only: connects with both `pyAgxArm` and `piper_sdk`, prints firmware
info, CAN fps, joint enable status, arm status, and joint angles. Sends no
motion commands.

## Scripts

- `scripts/find_all_can_port.sh` - lists connected CAN adapters and their USB bus addresses (from AgileX's `pyAgxArm` repo, `pyAgxArm/scripts/ubuntu/`).
- `scripts/can_activate.sh` - brings up a CAN adapter with a given name/bitrate (same source).
- `scripts/check_connection.py` - read-only SDK connectivity/status check.

## Known issue: arm status / joint angle frames not received

On this arm (firmware `S-V1.8-3`, hardware `H-V1.2-1`, node `ARM_MC`), after
CAN setup the interface comes up correctly and per-joint driver telemetry
flows fine, but the arm's high-level status range is **never observed on the
bus**:

- CAN IDs `0x2A1`-`0x2A8` (arm status, joint angles, end pose, gripper
  status) are absent from `candump` output.
- `pyAgxArm`'s `get_arm_status()` / `get_joint_angles()` return `None`
  indefinitely.
- `piper_sdk`'s `GetArmStatus()` / `GetArmJointMsgs()` return objects, but
  with `Hz: 0.0` and all-zero/default values - i.e. also no real frame ever
  received.
- CAN fps sits at ~1450, roughly half of AgileX's documented ~3040 fps for a
  healthy single-arm connection - consistent with that entire status range
  being silent.
- Confirmed with both SDKs (rules out an SDK-specific parsing bug), and
  confirmed after a full power cycle of the arm (rules out transient state).

To reproduce/check:

```bash
candump can0 -n 500 | awk '{print $2}' | sort | uniq -c | sort -rn
```

If `0x2A1`-`0x2A8` don't appear, low-level driver frames (`0x251`-`0x256`,
`0x3A0`-`0x3A7`, etc.) do, and `GetCanFps()`/`get_fps()` is roughly half the
expected rate, this is the same issue.

**Not resolved yet.** Suspected firmware/hardware-side condition on this
specific arm rather than a CAN or SDK config problem. Next steps: contact
AgileX support with the diagnostics above, check the teach pendant (if
present) for a fault indicator, and consider a firmware update given
`S-V1.8-3` is an early firmware revision.
