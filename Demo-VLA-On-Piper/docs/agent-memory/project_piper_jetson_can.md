---
name: project_piper_jetson_can
description: How the AgileX Piper connects over CAN on the Jetson AGX Thor (custom gs_usb build; arm is can4)
metadata: 
  node_type: memory
  type: project
  originSessionId: acabfc4d-beea-4306-a43a-cbd8b34f1fb9
  modified: 2026-08-07T22:31:45.112Z
---

Connecting the AgileX Piper arm to the **Jetson AGX Thor Developer Kit** over CAN. Setup repo: `~/Documents/SFF-AgileX-Piper` (see its README and `prepare-jetson/SETUP.md`). Software prereqs installed into venv at `~/venv` (pyAgxArm, piper_sdk, python-can, numpy, opencv, pyrealsense2 — all have aarch64 wheels).

**Key gotcha: the Thor kernel (`6.8.12-*-tegra`) ships NO USB-CAN drivers** — `drivers/net/can/usb/` is absent, so the candleLight/`gs_usb` USB-CAN adapter (`1d50:606f`) that ships with the Piper has no driver out of the box. The 4 native `can0`–`can3` are onboard `mttcan` controllers (not the arm).

**Fix that works: build `gs_usb` out-of-tree** (no rewiring needed). Kernel headers are present (`nvidia-l4t-kernel-headers`, build dir `/lib/modules/$(uname -r)/build`, Module.symvers already has can-dev symbols). Recipe in `~/gs_usb_build`:
- `curl -fsSL raw.githubusercontent.com/torvalds/linux/v6.8/drivers/net/can/usb/gs_usb.c`
- Makefile `obj-m += gs_usb.o`, `make -C /lib/modules/$(uname -r)/build M=$PWD modules`
- `sudo insmod ./gs_usb.ko` (unsigned → taints kernel, harmless).

**PERSISTENCE (done, survives reboot):**
- Module installed to `/lib/modules/$(uname -r)/updates/gs_usb.ko` + `sudo depmod -a`.
- `/etc/modules-load.d/gs_usb.conf` (contains `gs_usb`) → auto-loads at boot.
- `/usr/local/sbin/piper-can-up.sh` finds the CAN iface whose driver==gs_usb and brings it up @1Mbps (robust to index changes).
- `piper-can.service` (systemd oneshot, enabled) runs that script on boot (10x retry for USB enumeration).
- NOTE: out-of-tree module is tied to the exact kernel version — after a Jetson kernel upgrade, rebuild gs_usb (or use DKMS).
- Also folded into `prepare-jetson/setup.sh` as step 5 `ensure_gs_usb()`: skips if `modinfo gs_usb` works, else builds+installs persistently. Added build-essential+curl to apt step.

**STABLE NAME (done): udev rule `/etc/udev/rules.d/80-can-piper.rules`** matches USB ID 1d50:606f and renames the adapter to **`can_piper`** every boot (the enumeration order is a race with native mttcan — seen as both can4 and can0 across reboots, so the fixed name matters). Use `can_piper` in all SDK calls, e.g. `python3 prepare-piper/check_connection.py can_piper`. The bring-up service/script find the adapter by driver==gs_usb so they work regardless of name.

Verified across a full reboot: module auto-loads, service brings up can_piper @1Mbps, arm streams. Healthy = fps ~3040-3060, ArmStatus.Hz ~200, JointMsgs.Hz ~200, STANDBY, NORMAL, no errors, 0x2A1–0x2A8 present. EnableStatus all False (motors not energized — separate step, MOVES arm).

GOTCHA: a silent bus (0 frames, 0 errors) = arm-side issue (powered off / Master mode / CAN cable), NOT software — proven by full reboot cycle. Fix: power arm on, Joint-5 button OFF (Slave), reseat CAN cable, then arm streams immediately. An arm power-cycle can reset the adapter iface → bounce it (`ip link set can_piper down/up`).

`prepare-jetson/setup.sh` now reproduces the whole setup: `ensure_gs_usb()` (step 5, builds+installs gs_usb if missing) + `install_piper_can_service()` (step 6, writes udev rule + /usr/local/sbin/piper-can-up.sh + piper-can.service, enables it). Both idempotent. Uses `PIPER_IFACE` env (default can_piper). Related: [[project_asimov_robotics]].
