# piper-real-toolset — Asimov SDK ↔ the PHYSICAL AgileX Piper arm

Maps the Asimov SDK onto the **real** AgileX Piper 6-DOF arm over CAN
(`piper_sdk`), exposing the same MCP tool pattern as the sim toolset. A single
`PIPER_MODE` switch selects the backend, and motion is **observe-only by
default** — you must opt in to move real hardware.

```
piper-real-toolset/
├── backends.py   # MockPiperBackend + RealPiperBackend (piper_sdk over CAN)
├── main.py       # ToolsetMCPServer: PIPER_MODE switch + PIPER_ALLOW_MOTION gating
└── requirements.txt
piper-real-agent/ # RobotAgent (safety-first prompt) + local Ollama LLM
```

## Modes

| `PIPER_MODE` | Backend | Needs |
|---|---|---|
| `mock` (default) | `MockPiperBackend` | nothing — canned state, no CAN/hardware. Use it to exercise the agent + tools. |
| `real` | `RealPiperBackend` | the physical Piper on `can0`, `piper_sdk`, and the arm in **Slave mode + enabled**. |

## Safety model (mirrors the Asimov FR3 real toolset)

- **Reads are always allowed** (`get_status`, `get_joint_positions`, `get_ee_pose`,
  `get_gripper`, `get_connection`).
- **Motion is OBSERVE-ONLY unless `PIPER_ALLOW_MOTION=1`.** Without it,
  `enable_arm`, `move_*`, and gripper tools refuse with a clear message.
- **`emergency_stop` is always available** and halts the arm immediately.
- The `RealPiperBackend` reproduces AgileX's **joint 2/3/5 gravity-fall guard**
  before releasing an e-stop hold, and clamps joint targets to the arm limits.
- The agent's system prompt is safety-first: it checks status/connection before
  moving, stops on telemetry warnings, and moves one step at a time.

## Tool surface (identical style to the sim toolset)

Reads: `get_status`, `get_joint_positions`, `get_ee_pose`, `get_gripper`,
`get_connection`, `emergency_stop`.
Motion (gated): `enable_arm`, `home`, `move_to_joint_positions`, `move_to_pose`
(the arm does its own IK), `move_relative`, `open_gripper`, `close_gripper`,
`set_gripper` (+ auto `poll_status`/`cancel`/`pause`/`resume`).
Camera: `capture_image`, `get_scene_depth` (see below).

Units: joints in **degrees**, Cartesian x/y/z in **meters**, orientation
rx/ry/rz in **degrees**, gripper opening **0..1**.

## Camera (Intel RealSense D405 wrist cam)

Enabled by default (`PIPER_CAMERA=on`; set `off` to disable). On start the
toolset opens the D405 color+depth pipeline; if no camera is present it degrades
gracefully and the camera tools report unavailable.

- `capture_image` — grab a color snapshot (saved under `captures/`) and return
  its path + the depth (mm) at the image centre.
- `get_scene_depth(u, v)` — median depth (mm) at a normalized image point
  (0..1; 0.5,0.5 = centre), to gauge how far an object is.
- `GET /camera/snapshot` — live JPEG over HTTP (used by the admin UI / browser).

Runs independently of the arm — the camera works even while the arm is powered
off. `camera.py` + `pyrealsense2`/`opencv` back it; both are in `../.venv`.

## Run — MOCK (no hardware)

```bash
cd piper-real-toolset
PIPER_MODE=mock PIPER_ALLOW_MOTION=1 ../.venv/bin/python main.py    # MCP on :8212
# agent:
cd ../piper-real-agent && ../.venv/bin/python agent.py             # :8213 → /admin
```

## Run — REAL (physical arm) — do this at the robot

This toolset uses the `../.venv` (has `asimov` + `piper_sdk`). The CAN interface
and arm mode setup come from the sibling **`~/Documents/SFF-AgileX-Piper`**
project — follow its `prepare-jetson/SETUP.md` first. Summary:

1. **Slave mode**: turn OFF the button on Joint 5, then
   ```python
   from piper_sdk import *; p=C_PiperInterface_V2("can0"); p.ConnectPort()
   p.MasterSlaveConfig(0xFC,0,0,0)   # 0xFC = Slave
   ```
2. **Restart the arm** (power-cycle) so Slave mode takes effect.
3. **Bring up CAN** (re-run after every power cycle):
   ```bash
   bash ~/Documents/SFF-AgileX-Piper/prepare-piper/can_activate.sh can_piper 1000000
   ip -details link show can0        # expect: UP, ERROR-ACTIVE, bitrate 1000000
   ```
4. **Verify telemetry** (must be non-zero, else the arm is still in Master/teach mode):
   ```bash
   ~/venv/bin/python ~/Documents/SFF-AgileX-Piper/prepare-piper/check_connection.py can_piper
   ```
5. **Start the toolset in real mode.** Keep it observe-only until you've confirmed
   reads look right, then enable motion:
   ```bash
   # observe-only first (safe): reads work, motion refuses
   cd ~/Documents/piper-asimov/piper-real-toolset
   PIPER_MODE=real ../.venv/bin/python main.py

   # when ready to move (clear space around the arm, hand on power):
   PIPER_MODE=real PIPER_ALLOW_MOTION=1 PIPER_SPEED_PERCENT=30 ../.venv/bin/python main.py
   ```
6. Start the agent (`../.venv/bin/python ../piper-real-agent/agent.py`), open
   http://localhost:8213/admin, and command it — e.g. *"check status"*, then
   *"enable the arm"*, then small moves. Say *"stop"* to trigger emergency_stop.

> **⚠️ This drives a real robot.** Clear the workspace, keep the power switch in
> reach, start at low `PIPER_SPEED_PERCENT`, and prefer small verified moves.
> `move_to_pose` targets outside the reachable workspace are rejected at the CAN
> level — the tool reports the failure.

## Known issue on this specific arm

Per the [setup notes](../../../prepare-jetson/README.md), this unit (firmware `S-V1.8-3`) did not
broadcast the high-level status frames (CAN `0x2A1`–`0x2A8`) until put in Slave
mode + restarted — so `get_status`/`get_ee_pose` telemetry read `Hz=0`. If you
see that, the arm is still in Master/teach mode: redo Slave mode + restart +
CAN reactivate. The toolset surfaces this as a `get_connection` warning.

## Status of testing

The **mock** path is fully verified (agent → MCP → tools end-to-end). The
**real** backend is written to the `piper_sdk` API used by the SFF-AgileX-Piper
scripts (`prepare-piper/check_connection.py`, `useful-piper-scripts/read_current_pose.py`, `useful-piper-scripts/pick_can.py`,
`useful-piper-scripts/play_pos.py`) — unit conversions, enable/mode sequence, EndPose/Joint/Gripper
control, e-stop, and the gravity-fall guard — but has **not** been exercised
against the physical arm from here. Bring it up observe-only first and verify
reads before enabling motion.
