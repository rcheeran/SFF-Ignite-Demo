# Piper × LeRobot × pi0.5 — VR teleop → dataset → fine-tune → deploy

AgileX Piper 6-DoF arm on a Jetson AGX Thor, teleoperated with a Meta Quest 3, recorded into LeRobot datasets,
fine-tuned with LeRobot's `pi05` policy, deployed with the same robot class. A MuJoCo digital twin (`sim/`) stands in
for the arm and cameras behind `--sim`. Task for the smoke test: **"put the block in the bowl"**.

Everything the old team wrote (openpi patch, drag-teach recorder, scripted collector, custom pose policy runner) lives
untouched in `legacy/` for reference. Do not import it.

## Layout

```
piper/       config.py (every constant) · transforms.py (Euler/6-D, step clamps, fk_flange, tip_position) · robot.py
             (Piper: connect/enable, reads, faults, one-tick senders, blocking moves, clamp_pose, lock, rest) ·
             cameras.py (UsbCamera, RealSenseCamera, FakeCamera) · fake_arm.py (FakeArm, hardware-free Piper) ·
             lerobot_robot.py (PiperRobot = LeRobot Robot plugin)
teleop/      quest.py (QuestTeleop: HTTPS/WS/POST server + 30 Hz Cartesian control thread) · static/index.html (WebXR
             page) · certs/ (self-signed TLS, gitignored)
collect/     record.py   — LeRobotDataset recording loop driven by headset buttons (resume, one-frame-lag actions)
training/    train.sh (Thor wrapper) · cloud/ {launch,status,terminate,setup,train}.sh + hub_check.py + lambda_api.py
             (Lambda H100) · pull_checkpoint.sh · make_fake_dataset.py · NOTES.md
inference/   run_policy.py — pi05 checkpoint -> 15 Hz joint streaming (chunk prefetch, clamps, shadow mode, CSV log)
sim/         MuJoCo digital twin: piper_sim.py (SimPiper), cameras.py (SimCamera), scene.xml, expert.py (scripted task)
tools/       can_up.sh · check_arm.py · stream_test.py · rest.py · inspect_dataset.py · selftest.sh
tests/       pytest suite, all hardware-free (FakeArm, FakeCamera, --fake/--dry-run modes, gjs for the page)
data/        local datasets (LeRobot format), logs/, snapshots/, sim/   (gitignored)
legacy/      old code + old soda-can data
```

## Hardware / environment facts (verified)

| thing | value |
|---|---|
| machine | Jetson AGX Thor, aarch64, Ubuntu 24.04, Python 3.12, 122 GB unified, CUDA 13.2, torch `2.9.1+cu130` works |
| arm | AgileX Piper, `piper_sdk` 0.6.2, USB-CAN (gs_usb). Interface **`can_piper`**. Bring up: `bash tools/can_up.sh` (= `sudo /usr/local/sbin/piper-can-up.sh 1000000` + bus state; passwordless sudo works). `piper-can.service` does it at boot with retries. |
| wrist cam | Intel RealSense D405 (pyrealsense2 2.58), color 640×480 @30. Mounted on **top** of the gripper at J6 = 0 (the centre of the wrist's mechanical range, hard stops ≈ ±180°): the picture is upright when J6 ≈ 0. The old team's −165° "camera-up" roll was camera-*down* (15° short of inverted), which is the "almost upside down" picture the old snapshots show. Check: `tools/check_arm.py` snapshot + its joints line; twin: `python -m sim.piper_sim --snap` |
| overhead cams | `/dev/v4l/by-id/usb-WSD_usb_camera_202406051510-video-index0` (**HP**, 720p, 30 fps — the one in use, `TOP_CAM_DEV`) and `/dev/v4l/by-id/usb-Generic_USB_2.0_Camera_20220523-video-index0` (Realtek 5 MP, ~18 fps at 640×480). Both MJPG. Use the by-id path, node numbers shuffle. |
| network | Jetson WiFi `192.168.1.45` (wlP1p1s0), 0.3–0.5 s stalls seen; Ethernet preferred. Quest must be on the same WiFi. |
| envs | `~/lerobot-venv` = lerobot 0.6.1 + torch cu130 + piper_sdk + pyrealsense2 + mujoco (everything here). `~/pi05-venv` = the old openpi stack. **Do not modify it.** |
| HF | user `royc1981`, token in `~/Documents/HG-TOKEN.txt` |
| gotcha | a leftover `dual_camera_stream.py` (port 8090) may hold the cameras — `pkill -f dual_camera_stream` (done before motion) |

## Piper control facts (from the SDK + the old team's scars)

- Connect (`Piper.connect()`): checks `ip link` first (clear error if `can_piper` is down), `C_PiperInterface_V2("can_piper")`,
  `ConnectPort()`, waits up to `FEEDBACK_WAIT_S` = 2 s for joint feedback (`GetArmJointMsgs().Hz > 0`), then — before
  torque — one position-mode `MotionCtrl_2(0x01, 0x01, speed, 0x00)` so a driver left latched in MIT mode (`0xAD`) is
  never torqued in MIT mode, then loop `EnablePiper()` until `all(GetArmEnableStatus())`. Then `restore_joint_limits()`:
  query the motors' stored angle limits (`SearchAllMotorMaxAngleSpd` → `GetAllMotorAngleLimitMaxSpd`, 0.1° units) and
  write the factory limits back (`MotorAngleLimitMaxSpdSet(motor, max, min, 0x7FFF)`) into any joint that differs.
  The old code set J6 to [-179, -60] ("keep the camera up"); that limit is what both live sessions froze on, and the
  camera flange never needed it. Factory limits (`JOINT_LIMITS_DEG`): J1 ±150, J2 0..180, J3 -170..0, J4 ±100,
  J5 ±70, J6 ±120 (piper_sdk `piper_param`). `tools/check_arm.py` prints the stored limits with a factory verdict.
- Read: `GetArmJointMsgs().joint_state.joint_1..6` (0.001 deg), `GetArmEndPoseMsgs().end_pose.X_axis..RZ_axis`
  (0.001 mm / 0.001 deg), `GetArmGripperMsgs().gripper_state.grippers_angle` (0.001 mm, 0..70000),
  `GetArmLowSpdInfoMsgs().motor_6.motor_temp` (J6 overheats ≥66 °C under heavy wrist motion — pause at 46 °C).
- Status: `GetArmStatus().arm_status` carries `.arm_status` (enum NORMAL / EMERGENCY_STOP / NO_SOLUTION /
  SINGULARITY_POINT / TARGET_POS_EXCEEDS_LIMIT / JOINT_COMMUNICATION_ERR / JOINT_BRAKE_NOT_RELEASED /
  COLLISION_OCCURRED / ...), `.ctrl_mode`, `.mode_feed`, `.motion_status`, and `.err_status.joint_<n>_angle_limit` /
  `.communication_status_joint_<n>` booleans → `Piper.faults()` (list of strings, empty = healthy) and `Piper.health()`.
  Teleop polls it at 1 Hz into the ack; deploy stops on `FAULT_STATUS` (EMERGENCY_STOP, COLLISION_OCCURRED). Whether the
  firmware reports NO_SOLUTION while refusing a Cartesian target is still unknown — watch `fw=` in the teleop line.
- **Every command tick must start with `MotionCtrl_2(0x01, move_mode, speed_pct, 0x00)`** (0x01 = CAN control mode) or
  the arm silently ignores it, and the gripper must be re-sent every tick:
  - Cartesian (teleop): `MotionCtrl_2(0x01, 0x00, speed, 0x00)` then `EndPoseCtrl(x,y,z,rx,ry,rz)` in 0.001 mm /
    0.001 deg. Firmware IK; silently refuses unreachable/singular targets (arm just doesn't move).
  - Joint (policy deploy, homing): `MotionCtrl_2(0x01, 0x01, speed, 0x00)` then `JointCtrl(j1..j6)` in 0.001 deg.
    Stream the *same* target until reached — re-sending interpolated waypoints makes it jerky.
  - Gripper: `GripperCtrl(opening_um, 1000, 0x01, 0)` (effort 1000 verified sufficient; 0 = closed, 70000 = open).
    `gripper_code`: 0x00 = disable (de-torque, used by `rest()`), 0x01 = enable, 0x02/0x03 = disable/enable + clear error.
- Euler convention of RX/RY/RZ: R = Rz·Ry·Rx = scipy `from_euler('xyz', ..., degrees=True)` — see `piper/transforms.py`.
- **J6 zero lives in the motor** (`JointConfig(joint 6, set_zero)` = `Piper.zero_joint(6)`, persistent). It must be the
  physical centre of the wrist range with the camera on top; on this arm it had been taught ~180° off (camera down,
  on a mechanical stop), which is what the old team's −165° "camera-up" roll + custom limit papered over, and what
  made J6 = 0 stall the motor (driver cuts the joint → limp, firmware refuses everything). `tools/zero_j6.py` fixes it.
  `JointConfig(7, clear_err)` clears joint error codes (`Piper.clear_joint_errors()`); `EnableArm/DisableArm(n)` torque
  one motor (`Piper.enable_joint`).
- Joint poses (deg): HOME `[0, 0, 0, 0, 0, 0]` (folded, all joints at zero), REST `[0, 0, 0, 0, 25, 0]` (wrist in its
  gravity-settle band, zero drop on de-torque), READY `[0, 45, -25, 0, 40, 0]` (flange ≈ [128, 0, 232], gripper 50°
  down; where Cartesian streaming and every episode start). Speed 15 % for homing.
- Flange pose at zero joints ≈ `[56, 0, 213, ...]`; the table is at flange z ≈ 0; the gripper extends ~120 mm past the
  flange along the tool z-axis (`TOOL_LEN_MM`), so enforce a **tool-tip Z floor**, not a flange floor:
  `piper/transforms.tip_position(joints_deg)` = flange + TOOL_LEN_MM along tool z (the floor for joint-space targets).
- Lock: `/tmp/piper_arm.lock` (flock + pid). `connect_for_motion()` takes it, refuses if any other raw-CAN receiver is
  bound to `can_piper` (`/proc/net/can/rcvlist_all`), enables, counts down. A lock whose pid is dead is auto-cleared.
  `connect_readonly()` takes the lock without enabling (shadow mode). Never bypass either.

## Dataset contract (ONE source of truth — all code must match)

- LeRobot dataset (v3, videos), **fps = 15**, repo id `<hf_user>/piper_block_bowl`, local root `data/<name>`.
- `observation.images.top` and `observation.images.wrist`: uint8 480×640×3.
- `observation.state`: float32[7] = `[j1..j6 (deg), gripper_measured_mm]`.
- `action`: float32[7] = `[j1..j6 at the NEXT frame (deg), gripper_commanded_mm]`.
  Joint-space actions on purpose: the policy will be deployed with `JointCtrl`, which never refuses targets; the human
  teleop is Cartesian (firmware IK), so the recorder writes each frame with a one-frame lag so its action is the next
  observed joints (the last frame of an episode holds its own joints).
- `task`: `"put the block in the bowl"`.
- `training/cloud/hub_check.py dataset <repo>` verifies a Hub dataset against this contract before any GPU time is spent;
  `tools/inspect_dataset.py <root>` judges a local one (lengths, steps/frame vs the 10° deploy clamp, lag check, gaps).

## Teleop contract

- Quest Browser opens `https://192.168.1.45:8443/` (self-signed cert in `teleop/certs/`, accept the warning once), taps
  ENTER → WebXR `immersive-ar` (passthrough, so the operator sees the real arm; fall back to `immersive-vr`). Each XR
  frame the page sends `{t, head:{p,q}, right:{p,q,trigger,squeeze,a,b,thumb}}` (right controller `gripSpace` pose in
  `local-floor`, metres, WebXR frame: +X right, +Y up, −Z forward) over a WebSocket on the same origin, or — because the
  Quest browser refuses `wss://` under the self-signed cert — over same-origin `POST /pose` at ≤30 Hz. Only JSON objects
  are accepted (anything else is a 400 and an event); a packet with a malformed head/right is dropped by the control
  loop (`bad packet dropped` event) and counts as no packet.
- Python mapping (`teleop/quest.py`): WebXR → robot base (+X forward, +Y left, +Z up) = `x_r = −z_xr, y_r = −x_xr,
  z_r = y_xr`, with yaw re-zeroed to the headset's forward when the session starts (operator stands behind the arm
  looking along its +X).
- **Buttons**: A owns teleop — hold A 3 s = ARM / DISARM (`TELEOP_ARM_HOLD_S`; nothing moves while disarmed, leaving
  XR disarms), tap A = `a_tap` (the recorder discards the running episode). B owns episodes — hold B 3 s
  (`TELEOP_EPISODE_HOLD_S`) = `b_hold` (start), tap B = `b_tap` (stop and save). Both holds report progress in the ack
  (`arm_progress`, `b_progress`). Thumbstick click = READY.
- **Clutch = squeeze (grip button).** On press: store controller pose and the arm's *commanded* pose as references.
  While held the controller drives the **wrist centre** `wc = flange + R·[0, 0, −TELEOP_PIVOT_MM]` (91 mm behind the
  flange, where the J4/J5/J6 axes meet): `wc_target = wc_ref + SCALE·(ctrl_pos − ref_ctrl_pos)`,
  `target_rot = (ctrl_rot·ref_ctrl_rot⁻¹)^ROT_SCALE·ref_robot_rot`, `target_pos = wc_target + target_rot·[0, 0, TELEOP_PIVOT_MM]`.
  A pure hand rotation is therefore a wrist-only motion (twin: base-joint travel 306° → 4° for ±25° hand rotations).
  On release: hold the last target (snapped back to the measured pose if the target was pinned at the lead clamp).
  Trigger (analog) → gripper opening mm (1.0 = closed).
- Joint-limit projection on every tick (`piper/kinematics.reachable`): a damped-least-squares IK on the SDK's DH chain
  runs from the measured joints; when the target would push a joint past `JOINT_LIMITS_DEG` (or has no solution) the
  nearest reachable pose is sent instead and `limit` names the joints (`['J5']` in practice — wrist pitch ±70°). This is
  what the firmware would refuse (TARGET_POS_EXCEEDS_LIMIT / NO_SOLUTION); projecting first turns a freeze into a slide.
  The IK is weighted (`IK_JOINT_COST`: a degree of J1–J3 costs 3/3/2× a degree of wrist), so an unreachable target is
  approximated by leaving error rather than swinging the base.
- Safety on every tick: workspace box clamp, tool-tip Z floor, per-tick step clamp (≤15 mm, ≤8°), lead clamp (target
  never more than 30 mm / 20° ahead of the measured pose), stuck auto-declutch (at the lead clamp for 2 s → target snaps
  to the measured pose, clutch off, fresh squeeze required, headset shows STUCK), firmware `arm_status` at 1 Hz
  (COLLISION_OCCURRED / EMERGENCY_STOP → same declutch, target tracks the arm while the flag persists), 0.5 s watchdog
  (no packets → hold; an A-hold interrupted by a gap restarts), no-feedback hold (nothing streamed), J6 temperature
  pause, session_end → disarm, any control-loop exception → stop streaming + `rest()`, Ctrl-C → `rest()`.
- Threading: the aiohttp thread only writes the latest packet and flags (session_end, thumbstick); the 30 Hz control
  tick consumes them. Never mutate `engaged` / `_ref` / `armed` from the server thread.
- Control loop 30 Hz in its own thread; the recorder samples at 15 Hz.

## Digital twin (sim/)

- `sim/piper_sim.py` `SimPiper` subclasses `piper.robot.Piper`: same methods, same units; `pose()` = piper_sdk FK of the
  simulated joints (convention identical by construction); MuJoCo only does physics and rendering. MJCF qpos = reported
  joints + `SIM_JOINT_OFFSET_DEG` [0, +2, −2, 0, 0, 0] (the SDK/firmware FK carries a 2° J2/J3 zero offset); residual
  vs the SDK FK 1.4 mm / 0.002°. `connect_sim()` = the process-wide twin (arm + cameras share one physics thread).
  With `$DISPLAY` set (Jetson desktop) `connect()` opens a live `mujoco.viewer` window synced from the physics
  thread at 30 Hz (`SIM_VIEWER=0` disables); headless otherwise. `--sim` never touches the CAN bus.
- Model: mujoco_menagerie `agilex_piper` (MIT), meshes fetched by `sim/fetch_assets.sh` (32 MB, gitignored). Local
  edits to `piper.xml`: meshdir, a `wrist` camera on link6, J5/J6 kp 80.
- Scene (`sim/scene.xml`): table top at z = 0 (base plate on it), hollow bowl at `SIM_BOWL_XY_MM`, 30 mm block spawned
  in `SIM_BLOCK_SPAWN_MM`; cameras `top` and `wrist` render 480×640 BGR like the real ones (`sim/cameras.py`, EGL).
- Behaviour: `send_pose` = damped-least-squares IK (warm-started, ≤ `SIM_IK_ITERS`), refuses like the firmware (arm
  holds, `arm_status` `TARGET_POS_EXCEEDS_LIMIT(0x4)` / `NO_SOLUTION(0x2)`); `send_joints` clips to the joint limits
  (factory, J6 ±120); JointCtrl moves slew at speed % of `SIM_JOINT_VMAX_DEG_S`, EndPoseCtrl streaming follows at
  the full rate behind `SIM_LAG_S` (uncalibrated: check against `tools/stream_test.py`). Status strings use the real
  SDK's `NAME(0xN)` form; `faults()` returns bare names.
- Expert (`sim/expert.py`): privileged scripted pick-and-place through `clamp_pose` (the teleop clamps), fires the
  recorder's A-events, 10/10 on seed 0; `collect.record --sim --scripted` writes datasets with it.
- Measured (2026-08-22): at READY a ±15 mm / ±8° wiggle tracks with 0.5–1.4 mm RMS, 100–233 ms lag, 0 refused ticks
  (real arm at the older READY: ≤ 2.5 mm, 67–200 ms). The previous READY (flange 84 mm out, J6 at −119 under the
  old [−179, −60] limit) refused 28–48 % of ticks with J6 pinned at −179° — the live-session freeze.
- Gaps: IK is not the firmware's; fingertips 135 mm past the flange vs `TOOL_LEN_MM` 120; gripper effort not
  modelled; guessed camera intrinsics; no visual realism — real-data policies do not transfer to sim renders without
  domain randomization. Use it for pipeline validation, teleop practice and sim datasets.

## Runbook (hardware session)

```bash
source ~/lerobot-venv/bin/activate && cd ~/Documents/piper-openpi-V2
bash tools/can_up.sh                                   # once per boot / replug / after a crash (resets ERROR-PASSIVE)
python tools/check_arm.py                              # read-only: CAN + ARM verdicts, joints, faults, FK, camera snapshots
python tools/stream_test.py                            # home -> READY -> ±20 mm/±10° Cartesian wiggle -> rest; prints lag + TRACKS
python -m teleop.quest --scale 0.25                    # live teleop (moves to READY first)
python -m collect.record --repo-id <user>/piper_block_bowl --episodes 10 --push   # A start/stop, B discard; re-run = resume
python tools/inspect_dataset.py data/piper_block_bowl  # lengths, steps/frame, lag check, contact sheet
bash training/cloud/launch.sh --check <user>/piper_block_bowl <user>/pi05_piper   # then without --check
bash training/pull_checkpoint.sh <user>/pi05_piper
python -m inference.run_policy --checkpoint outputs/pi05_piper/pretrained_model --shadow     # no motion
python -m inference.run_policy --checkpoint outputs/pi05_piper/pretrained_model --max-steps 450
python tools/rest.py
```

Quest side: same WiFi, Quest Browser → `https://192.168.1.45:8443` → Advanced → proceed → ENTER (passthrough).
Stand behind the arm looking along its +X (yaw is re-zeroed to your head at session start). Squeeze = clutch.

Findings that changed the plan:
- The firmware IK **refuses every Cartesian target from the folded HOME** (arm sits still, no error) and goes sluggish
  (half amplitude, 1–2 s lag) at J5 = 0, the wrist singularity. Teleop and the self-test therefore go joint-space to
  READY first. From there streaming tracks 15 mm / 8° sinusoids with ≤2.5 mm RMS error and 67–200 ms lag; gripper
  tracks 20↔70 mm.
- Both live sessions froze with J6 pinned at −178°: the old code had narrowed J6 to a custom [−179, −60] limit (a
  "keep the camera up" precaution that was never needed) and parked the wrist at −119/−165, so ordinary tool yaw ran
  J6 into the limit, where the IK refuses silently while the commanded target keeps integrating away → violent
  catch-up when it unsticks. Fixed at the root (factory limits restored on every connect, J6 = 0 in every pose =
  ±120° of margin) and defended in depth: lead clamp, stuck auto-declutch, headset wrist-limit warning.
- HOME is the true zero pose; after de-torque the unpowered wrist pitch (J5) falls into a friction band of 10–31° and
  then stays put, so REST = J5 25° is commanded before disabling: measured drop after de-torque = 0.0° over 20 s.
- The Quest Browser never opened the `wss://` WebSocket under the self-signed cert (0 packets in the field); the page
  falls back to same-origin `POST /pose` at 30 Hz, and arming (hold A 3 s) is required before anything moves.
- Overhead camera = **HP** (`TOP_CAM_DEV`): 30 fps, tight well-exposed view of bowl + blocks.
- `lerobot-record --robot.type=piper` is not wired (needs an installed plugin package); `collect/record.py` is the recorder.


## Deploying a teach-collected policy

**`--n-action-steps` must be ~30, not the default 10.** Every episode from `collect/teach.py` opens with the arm
stationary at the start pose — `replay()` settles for `SETTLE_S`, and sampling begins before the operator starts
guiding. Measured on the 50-episode set: **median 42 frames (2.8 s) before any joint moves >3°, and 50/50 episodes
need more than 10**. So chunk actions 0-25 are the settle phase. With `N_ACTION_STEPS=10` the loop executes only
that prefix, re-predicts from an unchanged state, gets the same prefix, and the arm **stalls at the start pose
forever** — which looks exactly like a broken policy. It is not: on `royc1981/lerobot-pi05-base-50` the same
checkpoint reproduces recorded training actions to 0.3-6.2 deg and plans a 68 deg reach in its 50-action chunk.
The cost of 30 is 2 s of open-loop execution between re-plans instead of 0.67 s.

The real fix is data-side: trim the stationary prefix from each episode so the policy learns "at start pose -> go"
rather than "-> wait", and the default 10 works again. Do that before collecting the next batch.

**Inference on the Thor cannot use torch.compile.** lerobot's pi05 compiles `sample_actions`; Triton bundles a
CUDA 12.8 `ptxas` and Thor is `sm_110a`, so it dies with `ptxas fatal : Value 'sm_110a' is not defined`. There is
no system CUDA toolkit to fall back on. `inference/run_policy.py` sets `torch._dynamo.config.disable = True`
(override with `PIPER_COMPILE=1`); `training/train.sh` already passed `COMPILE=false` for the same reason.
Eager costs little here: 424 ms/call synthetic, 527 ms with real cameras, 0 late ticks at 15 Hz.

**Jittery motion on deploy is the chunk boundary, not the policy.** Measured on a live 31 s run (450 ticks,
44 re-plans): the commanded step is **10.6 deg mean / 45.9 max AT a boundary** against **1.08 deg elsewhere**.
A fresh chunk starts where the model thinks the arm should be, which is not where the last action put it;
`clamp_joints` caps the result at MAX_JOINT_STEP_DEG so it jerks (10 deg/tick = 150 deg/s) rather than lurches.
`ActionStream` now carries that discontinuity as an offset and decays it over `BLEND_TICKS` (default 6, joints
only - offsetting the gripper would blur the moment it closes). `--blend-ticks 0` restores the old behaviour.
Diagnose from the per-tick CSV: compare `action_j*` steps at ticks where `chunk_age_s` resets against the rest.
