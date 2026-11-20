# SFF AgileX Piper Setup

Setup and diagnostic scripts for controlling the AgileX Piper arm over CAN
from an Ubuntu machine.

Tested on Ubuntu 24.04.4 LTS / Python 3.12.3 with a single AgileX USB-CAN
adapter connected.

## Setup

Run commands from the repository root unless stated otherwise:

```bash
cd ~/SFF-Ignite-Demo
bash prepare-jetson/setup.sh
```

This installs `can-utils`/`ethtool`/`python3-venv`, creates a venv at
`~/venv`, installs `pyAgxArm` (from GitHub source - it is not published on
PyPI, so `pip install pyAgxArm` alone will fail) and `piper_sdk` (from
PyPI), installs Intel RealSense D405 permissions, and brings up the stable
`can_piper` interface at 1 Mbps. Python dependencies are pinned in
`prepare-jetson/requirements-setup.txt` for repeatable installations.

Requires `sudo` (interactive password prompt for `apt install` and the CAN
interface commands). Run the script as your normal user, not with `sudo`.
Log out and back in (or reboot) afterward so the new `video` and `plugdev`
group memberships take effect.

The script requires the standard Ubuntu package repositories to be enabled.
It stops with an actionable error instead of changing repository policy when
required packages are unavailable.

If more than one CAN adapter is plugged in, don't rely on the single-adapter
shortcut in `prepare-jetson/setup.sh` - run `prepare-piper/can_activate.sh <name> <bitrate>
"<usb-bus-address>"` manually per adapter instead (see `prepare-piper/can_activate.sh`
usage output, or `scripts/can_muti_activate.sh` upstream in the pyAgxArm repo
for activating several at once).

## Verifying the connection

```bash
source ~/venv/bin/activate
python3 prepare-piper/check_connection.py can_piper
```

Read-only: connects with both `pyAgxArm` and `piper_sdk`, prints firmware
info, CAN fps, joint enable status, arm status, and joint angles. Sends no
motion commands.

## Wrist-mounted D405 calibration

The active wrist-camera workflow is in
[`prepare-piper/hand-eye`](../prepare-piper/hand-eye/README.md). It calibrates the
D405-to-Piper rigid transform from one fixed ArUco marker observed at many
wrist poses, validates it from fresh poses, calibrates D405 depth scale/offset,
verifies corrected 3D coordinates, and provides a browser click-to-coordinate
test.

The older [`prepare-piper/homography`](../prepare-piper/homography/README.md)
workflow is retained only for a fixed camera pose where all four markers are
visible in one image. Moving the wrist invalidates that planar homography.

## Foundry YOLO Starburst-to-palm loop

The k3s [`k3s-OpenCV-service`](../k3s-OpenCV-service/README.md) exclusively owns the D405,
performs image/depth processing, and sends YOLO tensors to the Foundry Local
`candy-yolo11s` GPU deployment. Deploy it and the MediaPipe hand service first:

```bash
cd ~/SFF-Ignite-Demo
./k3s-OpenCV-service/build-and-deploy.sh
./k3s-hand-service/build-and-deploy.sh
curl http://127.0.0.1:8094/healthz
curl http://127.0.0.1:8095/healthz
```

Open `http://10.0.0.19:8094/` and confirm the live stream. The demo requires
the health response model to be `candy-yolo11s`; the OpenCV service calls
Foundry Local for GPU inference and performs NMS and D405 depth processing.

Teach one to three collision-free scan poses (default: three). For one view:

```bash
source ~/venv/bin/activate
python prepare-piper/Teach-scripts/teach_scan_poses.py --pose-count 1
```

The Piper SDK zero/home joint pose is `[0, 0, 0, 0, 0, 0]` degrees. The
recorder moves to that pose automatically immediately after connecting, then
verifies joint telemetry before allowing physical teach mode. The zero move uses the
SDK sequence `ModeCtrl(0x01, 0x01, speed, 0x00)` followed by
`JointCtrl(0, 0, 0, 0, 0, 0)`. After saving stable
joint/end telemetry to `prepare-piper/scan-poses/scan_poses.json`, it moves to
and verifies the same SDK zero pose again before replaying the requested poses.
It returns to zero after replay and keeps zero streamed until controller power
is off. Restart the controller and CAN interface afterward.

`--pose-count` accepts 1, 2, or 3. The demo reads the count from the saved file;
one pose means one view per scan cycle, without multi-view coverage. Re-teaching
overwrites the scan-pose output; use `--output` to keep a separate file and pass
that path to the demo with `--scan-poses`. The script saves after each accepted
pose, so finish the session before using its output.

Each taught pose must remain at least 2 degrees inside every factory joint
limit. Physical teach can move beyond limits accepted by CAN replay; such a
pose is rejected immediately with the violating joint and allowed range.

Teach the exact Starburst pickup height and gripper widths after the candy and
gripper alignment have been physically checked:

```bash
source ~/venv/bin/activate
python prepare-piper/Teach-scripts/teach_candy_grip.py
```

The script first moves to SDK zero and asks for physical teach mode. Place the
open gripper around the centered Starburst at the desired closing height and
orientation, then capture the stable pose. After the physical teach button is
pressed again, the arm holds that exact joint pose under CAN control while you
interactively test open and close commands. Keep fingers clear of the jaws and
keep the emergency stop accessible. Nothing is saved until both widths and the
final result are explicitly accepted.

Accepted values update `demo.grasp_tip_z_mm`, `demo.grip_open_mm`, and
`demo.grip_close_mm` in
`prepare-piper/hand-eye/config.json`. The full measured pose, commanded widths,
and measured widths are written to
`prepare-piper/hand-eye/calibration/candy_grip.json`.
The active demo grasp height is currently overridden to 9.5 mm, 0.5 mm above
the shared 9 mm tool-tip floor; the calibration record retains the original
10.468 mm taught measurement.

Run the supervised loop with all three explicit motion gates:

```bash
python Demo-Product-Day/demo.py \
  --allow-motion \
  --allow-pick \
  --allow-hand-delivery
```

To pick a Starburst and return it to the exact same grasp waypoint without
requiring a palm or contacting the hand service, omit `--allow-hand-delivery`:

```bash
python Demo-Product-Day/demo.py \
  --allow-motion \
  --allow-pick \
  --max-cycles 1
```

The robot continuously visits the saved one to three taught poses. At each pose it captures
Starburst in object slot 1-3 and an open palm in hand slot 4-6. The object and
hand service requests run concurrently after the pose settles, reducing scan
time without changing the one-second settling delay or pose-stability checks.
Demo scan, pickup, recovery, and delivery moves use the configurable
`demo.motion_speed_percent`, currently 20%, except pickup transit, approach,
grasp, and lift motions, which use `demo.pick_motion_speed_percent`, currently
15%. Each pose reports movement, settling, object inference,
hand inference, concurrent inference wall time, and total pose duration; these
values are also saved under `scan_timing_seconds` in both capture records.
Nothing is picked until both targets are valid and cross-view coordinates
agree. Object agreement uses full 3D position and orientation; palm agreement
uses X/Y only because delivery Z is fixed, while the detected palm Z is still
retained for workspace and release-clearance validation. After a verified
pickup, the selected open palm is reacquired before delivery. If it disappears
or fails the hand service's stable-cluster check,
the robot returns the held candy to the exact grasp waypoint, retreats, and
restarts the scan loop. The original-to-reacquired palm displacement is still
recorded, but its 15 mm rejection is temporarily disabled by
`demo.enforce_hand_reacquisition_limit=false`. The fresh palm coordinates
remain subject to workspace, release-clearance, IK, and path validation.

If post-lift gripper feedback indicates that no candy was retained, the robot
reopens and returns to scan pose 1, revisits the initial object scan slot with
the highest valid confidence, captures fresh coordinates, replans, and makes
one final grasp attempt. A second failed width check stops the demo and returns
the robot to rest safely. Any finite measured gripper width strictly above
0.0 mm counts as a retained candy; only zero, negative, or non-finite feedback
triggers this retry. The retry count is controlled by
`demo.maximum_grasp_retries`, currently 1.

For each Starburst detection, OpenCV estimates the wrapper's long axis and
draws it in magenta on the snapshot. `demo.py` transforms that axis into the
Piper base frame and rotates the grasp so the gripper finger direction is
parallel to the candy length and the jaws close across its width. Ambiguous
contours or cross-view orientation disagreement block pickup. Because the
candy's long axis is undirected, planning first tries the aligned yaw nearest
the nominal grasp and then its 180-degree-equivalent yaw. This lets the wrist
avoid J5/J6 limits without changing jaw alignment. If neither exact alignment
is safe, the planner searches yaw offsets in 5-degree steps through
`-20..20` degrees and chooses the smallest offset with a complete safe path.
The bounds are configured by `grasp_yaw_offset_step_deg` and
`maximum_grasp_yaw_offset_deg`.

Pickup planning first tries the configured 100 mm vertical approach. If that
waypoint is joint-limited, the planner searches inward high-transit waypoints
in 20 mm steps up to 120 mm toward the base. At each height, it prefers the
smallest inward offset, starting with a 50 mm pre-grasp and lowering in 5 mm
steps only as far as 30 mm. This preserves the most available clearance before
descending to the unchanged taught grasp pose. The held candy reverses the
same route. Sampled fallback paths must remain inside margined joint limits and
the flange workspace with the tool tip above its safety floor. These values are
configured by `approach_height_mm`, `pregrasp_height_mm`,
`minimum_pregrasp_height_mm`, `pregrasp_height_step_mm`,
`maximum_approach_inward_offset_mm`, and `approach_inward_step_mm`.

When multiple Starbursts are present, every valid box receives independent
depth and orientation processing. Detections from the three views are clustered
into distinct candies in Piper base coordinates. The demo sorts them by planar
distance to the robot base and chooses the closest candy with cross-view
agreement and a joint-limit-safe IK plan. If the nearest candy is unreachable,
the next closest safe candy is considered.

The palm target supplies release X/Y. Release Z remains the physically taught
absolute Piper-base tool-tip height in `demo.hand_release_tip_z_mm`. With the
current 73 mm value, the robot approaches at 93 mm and descends the final
20 mm before release. The hand service only accepts a stable open palm with all
four non-thumb fingers extended. Use `--max-cycles 1` for a bounded supervised
test; the default zero loops until Ctrl+C.

The accepted object workspace is `[100,-400,-50]..[500,400,100]` mm, and the
accepted palm workspace is `[100,-400,-30]..[500,400,30]` mm. These table-level
Z bands cover the expected measurements; every detected location is still
IK-prevalidated against the flange workspace
`[30,-400,40]..[500,400,480]` mm and margined factory joint limits before
motion. The flange minimum X remains 30 mm so folded HOME at approximately
X=46 mm stays inside the shared motion-safety envelope. Keep the hand centered
in the selected camera view, open and
stationary, keep the emergency stop accessible, and remove the hand
immediately if motion is unexpected.

Results are written to:

- `prepare-piper/hand-eye/calibration/demo_runs.json`

Annotated scan images and their metadata remain available in the vision
service UI and API.

## Scripts

- `useful-piper-scripts/find_all_can_port.sh` - lists connected CAN adapters and their USB bus addresses (from AgileX's `pyAgxArm` repo, `pyAgxArm/scripts/ubuntu/`).
- `prepare-piper/can_activate.sh` - brings up a CAN adapter with a given name/bitrate (same source).
- `prepare-piper/check_connection.py` - read-only SDK connectivity/status check.

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
candump can_piper -n 500 | awk '{print $2}' | sort | uniq -c | sort -rn
```

If `0x2A1`-`0x2A8` don't appear, low-level driver frames (`0x251`-`0x256`,
`0x3A0`-`0x3A7`, etc.) do, and `GetCanFps()`/`get_fps()` is roughly half the
expected rate, this is the same issue.

**Not resolved yet.** Suspected firmware/hardware-side condition on this
specific arm rather than a CAN or SDK config problem. Next steps: contact
AgileX support with the diagnostics above, check the teach pendant (if
present) for a fault indicator, and consider a firmware update given
`S-V1.8-3` is an early firmware revision.
