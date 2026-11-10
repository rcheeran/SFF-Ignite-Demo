# Wrist-mounted RealSense D435: fresh hand-eye calibration

This workflow is for D435 serial `263022078260` on the Piper wrist and the
new robot mounting surface. Do not copy the D405 calibration results.

The [configuration](config.json) uses the confirmed `DICT_4X4_50`, ID `1`
marker with a **101.6 mm black square** (excluding the white margin).
Keep it flat and stationary on the new surface during capture, validation,
and depth calibration. Keep the camera mount rigid.

## Separate environment and results

The existing environment has overlapping OpenCV installations and lacks
`calibrateHandEye`. Use an isolated environment with one pinned OpenCV build:

```bash
cd ~/productdaydemo/prepare-piper/hand-eye-realsense435
python3 -m venv ~/venv-hand-eye435
~/venv-hand-eye435/bin/python -m pip install -r requirements.txt
source ~/venv-hand-eye435/bin/activate
python calibrate.py check
```

`check` enumerates the camera and verifies imports, without opening a stream,
connecting to CAN, enabling motors, or moving the robot.

[calibrate.py](calibrate.py) reuses the camera-configurable scripts under
[hand-eye-realsense405](../hand-eye-realsense405/) without duplicating them.
It passes this directory's configuration and every input/output path
explicitly. All new JSON files and images stay under
[calibration](calibration/); the old D405 results are not used or overwritten.
Some messages in the shared scripts still say D405; the selected device is
the serial in this directory's configuration.
Later stages refuse to start unless their prerequisite calibration files
have passed quality checks.

## Before any physical calibration

- Keep the emergency stop accessible and the workspace clear.
- Stop any camera service/preview that owns this D435. Port `8094` must also
  be free before starting the calibration preview.
- Check the robot using
  `python ../check_connection.py can_piper`; resolve faults or missing
  telemetry before continuing.
- Support the arm whenever entering or leaving physical teach mode.
- Apart from `check` and offline `solve`, stages use teach/CAN hold control:
  they can enable motors and apply holding torque. They are not read-only.
- Do not run another CAN client while a calibration/hold tool is active.

## 1. Capture varied wrist poses

```bash
python calibrate.py capture
```

Open the printed preview URL. Follow the prompts for physical teach mode.
Capture at least **15 stable poses**, preferably 20, while keeping the full
marker visible. Include at least **100 mm translation span**, **25 degrees
rotation**, and rotations about **two independent axes**.

Vary position, distance, tilt, and wrist roll without moving the marker.
For depth stages, the initial accepted working band is **300-1000 mm**;
do not assume the D435 can use the D405's previous close-range poses.
Enter `q` when finished.

For teach exit, support the arm, press the Teach button a second time, wait
for stiffness / `STOP_RECORDING`, then follow the prompt to return to CAN
control while holding the measured current pose.

## 2. Solve offline

```bash
python calibrate.py solve
```

Require `acceptance.quality_passed: true` in `calibration/hand_eye.json`.
No robot or camera connection is made by this stage.

## 3. Independently validate

```bash
python calibrate.py validate
```

Use at least **five fresh poses** not used for fitting. Do not move the
marker or camera mount. Require `acceptance.quality_passed: true` in
`calibration/validation.json`.

## 4. Calibrate D435 depth

```bash
python calibrate.py depth
```

Collect at least **seven stable near/middle/far poses** with more than
**100 mm expected-depth span**. Keep the full marker visible. Require
`acceptance.quality_passed: true` in `calibration/depth_correction.json`.
Do not reuse the D405 scale or offset.

## 5. Verify corrected coordinates

```bash
python calibrate.py verify
```

Follow the teach/alignment and continuous-hold prompts. Require
`acceptance.quality_passed: true` in `calibration/depth_verification.json`.
After measurement, support the arm, turn controller power **off** when
prompted, and press Enter only after power is off. Restart the controller
after the script exits.

Optional browser click-to-coordinate check, after restarting the controller:

```bash
python calibrate.py click
```

It uses the same continuous-hold shutdown procedure.

## Before enabling the demo on the new surface

This folder deliberately has **no copied demo motion configuration**.
Calibration does not make the old motion settings safe on the new surface.

Recheck and reteach scan poses, pickup height, release height, workspace
bounds, tool-tip floor, and collision clearances. The teaching tools are under
[Teach-scripts](../Teach-scripts/). Review their configuration paths before
running them: the candy-grip defaults now target this D435 directory, while the
release-height tool still targets the older configuration.

For one scan pose, from the repository root in the isolated environment:

```bash
python prepare-piper/Teach-scripts/teach_scan_poses.py --pose-count 1
```

`--pose-count` supports 1–3 poses and defaults to 3. The demo accepts the
matching count in the saved `prepare-piper/scan-poses/scan_poses.json` file.
One pose sacrifices multi-view coverage; confirm the whole intended pickup
area is visible and within the D435 working range. No marker is needed for
scan-pose teaching.

This script **commands motion to SDK zero before teach mode**, then replays
the taught poses through zero after confirmation. Clear every path and keep
the emergency stop accessible. It preserves CAN hold until controller power
is confirmed off. It does not need the camera; OpenCV may stay stopped, or
serve a preview only after calibration has exited and released the camera.
No other calibration/CAN hold process may run alongside teaching.

Scan-pose teaching imports the shared hold helper from the renamed D405
source folder; it does not read D405 calibration results.

For candy-grip teaching:

```bash
python prepare-piper/Teach-scripts/teach_candy_grip.py
```

The script defaults to this directory's [config.json](config.json) and writes
the confirmed record to `calibration/candy_grip.json`. Saving does not modify
calibration configuration or runtime settings. Review and manually adopt the
taught height and jaw commands into `DEMO_SETTINGS` in
[Ignite's entry point](../../Demo-Ignite/demo.py).
Existing candy-grip records supply the initial jaw prompt values.
With no saved jaw settings, hardware open/closed limits are offered as initial
interactive prompt values, not assumed taught measurements.
It commands SDK zero before teaching; clear the path and keep fingers away
from the jaws during width tuning. Confirm saving and complete the continuous
hold/power-off procedure before starting any other CAN client.

The release-height tool still defaults to D405 configuration/output paths.
Ignite does not need its absolute height: release is palm Z + 76.2 mm.

## Ignite workspace, delivery orientation, and clearance teaching

From the repository root in the isolated environment:

```bash
python prepare-piper/Teach-scripts/teach_workspaces.py
```

[teach_workspaces.py](../Teach-scripts/teach_workspaces.py) defaults to this
directory's [config.json](config.json). It reads CAN feedback **without enabling
motors or commanding motion, gripper changes, teach transitions, or holds**.
It does not move to zero or replay measurements. Stop all other arm controllers;
keep emergency stop accessible and support the arm before using physical Teach.
Read-only does **not** prevent the arm from dropping in teach mode.

1. Follow the prompt to enter physical teach recording.
2. Hand-guide the gripper **tool tip** to candy-center workspace corners at both
   minimum and maximum permitted heights. Enter captures a stable corner;
   `u` removes the last corner and `q` finishes. Prefer all eight corners.
3. Repeat for a separate, empty marked **palm-center** region, with no person
   underneath. These coordinates are not release-tip positions.
4. Over a dummy palm position, teach the downward gripper orientation at the
   intended release location. Ignite separately adds **76.2 mm in base Z** to
   each detected palm center; teaching orientation does not change this offset.
5. Keep supporting the arm, stop teaching with the physical button, turn
   controller power off, and type `OFF`. The script never takes over holding.
   Ctrl+C during measurement still requires this manual power-off confirmation.
6. With power off, enter physically reviewed pickup-approach and pre-grasp
   offsets **above the grasp tip**, and hand-approach offset **above the release
   point**. No new clearance values are guessed. Pre-grasp must be below pickup
   approach. Review these distances in the empty setup before accepting them.
7. Review the printed settings and explicitly confirm saving.

The operator chose to reuse Product Day's planning clearances in Ignite's
`DEMO_SETTINGS`: pickup approach **100 mm** above grasp, pre-grasp **50 mm**
above grasp, and hand approach **20 mm** above release. These are reused
parameters, not newly measured or collision-validated clearances.

The operator also approved reusing Product Day's remaining required planner,
timing, and perception parameters: 30 mm minimum pre-grasp with 5 mm steps;
120 mm inward search with 20 mm steps; 20-degree yaw search with 5-degree steps;
2 mm / 0.75-degree scan tolerance; 25 mm / 20-degree cross-view agreement;
15 mm palm reacquisition tolerance; 0.05 detection confidence; 8-second move
timeout; and 1-second scan settling, gripper hold, and retry delay. The taught
workspaces, orientations, grip values, and 10% speeds remain unchanged.
These reused parameters do not establish physical path safety.

Saving writes only `calibration/workspace_teaching.json`, containing measured
workspace bounds, delivery RPY, clearances, samples, and provenance. Calibration
configuration, runtime settings, scan poses, and candy-grip records are unchanged.
Existing `DEMO_SETTINGS` supply clearance prompt defaults and pre-grasp search
constraints. Review the saved measurements and manually update the workspace
variables or `DEMO_SETTINGS` in [Ignite's entry point](../../Demo-Ignite/demo.py)
before using newly taught values. Record writes are atomic.

The measured envelope is an axis-aligned box, **not** a reachability or
collision-free certificate. Capturing two opposite corners alone does not
validate the remaining corners, interior, paths, or clearances. Review those
physically; the demo still checks geometry and can reject targets. Additional
planner search limits, timings, and perception thresholds must also be reviewed
before operation; the approved reused values above provide the current settings.

## Compare a candy-surface point against camera coordinates

```bash
~/venv-hand-eye435/bin/python prepare-piper/Teach-scripts/verify_candy_point.py
```

[verify_candy_point.py](../Teach-scripts/verify_candy_point.py) uses feedback-only
CAN and the running calibrated camera service. It never enables motors, moves
the arm, opens/closes the gripper, or sends holding commands. Stop other arm
controllers, keep supporting the arm in physical teach mode, and use an empty
marked stationary surface at candy-top height, with nobody underneath.

Hand-guide the tool tip to the exact marked point and record it without
compressing the surface. Then hand-guide to a viewing pose with the point
unobstructed and at least 300 mm from the D435. A known, independently surveyed
base XYZ can instead be supplied with `--reference-point-mm X Y Z`; do not use
camera coordinates as their own reference. Never lower the tip below the safety
floor to collect a reference; use an independently surveyed point if necessary.

The script freezes a token-bound PNG/depth frame and the wrist transform at
capture. Follow the manual support/teach-stop/controller-power-off prompt
**before** opening the generated local `candy_point_verification.html` in a
browser. Click the exact mark and copy the displayed `u v` to the terminal.
No web server or external image upload is involved. Select a flat patch away
from edges: the camera's pixel deprojection samples a 17-by-17 depth window,
which may mix candy and table depths on a small candy. This pixel estimator
differs from YOLO box localization, so verify on a sufficiently large marked
surface and do not treat it as proof of each detection's accuracy.

The report shows reference XYZ, measured camera-derived base XYZ, signed
camera-minus-reference XYZ, and 3D error. The default comparison threshold is
10 mm (`--maximum-error-mm` can override it); it is not a certified safety
tolerance. Confirm saving to `calibration/candy_point_verification.json`.
Failed comparisons can be saved and return a failing exit status. An expired
camera token produces an explicit error; rerun rather than substituting a live
frame. Repeat at several known points across the candy region before deciding
on detection Z bounds. No demo configuration, workspace bounds, or accepted
calibration artifacts are automatically changed.

Only after all calibration quality checks and physical checks pass should the
demo and vision service be pointed at the D435 serial, new transform, new
depth correction, and newly verified motion settings. This workflow does not
automatically switch or start the demo.

## Hardware-free validation

```bash
python -m unittest -q test_calibrate
cd ../hand-eye-realsense405
python -m unittest discover -q
```
