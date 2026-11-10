# Wrist-mounted D405 calibration for the AgileX Piper

This is the active calibration workflow for the wrist-mounted Intel RealSense
D405.

Although this is sometimes called "building the homography," the runtime
mapping is not a single planar homography. The wrist moves, so the correct
model is:

1. A rigid **eye-in-hand transform** from D405 optical coordinates to the
   Piper end frame.
2. The live Piper end pose, which places the camera in the Piper base frame.
3. Aligned D405 depth, corrected with a multi-distance scale/offset model.

At runtime:

```text
base_T_camera = base_T_gripper @ gripper_T_camera
base_point = base_T_camera @ camera_point
```

`gripper_T_camera` is stored as `camera_to_gripper` in `hand_eye.json`.

The older four-marker planar homography is valid only for a completely fixed
camera pose with every marker visible in one image. It is not the correct
runtime model for this wrist camera.

## Generated artifacts

Hardware-specific files are written under `calibration/` and ignored by Git.

| Artifact | Purpose |
|---|---|
| `samples.json` | Multi-pose Piper and ArUco observations |
| `hand_eye.json` | D405-to-Piper rigid transform |
| `validation.json` | Independent hand-eye validation |
| `depth_samples.json` | Multi-distance raw/expected D405 depths |
| `depth_correction.json` | D405 depth scale and offset |
| `depth_verification.json` | Corrected depth-coordinate validation |
| `click_coordinates.json` | Arbitrary clicked pixel-to-base measurements |

Annotated JPEGs are saved beside these files.

## Safety and teach-mode lifecycle

Keep the emergency stop accessible and the workspace clear. Hold the arm
whenever entering or leaving physical teach mode.

This Piper firmware requires a two-stage teach exit:

1. While supporting the arm, press the physical Teach button a second time.
2. Wait for the arm to become stiff. Telemetry changes to
   `STOP_RECORDING`, although control mode may still report teach mode.
3. Press Enter when the script asks to take CAN control back.
4. The software snapshots the six taught joints before changing modes and
   streams that exact pose immediately, preventing a stale zero target.

For continuous-hold tools such as corrected depth verification and
click-to-coordinate:

1. The same Piper connection remains open through measurement.
2. The taught pose is streamed continuously.
3. At shutdown, keep supporting the arm and turn controller power **off**.
4. Press Enter only after power is off.
5. Restart the controller after the script has disconnected.

Do not start another CAN client while a continuous-hold tool is running.

## Hardware setup

- D405 serial: `126122270802`
- D405 stream: 640x480 at 30 FPS
- CAN interface: `can_piper`, 1 Mbps
- Marker dictionary: `DICT_4X4_50`
- Calibration marker: ID 1
- Black ArUco square size: exactly 101.6 mm (4 inches)

Keep Marker ID 1 flat, stationary, and well lit throughout hand-eye and depth
calibration. Do not move the D405 mount afterward.

## 1. Activate the environment and verify hardware

Use Bash because `source` is not available in `/bin/sh`:

```bash
bash
source ~/venv/bin/activate
cd ~/SFF-Ignite-Demo
python prepare-piper/check_connection.py can_piper
```

Healthy telemetry is approximately:

- CAN FPS: 3040-3060
- Arm status: `NORMAL`
- Joint/status telemetry: 200 Hz
- No communication or angle-limit errors

Resolve hardware or CAN errors before calibration.

## 2. Collect eye-in-hand samples

```bash
python prepare-piper/hand-eye-realsense405/capture_samples.py
```

Procedure:

1. Open the browser URL printed by the script.
2. Support the arm and press the physical Teach button.
3. Keep Marker ID 1 fully visible.
4. Capture at least 15 stable poses; 20 is preferred.
5. Vary camera position, distance, tilt, and wrist roll.
6. Include at least 100 mm translation span, at least 25 degrees rotation, and
   rotation around at least two independent axes.
7. Enter `q` after enough accepted samples.
8. Follow the two-stage teach-exit prompts exactly.

Each accepted observation is saved immediately in `samples.json`, so completed
samples survive an interruption. Running the collector again starts a new
sample set when its first observation is accepted.

## 3. Solve the eye-in-hand transform

This step is offline and does not connect to the robot:

```bash
python prepare-piper/hand-eye-realsense405/solve_hand_eye.py
```

The solver runs Tsai, Park, Horaud, Andreff, and Daniilidis methods. It selects
the transform that makes the fixed marker most consistent in Piper base
coordinates.

Expected output:

```text
Selected <METHOD>: translation RMSE ...mm, rotation RMSE ...deg
Hand-eye calibration written to .../calibration/hand_eye.json
```

Do not use `--allow-failed-quality` output for robot targeting.

## 4. Validate with fresh wrist poses

Do not move Marker ID 1 or the D405 mount. Use at least five poses that were
not part of calibration:

```bash
python prepare-piper/hand-eye-realsense405/validate_hand_eye.py
```

Vary distance, tilt, position, and wrist roll. After the final pose, follow the
two-stage teach-exit prompts.

The script uses the solved transform without recalculating it and checks that:

```text
base_T_target =
    base_T_gripper @ gripper_T_camera @ camera_T_target
```

remains stationary. Require `quality_passed: true` in `validation.json`.

## 5. Calibrate D405 depth scale and offset

Hand-eye calibration uses ArUco geometry; arbitrary objects require metric D405
depth. Collect the fixed marker at multiple distances:

```bash
python prepare-piper/hand-eye-realsense405/calibrate_depth.py
```

Procedure:

1. Enter physical teach mode when prompted.
2. Capture at least seven stable poses; more are encouraged.
3. Include near, middle, and far views.
4. Continue until the printed expected-depth span exceeds 100 mm.
5. Keep the full marker visible and avoid grazing angles.
6. Enter `q`.
7. Follow the two-stage teach-exit prompts.

The fitted model is:

```text
corrected_z_mm = scale * raw_z_mm + offset_mm
```

Acceptance includes fit RMSE and leave-one-pose-out cross-validation. Require
`quality_passed: true` in `depth_correction.json`.

The currently measured correction is:

```text
scale = 1.13525633
offset = -20.554 mm
cross-validation RMSE = 4.73 mm
```

Recalibrate rather than copying these values to another camera or mount.

## 6. Verify corrected depth coordinates

Use the continuous hold workflow:

```bash
python prepare-piper/hand-eye-realsense405/verify_depth_coordinates.py --align-in-teach
```

Procedure:

1. Align Marker ID 1 in the browser preview.
2. Complete the two-stage teach exit.
3. Do not touch the arm while 30 observations are collected.
4. After the report is written, keep supporting the arm.
5. Turn controller power off when prompted.
6. Press Enter after power is off.
7. Restart the controller only after the script exits.

The verifier automatically applies `depth_correction.json` and uses temporal
consensus rather than trusting one depth frame.

Acceptance requires:

- Consensus base-coordinate error at most 10 mm
- Temporal depth MAD at most 10 mm
- At least 80% temporal depth inliers
- Robot position deviation at most 1 mm
- Robot rotation deviation at most 0.5 degrees

Single-frame depth spikes remain in the report for diagnosis but are excluded
from the runtime target through temporal consensus.

The accepted hardware run produced:

```text
28/30 temporal inliers
depth MAD = 7.85 mm
consensus base-coordinate error = 4.95 mm
```

## 7. Test arbitrary pixels in the browser

This diagnostic exercises the exact coordinate pipeline that object detection
will use:

```bash
python prepare-piper/hand-eye-realsense405/click_to_coordinate.py
```

Procedure:

1. Enter physical teach mode and position the wrist so the test objects are
   visible.
2. Complete the two-stage teach exit.
3. Wait for browser status `ready`.
4. Click the middle of a stable surface.
5. Avoid silhouettes, holes, reflections, and depth discontinuities.
6. Review raw/corrected depth, camera XYZ, Piper base XYZ, inlier count, and
   depth MAD.
7. Repeat as needed.
8. Select **Finish session**.
9. Keep supporting the arm, turn controller power off, and press Enter.

Outputs:

- `click_coordinates.json`
- `click_coordinate_NNN.jpg`

The accepted test session recorded 18 successful clicks. Every measurement met
the configured depth-quality and robot-stability limits.

## 8. Run software tests

```bash
cd ~/SFF-Ignite-Demo/prepare-piper/hand-eye-realsense405
python -m unittest -v \
  test_hand_eye.py \
  test_depth_calibration.py \
  test_depth_coordinates.py \
  test_click_to_coordinate.py
```

These tools use the Piper control package bundled in `Demo-Ignite/piper`.

## 9. Run the text-prompted pick/drop demo

The k3s [OpenCV service](../../k3s-OpenCV-service/README.md) owns the D405 and
returns corrected camera-frame coordinates to [`demo.py`](../../Demo-Product-Day/demo.py).
Teach three scan poses with [`teach_scan_poses.py`](../../Demo-Product-Day/Teach-scripts/teach_scan_poses.py),
then follow the scan-only and supervised one-pick sequence in the project
[setup instructions](../../prepare-jetson/README.md). Do not begin with an unlimited motion loop.

## Recalibrate when

Repeat hand-eye and depth calibration if:

- The D405 mount moves or flexes.
- The D405 is replaced.
- Camera resolution, alignment, or stream settings change.
- The Piper end-frame convention changes.
- The fixed calibration marker moves before validation finishes.
- Independent validation exceeds its limits.
- Clicked coordinates become inconsistent with physical surfaces.

Depth-only recalibration is sufficient when the rigid camera mount is
unchanged but the D405 depth bias changes.

## Troubleshooting

### Browser URL times out

Confirm TCP port 8094 is permitted from the LAN as documented in
`prepare-jetson/SETUP.md`.

### Marker is not visible after teach exit

Do not disconnect and reconnect between alignment and measurement. Use the
continuous `--align-in-teach` workflow.

### Arm moves toward zero when CAN control returns

Stop immediately and support the arm. Current scripts snapshot the taught
joints before mode transition and stream that pose on the first CAN frame. Do
not use an older copy of the scripts.

### Controller reasserts teach mode or disables varying joints

Support the arm and power-cycle safely. For continuous-hold tools, power off
only when the terminal asks, press Enter after power is off, and restart after
the script disconnects.

### Depth has occasional large spikes

Never use one depth frame for targeting. Use the configured 30-frame temporal
median, MAD-based outlier rejection, and minimum inlier fraction.
