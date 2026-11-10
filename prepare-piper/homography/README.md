# Fixed-Pose Piper/D405 Homography

> **Not the active wrist-camera workflow.** This planar homography requires the
> wrist camera to remain at one fixed scan pose and all calibration markers to
> be visible in the same image. The current D405 installation sees one marker
> at a time and moves with the wrist, so use the
> [eye-in-hand calibration workflow](../hand-eye/README.md) instead.

This toolkit calibrates a planar mapping from RealSense color pixels `(u, v)`
to Piper base coordinates `(X, Y)` in millimetres:

```text
[X, Y, W]^T = H [u, v, 1]^T
X_robot = X / W
Y_robot = Y / W
```

It is intended for a D405 mounted on the Piper wrist. The matrix is valid only
when:

- The camera is at the recorded scan pose.
- The board and markers have not moved.
- The configured camera, resolution, and stream settings are unchanged.
- Target points lie on the calibrated board plane.

Moving the wrist changes the camera-to-board relationship and invalidates this
fixed homography. Arbitrary wrist poses are supported by the active eye-in-hand
calibration.

## Safety

The fixed-pose tools do not autonomously move the robot:

- `capture_markers.py` uses only the RealSense.
- `record_scan_pose.py` reads Piper joint and Cartesian telemetry.
- `collect_robot_points.py` reads Piper Cartesian telemetry.
- `solve_homography.py` and `validate_homography.py` use saved files only.

The operator still enters physical teach mode and must use the measured
two-stage exit sequence: support the arm, press the physical Teach button
again, wait for stiffness and `STOP_RECORDING`, then allow software to take CAN
control when prompted. Keep the emergency stop accessible. Never press the
gripper into the board; lightly touch each center with the same physical tool
point.

## Files

| File | Purpose | Expected output |
|---|---|---|
| `config.json` | Active camera, marker, sampling, and acceptance settings | Input only |
| `capture_markers.py` | Average subpixel ArUco centers over many D405 frames | `calibration/marker_pixels.json`, `marker_capture.jpg` |
| `record_scan_pose.py` | Stream annotated D405 video, guide physical teach-mode alignment, then save the fixed wrist pose | Live FFmpeg MJPEG URL and `calibration/scan_pose.json` |
| `collect_robot_points.py` | Record repeated Piper poses while touching tag centers | `calibration/robot_points.json` |
| `solve_homography.py` | Fit the matrix and enforce fit/planarity limits | `calibration/homography.json` |
| `validate_homography.py` | Measure error at independent holdout points | `calibration/validation_report.json` |
| `test_common.py` | Verify geometry using synthetic known data | An `OK` unit-test result |

Every executable script also contains a top-level docstring stating its exact
purpose, safety behavior, expected output, and an example command.

Generated calibration files are ignored by Git because they are specific to
one physical camera, board placement, scan pose, and robot.

## Before calibration

1. Complete [Jetson setup](../../prepare-jetson/SETUP.md).
2. Verify the Piper connection:

   ```bash
   cd /home/clouduser/SFF-Ignite-Demo
   source ~/venv/bin/activate
   python3 prepare-piper/check_connection.py can_piper
   ```

3. Confirm approximately 3060 CAN FPS and 200 Hz joint/end-pose telemetry.
4. Log out and back in after setup so `video` and `plugdev` groups apply.
5. Fix tags 0, 1, 2, and 3 flat around the picking area:

   ```text
   ID 0 ---------------- ID 1
    |                      |
    |     picking area     |
    |                      |
   ID 3 ---------------- ID 2
   ```

6. Ensure all tag centers and the intended object contact plane are coplanar.
7. Choose one exact point centered between the closed gripper tips as the tool
   contact point. Use that same point for every measurement.

If your IDs or camera differ, edit
[config.json](./config.json). The configured serial currently matches the
connected D405, `126122270802`.

## 1. Position the wrist at the scan pose

Run the interactive alignment script:

```bash
python3 prepare-piper/homography/record_scan_pose.py
```

It starts an FFmpeg-backed MJPEG server and prints a URL such as:

```text
http://<jetson-ip>:8094/
```

Open the URL in a browser. The live image identifies visible tags and changes
its status to `READY` when all required IDs are visible. The server supports
one browser stream at a time.

Then follow the prompts:

1. Hold the arm and press the physical teach button.
2. Wait for the script to confirm `TEACH MODE detected`.
3. Hand-guide the wrist until the board fills most of the image and the
   overlay says `READY`. Avoid a grazing viewing angle.
4. Press Enter in the terminal to accept the alignment.
5. Keep holding the arm and press the physical teach button again.
6. Do not release the arm until holding torque returns and the script confirms
   that teach mode has exited.

FFmpeg performs the JPEG encoding, while the script's HTTP server exposes the
browser page and supports multiple viewers.

The script waits one second for the arm to settle, samples the pose, writes:

```text
prepare-piper/homography/calibration/scan_pose.json
```

and then stops the D405 and FFmpeg server automatically.

Do not move the arm after that. Capture the calibration pixels from the same
pose:

```bash
python3 prepare-piper/homography/capture_markers.py
```

Inspect:

```text
prepare-piper/homography/calibration/marker_pixels.json
prepare-piper/homography/calibration/marker_capture.jpg
```

The marker script requires at least 20 observations per ID and less than one
pixel of center standard deviation by default. Review the annotated JPEG and
confirm each label is at the physical tag center.

## 2. Touch and collect marker coordinates

Use the approved physical procedure to put the arm into teach/drag mode, then
run:

```bash
python3 prepare-piper/homography/collect_robot_points.py
```

For each requested ID:

1. Move the same tool contact point to the exact printed marker center.
2. Keep the tool approximately perpendicular to the board.
3. Hold the arm still.
4. Press Enter.

The script takes 20 samples and accepts the point only when the maximum X/Y/Z
standard deviation is at most 1 mm. It does not change robot mode. Exit
teach/drag mode using the approved procedure after collection.

## 3. Solve the matrix

```bash
python3 prepare-piper/homography/solve_homography.py
```

The default requirements are:

- Fit RMSE at most 8 mm.
- Maximum fit error at most 15 mm.
- Marker contact Z range at most 5 mm.
- Every correspondence must be a RANSAC inlier when more than four markers
  are configured.

The solver refuses to write `homography.json` when a limit fails. The
`--allow-failed-quality` option exists for diagnosis only; such output must
not be used for robot motion.

With exactly four points, a homography can fit the calibration points almost
perfectly even when a measurement is wrong. Independent validation is
therefore mandatory.

## 4. Validate independent points

Copy the template:

```bash
cp prepare-piper/homography/validation-points.example.json \
   prepare-piper/homography/calibration/validation_points.json
```

Replace its example values with at least three board points that were not used
to solve the matrix:

1. Record each point's image pixel while the camera is at the saved scan pose.
2. Touch the same physical point with the calibrated gripper contact point.
3. Record the Piper base X/Y values.

Run:

```bash
python3 prepare-piper/homography/validate_homography.py
```

The defaults require validation RMSE at most 10 mm and maximum error at most
15 mm. Tighten these limits if the intended object or grasp tolerance is
smaller.

## 5. Verify the software without hardware

```bash
cd /home/clouduser/SFF-Ignite-Demo/prepare-piper/homography
~/venv/bin/python -m unittest -v test_common.py
```

This synthetic test proves that the solver recovers a known projective
mapping, rejects collinear inputs, and reports errors in millimetres.

## Recalibrate when

Discard the generated calibration and repeat all steps if:

- The wrist camera mount moves.
- The board or any marker moves.
- The scan pose changes.
- Camera resolution or stream selection changes.
- The D405 is replaced.
- Runtime tag centers no longer align with their calibrated pixels.
