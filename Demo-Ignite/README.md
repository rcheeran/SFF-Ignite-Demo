# Ignite: Starburst-to-palm loop

[demo.py](demo.py) is a supervised entry point with its own perception
orchestration, grasp planning, calibration loading, and motor-health checks.
It does not import code from either older demo folder. It defaults to D435
calibration, not D405.

## Flow

1. Print independent checks for the OpenCV camera, YOLO/Foundry CUDA model,
   and MediaPipe hand service. Require calibrated depth from the configured
   camera; preview-only mode is rejected.
2. Visit the taught scan pose(s). The application sends the token-bound PNG to
   the detection client for Starbursts; the hand service independently captures
   synchronized RGB-D for open palms. Reject movement during inference.
3. Pick an orientation-valid, reachable candy using measured XY and the
   reviewed/taught grasp-tip Z. After lift, the measured jaw width must be at
   least `minimum_held_grip_mm` (**8 mm**; an empty gripper closes to ~5 mm and a
   held Starburst measured 22.4 mm). Below that, the demo assumes no candy,
   prints `No candy in gripper: ...`, opens the gripper where it is (above the
   table, so a loosely held candy may drop), returns to the scan pose, and
   continues the loop to find another candy.
   Pickup XY comes from deprojecting the midpoint of the detected long-axis
   image endpoints using corrected box-filtered depth, then transforming that
   point into robot-base coordinates. It no longer uses the depth-cluster pixel
   median as the pickup location. Missing/invalid endpoints reject the candidate.
4. Return to the scan pose holding the candy and look for a new open palm. If
   the palm is missing, not an open palm, nonfinite, outside the workspace,
   stale, unreachable (for example, no IK solution for the hand approach), or its
   delivery path is rejected, the arm **waits at the scan pose holding the candy**,
   prints `Waiting at scan pose with candy for a reachable open palm (attempt N):
   <reason>`, and captures again every `retry_delay_seconds` (1 s) with no motion.
   Hand-service request failures are retried the same way. There is no time
   limit; arm health is checked before every attempt, and a fault or Ctrl+C stops
   waiting. As soon as a fresh palm has an accepted delivery plan, it prints
   `Palm detected after pickup` and continues. The demo does not check
   handedness or stability metadata, compare earlier palm coordinates, or
   capture again during approach.
5. Approach and descend using that accepted detection. Release the **gripper
   tool tip at palm center Z + 76.2 mm (3 inches)** in the robot base frame.
   This is not an absolute table height or distance along the camera axis.
   If any other demo error occurs while the candy may be gripped, the candy is
   returned (see below) and the loop continues.
6. Retreat, return to scan pose, and repeat. `--max-cycles 0` loops until Ctrl+C;
   a cycle with no accepted candy/palm pair waits and scans again.

### Returning a held candy after an error

From the moment the gripper closes at the grasp pose until it starts opening
at the palm, the demo records every waypoint the arm reaches. If a demo error
occurs in that window (for example during lift, return to scan, hand approach,
or descent, or a palm that goes stale after the approach starts), it prints
`ERROR while holding candy: ...`, then **retraces those waypoints in reverse**
to the grasp pose, opens the gripper there, retreats via the pickup
lift/transit waypoints to the scan pose, and continues with the next cycle. A
missing or unreachable palm before the approach does not return the candy; the
arm waits at the scan pose instead (step 4). Retracing only re-crosses segments
already traversed and path-checked; the whole return route is sampled from the
measured joints before the first move, each segment is rechecked before it is
commanded, and waypoints already within 2 degrees are skipped. The gripper
stays closed until the pickup location and never opens near the hand during
recovery. Return moves use the pickup speed.

The loop still **stops** (with the usual zero return and disconnect) for
Ctrl+C, for errors when nothing is gripped (before the gripper closes, after a
missed grasp, or once the gripper has started opening at the palm), and if the
return itself fails (reported as `candy return failed after ...`). An unhealthy
arm fails the return's health checks and stops. Recovery motion can follow a
failed move, including one caused by contact; keep the emergency stop accessible.
An arm fault while waiting for a palm triggers this return, which then fails its
own health check and stops the demo with the candy still gripped.

This is not continuous human tracking or certified collision avoidance.
Every candy detected in a scan is considered, nearest to the base first. A
candy whose IK, grasp-orientation, or full pickup-route checks fail is skipped
(`Skipping unreachable candy at ...: <first reason>`) and the next one is
tried. If none can be planned, the demo prints `No reachable candy in this
scan; scanning again` with one line per candy and keeps looping instead of
stopping; `--verbose` adds the full per-orientation details. Passing the
detection workspace check does not guarantee reachability.
The person must keep the palm stationary through approach/release, and the
operator must supervise with the emergency stop accessible. There is **no
approach-time hand recheck**: movement or loss of the palm after the post-pickup
capture is not detected. The maximum palm age is fixed at 10 seconds (not a CLI
option), checked before pickup, in the delivery plan, and again before approach,
descent, and opening. This age check does not verify that the palm is still there.
The hand service's own open-palm detection and temporal filtering remain unchanged;
service-level rejected observations are not overridden by the demo.

Midpoint pickup requires the updated OpenCV camera service. Ignite explicitly
requests `pickup_point: long_axis_midpoint` and refuses detected results that do
not confirm that mode; deploy [the camera service](../k3s-OpenCV-service/README.md)
before running the updated demo. The snapshot cross marks the selected midpoint.
Depth still comes from the dominant cluster inside the detection box; this
change does not eliminate depth errors or correct camera-to-robot calibration.

All commanded joint segments are sampled for joint limits, flange workspace,
and tool-tip floor before motion. Planned waypoints stay 2 degrees inside the
factory joint limits. A segment that starts from the arm's **measured** joints
tolerates settling up to the 2-degree arrival tolerance past that margin (for
example J5 reading 68.3 degrees after a waypoint planned at 68.0); the reading
is clamped onto the margin for sampling, so it never exceeds the factory limit.
Larger excursions are rejected. Joint-limit errors name the joint, its angle,
and the limit, for example `crosses a joint limit at the segment start:
J5=68.30deg outside [-68.0, 68.0]deg`. These geometric checks do not model
self-collision, the table, people, or other obstacles; review every path and
clearance physically. With nothing gripped, missing hands, unsafe paths,
planning errors, and service failures stop the loop without automatic release;
with a candy gripped, the candy is first returned as described above.
**Every exit after connection (normal completion, error, or Ctrl+C) attempts a
slow return to SDK zero (all six joints at zero), then opens the gripper there
(`grip_open_mm`, 60 mm) and disconnects.** The startup zero move also opens the
gripper at zero before moving to the first scan pose. The gripper does not
change during the move itself; it opens only after zero is reached and
verified, so a candy still gripped at shutdown drops from the zero-pose tool
tip (about `[176, 0, 224]` mm, roughly 22 cm above the table in front of the
base), not over a hand. The measured width must come within 5 mm of the open
command, otherwise the startup stops or the shutdown reports the error.
Clear and review that return path too. The return checks
arm health, sampled flange workspace, tool-tip floor, and factory joint limits
(zero is on the J2/J3 boundaries, outside the normal inward-margined limits).
If those checks fail, the return times out, or Ctrl+C interrupts the return,
the script reports it and disconnects without retrying or commanding further motion.
There is no hold loop or power-off confirmation in the main demo. Disconnecting
closes CAN communication; it does not disable motors or switch off the controller.
Remain ready to support the arm and use emergency stop if the return fails.

## Required setup

### Jetson storage

The repository is stored at `/mnt/k3s-data/productdaydemo` on the encrypted
data filesystem and persistently bind-mounted at `/home/clouduser/productdaydemo`.
Existing commands and Kubernetes calibration host paths therefore stay unchanged.
The `home-clouduser-productdaydemo.mount` unit is required by k3s at startup,
preventing fallback to an empty root-filesystem directory if the mount fails.

The separate Docker containerd store is at `/mnt/k3s-data/docker-containerd`,
bind-mounted at `/var/lib/containerd` by `var-lib-containerd.mount`, which
`containerd.service` requires. This is distinct from k3s's embedded containerd:
k3s images and volumes retain their existing data-filesystem mounts, and its
core state retains the separate encrypted infrastructure filesystem.

Original migration backups are retained at
`/mnt/k3s-data/productdaydemo-original-backup` and
`/mnt/k3s-data/docker-containerd-original-backup`; they are not active stores.
Persistent mounts and service ordering were validated without rebooting.

The implementation does **not** deploy services, enable calibrated camera mode,
copy D405 motion settings, or start the robot.

- Accepted D435 `hand_eye.json`, `validation.json`, `depth_correction.json`,
  and `depth_verification.json`.
- Reviewed `DEMO_SETTINGS` in [demo.py](demo.py), validated by
  `validate_demo_settings`. Runtime settings are separate from the D435 calibration configuration.
  Shared candy/palm workspace limits come from `WORKSPACE_CORNERS_MM`
  in [demo.py](demo.py), not configuration keys.
  C2/C4/C6/C8 define an axis-aligned XY rectangle: X **144-500 mm**, Y
  **-170 to 206.317 mm** (all workspace constants are rounded to 3 decimals).
  X was extended from 440 mm (the measured C6 corner) to 500 mm; confirm
  the table surface and clearances physically out to X = 500 mm. Offline
  pickup planning succeeds across the extended area at the taught scan pose,
  except some candy angles near X 500 / Y 0, which the planner rejects with
  reasons. Candy detection acceptance checks XY against the workspace; detected
  Z is not compared with workspace Z limits but must lie in the plausible
  table-height window `candy_min_z_mm`..`candy_max_z_mm` (**-30 to +30 mm**;
  the table plane is ~-3 mm and real candies measured -12 to +7 mm), which
  rejects bad depth readings such as a YOLO detection localized at Z -348 mm.
  `candy_xy_correction_mm` (default **[0, 0]**) is added to each candy's base
  XY before the workspace check and planning; see
  [Measuring the candy XY correction](#measuring-the-candy-xy-correction).
  Pickup uses the taught `grasp_tip_z_mm` (**16.584 mm**, re-taught 2026-11-04
  with `teach_candy_grip.py`; jaws open 60 mm, closed on a candy at 20.9 mm),
  not detected Z. The previous record is kept as
  `calibration/candy_grip.backup-20261102.json` (18.012 mm).
  Grasp roll/pitch are set to **180 / +20 degrees** (20 degrees from vertical),
  not the hand-held teaching tilt (~9 degrees), with candy-aligned yaw and
  bounded yaw alternatives. The lean follows the candy's length, so reach
  depends on candy angle; across 12 candy angles at X 250-500 mm, offline
  planning reaches about **98%** with 20 degrees versus 74% at 5 degrees, 54%
  with the ~9 degree teaching tilt, and 50% vertical (vertical fails beyond
  X ~400 mm). Pickup approach clearance stays **100 mm**. Review the tool
  tilt and clearances physically before motion.
  Exception: when the 20 degree grasp's fingertips would point forward (within
  ~45 degrees of +X, `forward_lean_vertical_threshold` **0.7**; candies lying
  along X), the planner grasps **vertically** instead if that is reachable and
  keeps the tilted grasp otherwise. See
  [2026-11-05 offset readings](#2026-11-05-offset-readings-lean-direction-not-camera-calibration).
  Palms use the same XY rectangle plus `WORKSPACE_MIN_Z_MM`/`WORKSPACE_MAX_Z_MM`:
  **17.423-263.323 mm**. The corner Z=18 mm does not collapse this height range.
  These Z limits also remain for the visual workspace overlay and
  standalone motion replay; they do not reject candy detections.
  Obsolete `target_workspace_min_mm`/`max_mm` configuration
  entries have been removed; internal validation bounds are derived from these variables.
  Separate palm bounds have been removed from `DEMO_SETTINGS`; grasp-tip height
  remains there. Palm XYZ bounds,
  robot joint/flange limits, and tool-tip floor checks remain enforced.
  Ignoring detected candy Z does not correct camera localization errors in XY or Z.
  Review the runtime settings against verified new-surface grasp height, gripper widths, workspaces,
  speeds, approach heights, settling limits, and other shared planning settings.
  Do not simply copy old mounting-surface values.
  The read-only [workspace teaching tool](../prepare-piper/Teach-scripts/teach_workspaces.py)
  records candy/palm-center bounds, downward delivery orientation, and reviewed
  approach clearances without overwriting grip or scan teaching. Follow its
  [physical-support and power-off procedure](../prepare-piper/hand-eye-realsense435/README.md#ignite-workspace-delivery-orientation-and-clearance-teaching);
  it does not hold or move the arm. Teaching saves measurement records only;
  review and manually adopt their values into the workspace variables or `DEMO_SETTINGS`.
- Reviewed Piper tool length, joint limits, flange bounds, and tool-tip floor.
- One to three saved scan poses from
  [the teaching script](../prepare-piper/Teach-scripts/teach_scan_poses.py).
- Camera on 8094 with `CAMERA_PREVIEW_ONLY=false`, accepted D435 depth correction,
  and health-reported serial matching `hand_eye.json`. Startup checks calibration
  acceptance and matching depth-verification serial. Ignite reads accepted calibration
  artifacts, not the calibration tools' configuration file.
- [Detection](../k3s-vision-detection/README.md) on 8096 with a loaded CUDA candy
  model; [hand service](../k3s-hand-service/README.md) on 8095.
- No other calibration/teaching/CAN process running.

Legacy `hand_release_tip_z_mm`, `drop_approach_height_mm`,
`maximum_grasp_retries`, `enforce_hand_reacquisition_limit`, and
`maximum_hand_reacquisition_difference_mm` settings are not
required or used. Release height is always relative to the newly measured palm;
a missed grasp returns to scanning rather than using the older demo's retry workflow.

## Preflight without CAN or motion

From the repository root in the existing isolated environment:

```bash
source ~/venv-hand-eye435/bin/activate
python Demo-Ignite/demo.py --check-only
```

Services are checked independently and printed before motion setup is loaded,
so missing detection or preview-only camera mode is explicit. Check-only also
requires accepted calibration and reviewed motion configuration; it is not a
success-shaped bypass of missing prerequisites.

## Supervised execution, only after preflight and physical review

For the full pickup/delivery cycle:

```bash
python Demo-Ignite/demo.py \
  --allow-motion --allow-pick --allow-hand-delivery --setup-reviewed \
  --max-cycles 1
```

The four flags explicitly permit motor enablement, pickup, hand delivery, and
confirm physical setup review. A five-second countdown precedes connection.
`connect_robot()` takes exclusive CAN control and enables the arm without
stopping camera services (`connect_for_teach` remains a compatibility alias for
existing teaching tools). After connection, the demo first executes a checked
**SDK-zero move**, verifies arrival, and moves to the first scan
pose without gripper commands. Review both startup paths; zero is a joint pose,
not tool-tip XYZ `(0,0,0)`. Zero-adjacent segments use factory joint limits;
subsequent normal motion retains inward-margined limits. Failure stops startup
and enters the same checked zero-return/disconnect shutdown.
Omit `--max-cycles 1` only when ready for the continuous supervised loop.
URL overrides: `--vision-url`, `--detection-url`, `--hand-url`.
File overrides: `--calibration-dir`, `--scan-poses`. Runtime settings are edited in
[demo.py](demo.py); the obsolete `--config` option has been removed.

Add `--verbose` to the execution command to trace each localized candy and palm:
camera XYZ -> gripper-frame XYZ -> robot-base XYZ. Output includes the saved
camera-to-gripper matrix and the capture-time gripper-to-base matrix, with XYZ
and matrix translations in millimetres. Printed values are rounded to three
decimal places; calculations retain full precision. Verbose mode does not
change motion permissions or planning. `--check-only --verbose` still performs
only preflight; it does not capture detections or print point transformations.
Joint-limit summaries, enable-time joint/twitch diagnostics, connection status,
zero-motion announcements, and the initial scan-pose motion announcement are
also verbose-only in the main demo. Errors,
safety warnings, joint-limit restoration warnings, and zero-arrival confirmation
remain visible without `--verbose`. Standalone teaching/replay tools retain their
existing diagnostic output.

Verbose mode also prints two pickup diagnostics. `PICK PLAN:` names the saved
camera frame (`frame=images/<UTC timestamp>_....jpg` under
`k3s-OpenCV-service/datasets/collected-starburst/`), capture time, detector
score and size, candy axis yaw, the jaw angle to the candy axis (90 = across),
tool tilt, and the route. `GRASP ARRIVAL:` compares the planned and measured
tool tip at the grasp pose before the gripper closes. When some candies in a
scan are accepted, verbose mode also lists each rejected candy with its score
and reason (for example, outside the plausible table height).

To save a run for later review, log to `Demo-Ignite/logs/` (git-ignored).
`python -u` keeps output unbuffered and `tee -i` keeps logging through Ctrl+C,
so the shutdown lines are captured:

```bash
python -u Demo-Ignite/demo.py --allow-motion --allow-pick --allow-hand-delivery \
  --setup-reviewed --detector opencv --verbose --max-cycles 10 2>&1 \
  | tee -i Demo-Ignite/logs/run-$(date +%Y%m%d-%H%M%S).log
```

### Measuring the candy XY correction

The 2026-11-04 ten-cycle runs (YOLO 3/10, OpenCV 5/10 pickups) reached every
planned grasp within 0.6 mm with the jaws across the candy, yet candies lying
along base Y were picked 6/7 times while diagonal or X-aligned candies mostly
failed. A constant candy-position error of roughly 30-35 mm (sign unknown)
explains 17 of 19 valid attempts: the error is harmless along the candy's
length but exceeds the ~13 mm jaw clearance across its width.

`--pause-before-grip` measures that error directly. After the arm reaches each
grasp pose (jaws open, before closing) it holds position and prints the candy
estimate, candy yaw, and jaw point, then asks where the candy centre really is
relative to the midpoint between the open fingertips, in robot base directions:
**+X** straight out from the base (toward the far edge), **+Y** to the
operator's left when standing behind the robot facing +X. Type `dx dy` in mm
(for example `0 -30`), or press Enter to grip without recording. Invalid input
re-prompts; Ctrl+C stops the demo with the normal zero return. The arm keeps
streaming the hold pose with health checks while waiting, then the pickup
continues as usual.

Each reading prints its component across the jaws and along the candy, then a
`GRASP OFFSET SUMMARY` with the run's mean and the suggested
`"candy_xy_correction_mm"` (the correction in effect plus the mean offset). Every
reading is also appended to `Demo-Ignite/logs/grasp_offsets.jsonl` with the
candy point, yaw, jaw point and axis, offset, and correction in effect. Take
readings with the candy at several orientations (along X, along Y, diagonal),
set `candy_xy_correction_mm` in `DEMO_SETTINGS` to the suggested value
(each component limited to +/-60 mm), and rerun to confirm the offsets shrink.
The correction applies to candies only, not palms. The option needs an
interactive terminal; `| tee -i` logging keeps the keyboard connected:

```bash
python -u Demo-Ignite/demo.py --allow-motion --allow-pick --allow-hand-delivery \
  --setup-reviewed --pause-before-grip --verbose --max-cycles 6 2>&1 \
  | tee -i Demo-Ignite/logs/offsets-$(date +%Y%m%d-%H%M%S).log
```

#### 2026-11-05 offset readings: lean direction, not camera calibration

Twelve paused readings plus all 33 logged pickups showed no constant camera
offset: grasps leaning sideways or back landed within 0-4 mm of the candy, and
the scan frames show the estimated centre on the candy. Grasps where the 20
degree lean pointed forward (fingertips toward +X, used for candies lying along
X) were off by 4-31 mm along the candy and picked **2/18**, versus **12/15**
for other leans. Keep `candy_xy_correction_mm` at [0, 0]; the run mean
(about [-7, -0.6] mm) would push sideways-gripped candies across the jaws.
Verbose `PICK PLAN` lines now report the lean direction (`leaning [x, y]`, or
`(vertical)`).

To compare grasp tilts physically, `--grasp-tilt-deg` overrides the
`grasp_rpy_deg` pitch for one run (0-30 degrees; 0 = vertical) without editing
`DEMO_SETTINGS`. Offline planning reaches vertical grasps for candies lying
along X out to about X 420 mm. Run the same candy placement once with the
default and once with `--grasp-tilt-deg 0`, both with `--pause-before-grip`:

```bash
python -u Demo-Ignite/demo.py --allow-motion --allow-pick --allow-hand-delivery \
  --setup-reviewed --pause-before-grip --verbose --max-cycles 3 --grasp-tilt-deg 0 2>&1 \
  | tee -i Demo-Ignite/logs/tilt0-$(date +%Y%m%d-%H%M%S).log
```

The 2026-11-06 comparison on one candy lying along X (X ~299, Y ~101 mm):
the forward-leaning 20 degree grasp picked 1/2 (offset -4 mm, and the second
attempt clipped and toppled the candy on the way down); the vertical grasp
picked **3/3** with 0 mm offsets. Each descent is a single joint-space move, and
the forward-leaning descent also rotates J4/J6, so the fingertips can sweep
sideways below grasp height before arriving; the vertical descent moves only
J2/J3/J5. The planner therefore grasps vertically whenever the tilted grasp
would lean forward and a vertical grasp is reachable. Verbose output reports
`Grasp leans forward (lean X ...); using a vertical grasp instead`, or that
the tilted grasp was kept because vertical is unreachable.

### Choosing the candy detector

`--detector yolo` (default) uses the CUDA YOLO detection service on 8096.
`--detector opencv` uses the local CPU colour detector in
[opencv_detector.py](opencv_detector.py) instead; the YOLO service is then not
contacted or required. Both return the same box, score, and long-axis fields, so
camera localization, long-axis midpoint pickup, workspace checks, and planning
are unchanged. Both ignore detections scoring below
`detection_confidence_threshold` (**0.5**, raised from 0.05): in the 2026-11-04
YOLO run the picks scored 0.05-0.49 were gripper-occluded or bad-depth
detections, while real candies scored 0.84-0.99. OpenCV fill scores are already
at least 0.6, so the threshold does not change OpenCV results.
Preflight reports which detector is active:

```bash
python Demo-Ignite/demo.py --check-only --detector opencv
```

The OpenCV detector masks saturated, bright warm colours (hue 0-30 or 165-179,
S >= 100, V >= 100), then accepts blobs of 1,500-9,000 px with rotated-rectangle
aspect 2.0-3.5 and fill >= 0.6. Its score is that fill ratio (a shape score, not
a learned confidence). After camera localization, each OpenCV candidate must
also be physically **shorter than 4 in (101.6 mm) and narrower than 2 in
(50.8 mm)**, so colour alone cannot select a target. Size is measured by
projecting the wrapper's long-axis ends and width edges through the accepted
camera intrinsics and capture-time camera pose onto the horizontal plane at
the candy's localized base Z; width uses the component perpendicular to the
table length axis to remove perspective skew. At the taught scan pose (~27
degrees tilt, ~418 mm high), modelled errors are overestimates only: length
0-9 mm (largest near image corners), width under 2 mm, so error can reject a
near-limit candy but cannot pass an oversized object. Captured wrappers
measured 78-89 x 32-37 mm. Oversized candidates are rejected with their size
in the error; accepted sizes appear in the `SCAN` line. Startup requires
camera intrinsics in `hand_eye.json` when `--detector opencv` is selected. The
YOLO path is unchanged and does not apply the size check.
Touching or partly occluded wrappers usually fail the
shape limits and are skipped rather than merged. Tuning used one scan-pose
scene of three wrappers (all detected; axis midpoints within about 3 mm of the
YOLO path) and 80 live frames of a cluttered room (no detections after tuning).
**It has not been validated with hands, skin, or other warm-coloured objects in
the workspace**; skin overlaps the hue range. Validate with
[capture rounds](#capture-scan-pose-images-for-detector-tuning) before physical
use, and expect lighting changes to require re-tuning. Evaluate saved captures
offline (no robot or services):

```bash
python Demo-Ignite/opencv_detector.py Demo-Ignite/captures
```

This prints per-image OpenCV vs YOLO-reference agreement and measured sizes,
lists rejected blobs with reasons (including size), writes annotated images to
`captures/opencv-eval/`, and exits 1 if any image disagrees with its YOLO
reference. Size uses each capture's saved camera pose and intrinsics with a
table plane at `--plane-z-mm` (default 12 mm; ±8 mm changes sizes by ~2 mm).

## Replay originally taught candy-workspace corners

List and validate the saved corners without connecting to CAN:

```bash
python Demo-Ignite/workspace_motion.py --check-only
```

[workspace_motion.py](workspace_motion.py) reads the candy poses from the D435
`calibration/workspace_teaching.json`, verifies their current tool-tip FK,
workspace bounds, tool length, and endpoint geometry, and lists the corner
numbers. It replays recorded joint poses, not the eight synthetic corners of
the axis-aligned workspace box.

After clearing and reviewing the path to one selected corner:

```bash
python Demo-Ignite/workspace_motion.py --corner 1 --allow-motion --setup-reviewed
```

To visit **all recorded corners in order** in a single session:

```bash
python Demo-Ignite/workspace_motion.py --all-corners --allow-motion --setup-reviewed
```

To select an ordered subset, use `--corners 2 4 6 8` instead of `--all-corners`.
The selection options are mutually exclusive; duplicates and invalid indices
are rejected. Add `--check-only` and omit motion permissions to validate a
selection offline. Do not retry physical motion after CAN transmission failures
until the communication fault is resolved.

Restart the controller after teaching; remove candy and people from the test
region and keep emergency stop accessible. Type `MOVE` once to confirm the whole
selected route. The utility enables motors and visits the selected corner(s)
at no more than 10% speed, without gripper commands. **Every replay first moves
to SDK zero (all six joints zero)** at the Piper rest speed (currently 10%),
verifies zero arrival, then visits the selection. It checks the entire
current-to-zero, zero-to-first-corner, and corner-to-corner route before the first move, then
rechecks each actual current-to-target segment and verifies arrival joints within
2 degrees. Each corner has a one-second pause, with no power-off between corners.
These checks do not
detect table, person, or self-collisions, and a taught endpoint does not prove
the path to it is clear.

The two zero-adjacent segments use factory joint limits because SDK zero lies
on J2/J3 boundaries outside the usual inward margin. Taught endpoints and
corner-to-corner segments retain inward-margined limits. A rejected zero path,
failed zero convergence, or unsafe later segment prevents corner replay.

Experimental settings are kept in this script's `PREVIEW_CORNER_OFFSETS_MM`
and `PREVIEW_MAXIMUM_Y_MM`, not the live demo configuration. C2 currently has
`[0, 70, -8.20008306013348]` mm: its requested tip is approximately
`[144.64, 206.32, 18.00]` mm, with a script-local preview maximum Y of
206.32 mm. Original teaching records and the demo's candy maximum Y
(178.20 mm in the configuration file) remain unchanged. Ignite's independent
code-defined limits are described under Required setup. These preview settings
are not new physical measurements or collision certification.
The script-local `PREVIEW_MINIMUM_XY_MM` is `[144, -170]` mm to permit C4's
requested target `[144, -170, 18]` mm. C4's offset is relative to its unchanged
original taught point, and its taught wrist orientation is preserved.
C6 requests `[440, -170, 18]` mm and C8 requests `[440, 206, 18]` mm, also
through script-local offsets with their original taught orientations preserved.
Adjusted targets are solved with IK while
preserving the originally taught orientation, then checked within 1 mm and
0.5 degrees. Requested XYZ must lie inside candy bounds; achieved XYZ is
allowed only the stated 1 mm IK tolerance, including at a boundary. Flange,
joint-limit, and tool-tip floor checks remain strict. Listing shows the
achieved IK tip and the applied offset.
The earlier coordinate diagram shows the unadjusted teaching record.

Run from an interactive terminal. On success, error, or Ctrl+C it holds the
measured current pose until manual controller power-off confirmation. Unlike
the main demo, Ctrl+C does **not** request automatic zero return. The power-off
prompt occurs only after the selected sequence finishes or stops; it safely ends
the holding session, not each waypoint. No configuration or teaching records are changed. This checks
physical taught bounds only, not the accuracy of camera localization.

## Capture scan-pose images for detector tuning

[capture_scan_images.py](capture_scan_images.py) saves lossless frames from a
taught scan pose for offline candy-detector tuning (for example, an OpenCV
colour detector). By default it only validates calibration, the scan pose, and
the camera service; it never connects to CAN:

```bash
python Demo-Ignite/capture_scan_images.py
```

To capture, run from an interactive terminal with the demo stopped:

```bash
python Demo-Ignite/capture_scan_images.py --allow-motion --setup-reviewed
```

After typed `MOVE` confirmation and a 5-second countdown, it moves current ->
SDK zero -> scan pose with the same checks as demo startup, with **no gripper
commands**. The arm stays enabled and holding while you arrange the scene. At
each prompt: Enter captures a round, a label (letters, digits, `-`, `_`) captures
a labelled round (e.g. `hand`, `three-candies`), and `q` finishes. On finish,
error, or Ctrl+C it attempts the checked SDK-zero return, then disconnects.

Each round saves `--frames` frames (default 3) to `Demo-Ignite/captures/`
(ignored by git): a PNG plus JSON metadata with token, measured joints/pose,
`camera_to_gripper`, `gripper_to_base`, `camera_to_base`, camera intrinsics,
workspace corners/Z limits, and YOLO reference detections (`--no-yolo` skips
them; an unavailable detector is reported but does not stop capture). A frame is
rejected if the arm moves beyond the scan stillness limits; existing files are
never overwritten. Options: `--scan`, `--frames`, `--frame-interval`, `--label`,
`--output-dir`, `--confidence`, `--verbose`. Images are never used as robot targets.

## Hardware-free tests

The local [vision_client.py](vision_client.py) handles camera, candy-detection,
and hand-service HTTP requests. The local [piper/robot.py](piper/robot.py)
supplies arm connection and control, using local [config.py](piper/config.py),
[transforms.py](piper/transforms.py), and [kinematics.py](piper/kinematics.py).
The required higher-level planning and calibration helpers live directly in
[demo.py](demo.py); neither `Demo-Product-Day` nor `Demo-VLA-On-Piper` is needed.
The demo still requires installed Python dependencies, the detection HTTP client
from [k3s-vision-detection](../k3s-vision-detection/detection_client.py), camera-specific
configuration/calibration files, taught scan poses, and the running services.

```bash
python -m unittest discover -s Demo-Ignite -v
```

Tests mock all robot connections and live service requests, covering the
three-inch relative height, updated palm positions, missing/moved/stale hands,
motion gates, path rejection, grasp failures, looping, and zero-return/disconnect shutdown.
Geometry/planner regressions exercise local IK and coordinate transforms.
An isolated-copy test runs preflight with neither older demo folder present.
