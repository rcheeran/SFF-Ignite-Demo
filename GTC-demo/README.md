# GTC demo: candy from the table to an open hand

The Piper arm picks a cuboid Christmas candy (about 1 x 1 x 0.75 inch) from
the table and drops it on an open hand, using the wrist D435 camera, OpenCV,
and the MediaPipe hand service. It reuses the Ignite calibration, scan pose,
workspace, k3s camera/hand services, and motion safety checks.

The earlier bowl version is parked: [bowl_vision.py](bowl_vision.py) is kept
for later but no longer used by the demo, apart from shared helpers.

## Flow

Each cycle starts at the taught scan pose:

1. **Candies** ([table_vision.py](table_vision.py)). The photo is warped to a
   top-down view on the plane through the candy tops. A candy is a blob of the
   foil colours (red, green, gold, merged so a ribbon in another colour does
   not split it). The camera also sees each candy's near sides, which stretch
   the blob toward the camera; they are trimmed geometrically so the box sits
   on the top face. Blobs cut off by the photo edge, the gripper fingers (the
   dark shape at the bottom of every photo), or the workspace edge are
   skipped. Each candy's top is then measured from depth: it must be 10-32 mm
   above the table (a print on the board or a hand is not a candy), and depth
   where each open finger would land must be below the fingertips, so the
   gripper does not come down on a neighbour. The candy nearest the robot wins.
2. **Pick** ([demo.py](demo.py)). No pick starts while the hand service sees
   any hand, open or closed. The gripper goes 60 mm above the candy, moves
   straight in along its own axis, grips, moves back out, and returns to the
   scan pose; it never dips below that approach height on the way over other
   candies. It is vertical when that reaches; farther candies use the
   smallest tilt (10, 20, or 30 degrees, about the jaws' closing axis so they
   still close level). A width check confirms the candy is held; a miss opens
   the gripper and rescans.
3. **Wait for a hand** ([hand_target.py](hand_target.py)). Holding the candy at
   the scan pose, the demo waits (no timeout) for an open palm whose finger
   base (centre of the four knuckles) is visible by depth and inside the hand
   workspace.
4. **Drop**: 35 mm above the palm centre, or above the finger base when the
   palm centre is not visible. Then the next cycle picks another candy.

If anything fails while the candy is held, it is carried back along the
reached waypoints and dropped onto its spot from the approach height (the arm
does not come back down to the table). Ctrl+C returns the arm to SDK zero and
opens the gripper there.

## Candy placement

Place candies one at a time, not touching, where the scan pose sees them
whole: roughly X 230-550 mm and inside the workspace (X 144-595, Y -170 to 206
mm). Candies at the bottom edge of the photo, behind the gripper fingers, or
across the workspace edge are skipped.

| X (mm) | Grasp |
|---|---|
| up to ~370 | straight down (most reliable) |
| ~375-525 | tilted 10-30 degrees; most candy orientations reach |
| ~550-595 | tilted; about two thirds of orientations reach, at nearly full arm extension |

The flange workspace limit in [piper/config.py](piper/config.py) is X 600 mm
(500 mm in Demo-Ignite), so the arm's joint limits set the reach: with 500 mm,
about a third of orientations reached at X 550 and none beyond ~575. Drops
reach palms out to X ~560-570 mm near the centre line and ~550 mm at the sides
(~510-520 mm at the far left, Y 180-206); a palm farther out keeps the arm
waiting with `no joint-limit-safe IK solution for hand approach`.

## Run

From the repository root with `source ~/venv-ignite-demo/bin/activate`:

```bash
# Preflight: services, calibration, scan pose; no CAN, no motion
python GTC-demo/demo.py --check-only

# Move to the scan pose and report candies and hand every cycle; never picks
python GTC-demo/demo.py --allow-motion --scan-only --verbose

# Pick a candy and drop it back on its spot; no open palm needed
python GTC-demo/demo.py --allow-motion --pick-only --verbose --pause-before-grip

# Full demo
python -u GTC-demo/demo.py --allow-motion --verbose 2>&1 | tee -i GTC-demo/logs/run-$(date +%Y%m%d-%H%M%S).log
```

`--max-cycles N` stops after N cycles (default 0: until Ctrl+C). Every mode
needs the hand service, which confirms no hand is in view before each pick.
Each table scan writes an annotated photo to
`GTC-demo/logs/latest_candy_scan.jpg`: pickable candies in green, skipped ones
in red.

## Position correction (camera to robot)

The hand-eye calibration places candies too far out in X: +20 mm at X 300
(2026-11-23) and +32 mm at X 372 (the failed 2026-11-03 candy-point check in
`prepare-piper/hand-eye-realsense435/calibration/candy_point_verification.json`).
A camera-angle correction does not explain both, so the demo applies a
measured correction to every candy and hand position seen from the scan pose:

    true XY = candy_xy_correction_matrix @ camera XY + candy_xy_correction_mm

`[-20, 0]` with the identity matrix is provisional. To fit it, spread 5-6
candies over the area the demo will use (not touching), then:

```bash
python GTC-demo/demo.py --allow-motion --measure-offsets --verbose --max-cycles 1
```

From one scan it visits each reachable candy in turn, hovering the open
fingertips 5 mm above it (nothing is gripped or touched), and asks for the
candy centre's offset from the midpoint between the fingertips in robot
directions (`dx dy` in mm: +X away from the robot, +Y to your left facing +X).
Keep your hand out of the camera view while the arm moves, and do not move the
candies until the cycle ends. Then fit the readings taken since the run
started (local time) and paste the two printed settings into `DEMO_SETTINGS`:

```bash
python GTC-demo/fit_position_correction.py --since 2026-11-23T15:40
```

Four or more readings fit the full matrix; fewer fit the offset only.

## Tuning with photos

Capture photos at the scan pose (moves the arm; type `MOVE` to confirm):

```bash
python GTC-demo/capture_scan_images.py --allow-motion --setup-reviewed --no-yolo --frames 1 --label table
```

Evaluate them offline (no robot or services) and review the annotated images
in `GTC-demo/captures/table-eval/`:

```bash
python GTC-demo/table_vision.py GTC-demo/captures
```

Detection settings are in `TABLE_SETTINGS` in [table_vision.py](table_vision.py)
(the foil colours come from `BOWL_SETTINGS` in [bowl_vision.py](bowl_vision.py));
motion settings in `DEMO_SETTINGS` in [demo.py](demo.py).

## To measure on the arm

These values are estimates until checked with the real candy:

| Setting | Current | How to check |
|---|---|---|
| `grasp_tip_above_candy_top_mm` | 5 mm | `--pick-only --pause-before-grip`: fingertips ~10 mm down the candy's side |
| `candy_xy_correction_mm`, `candy_xy_correction_matrix` | -20, 0; identity | `--measure-offsets`, then `fit_position_correction.py` (see above) |
| `minimum_held_grip_mm` | 12 mm | Held candy should read about 20-25 mm, an empty close about 1 mm |
| `pick_open_mm` | 60 mm | Jaw opening from "above candy" down to the grasp (the scan pose and drops use `grip_open_mm`, 40 mm). Candies closer than about 40 mm centre to centre are skipped by the finger-landing check |
| `finger_thickness_mm` | 10 mm | Measure the jaws; sets where the finger-landing depth is read |

The candy tops measure about 21-23 mm across in the scan photos rather than
25.4 mm; the 60 mm pick opening leaves about 19 mm either side.

## Files

| File | Purpose |
|---|---|
| [demo.py](demo.py) | The demo loop and its modes |
| [table_vision.py](table_vision.py) | Candy detection on the table and grasp ordering; offline evaluation |
| [fit_position_correction.py](fit_position_correction.py) | Fits the camera-to-robot XY correction from `--measure-offsets` readings |
| [bowl_vision.py](bowl_vision.py) | Parked bowl detection; its top-down view, colour segmentation, and drawing helpers are shared |
| [hand_target.py](hand_target.py) | Finger-base gate and palm-centre / finger-base choice |
| [vision_client.py](vision_client.py) | Camera (photos, depth at a pixel) and hand service clients |
| [capture_scan_images.py](capture_scan_images.py) | Scan-pose photo capture for tuning |
| [workspace_motion.py](workspace_motion.py) | Replay the taught workspace corners |

The hand service ([k3s-hand-service](../k3s-hand-service/README.md)) reports
the palm centre and finger base with depth visibility in its `points` block,
and how many of its samples showed any hand.

Tests (hardware-free): `python -m unittest discover -s GTC-demo` from the
repository root.
