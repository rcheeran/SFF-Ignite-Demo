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
| ~550 | tilted; about a third of orientations reach |
| beyond ~575 | out of reach |

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
| `candy_xy_correction_mm` | 0, 0 | `--pause-before-grip` prints the suggested value after each typed offset |
| `minimum_held_grip_mm` | 12 mm | Held candy should read about 20-25 mm, an empty close about 1 mm |
| `finger_thickness_mm` | 10 mm | Measure the jaws; sets where the finger-landing depth is read |

The candy tops measure about 21-23 mm across in the scan photos rather than
25.4 mm; the 40 mm jaw opening leaves about 9 mm either side.

## Files

| File | Purpose |
|---|---|
| [demo.py](demo.py) | The demo loop and its modes |
| [table_vision.py](table_vision.py) | Candy detection on the table and grasp ordering; offline evaluation |
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
