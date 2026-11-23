# GTC demo: candy from a bowl to an open hand

The Piper arm picks a cuboid Christmas candy (1 x 1 x 0.75 inch) from a
6-inch bowl and drops it on an open hand, using the wrist D435 camera, OpenCV,
and the MediaPipe hand service. It reuses the Ignite calibration, scan pose,
workspace, k3s camera/hand services, and motion safety checks.

## Flow

Each cycle starts at the taught scan pose:

1. **Bowl and candy** ([bowl_vision.py](bowl_vision.py)). The photo is warped
   to a top-down view at rim height, where the 6-inch rim is a circle: Hough
   circles confirmed by edge support (no colour). Depth just inside the far
   rim must read a raised bowl, which rejects flat circles. Candies are regions
   inside the rim whose colour differs from the bowl's own, kept when they are
   candy-sized rectangles (touching candies are split). Each candy's top
   centre comes from depth.
2. **Pick** ([demo.py](demo.py)). The highest candy whose open jaws stay clear
   of the bowl wall is picked straight down: above the candy 30 mm over the
   rim, vertical descent, grip, vertical lift, back to the scan pose. A width
   check confirms the candy is held; a miss opens above the bowl and rescans.
3. **Wait for a hand** ([hand_target.py](hand_target.py)). Holding the candy at
   the scan pose, the demo waits (no timeout) for an open palm whose finger
   base (centre of the four knuckles) is visible by depth and inside the hand
   workspace.
4. **Drop**: 35 mm above the palm centre, or above the finger base when the
   palm centre is not visible. Then the next cycle.

If anything fails while the candy is held, it is carried back along the
reached waypoints and dropped into the bowl. Ctrl+C returns the arm to SDK
zero and opens the gripper there.

## Bowl placement

With the current scan pose (shared with Ignite), the whole rim is in the photo
only when the bowl centre is beyond about X 280 mm, and straight-down grasps
reach only to about X 300-350 mm (wrist pitch J5 limit). The usable band is:

| Bowl centre X | Bowl centre Y |
|---|---|
| 280-300 mm | about -80 to +100 mm |

A dedicated scan pose that looks closer to the robot would allow X 220-280 mm.

## Run

From the repository root with `source ~/venv-ignite-demo/bin/activate`:

```bash
# Preflight: services, calibration, scan pose; no CAN, no motion
python GTC-demo/demo.py --check-only

# Move to the scan pose and report bowl, candies, and hand every cycle; never picks
python GTC-demo/demo.py --allow-motion --scan-only --verbose

# Pick from the bowl and drop the candy back in; no hand needed
python GTC-demo/demo.py --allow-motion --pick-only --verbose

# Full demo
python -u GTC-demo/demo.py --allow-motion --verbose 2>&1 | tee -i GTC-demo/logs/run-$(date +%Y%m%d-%H%M%S).log
```

`--max-cycles N` stops after N cycles (default 0: until Ctrl+C). Each bowl scan
writes an annotated photo to `GTC-demo/logs/latest_bowl_scan.jpg`: rim in blue,
candies in green, rejected regions in red, the chosen candy in yellow.

## Tuning with photos

Capture photos at the scan pose (moves the arm; type `MOVE` to confirm, Enter
to capture, a word to label the photo, `q` to finish):

```bash
python GTC-demo/capture_scan_images.py --allow-motion --setup-reviewed --no-yolo --frames 1
```

Evaluate them offline (no robot or services) and review the annotated images
in `GTC-demo/captures/bowl-eval/`:

```bash
python GTC-demo/bowl_vision.py GTC-demo/captures
```

Detection settings are in `BOWL_SETTINGS` in [bowl_vision.py](bowl_vision.py);
motion settings in `DEMO_SETTINGS` in [demo.py](demo.py).

## To measure on the arm

These values are estimates until checked with the real candy and bowl:

| Setting | Current | How to check |
|---|---|---|
| `maximum_descent_below_rim_mm` (bowl_vision) | 40 mm | Lower the gripper into the bowl by hand; the deepest candy top the wrist and camera can reach without touching the rim |
| `grasp_tip_above_candy_top_mm` | 0 mm | `--pick-only --pause-before-grip`: jaws should straddle the candy's middle |
| `candy_xy_correction_mm` | 0, 0 | `--pause-before-grip` prints the suggested value after each typed offset |
| `minimum_held_grip_mm` | 12 mm | Held candy should read about 19-25 mm, an empty close about 1 mm |
| `finger_thickness_mm`, `finger_width_mm` (bowl_vision) | 10, 20 mm | Measure the jaws; they set the wall clearance |

## Files

| File | Purpose |
|---|---|
| [demo.py](demo.py) | The demo loop and its modes |
| [bowl_vision.py](bowl_vision.py) | Bowl and candy detection, rim check, grasp ordering; offline evaluation |
| [hand_target.py](hand_target.py) | Finger-base gate and palm-centre / finger-base choice |
| [vision_client.py](vision_client.py) | Camera (photos, depth at a pixel) and hand service clients |
| [capture_scan_images.py](capture_scan_images.py) | Scan-pose photo capture for tuning |
| [workspace_motion.py](workspace_motion.py) | Replay the taught workspace corners |

The hand service ([k3s-hand-service](../k3s-hand-service/README.md)) reports
the palm centre and finger base with depth visibility in its `points` block.

Tests (hardware-free): `python -m unittest discover -s GTC-demo` from the
repository root.
