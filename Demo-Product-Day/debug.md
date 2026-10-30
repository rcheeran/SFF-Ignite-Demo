# Demo hand-targeting investigation

Investigation date: 2026-10-08

## Reported behavior

After approximately cycle 290, the Piper arm appeared to attempt picking the
presented hand instead of a Starburst candy.

## Available evidence

The run metadata for this period is no longer intact:

- `prepare-piper/hand-eye/calibration/demo_runs.json` was reset and currently
  contains only a later 10-cycle run.
- `prepare-piper/hand-eye/calibration/.demo_runs.json.b_gre2pp` is an interrupted
  atomic-write temporary file. It contains complete records through cycle 143
  and truncates during cycle 144.
- The YOLO dataset images and labels survived under
  `/var/lib/rancher/k3s/storage/productdaydemo-yolo/datasets/collected-starburst`.

Cycle 144 starts with image
`20261008T185202_898743Z_e57ae206.jpg`. Using the subsequent three-image scan
groups, cycle 290 maps to approximately 20:00:32 UTC. This mapping assumes there
were no extra grasp-retry captures after cycle 144; the regular timestamp groups
and the reported onset support that assumption.

## Confirmed findings

Cycle 290 corresponds approximately to these images:

- `20261008T200032_781865Z_19168e12.jpg`
- `20261008T200039_542173Z_f8f2503c.jpg`
- `20261008T200046_949672Z_533c5ba1.jpg`

An open palm becomes prominent in scan pose 2 shortly afterward. The first
confirmed palm false positives are:

### Approximate cycle 293, scan pose 2

- Image: `images/20261008T200156_547325Z_d43d8d6c.jpg`
- Label: `labels/20261008T200156_547325Z_d43d8d6c.txt`
- YOLO emitted class `8` (`starburst`) over the palm:

```text
8 0.532559 0.534183 0.168594 0.269208
```

At 640x480 resolution, this is a box of approximately 108x129 pixels centered
on the open palm.

### Approximate cycle 294, scan pose 2

- Image: `images/20261008T200221_295084Z_12558913.jpg`
- Label: `labels/20261008T200221_295084Z_12558913.txt`
- YOLO again emitted class `8` over the palm:

```text
8 0.628486 0.504941 0.171060 0.389544
```

This is a still taller palm-sized box, approximately 109x187 pixels.

The images visually confirm that these boxes cover the presented hand rather
than candy. The missing run JSON prevents recovering the exact score, selected
base coordinate, and executed plan for these cycles, but the detector failure
itself is conclusive.

## Failure chain

1. The candy YOLO model classified an open palm as the `starburst` class.
2. `decode_yolo_output()` in `k3s-OpenCV-service/vision.py` accepts detections using
   the requested class score threshold and NMS. It has no hand exclusion.
3. The HSV contour/orientation check can accept saturated, elongated regions
   formed by skin and fingers.
4. `choose_target()` in `demo.py` assigns a lone detection a cross-view distance
   of zero. A one-view false positive therefore passes cross-view agreement.
5. `distinct_object_targets()` sorts accepted targets by base distance.
6. `select_reachable_object_target()` chooses the first target with a safe IK
   plan, so a nearby reachable palm false positive can outrank real candy.
7. Object and hand detections are processed separately. There is no final veto
   for an object candidate that overlaps or lies near the detected palm.

## Recommended follow-up

Before another supervised run:

1. Require at least two agreeing object views before permitting a pick.
2. Reject object targets within a conservative 3D distance of any detected
   palm, including per-view detections rather than only the selected hand target.
3. Reject implausibly large Starburst boxes or physical dimensions.
4. Add palm false-positive images from cycles 293 and 294 as hard negatives for
   model evaluation or retraining.
5. Preserve each run in a timestamped output file instead of overwriting
   `demo_runs.json`, so future incidents retain exact target and motion records.
6. Replay the retained images through the deployed model and record confidence,
   post-orientation filtering, transformed coordinates, and target ranking.

Do not resume autonomous picking until at least the multi-view requirement and
hand-proximity veto are implemented and tested.