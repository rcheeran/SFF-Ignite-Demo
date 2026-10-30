#!/usr/bin/env python3
"""Where to put the block: a live top-camera frame with the usable placement area drawn on it.

    python -m tools.placement_guide                 # -> data/snapshots/placement_guide.jpg
    python -m tools.placement_guide --live          # refresh every 2 s until Ctrl-C (watch while you place it)

Open-loop replay teaches the policy nothing about position by itself - the VARIATION between episodes is
the whole signal for "find the block". The first six episodes all landed in two clusters spanning 120 mm
in x and 56 mm in y, all at positive y, so this draws the reachable band and marks where those grasps
happened, to make the unused area obvious.

The robot-XY <-> pixel map comes from auto_data/calib.json. If the top camera has been moved since that
was fitted the overlay will be wrong - the CAMERA VIEW itself is still correct, so the frame edges always
tell you what is in shot even when the overlay does not.
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from piper import config as C                    # noqa: E402
from piper.transforms import tip_position        # noqa: E402

CALIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "auto_data", "calib.json")
OUT = os.path.join(C.SNAP_DIR, "placement_guide.jpg")
REACH_X = (200.0, 480.0)      # conservative slice of the measured reach (x 150-560), clear of the base
REACH_Y = (-220.0, 220.0)     # measured +-260; pulled in so the arm is not fighting its own limits
GRID_MM = 50.0


def to_px(Ainv, x, y):
    p = cv2.transform(np.array([[[float(x), float(y)]]], np.float64), Ainv)[0, 0]
    return int(round(p[0])), int(round(p[1]))


def past_grasps(dataset):
    """(x, y) of the tool tip when the jaws first closed on something, per episode."""
    import pandas as pd
    files = sorted(glob.glob(os.path.join(dataset, "data", "**", "*.parquet"), recursive=True))
    if not files:
        return []
    df = pd.concat([pd.read_parquet(f) for f in files])
    out = []
    for _, sub in df.groupby("episode_index"):
        s = np.stack(sub.sort_values("frame_index")["observation.state"].values)
        hold = np.flatnonzero((s[:, 6] > 5) & (s[:, 6] < 60))
        if len(hold):
            p = tip_position(s[hold[0], :6])
            out.append((float(p[0]), float(p[1])))
    return out


def annotate(frame, Ainv, grasps):
    v = frame.copy()
    # the reachable band, as a filled-corner quad (the affine keeps straight lines straight)
    corners = [to_px(Ainv, REACH_X[0], REACH_Y[0]), to_px(Ainv, REACH_X[1], REACH_Y[0]),
               to_px(Ainv, REACH_X[1], REACH_Y[1]), to_px(Ainv, REACH_X[0], REACH_Y[1])]
    cv2.polylines(v, [np.array(corners, np.int32)], True, (0, 220, 0), 2)
    shade = v.copy()
    cv2.fillPoly(shade, [np.array(corners, np.int32)], (0, 90, 0))
    v = cv2.addWeighted(shade, 0.18, v, 0.82, 0)

    for x in np.arange(REACH_X[0], REACH_X[1] + 1, GRID_MM):          # grid, so spacing is judgeable by eye
        cv2.line(v, to_px(Ainv, x, REACH_Y[0]), to_px(Ainv, x, REACH_Y[1]), (0, 140, 0), 1)
    for y in np.arange(REACH_Y[0], REACH_Y[1] + 1, GRID_MM):
        cv2.line(v, to_px(Ainv, REACH_X[0], y), to_px(Ainv, REACH_X[1], y), (0, 140, 0), 1)

    for i, (gx, gy) in enumerate(grasps):                              # where the block has already been
        px = to_px(Ainv, gx, gy)
        cv2.drawMarker(v, px, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
        cv2.putText(v, str(i), (px[0] + 10, px[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.putText(v, "GREEN = place the block anywhere in here", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
    cv2.putText(v, f"RED = the {len(grasps)} places already used - avoid repeating them", (10, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.putText(v, f"grid {GRID_MM:.0f} mm   x {REACH_X[0]:.0f}-{REACH_X[1]:.0f}   y {REACH_Y[0]:.0f}..{REACH_Y[1]:.0f}",
                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join(C.DATA_DIR, "piper_block_bowl"))
    ap.add_argument("--live", action="store_true", help="refresh every 2 s until Ctrl-C")
    args = ap.parse_args()

    from piper.cameras import open_cameras
    Ainv = cv2.invertAffineTransform(np.array(json.load(open(CALIB))["A"], np.float64))
    grasps = past_grasps(args.dataset) if os.path.isdir(args.dataset) else []
    os.makedirs(C.SNAP_DIR, exist_ok=True)
    cams = open_cameras(names=("top",))
    try:
        if not cams["top"].wait(5.0):
            raise SystemExit("ABORT: no frames from the top camera")
        while True:
            cv2.imwrite(OUT, annotate(cams["top"].read(), Ainv, grasps), [cv2.IMWRITE_JPEG_QUALITY, 88])
            print(f"  {OUT}   ({len(grasps)} past grasps marked)" + ("   Ctrl-C to stop" if args.live else ""))
            if not args.live:
                return
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        for c in cams.values():
            c.close()


if __name__ == "__main__":
    main()
