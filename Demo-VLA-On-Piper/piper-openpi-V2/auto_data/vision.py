#!/usr/bin/env python3
"""Find the cubes, bowls and bin in the top camera, so the collector can work a scene it did
not lay out itself.

Colour is decided by CHANNEL DOMINANCE (R clearly above G and B, etc), not an HSV hue window.
The table is warm plywood under changing light: a hue window either misses a washed-out cube or
swallows the whole board, and we have been bitten by both. Dominance survives both.

Class comes from size + roundness, measured on this rig at 640x480:
    bowl   area > 3000, circularity > 0.65      (red bowl 6310/0.87, blue bowl 5684/0.96)
    cube    200 < area < 2000, circularity > 0.55   (blue cube 768/0.72, red cube 366/0.70)
    bin    area > 3000, 0.35 < circularity < 0.65   (4058/0.46; the ARM is 15110/0.27 - rejected)

The arm itself is the largest dark blob in frame, so `detect()` is only trustworthy with the arm
parked out of the way (HOME). The collector detects between episodes, when it is parked anyway.

    from auto_data.vision import detect
    objs = detect(bgr_frame)        # [{'label':'red_cube','px':279,'py':236,'area':366,...}, ...]
"""
import cv2
import numpy as np

MIN_AREA = 200
BOWL_AREA = 3000
CUBE_AREA_MAX = 2000
BOWL_CIRC = 0.65
CUBE_CIRC = 0.55
CUBE_ASPECT_MAX = 1.5   # a cube from above is square (measured 1.02-1.03); the cable and frame-edge
                        # artifacts that used to pass as cubes are 2.1-3.5
BIN_CIRC = (0.35, 0.65)

DOMINANCE = 28          # a channel must beat both others by this much. Tuned on two real scenes:
                        # a cube in the bowl's shade reads +69, the same cube on the bright table
                        # only +32 (pale cyan), so 35 missed it. Going lower is NOT better - at 22
                        # the blue regions grow until bowl and cube merge into one unclassifiable
                        # blob, and both are lost.
MIN_BRIGHT = 60         # ... and be at least this bright, so shadows are not "blue"
DARK_MAX = 70           # every channel below this = "dark" (the bin)
CLOSE_COLOUR = 7        # morphological close for red/blue: small, so a cube next to a bowl stays separate
CLOSE_DARK = 13         # ... and larger for the bin, which is ragged and would otherwise fragment


def white_balance(bgr):
    """Divide out the illuminant so dominance measures the OBJECT, not the lamp.

    Necessary, not cosmetic: changing the room light moved the bare table from R-B +15 (warm) to
    -22 (cool). At -22 the table is only 6 units short of the blue threshold - one notch bluer and
    the whole board registers as one giant blue object. The reference is the per-channel MEDIAN,
    which the table dominates; a mean would be dragged around by a big red bowl.
    """
    med = np.median(bgr.reshape(-1, 3), axis=0)
    scale = med.mean() / np.maximum(med, 1.0)
    return np.clip(bgr.astype(np.float32) * scale, 0, 255).astype(np.uint8)


def _masks(bgr):
    bgr = white_balance(bgr)
    b, g, r = (bgr[:, :, i].astype(np.int16) for i in range(3))
    return {
        "red": ((r - np.maximum(g, b)) > DOMINANCE) & (r > MIN_BRIGHT + 10),
        "blue": ((b - np.maximum(g, r)) > DOMINANCE) & (b > MIN_BRIGHT),
        "dark": np.maximum(np.maximum(b, g), r) < DARK_MAX,
    }


def _blobs(mask, close_k):
    m = cv2.morphologyEx((mask * 255).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = float(cv2.contourArea(c))
        if a < MIN_AREA:
            continue
        (cx, cy), rad = cv2.minEnclosingCircle(c)
        if rad <= 0:
            continue
        (rx, ry), (rw, rh), ang = cv2.minAreaRect(c)
        # a square has 4-fold symmetry, so any opposite face pair is as good as another: fold the
        # angle into +-45 deg and the wrist never has to turn further than that to line up
        out.append({"px": float(cx), "py": float(cy), "area": a,
                    "radius": float(rad), "circ": a / (np.pi * rad * rad),
                    "angle": float(((ang + 45.0) % 90.0) - 45.0),
                    "aspect": float(max(rw, rh) / max(min(rw, rh), 1e-6))})
    return out


def detect(bgr, roi=None):
    """Objects visible in a top-camera frame. `roi` = (x0, y0, x1, y1) to ignore the arm side."""
    objs = []
    for colour, mask in _masks(bgr).items():
        # The close kernel has to differ by class. A cube sitting beside a bowl leaves an ~11 px
        # gap: closing with 11 bridges it and the two merge into one blob that is neither (that
        # cost us both objects). The bin is the opposite - large, dark and ragged, and needs the
        # bigger kernel to come out as one piece.
        for b in _blobs(mask, CLOSE_DARK if colour == "dark" else CLOSE_COLOUR):
            if roi and not (roi[0] <= b["px"] <= roi[2] and roi[1] <= b["py"] <= roi[3]):
                continue
            label = None
            if colour == "dark":
                if b["area"] > BOWL_AREA and BIN_CIRC[0] < b["circ"] < BIN_CIRC[1]:
                    label = "black_bin"
            elif b["area"] > BOWL_AREA and b["circ"] > BOWL_CIRC:
                label = f"{colour}_bowl"
            elif (MIN_AREA <= b["area"] <= CUBE_AREA_MAX and b["circ"] > CUBE_CIRC
                  and b["aspect"] <= CUBE_ASPECT_MAX):
                label = f"{colour}_cube"
            if label:
                objs.append({"label": label, **b})
    # one of each: the biggest candidate wins (a reflection never beats the real thing)
    best = {}
    for o in objs:
        if o["label"] not in best or o["area"] > best[o["label"]]["area"]:
            best[o["label"]] = o
    return sorted(best.values(), key=lambda o: -o["area"])


def annotate(bgr, objs):
    out = bgr.copy()
    for o in objs:
        c = {"red": (0, 0, 255), "blue": (255, 120, 0), "black": (200, 200, 200)}[o["label"].split("_")[0]]
        cv2.circle(out, (int(o["px"]), int(o["py"])), int(o["radius"]), c, 2)
        cv2.drawMarker(out, (int(o["px"]), int(o["py"])), (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(out, f"{o['label']} {int(o['area'])}", (int(o["px"]) - 55, int(o["py"]) - int(o["radius"]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    return out


if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1])
    found = detect(img)
    for o in found:
        print(f"  {o['label']:11s} px ({o['px']:6.1f},{o['py']:6.1f})  area {o['area']:7.0f}  "
              f"circ {o['circ']:.2f}  angle {o['angle']:+6.1f}  aspect {o['aspect']:.2f}")
    if len(sys.argv) > 2:
        cv2.imwrite(sys.argv[2], annotate(img, found))
        print("annotated ->", sys.argv[2])
