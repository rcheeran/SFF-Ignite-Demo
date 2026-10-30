#!/usr/bin/env python3
"""Visual servoing: line the JAWS up with the OBJECT using the overhead camera, closed loop.

Why this replaces open-loop reaching. Predicting where the jaws will land needs a perfect chain -
camera calibration, tool length, the grasp point's offset from the flange, and how all of that
rotates with wrist yaw. Every one of those has been wrong at some point in this project, and the
errors only showed up as the jaws closing beside the cube. Measured evidence: the cube detected
1 mm from the commanded spot, reached for at that spot, and still missed by tens of millimetres.

Closed loop needs none of it. Both the jaws and the cube appear in the SAME overhead frame, so
their pixel difference is a direct measurement of the error. Drive that difference to zero and the
jaws are on the cube, whatever the calibration thinks. The affine is still used, but only to turn
a small pixel error into a small millimetre step - a rough scale is enough for that.

Servo at a height where the jaw tips and the object top are close together: the camera is not
directly overhead, so objects at different heights sit at different pixels even when they are at
the same place on the table.
"""
import cv2
import numpy as np

from auto_data.vision import white_balance

DARK_MAX = 70            # matt-black jaws against a lit table
MIN_PIX = 40


def find_jaws(bgr, expect_px, radius=55):
    """(cx, cy, n_pixels) of the jaw cloud near expect_px, or None.

    Same tuning as auto_data.jaw_direct: a tight radius, because a wide one swallows the wrist
    motor and cables and returns their centroid instead of the fingers'.
    """
    x0 = max(0, int(expect_px[0] - radius)); x1 = min(bgr.shape[1], int(expect_px[0] + radius))
    y0 = max(0, int(expect_px[1] - radius)); y1 = min(bgr.shape[0], int(expect_px[1] + radius))
    patch = white_balance(bgr)[y0:y1, x0:x1]
    dark = (patch.max(axis=2) < DARK_MAX).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    pts = np.column_stack(np.where(dark > 0))[:, ::-1]
    if len(pts) < MIN_PIX:
        return None
    c = pts.mean(0)
    return float(c[0] + x0), float(c[1] + y0), int(len(pts))


def pixel_step_to_mm(A, dpx, dpy):
    """Turn a pixel error into a millimetre step using the calibration's linear part only.
    The translation is irrelevant for a difference, so calibration OFFSET errors cancel here."""
    v = np.asarray(A, float)[:, :2] @ np.array([float(dpx), float(dpy)])
    return float(v[0]), float(v[1])


def align(robot, cam, A, joints_for_tip, stream, target_px, tip_xy, tip_z, yaw_deg=0.0,
          tol_px=4.0, max_iters=4, gain=0.8, log=print):
    """Nudge the arm until the jaw centroid sits on target_px. Returns the corrected (x, y).

    `target_px` is where the OBJECT is in the overhead frame; `tip_xy` is the current best guess
    of its table position. Each iteration measures the jaw centroid, converts the pixel error to
    millimetres and steps a fraction of it (gain < 1 keeps it from oscillating when the scale is
    imperfect).
    """
    x, y = float(tip_xy[0]), float(tip_xy[1])
    for i in range(max_iters):
        frame = cam.read()
        jaws = find_jaws(frame, target_px)
        if jaws is None:
            log(f"    servo {i}: jaws not visible - keeping ({x:.0f},{y:.0f})")
            return x, y, None
        dpx, dpy = target_px[0] - jaws[0], target_px[1] - jaws[1]
        err_px = float(np.hypot(dpx, dpy))
        if err_px <= tol_px:
            log(f"    servo {i}: aligned, {err_px:.1f} px")
            return x, y, err_px
        dx, dy = pixel_step_to_mm(A, dpx * gain, dpy * gain)
        x, y = x + dx, y + dy
        log(f"    servo {i}: jaws off by {err_px:5.1f} px -> step ({dx:+5.1f},{dy:+5.1f}) mm "
            f"-> ({x:.0f},{y:.0f})")
        stream(robot, joints_for_tip([x, y, tip_z], yaw_deg=yaw_deg), 2.5, None)
    return x, y, err_px
