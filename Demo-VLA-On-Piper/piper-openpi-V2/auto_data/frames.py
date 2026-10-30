#!/usr/bin/env python3
"""Coordinate transforms between the top camera and the robot, plus the calibration fit.

Lives on its own because both the calibration procedure and the collector need it, and having
them import each other for it produced circular imports twice.

The map is an AFFINE from top-camera pixels to robot XY on the table, stored in calib.json.
Affine rather than a homography on purpose: with the spot spreads available here a homography's
8 DOF fit noise (84 mm mean residual on real data) while an affine on the same points gave 0.8 mm.
"""
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_JSON = os.path.join(HERE, "calib.json")
ROI = (60, 0, 455, 480)        # ignore the arm side of the frame when detecting


def load_calib(path=CALIB_JSON):
    """The 2x3 affine, or None if the robot has not been calibrated yet."""
    if not os.path.exists(path):
        return None
    return np.asarray(json.load(open(path))["A"], float)


def px_to_xy(A, px, py):
    """Top-camera pixel -> robot XY of a thing sitting ON the table.

    Only valid for objects on the table. The camera is ~35 deg off vertical, so anything raised
    (the jaws, a carried cube) appears displaced by roughly 0.7 mm per mm of height - which is why
    comparing pixels of the jaws against pixels of a cube is not a measurement of their distance.
    """
    p = cv2.transform(np.array([[[float(px), float(py)]]]), np.asarray(A, float))
    return float(p[0, 0, 0]), float(p[0, 0, 1])


def image_angle_to_robot(A, angle_deg):
    """An edge direction measured in the image, expressed in the robot's XY frame."""
    A = np.asarray(A, float)
    d = np.array([np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))])
    v = A[:, :2] @ d
    a = np.degrees(np.arctan2(v[1], v[0]))
    return ((a + 45.0) % 90.0) - 45.0          # a square looks the same every quarter turn


def fit(pairs):
    """pairs = [((px,py),(x,y)), ...] -> (2x3 affine, per-point residual mm).

    Plain least squares over EVERY pair: cv2's robust estimators discard up to half the points,
    which with 4-6 validated pairs throws away data. A bad pair shows in the residuals instead.
    """
    src = np.array([p[0] for p in pairs], np.float64)
    dst = np.array([p[1] for p in pairs], np.float64)
    P = np.column_stack([src, np.ones(len(src))])
    M = np.linalg.lstsq(P, dst, rcond=None)[0].T
    return M, np.linalg.norm(P @ M.T - dst, axis=1)


def loo_error(pairs):
    """Leave-one-out error in mm - the honest accuracy.

    In-sample residual is not: two different 4-pair sets each fitted their own points to 0.7 mm
    while disagreeing on scale by 30 %, because a near-collinear set bends to its own data and is
    wrong everywhere else. Their LOO errors were 27 and 45 mm.
    """
    if len(pairs) < 4:
        return float("nan")
    errs = []
    for i in range(len(pairs)):
        M, _ = fit(pairs[:i] + pairs[i + 1:])
        px, py = pairs[i][0]
        errs.append(float(np.linalg.norm(M @ np.array([px, py, 1.0]) - np.asarray(pairs[i][1], float))))
    return float(np.mean(errs))


def conditioning(A):
    """(isotropic scale mm/px, anisotropy). Anisotropy far from 1 means the points were too close
    to collinear to pin the fit down - and the residual on those same points will look fine."""
    A = np.asarray(A, float)[:, :2]
    n0, n1 = np.linalg.norm(A[:, 0]), np.linalg.norm(A[:, 1])
    return float(np.sqrt(abs(np.linalg.det(A)))), float(max(n0, n1) / max(min(n0, n1), 1e-9))
