"""Rotation / image helpers shared by teleop, collection, training and inference.

Piper end-pose Euler angles (RX, RY, RZ) decompose the tool rotation as
R = Rz(RZ) * Ry(RY) * Rx(RX)  (verified against piper_sdk.kinematics.piper_fk),
which is scipy's extrinsic 'xyz' convention.
"""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from piper import config as C

EULER = "xyz"


def euler_to_rot(rpy_deg) -> Rot:
    return Rot.from_euler(EULER, rpy_deg, degrees=True)


def rot_to_euler(r: Rot) -> np.ndarray:
    return r.as_euler(EULER, degrees=True)


def rot_to_6d(r: Rot) -> np.ndarray:
    """First two columns of the rotation matrix (continuous, no wrap-around)."""
    m = r.as_matrix()
    return np.concatenate([m[:, 0], m[:, 1]]).astype(np.float32)


def sixd_to_rot(r6) -> Rot:
    """Gram-Schmidt the two columns back into a proper rotation."""
    a1, a2 = np.asarray(r6[:3], np.float64), np.asarray(r6[3:6], np.float64)
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 /= np.linalg.norm(b2)
    return Rot.from_matrix(np.stack([b1, b2, np.cross(b1, b2)], axis=1))


def pose7_to_10d(pose7) -> np.ndarray:
    """[x,y,z, rx,ry,rz (deg), grip] -> [x,y,z, R6d(6), grip] (policy state/action)."""
    p = np.asarray(pose7, np.float64)
    return np.concatenate([p[:3], rot_to_6d(euler_to_rot(p[3:6])), p[6:7]]).astype(np.float32)


def pose10_to_7(pose10) -> np.ndarray:
    p = np.asarray(pose10, np.float64)
    return np.concatenate([p[:3], rot_to_euler(sixd_to_rot(p[3:9])), p[9:10]])


def clamp_step(prev_xyz, target_xyz, max_mm):
    """Move from prev toward target by at most max_mm (per axis)."""
    prev_xyz, target_xyz = np.asarray(prev_xyz, np.float64), np.asarray(target_xyz, np.float64)
    return prev_xyz + np.clip(target_xyz - prev_xyz, -max_mm, max_mm)


def clamp_rot_step(prev_rot: Rot, target_rot: Rot, max_deg) -> Rot:
    """Rotate from prev toward target by at most max_deg (geodesic, in SO(3))."""
    rel = prev_rot.inv() * target_rot
    rv = rel.as_rotvec(degrees=True)
    ang = np.linalg.norm(rv)
    if ang > max_deg:
        rel = Rot.from_rotvec(rv / ang * max_deg, degrees=True)
    return prev_rot * rel


def fk_flange(joints_deg) -> np.ndarray:
    """[x, y, z (mm), rx, ry, rz (deg)] of the flange for the given joints (piper_sdk forward kinematics)."""
    from piper_sdk.kinematics import C_PiperForwardKinematics
    return np.array(C_PiperForwardKinematics().CalFK(np.radians(joints_deg).tolist())[-1], np.float64)


def tip_position(joints_deg) -> np.ndarray:
    """Gripper tip [x, y, z] mm: the flange plus TOOL_LEN_MM along the tool z-axis (the tool-tip floor
    for joint-space targets, e.g. policy deploy)."""
    p = fk_flange(joints_deg)
    return p[:3] + euler_to_rot(p[3:]).as_matrix()[:, 2] * C.TOOL_LEN_MM


def resize_with_pad(img, size=224):
    """Letterbox to a square, then resize - keeps aspect ratio (what pi0.5 was trained on)."""
    h, w = img.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    sq = cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left,
                            cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)
