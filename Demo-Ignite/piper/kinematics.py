"""Piper forward and inverse kinematics using the SDK's DH convention."""

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from . import config as C

JOINT_LIMITS_DEG = np.array(C.JOINT_LIMITS_DEG, np.float64)

_DH_A = np.array([0.0, 0.0, 285.03, -21.98, 0.0, 0.0])
_DH_ALPHA = np.array([0.0, -np.pi / 2, 0.0, np.pi / 2, -np.pi / 2, np.pi / 2])
_DH_THETA = np.radians([0.0, -172.22, -102.78, 0.0, 0.0, 0.0])
_DH_D = np.array([123.0, 0.0, 0.0, 250.75, 0.0, 91.0])


def fk(q_deg) -> np.ndarray:
    """Return the flange transform (mm) matching SDK forward kinematics."""
    T = np.eye(4)
    for a, al, th, d in zip(_DH_A, _DH_ALPHA, np.radians(q_deg) + _DH_THETA, _DH_D):
        ct, st, ca, sa = np.cos(th), np.sin(th), np.cos(al), np.sin(al)
        T = T @ np.array([[ct, -st, 0.0, a], [st * ca, ct * ca, -sa, -sa * d],
                          [st * sa, ct * sa, ca, ca * d], [0.0, 0.0, 0.0, 1.0]])
    return T


def _pose_err(q, xyz, Rt):
    """Return position (mm) and rotation-vector (deg) error at joint angles q."""
    T = fk(q)
    return np.concatenate([xyz - T[:3, 3], Rot.from_matrix(Rt @ T[:3, :3].T).as_rotvec(degrees=True)])


def ik(xyz, rot: Rot, q0_deg, limits=JOINT_LIMITS_DEG, iters=C.SIM_IK_ITERS,
       tol_mm=C.SIM_IK_TOL_MM, tol_deg=C.SIM_IK_TOL_DEG):
    """Solve seeded, joint-limit-constrained IK with damped least squares."""
    xyz, Rt, h = np.asarray(xyz, np.float64), rot.as_matrix(), 0.05
    w, damp = np.array([1.0] * 3 + [C.SIM_IK_ROT_WEIGHT] * 3), C.SIM_IK_DAMP ** 2 * np.eye(6)
    winv = 1.0 / np.asarray(C.IK_JOINT_COST, np.float64)
    q = np.clip(np.asarray(q0_deg, np.float64), limits[:, 0], limits[:, 1])
    for _ in range(iters):
        e = _pose_err(q, xyz, Rt)
        if np.linalg.norm(e[:3]) < tol_mm and np.linalg.norm(e[3:]) < tol_deg:
            return q, True
        J = np.array([(e - _pose_err(q + np.eye(6)[i] * h, xyz, Rt)) / h for i in range(6)]).T * w[:, None]
        step = winv * (J.T @ np.linalg.solve(J @ (winv[:, None] * J.T) + damp, e * w))
        pinned = ((q <= limits[:, 0]) & (step < 0)) | ((q >= limits[:, 1]) & (step > 0))
        if pinned.any():
            J[:, pinned] = 0.0
            step = winv * (J.T @ np.linalg.solve(J @ (winv[:, None] * J.T) + damp, e * w))
        q = np.clip(q + step * min(1.0, 20.0 / max(np.abs(step).max(), 1e-9)), limits[:, 0], limits[:, 1])
    e = _pose_err(q, xyz, Rt)
    return q, bool(np.linalg.norm(e[:3]) < tol_mm and np.linalg.norm(e[3:]) < tol_deg)


MARGINED_LIMITS = JOINT_LIMITS_DEG + np.array([C.IK_LIMIT_MARGIN_DEG, -C.IK_LIMIT_MARGIN_DEG])
