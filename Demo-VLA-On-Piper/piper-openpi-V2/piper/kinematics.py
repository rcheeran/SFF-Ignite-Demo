"""Piper kinematics on the piper_sdk DH chain (pure numpy): forward kinematics, a damped-least-squares IK that
respects the joint limits, and reachable() - the teleop's answer to the firmware refusing a target.

The firmware IK refuses any Cartesian target that would push a joint past its range (arm_status
TARGET_POS_EXCEEDS_LIMIT) or that it cannot solve (NO_SOLUTION): the arm just holds. reachable() runs the same
problem here first, from the measured joints, and when it does not converge returns the nearest pose the arm CAN
take (the limit-clipped solution's FK) plus the joints sitting on their limits - so teleop slides along a limit
instead of freezing, and the headset can say which joint it is.
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from piper import config as C

JOINT_LIMITS_DEG = np.array(C.JOINT_LIMITS_DEG, np.float64)

# piper_sdk DH chain (C_PiperForwardKinematics, dh_is_offset=1) = the convention the real arm reports in
_DH_A = np.array([0.0, 0.0, 285.03, -21.98, 0.0, 0.0])
_DH_ALPHA = np.array([0.0, -np.pi / 2, 0.0, np.pi / 2, -np.pi / 2, np.pi / 2])
_DH_THETA = np.radians([0.0, -172.22, -102.78, 0.0, 0.0, 0.0])
_DH_D = np.array([123.0, 0.0, 0.0, 250.75, 0.0, 91.0])
WRIST_CENTRE_MM = _DH_D[5]   # the J4/J5/J6 axes meet this far behind the flange along tool z (a spherical wrist)


def fk(q_deg) -> np.ndarray:
    """4x4 flange transform (mm) of the piper_sdk DH chain - same numbers as C_PiperForwardKinematics."""
    T = np.eye(4)
    for a, al, th, d in zip(_DH_A, _DH_ALPHA, np.radians(q_deg) + _DH_THETA, _DH_D):
        ct, st, ca, sa = np.cos(th), np.sin(th), np.cos(al), np.sin(al)
        T = T @ np.array([[ct, -st, 0.0, a], [st * ca, ct * ca, -sa, -sa * d],
                          [st * sa, ct * sa, ca, ca * d], [0.0, 0.0, 0.0, 1.0]])
    return T


def _pose_err(q, xyz, Rt):
    """[dx, dy, dz (mm), rotation vector (deg)] from fk(q) to the target."""
    T = fk(q)
    return np.concatenate([xyz - T[:3, 3], Rot.from_matrix(Rt @ T[:3, :3].T).as_rotvec(degrees=True)])


def ik(xyz, rot: Rot, q0_deg, limits=JOINT_LIMITS_DEG, iters=C.SIM_IK_ITERS,
       tol_mm=C.SIM_IK_TOL_MM, tol_deg=C.SIM_IK_TOL_DEG):
    """Damped least squares on a numeric 6x6 Jacobian (mm / deg per deg), warm-started from q0, joints clipped
    to `limits` every iteration. Returns (q_deg, converged)."""
    xyz, Rt, h = np.asarray(xyz, np.float64), rot.as_matrix(), 0.05
    w, damp = np.array([1.0] * 3 + [C.SIM_IK_ROT_WEIGHT] * 3), C.SIM_IK_DAMP ** 2 * np.eye(6)
    winv = 1.0 / np.asarray(C.IK_JOINT_COST, np.float64)   # weighted least norm: expensive joints move less
    q = np.clip(np.asarray(q0_deg, np.float64), limits[:, 0], limits[:, 1])
    for _ in range(iters):
        e = _pose_err(q, xyz, Rt)
        if np.linalg.norm(e[:3]) < tol_mm and np.linalg.norm(e[3:]) < tol_deg:
            return q, True
        J = np.array([(e - _pose_err(q + np.eye(6)[i] * h, xyz, Rt)) / h for i in range(6)]).T * w[:, None]
        step = winv * (J.T @ np.linalg.solve(J @ (winv[:, None] * J.T) + damp, e * w))
        pinned = ((q <= limits[:, 0]) & (step < 0)) | ((q >= limits[:, 1]) & (step > 0))
        if pinned.any():   # a joint on its limit cannot help in that direction: drop its column
            J[:, pinned] = 0.0
            step = winv * (J.T @ np.linalg.solve(J @ (winv[:, None] * J.T) + damp, e * w))
        q = np.clip(q + step * min(1.0, 20.0 / max(np.abs(step).max(), 1e-9)), limits[:, 0], limits[:, 1])
    e = _pose_err(q, xyz, Rt)
    return q, bool(np.linalg.norm(e[:3]) < tol_mm and np.linalg.norm(e[3:]) < tol_deg)




MARGINED_LIMITS = JOINT_LIMITS_DEG + np.array([C.IK_LIMIT_MARGIN_DEG, -C.IK_LIMIT_MARGIN_DEG])


def reachable(xyz, rot: Rot, q0_deg, limits=MARGINED_LIMITS):
    """(xyz, rot, pinned): the target itself when the IK from q0 reaches it, else the nearest reachable pose and
    the names of the joints on their limits (e.g. ['J5']); pinned is [] and the pose unchanged when reachable.
    The limits carry IK_LIMIT_MARGIN_DEG of margin: the firmware refuses poses whose own solution lands exactly
    on a factory limit, so the projection must stay strictly inside."""
    q, ok = ik(xyz, rot, q0_deg, limits)
    if ok:
        return np.asarray(xyz, np.float64), rot, []
    T = fk(q)
    pinned = [f"J{i + 1}" for i in range(6) if q[i] <= limits[i, 0] + 0.5 or q[i] >= limits[i, 1] - 0.5]
    return T[:3, 3].copy(), Rot.from_matrix(T[:3, :3]), pinned
