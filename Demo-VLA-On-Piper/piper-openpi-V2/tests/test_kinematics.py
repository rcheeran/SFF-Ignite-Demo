"""piper/kinematics.py: the SDK DH chain, the limit-aware IK and reachable() (no hardware, no mujoco)."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from piper import config as C
from piper.kinematics import JOINT_LIMITS_DEG, fk, ik, reachable
from piper.transforms import euler_to_rot, fk_flange


def test_fk_matches_the_sdk():
    rng = np.random.default_rng(1)
    for q in rng.uniform(JOINT_LIMITS_DEG[:, 0], JOINT_LIMITS_DEG[:, 1], (100, 6)):
        sdk, T = fk_flange(q), fk(q)
        assert np.allclose(T[:3, 3], sdk[:3], atol=1e-6)
        assert np.allclose(T[:3, :3], euler_to_rot(sdk[3:]).as_matrix(), atol=1e-6)


def test_ik_round_trip_near_the_start():
    rng = np.random.default_rng(2)
    q0 = np.array(C.READY_JOINTS_DEG)
    for _ in range(30):
        q_t = np.clip(q0 + rng.uniform(-15, 15, 6), JOINT_LIMITS_DEG[:, 0] + 1, JOINT_LIMITS_DEG[:, 1] - 1)
        T = fk(q_t)
        q, ok = ik(T[:3, 3], Rot.from_matrix(T[:3, :3]), q0, iters=30)
        assert ok and np.linalg.norm(fk(q)[:3, 3] - T[:3, 3]) < 1.0


def test_reachable_projects_onto_the_joint_limit():
    q0 = np.array(C.READY_JOINTS_DEG)
    p = fk_flange(q0)
    xyz, rot = p[:3], euler_to_rot(p[3:])
    x2, r2, pinned = reachable(xyz + [20, 0, 0], rot, q0)                  # plain reach: untouched
    assert pinned == [] and np.allclose(x2, xyz + [20, 0, 0])
    far = Rot.from_euler("y", 60, degrees=True) * rot                        # pitch the tool 60 deg further: J5 > 70
    x3, r3, pinned = reachable(xyz, far, q0)
    assert pinned == ["J5"]
    q, _ = ik(x3, r3, q0, iters=30)
    assert (q >= JOINT_LIMITS_DEG[:, 0] - 0.6).all() and (q <= JOINT_LIMITS_DEG[:, 1] + 0.6).all()
    assert 0 < (r3.inv() * far).magnitude() * 180 / np.pi < 60               # short of the request, not frozen


def test_reachable_keeps_a_margin_inside_the_factory_limits():
    """The firmware refuses poses whose solution sits ON a limit (live 2026-08-30, J5 at +70): the projection
    must land strictly inside, so what we send is never borderline."""
    q0 = np.array(C.READY_JOINTS_DEG)
    T = fk(q0)
    down = Rot.from_matrix(T[:3, :3]) * Rot.from_euler("y", 80, degrees=True)   # far past J5's range
    x, r, pinned = reachable(T[:3, 3], down, q0)
    assert "J5" in pinned
    q, _ = ik(x, r, q0, JOINT_LIMITS_DEG)
    assert (q >= JOINT_LIMITS_DEG[:, 0] + C.IK_LIMIT_MARGIN_DEG - 0.6).all()
    assert (q <= JOINT_LIMITS_DEG[:, 1] - C.IK_LIMIT_MARGIN_DEG + 0.6).all()
