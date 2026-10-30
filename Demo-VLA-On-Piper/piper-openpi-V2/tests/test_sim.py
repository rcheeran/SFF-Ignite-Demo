"""MuJoCo digital twin (sim/): model, FK convention, IK, Cartesian tracking, limits, refusal, gripper, cameras,
scripted expert. Runs headless (EGL) in well under 90 s; one physics thread is shared by the whole module."""
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from piper import config as C
from piper.transforms import euler_to_rot
from sim import piper_sim                      # sets MUJOCO_GL before mujoco is imported (headless EGL)
from sim.piper_sim import SimPiper, fk, ik, wiggle
from piper.fake_arm import fk_pose

import mujoco  # noqa: E402


@pytest.fixture(scope="module")
def sim():
    arm = SimPiper().connect()
    yield arm
    arm.disconnect()


def test_model_and_fk_convention(sim):
    """Own DH FK == piper_sdk FK exactly; MJCF link6 (with the J2/J3 offset map) matches it to < 2 mm / 0.05 deg."""
    assert sim.m.nu == 7 and [mujoco.mj_id2name(sim.m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(sim.m.ncam)] == ["top", "wrist"]
    d, b6 = mujoco.MjData(sim.m), mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_BODY, "link6")
    rng, dp, dr = np.random.default_rng(0), [], []
    for q in rng.uniform(sim.limits[:, 0], sim.limits[:, 1], (300, 6)):
        sdk, T = fk_pose(q), fk(q)
        assert np.allclose(T[:3, 3], sdk[:3], atol=1e-6)
        assert np.allclose(T[:3, :3], euler_to_rot(sdk[3:]).as_matrix(), atol=1e-6)
        d.qpos[sim._qadr[:6]] = np.radians(q + piper_sim.OFFSET)
        mujoco.mj_kinematics(sim.m, d)
        dp.append(np.linalg.norm(d.xpos[b6] * 1000 - sdk[:3]))
        dr.append(np.linalg.norm((Rot.from_matrix(d.xmat[b6].reshape(3, 3)).inv() * euler_to_rot(sdk[3:])).as_rotvec(degrees=True)))
    assert max(dp) < 2.0 and max(dr) < 0.05, (max(dp), max(dr))


def test_ik_round_trip(sim):
    rng = np.random.default_rng(1)
    for _ in range(50):
        q = rng.uniform(sim.limits[:, 0], sim.limits[:, 1])
        T = fk(q)
        q0 = np.clip(q + rng.uniform(-15, 15, 6), sim.limits[:, 0], sim.limits[:, 1])   # warm start, like streaming
        qi, ok = ik(T[:3, 3], Rot.from_matrix(T[:3, :3]), q0, sim.limits, iters=50, tol_mm=0.2, tol_deg=0.1)
        Ti = fk(qi)
        assert ok and np.linalg.norm(Ti[:3, 3] - T[:3, 3]) < 1.0
        assert np.linalg.norm(Rot.from_matrix(Ti[:3, :3] @ T[:3, :3].T).as_rotvec(degrees=True)) < 0.5


TRACK_POSE = [0.0, 40.0, -20.0, 0.0, 25.0, 0.0]   # the READY the real arm's tracking was measured at (flange
                                                     # 126 mm out). At READY_JOINTS_DEG (flange 84 mm from the
                                                     # base axis) +-15 mm of y runs J6 into its -179 limit.


def test_send_pose_tracks(sim):
    assert sim.ready(C.GRIP_OPEN_MM)
    assert np.allclose(sim.pose(), fk_pose(sim.joints_deg()))          # pose() is the SDK FK of the sim joints
    assert sim.move_joints(TRACK_POSE, C.HOME_TIMEOUT_S, None, speed=C.SPEED_HOME)
    out = wiggle(sim, secs=6.0, amp_mm=15.0, amp_deg=8.0)            # teleop-tick amplitude
    assert out.pop("refused") == 0.0
    for axis, (cmd_amp, meas_amp, lag_ms, rms) in out.items():
        assert meas_amp >= 0.8 * cmd_amp and lag_ms < 300, (axis, out[axis])


def test_j6_limit_and_refusal(sim):
    sim.send_joints([0, 30, -15, 0, 40, 150], None)
    assert sim._q_cmd[5] == C.JOINT_LIMITS_DEG[5][1]          # factory J6 limit (+120)
    sim.send_joints(C.READY_JOINTS_DEG, None)
    time.sleep(1.5)
    p = sim.pose()
    for _ in range(5):
        sim.send_pose(p[:3] + [2000, 0, 0], p[3:], C.GRIP_OPEN_MM)
    assert sim.arm_status() != piper_sim.NORMAL and sim.faults() == [sim.arm_status().split("(")[0]]
    time.sleep(0.5)
    assert np.linalg.norm(sim.pose()[:3] - p[:3]) < 3.0                # refused = held
    sim.send_pose(p[:3], p[3:], C.GRIP_OPEN_MM)
    assert sim.arm_status() == piper_sim.NORMAL and sim.faults() == []


def test_gripper_mapping(sim):
    for mm in (20.0, 70.0, 0.0):
        sim.send_gripper(mm)
        time.sleep(0.9)
        assert abs(sim.grip_mm() - mm) < 2.0
        assert np.isclose(sim.grip_mm(), sim.d.qpos[sim._qadr[6]] * 2000.0)   # 0..70 mm opening = 2 x finger travel
    assert sim.state7().shape == (7,) and sim.state7().dtype == np.float32


def test_cameras(sim):
    from sim.cameras import open_sim_cameras
    cams = open_sim_cameras()
    frames = {n: cam.read() for n, cam in cams.items()}
    for n, f in frames.items():
        assert f.shape == (C.IMG_H, C.IMG_W, 3) and f.dtype == np.uint8 and f.std() > 10, n
    assert not np.array_equal(frames["top"], frames["wrist"])
    for cam in cams.values():
        cam.close()


def test_expert_episode(sim):
    from sim.expert import run_episodes
    assert run_episodes(sim, 1, seed=0) == [True]


def test_teleop_slides_along_a_joint_limit_instead_of_freezing(sim):
    """Headset pitches the hand far beyond what J5 can give while moving: the target is projected onto the
    nearest reachable pose every tick - the twin keeps following (no refusal, no STUCK), reports limit=J5."""
    from scipy.spatial.transform import Rotation as Rot
    from teleop.quest import QuestTeleop
    sim.ready(C.GRIP_OPEN_MM)
    t = QuestTeleop(sim, armed=True).start()
    head = {"p": [0.0, 1.6, 0.0], "q": [0.0, 0.0, 0.0, 1.0]}
    p0, seen, refused, j5_dev, overshoot = np.array([0.2, 1.2, -0.4]), set(), 0, 0.0, 0.0
    try:
        t0 = time.monotonic()
        while (s := time.monotonic() - t0) < 4.0:
            ang = 120.0 * np.sin(2 * np.pi * s / 4.0)                        # +-120 deg pitch: past J5 both ways
            q = Rot.from_euler("x", ang, degrees=True).as_quat().tolist()    # XR x = robot -y: a tool pitch
            p = (p0 + [0.0, 0.0, -0.05 * min(1.0, s / 2.0)]).tolist()         # and 5 cm forward
            t.feed({"t": s, "head": head, "right": {"p": p, "q": q, "trigger": 0.0, "squeeze": 1.0,
                                                     "a": False, "b": False, "thumb": False, "axes": [0, 0]}}, "test")
            seen.update(t.limit)
            refused += sim.arm_status() != piper_sim.NORMAL
            j = sim.joints_deg()
            j5_dev = max(j5_dev, abs(j[4] - C.READY_JOINTS_DEG[4]))
            overshoot = max(overshoot, float(np.maximum(sim.limits[:, 0] - j, j - sim.limits[:, 1]).max()))
            time.sleep(1 / 60)
    finally:
        t.stop()
    assert "J5" in seen, seen                                                # the wrist pitch limit was reached
    assert refused == 0, refused                                             # and never refused: projected targets
    assert "stuck" not in [e for e in list(t.events.queue)]
    assert j5_dev > 20 and overshoot < 3.0, (j5_dev, overshoot)             # followed up to the limit; the PD joints
                                                                             # overshoot a clipped command by < 3 deg


def test_wrist_camera_sits_on_top_at_j6_zero(sim):
    """The real D405 is on top of the gripper at J6 = 0 (centre of the wrist's range): the twin's camera must be
    too - on link6's -x side (world up at HOME) with the image's up vector pointing up."""
    assert sim.move_joints(C.HOME_JOINTS_DEG, 8.0, None)                     # blocking: arrive first
    cid = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
    b6 = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_BODY, "link6")
    with sim.lock:
        R6, p6 = sim.d.xmat[b6].reshape(3, 3), sim.d.xpos[b6]
        Rc, pc = sim.d.cam_xmat[cid].reshape(3, 3), sim.d.cam_xpos[cid]
    assert (R6.T @ (pc - p6))[0] < -0.05                                     # on the -x (top) side of the flange
    assert Rc[2, 1] > 0.7 and -Rc[0, 2] > 0.7                                 # image up ~ world up, looking forward
