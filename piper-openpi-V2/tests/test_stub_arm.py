import numpy as np

from piper import config as C
from piper.fake_arm import FakeArm, fk_pose


def test_stub_follows_pose_and_freezes():
    arm = FakeArm()
    p0 = arm.pose()
    for _ in range(30):
        arm.send_pose(p0[:3] + [20, 0, 0], p0[3:], 70)
    assert abs(arm.pose()[0] - (p0[0] + 20)) < 0.5
    frozen = FakeArm(frozen=True)
    for _ in range(30):
        frozen.send_pose(p0[:3] + [20, 0, 0], p0[3:], 70)
    assert np.allclose(frozen.pose(), p0)


def test_stub_joints_drive_fk():
    arm = FakeArm()
    assert np.allclose(arm.pose(), fk_pose(C.READY_JOINTS_DEG))
    for _ in range(40):
        arm.send_joints(C.HOME_JOINTS_DEG, 70)
    assert np.allclose(arm.joints_deg(), C.HOME_JOINTS_DEG, atol=0.1)
