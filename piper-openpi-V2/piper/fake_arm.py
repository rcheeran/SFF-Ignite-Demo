"""Hardware-free stand-in for piper.robot.Piper (tests, inference --dry-run / --fake-arm).

Cartesian targets (send_pose) are followed with a first-order lag unless frozen=True, which mimics the
firmware IK silently refusing targets (the arm just stays put). Joint targets (send_joints) move the
joints and recompute the pose with the SDK forward kinematics, exactly like the real arm reports it.
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from piper import config as C
from piper.robot import JOINT_LIMITS_DEG
from piper.transforms import euler_to_rot, fk_flange, rot_to_euler


fk_pose = fk_flange   # [x, y, z (mm), rx, ry, rz (deg)] of the flange, piper_sdk FK


class FakeArm:
    def __init__(self, joints=C.READY_JOINTS_DEG, frozen=False, lag=0.35, temp=32, grip_mm=C.GRIP_OPEN_MM):
        self.j = np.array(joints, np.float64)
        self._pose = fk_pose(self.j)
        self.frozen, self.lag, self.temp = frozen, lag, temp
        self.grip = float(grip_mm)
        self.status = "ArmStatus.NORMAL"
        self.sent_poses, self.sent_joints = [], []   # every command, for assertions
        self.rested = False
        self._p = object()                           # "connected"

    # ---- feedback (same names/units as piper.robot.Piper) ----
    def pose(self):
        return self._pose.copy()

    def joints_deg(self):
        return self.j.copy()

    def grip_mm(self):
        return self.grip

    def j6_temp(self):
        return self.temp

    def arm_status(self):
        return self.status

    def ctrl_mode(self):
        return "CtrlMode.CAN_CTRL"

    def enabled(self):
        return [True] * 6

    def faults(self):
        return []

    def feedback_hz(self):
        return 200.0

    def state7(self):
        return np.append(self.j, self.grip).astype(np.float32)

    # ---- commands ----
    def send_pose(self, xyz, rpy_deg, grip_mm, speed=C.SPEED_TELEOP):
        self.sent_poses.append((np.array(xyz, np.float64), np.array(rpy_deg, np.float64), grip_mm))
        if not self.frozen:
            self._pose[:3] += self.lag * (np.asarray(xyz, np.float64) - self._pose[:3])
            r0, r1 = euler_to_rot(self._pose[3:]), euler_to_rot(rpy_deg)
            rv = (r0.inv() * r1).as_rotvec()
            self._pose[3:] = rot_to_euler(r0 * Rot.from_rotvec(self.lag * rv))
        self._set_grip(grip_mm)

    def send_joints(self, j_deg, grip_mm, speed=C.SPEED_JOINT):
        j_deg = np.clip(np.asarray(j_deg, np.float64), JOINT_LIMITS_DEG[:, 0], JOINT_LIMITS_DEG[:, 1])   # like Piper
        self.sent_joints.append((j_deg.copy(), grip_mm))
        if not self.frozen:
            self.j += self.lag * (j_deg - self.j)
            self._pose = fk_pose(self.j)
        self._set_grip(grip_mm)

    def send_gripper(self, grip_mm, speed=C.SPEED_JOINT):
        self._set_grip(grip_mm)

    def _set_grip(self, grip_mm):
        if grip_mm is not None and not self.frozen:
            self.grip += 0.5 * (float(np.clip(grip_mm, C.GRIP_CLOSED_MM, C.GRIP_OPEN_MM)) - self.grip)

    def move_joints(self, target, secs, grip_mm=None, speed=C.SPEED_JOINT, tol=C.HOME_TOL_DEG, settle=0.3):
        self.j = np.array(target, np.float64)
        self._pose = fk_pose(self.j)
        return True

    def gripper_enable(self, clear_errors=True):
        self.gripper_enabled = True

    def in_teach(self) -> bool:
        return getattr(self, "teaching", False)

    def wait_teach(self, on: bool, timeout=120.0, poll=0.2) -> bool:
        self.teaching = on           # no button to press: pretend it was pressed at once
        return True

    def teach_stop(self, timeout=6.0) -> bool:
        self.teaching = False
        return True

    def home(self, grip_mm=None):
        return self.move_joints(C.HOME_JOINTS_DEG, 0)

    def ready(self, grip_mm=None):
        return self.move_joints(C.READY_JOINTS_DEG, 0)

    def rest(self, soft=False):
        self.rested = True
        self._p = None

    def disconnect(self):
        self._p = None
