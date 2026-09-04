"""LeRobot Robot plugin: joint-space actions (JointCtrl) + two cameras, per the README dataset contract.

    from piper.lerobot_robot import PiperRobot, PiperRobotConfig
    robot = PiperRobot(PiperRobotConfig()); robot.connect()   # arm lock + enable (connect_for_motion)
    obs = robot.get_observation()            # j1..j6, gripper_mm, top, wrist
    robot.send_action({"j1": .., ..., "gripper_mm": ..})
    robot.disconnect()                       # = arm.rest(): home, de-torque, unlock
"""
from dataclasses import dataclass

import cv2
import numpy as np
from lerobot.robots import Robot, RobotConfig

from piper import config as C
from piper.cameras import FakeCamera, open_cameras
from piper.robot import Piper, clamp_joints, connect_for_motion


@RobotConfig.register_subclass("piper")
@dataclass
class PiperRobotConfig(RobotConfig):
    can_iface: str = C.CAN_IFACE
    top_cam: str = C.TOP_CAM_DEV
    speed: int = C.SPEED_JOINT
    fake_cameras: bool = False


class PiperRobot(Robot):
    config_class = PiperRobotConfig
    name = C.ROBOT_TYPE

    def __init__(self, config: PiperRobotConfig):
        super().__init__(config)
        self.config = config
        self.arm = Piper(config.can_iface)
        self.cameras = {}
        self._prev = None    # last commanded joints (clamp_joints reference); the first tick starts from the measured joints

    def __del__(self) -> None:
        pass   # lerobot's Robot.__del__ would run rest() (a HOME move) during interpreter teardown; rest is explicit via disconnect()

    @property
    def observation_features(self) -> dict:
        return {**dict.fromkeys(C.STATE_NAMES, float),
                **{cam: (C.IMG_H, C.IMG_W, 3) for cam in C.CAMERA_NAMES}}

    @property
    def action_features(self) -> dict:
        return dict.fromkeys(C.STATE_NAMES, float)

    @property
    def is_connected(self) -> bool:
        return self.arm._p is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        # cameras first: a camera failure must not leave the arm enabled (same order as record.py / run_policy.py)
        self.cameras = ({n: FakeCamera(n) for n in C.CAMERA_NAMES} if self.config.fake_cameras
                        else open_cameras(C.CAMERA_NAMES, top_dev=self.config.top_cam))
        self.arm = connect_for_motion(iface=self.config.can_iface)

    def get_observation(self) -> dict:
        obs = dict(zip(C.STATE_NAMES, self.arm.state7().tolist()))
        for name, cam in self.cameras.items():
            obs[name] = cv2.cvtColor(cam.read(), cv2.COLOR_BGR2RGB)
        return obs

    def send_action(self, action: dict) -> dict:
        meas = self.arm.joints_deg()
        target = [float(action[n]) for n in C.STATE_NAMES[:6]]
        joints, _ = clamp_joints(meas if self._prev is None else self._prev, meas, target)   # same clamps as deploy
        grip = float(np.clip(action["gripper_mm"], C.GRIP_CLOSED_MM, C.GRIP_OPEN_MM))
        self.arm.send_joints(joints, grip, self.config.speed)
        self._prev = joints
        return {**dict(zip(C.STATE_NAMES[:6], joints.tolist())), "gripper_mm": grip}

    def disconnect(self) -> None:
        for cam in self.cameras.values():
            cam.close()
        self.cameras = {}
        self._prev = None
        self.arm.rest()


if __name__ == "__main__":
    import time
    robot = PiperRobot(PiperRobotConfig())
    print("observation_features:", robot.observation_features)
    print("action_features:", robot.action_features)
    try:
        robot.connect()
        t0 = time.time()
        obs = robot.get_observation()
        print(f"get_observation {1000 * (time.time() - t0):.0f} ms:",
              {k: (v.shape if hasattr(v, "shape") else round(v, 2)) for k, v in obs.items()})
    finally:
        robot.disconnect()   # rest() is a no-op when the arm never connected
