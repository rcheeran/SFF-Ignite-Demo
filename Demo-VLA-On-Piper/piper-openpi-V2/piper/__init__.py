def __getattr__(name):
    """Lazy export so lerobot's make_robot_from_config(PiperRobotConfig()) finds PiperRobot here."""
    if name in ("PiperRobot", "PiperRobotConfig"):
        from piper import lerobot_robot
        return getattr(lerobot_robot, name)
    raise AttributeError(name)
