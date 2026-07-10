#!/usr/bin/env python3
"""Read-only connectivity check for the Piper arm over CAN.

No motion commands are sent. Connects with both pyAgxArm and piper_sdk,
reads firmware/status/joint angles, and reports whether the arm is
broadcasting its high-level status frames (CAN IDs 0x2A1-0x2A8: arm
status, joint angles, end pose, gripper status) in addition to its
low-level per-joint driver frames (0x251-0x256, 0x3A0-0x3A7, ...).

Usage: python3 scripts/check_connection.py [can_iface]
"""
import sys
import time

CAN_IFACE = sys.argv[1] if len(sys.argv) > 1 else "can0"


def check_pyagxarm():
    from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW

    print(f"\n=== pyAgxArm ({CAN_IFACE}) ===")
    cfg = create_agx_arm_config(robot=ArmModel.PIPER, firmeware_version=PiperFW.V183, channel=CAN_IFACE)
    robot = AgxArmFactory.create_arm(cfg)
    robot.connect()
    time.sleep(1.0)

    print("is_ok:", robot.is_ok())
    print("fps:", robot.get_fps())
    print("firmware:", robot.get_firmware())
    print("joint_enable_status:", robot.get_joints_enable_status_list())

    deadline = time.time() + 3
    arm_status = joint_angles = None
    while time.time() < deadline and (arm_status is None or joint_angles is None):
        arm_status = arm_status or robot.get_arm_status()
        joint_angles = joint_angles or robot.get_joint_angles()
        time.sleep(0.02)

    print("arm_status:", arm_status.msg if arm_status else "None (no 0x2A1 frame received)")
    print("joint_angles:", joint_angles.msg if joint_angles else "None (no 0x2A5/2A6/2A7 frame received)")

    robot.disconnect()


def check_piper_sdk():
    from piper_sdk import C_PiperInterface_V2

    print(f"\n=== piper_sdk ({CAN_IFACE}) ===")
    piper = C_PiperInterface_V2(can_name=CAN_IFACE)
    piper.ConnectPort()
    time.sleep(1.0)

    print("fps:", piper.GetCanFps())

    status = piper.GetArmStatus()
    angles = piper.GetArmJointMsgs()
    print("ArmStatus.Hz:", getattr(status, "Hz", "n/a"), "(0.0 means no real 0x2A1 frame received)")
    print(status)
    print("JointMsgs.Hz:", getattr(angles, "Hz", "n/a"))
    print(angles)


if __name__ == "__main__":
    check_pyagxarm()
    check_piper_sdk()

    print(
        "\nNote: if fps is ~1400-1500 but arm_status/joint_angles come back "
        "empty/zeroed from both SDKs, the arm's main controller is not "
        "transmitting its 0x2A1-0x2A8 status range. Confirm with:\n"
        f"  candump {CAN_IFACE} -n 500 | awk '{{print $2}}' | sort | uniq -c | sort -rn\n"
        "See README.md 'Known issue' section."
    )
