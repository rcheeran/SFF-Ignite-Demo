#!/usr/bin/env python3
"""Validate the Intel RealSense camera: confirm it's detected, and that
both color and depth streams produce real frames.

No arm/CAN commands are sent - this only touches the camera.

Saves a snapshot (camera_snapshot.jpg, next to this script) so you can
visually confirm what the camera sees.

Usage: python3 scripts/test-camera.py
"""
import os
import pyrealsense2 as rs
import numpy as np
import cv2

SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), "camera_snapshot.jpg")
NUM_FRAMES = 15  # frames to pull before trusting readings/snapshot (let auto-exposure settle)


def main():
    ctx = rs.context()
    devices = ctx.query_devices()
    print(f"Found {len(devices)} RealSense device(s)")
    if len(devices) == 0:
        print("ERROR: no RealSense device detected - check the USB connection")
        return
    for dev in devices:
        print(f"  Name: {dev.get_info(rs.camera_info.name)}")
        print(f"  Serial: {dev.get_info(rs.camera_info.serial_number)}")
        print(f"  Firmware: {dev.get_info(rs.camera_info.firmware_version)}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    print(f"\nStarting stream, pulling {NUM_FRAMES} frames...")
    profile = pipeline.start(config)
    try:
        color_img = depth_img = None
        for i in range(NUM_FRAMES):
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                print(f"  frame {i}: missing color or depth frame")
                continue
            color_img = np.asanyarray(color_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())
            center_depth_mm = depth_img[depth_img.shape[0] // 2, depth_img.shape[1] // 2]
            print(f"  frame {i}: color shape={color_img.shape}  depth shape={depth_img.shape}  "
                  f"center depth={center_depth_mm}mm")

        if color_img is None:
            print("\nERROR: never received a valid color+depth frame pair")
            return

        cv2.imwrite(SNAPSHOT_PATH, color_img)
        print(f"\nOK: streams validated. Snapshot saved to {SNAPSHOT_PATH}")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
