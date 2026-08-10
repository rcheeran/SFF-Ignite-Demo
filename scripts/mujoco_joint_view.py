#!/usr/bin/env python3
"""mujoco_joint_view.py - render the AgileX Piper (real STL meshes) in MuJoCo
with each joint's rotation axis drawn as an arrow and labeled (joint1..joint6).

Headless offscreen software render (OSMesa), so it works over SSH / on the
Jetson without a display. No arm/CAN access - pure simulation.

Requires:
    pip install mujoco pillow numpy
    sudo apt install libosmesa6 libosmesa6-dev
Needs the Piper MJCF + meshes from the piper-asimov sim toolset
(piper_arm.xml + meshes/). By default it looks in the usual location;
override with --model.

Usage:
    python3 scripts/mujoco_joint_view.py
    python3 scripts/mujoco_joint_view.py --model /path/to/piper_arm.xml --out piper_joints.png
"""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np  # noqa: E402
import mujoco  # noqa: E402
from PIL import Image  # noqa: E402

MODEL_CANDIDATES = [
    os.path.expanduser("~/Documents/piper-asimov/piper-sim-toolset/assets/piper_arm.xml"),
    os.path.expanduser("~/piper-asimov/piper-sim-toolset/assets/piper_arm.xml"),
]


def resolve_model(path):
    if path:
        if not os.path.exists(path):
            sys.exit(f"model not found: {path}")
        return os.path.abspath(path)
    for c in MODEL_CANDIDATES:
        if os.path.exists(c):
            return c
    sys.exit("Piper MJCF not found. Pass --model /path/to/piper_arm.xml")


def main():
    ap = argparse.ArgumentParser(description="Render the Piper with labeled joint axes (MuJoCo).")
    ap.add_argument("--model", help="path to piper_arm.xml")
    ap.add_argument("--out", default="piper_joints.png", help="output image path")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    model_path = resolve_model(args.model)
    os.chdir(os.path.dirname(model_path))  # so meshdir="meshes/" resolves

    m = mujoco.MjModel.from_xml_path(os.path.basename(model_path))
    d = mujoco.MjData(m)

    m.vis.headlight.active = 1
    m.vis.headlight.ambient[:] = [0.6, 0.6, 0.6]
    m.vis.headlight.diffuse[:] = [0.72, 0.72, 0.72]
    m.vis.headlight.specular[:] = [0.15, 0.15, 0.15]
    m.vis.scale.jointlength = 2.2
    m.vis.scale.jointwidth = 0.16

    d.qpos[0:6] = [0.0, 0.75, -1.0, 0.0, 0.85, 0.0]  # spread the joints
    mujoco.mj_forward(m, d)

    W, H = 1000, 780
    m.vis.global_.offwidth = W
    m.vis.global_.offheight = H
    r = mujoco.Renderer(m, H, W)

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.06, 0.0, 0.30]
    cam.distance = 0.78
    cam.azimuth = 140
    cam.elevation = -10

    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    opt.label = mujoco.mjtLabel.mjLABEL_JOINT.value

    r.update_scene(d, camera=cam, scene_option=opt)
    Image.fromarray(r.render()).save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
