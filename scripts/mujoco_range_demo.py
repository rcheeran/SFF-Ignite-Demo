#!/usr/bin/env python3
"""mujoco_range_demo.py - the joint range demo, in simulation. Sweeps each
Piper joint through its full mechanical range one at a time and writes an
animated GIF, with a top bar showing the joint, live angle, and range.

The safe counterpart to scripts/joint_range_demo.py (which moves the real arm):
this runs entirely in MuJoCo - no arm, no CAN, no tipping.

Requires:
    pip install mujoco imageio pillow numpy
    sudo apt install libosmesa6 libosmesa6-dev
Needs the Piper MJCF + meshes from the piper-asimov sim toolset. Override the
model path with --model.

Usage:
    python3 scripts/mujoco_range_demo.py
    python3 scripts/mujoco_range_demo.py --out piper_range_demo.gif --fps 11
"""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np  # noqa: E402
import mujoco  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

MODEL_CANDIDATES = [
    os.path.expanduser("~/Documents/piper-asimov/piper-sim-toolset/assets/piper_arm.xml"),
    os.path.expanduser("~/piper-asimov/piper-sim-toolset/assets/piper_arm.xml"),
]
NAMES = {0: "base yaw", 1: "shoulder", 2: "elbow",
         3: "forearm roll", 4: "wrist pitch", 5: "wrist roll"}


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
    ap = argparse.ArgumentParser(description="Animate each Piper joint's range in MuJoCo.")
    ap.add_argument("--model", help="path to piper_arm.xml")
    ap.add_argument("--out", default="piper_range_demo.gif", help="output GIF path")
    ap.add_argument("--fps", type=int, default=11, help="GIF frame rate (lower = slower)")
    ap.add_argument("--frames-per-leg", type=int, default=13, help="more = slower, smoother")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    model_path = resolve_model(args.model)
    os.chdir(os.path.dirname(model_path))

    m = mujoco.MjModel.from_xml_path(os.path.basename(model_path))
    d = mujoco.MjData(m)
    m.vis.headlight.active = 1
    m.vis.headlight.ambient[:] = [0.55, 0.55, 0.55]
    m.vis.headlight.diffuse[:] = [0.7, 0.7, 0.7]
    m.vis.headlight.specular[:] = [0.15, 0.15, 0.15]

    W, H = 520, 400
    m.vis.global_.offwidth = W
    m.vis.global_.offheight = H
    r = mujoco.Renderer(m, H, W)

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.32]
    cam.distance = 1.15
    cam.azimuth = 140
    cam.elevation = -12
    opt = mujoco.MjvOption()

    K = args.frames_per_leg
    HOLD = 6

    def grab(text):
        r.update_scene(d, camera=cam, scene_option=opt)
        pim = Image.fromarray(r.render())
        dr = ImageDraw.Draw(pim)
        dr.rectangle([0, 0, W, 26], fill=(20, 20, 22))
        dr.text((10, 8), text, fill=(240, 240, 240))
        return np.array(pim)

    frames = []
    for j in range(6):
        lo, hi = float(m.jnt_range[j][0]), float(m.jnt_range[j][1])
        path = np.concatenate([
            np.full(HOLD, 0.0),
            np.linspace(0.0, lo, K),
            np.linspace(lo, hi, 2 * K),
            np.linspace(hi, 0.0, K),
        ])
        for a in path:
            d.qpos[:] = 0.0
            d.qpos[j] = a
            mujoco.mj_forward(m, d)
            frames.append(grab(f"J{j+1}  {NAMES[j]}   {np.degrees(a):+4.0f} deg   "
                               f"(range {np.degrees(lo):.0f} to {np.degrees(hi):.0f})"))

    imageio.mimsave(out, frames, fps=args.fps, loop=0)
    print(f"saved {out}  frames={len(frames)}  {os.path.getsize(out)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
