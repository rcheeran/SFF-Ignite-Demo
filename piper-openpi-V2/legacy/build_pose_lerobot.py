#!/usr/bin/env python3
"""Convert the raw drag-teach episodes into an openpi-format LeRobot dataset.

Per frame:
  wrist_image : D405 RGB
  state       : [x, y, z, rx, ry, rz, gripper]   (mm, deg, m)
  actions     : the NEXT frame's state (absolute pose+gripper target)
  task        : the language prompt

The last frame of each episode is dropped (no next state to be its action).

Run in ~/pi05-venv (has lerobot). Output: a local LeRobot dataset (no hub push).
"""
import glob
import json
import os
import shutil

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def euler_to_6d(rpy_deg):
    """Euler (rx,ry,rz deg, 'xyz') -> continuous 6D rotation (first 2 matrix cols)."""
    M = Rot.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    return np.concatenate([M[:, 0], M[:, 1]]).astype(np.float32)


def state_to_10d(s7):
    """[x,y,z, rx,ry,rz(deg), grip] -> [x,y,z, R6d(6), grip]."""
    return np.concatenate([s7[:3], euler_to_6d(s7[3:6]), s7[6:7]]).astype(np.float32)

RAW = os.path.expanduser("~/Documents/piper-openpi/data/raw/pick_soda_can")
ROOT = os.path.expanduser("~/Documents/piper-openpi/lerobot/pick_soda_can")
REPO = "piper/pick_soda_can"
PROMPT = "pick up the soda can"
FPS = 15
USE_VIDEOS = False   # store images (avoids aarch64 video-decode issues at train time)
IMG = 224            # store at the model's input size (aspect-preserving pad)


def resize_with_pad(bgr_or_rgb, size=IMG):
    h, w = bgr_or_rgb.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    sq = cv2.copyMakeBorder(bgr_or_rgb, top, s - h - top, left, s - w - left,
                            cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)

eps = sorted(e for e in glob.glob(os.path.join(RAW, "episode_*")) if os.path.isdir(e))
print(f"{len(eps)} episodes -> {ROOT}")

# --- sanity: rotation wraparound check (Euler angles) ---
maxjump = 0.0
for ep in eps:
    S = np.array([json.loads(l)["state"] for l in open(os.path.join(ep, "obs.jsonl"))], dtype=np.float32)
    if len(S) > 1:
        maxjump = max(maxjump, float(np.abs(np.diff(S[:, 3:6], axis=0)).max()))
print(f"raw-Euler max per-step rotation change: {maxjump:.1f} deg "
      f"-> converting rotation to continuous 6D (handles the +/-180 wraparound).")

if os.path.exists(ROOT):
    shutil.rmtree(ROOT)

ds = LeRobotDataset.create(
    repo_id=REPO,
    fps=FPS,
    root=ROOT,
    robot_type="piper",
    features={
        "wrist_image": {"dtype": "image", "shape": (IMG, IMG, 3), "names": ["height", "width", "channel"]},
        "state": {"dtype": "float32", "shape": (10,), "names": ["state"]},   # [xyz, R6d, grip]
        "actions": {"dtype": "float32", "shape": (10,), "names": ["actions"]},
    },
    use_videos=USE_VIDEOS,
    image_writer_threads=8,
)

MIN_TRANS_MM = 1.0   # keep a frame once the arm has moved this far since the last kept one
MIN_ROT_DEG = 0.5    # ... or rotated this much
def rot_dist_deg(e1, e2):
    r1 = Rot.from_euler("xyz", e1, degrees=True)
    r2 = Rot.from_euler("xyz", e2, degrees=True)
    return np.degrees((r1.inv() * r2).magnitude())


total = 0
for ep in eps:
    L = [json.loads(l) for l in open(os.path.join(ep, "obs.jsonl"))]
    S = np.array([r["state"] for r in L], dtype=np.float32)          # (T, 7) euler
    # Distance-based resampling: drag-teach pauses make ~half the frames
    # near-stationary, which teaches the model "do nothing". Keep a frame only
    # once real motion (or a gripper toggle) happened since the last kept one.
    keep = [0]
    for t in range(1, len(S)):
        moved = np.linalg.norm(S[t, :3] - S[keep[-1], :3]) >= MIN_TRANS_MM
        rotated = rot_dist_deg(S[t, 3:6], S[keep[-1], 3:6]) >= MIN_ROT_DEG
        gripped = S[t, 6] != S[keep[-1], 6]
        if moved or rotated or gripped:
            keep.append(t)
    S10 = np.array([state_to_10d(S[t]) for t in keep], dtype=np.float32)  # (K, 10)
    K = len(keep)
    for i in range(K - 1):
        f = os.path.join(ep, "frames", f"frame_{keep[i]:06d}.jpg")
        img = resize_with_pad(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB))
        ds.add_frame({
            "wrist_image": img,
            "state": S10[i],
            "actions": S10[i + 1],   # absolute next KEPT pose (config makes it a delta)
            "task": PROMPT,
        })
    ds.save_episode()
    total += K - 1
    print(f"  {os.path.basename(ep)}: kept {K - 1}/{len(S) - 1} frames")

print(f"\nDONE: {len(eps)} episodes, {total} frames -> {ROOT}")
