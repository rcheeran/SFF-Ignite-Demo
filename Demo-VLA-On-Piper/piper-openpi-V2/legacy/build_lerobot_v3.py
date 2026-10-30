#!/usr/bin/env python3
"""Convert auto_collect episodes into an openpi-ready LeRobot dataset (3 cameras).

Per frame:
  top_image    Realtek static  -> base_0_rgb        (global reference)
  wrist_image  D405            -> left_wrist_0_rgb  (close-range alignment)
  side_image   HP top-down     -> right_wrist_0_rgb (overhead layout)
  state        [x, y, z, R6d(6), gripper]           10-D
  actions      the NEXT kept state (absolute; the train config turns it into a delta)

Rotation is stored as continuous 6D (first two columns of the rotation matrix),
not Euler: raw Euler wraps +/-180 and produced systematic jumps in the human
demos. Frames are distance-resampled - drag/pause frames where nothing moves
teach the model to do nothing.

Episodes whose meta.json says ok=false (grasp missed) are skipped.

Run in ~/pi05-venv:  python build_lerobot_v3.py [--raw DIR] [--repo-id ID]
"""
import argparse
import glob
import json
import os
import shutil

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 15
IMG = 224
MIN_TRANS_MM = 1.0
MIN_ROT_DEG = 0.5
CAMS = ("top", "wrist", "side")


def euler_to_6d(rpy_deg):
    M = Rot.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    return np.concatenate([M[:, 0], M[:, 1]]).astype(np.float32)


def state_to_10d(s7):
    return np.concatenate([s7[:3], euler_to_6d(s7[3:6]), s7[6:7]]).astype(np.float32)


def rot_dist_deg(e1, e2):
    r1 = Rot.from_euler("xyz", e1, degrees=True)
    r2 = Rot.from_euler("xyz", e2, degrees=True)
    return np.degrees((r1.inv() * r2).magnitude())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(HERE, "data", "raw", "pick_soda_can_v3"))
    ap.add_argument("--root", default=os.path.join(HERE, "lerobot", "pick_soda_can_v3"))
    ap.add_argument("--repo-id", default="piper/pick_soda_can_v3")
    ap.add_argument("--prompt", default="pick up the soda can")
    ap.add_argument("--keep-bad", action="store_true", help="include episodes flagged ok=false")
    args = ap.parse_args()

    eps = sorted(e for e in glob.glob(os.path.join(args.raw, "episode_*")) if os.path.isdir(e))
    if not eps:
        raise SystemExit(f"no episodes in {args.raw}")

    kept_eps, skipped = [], []
    for ep in eps:
        mp = os.path.join(ep, "meta.json")
        ok = True
        if os.path.exists(mp):
            ok = bool(json.load(open(mp)).get("ok", True))
        (kept_eps if (ok or args.keep_bad) else skipped).append(ep)
    print(f"{len(eps)} episodes: {len(kept_eps)} usable, {len(skipped)} skipped (failed grasp)")

    if os.path.exists(args.root):
        shutil.rmtree(args.root)

    feats = {f"{c}_image": {"dtype": "image", "shape": (IMG, IMG, 3),
                            "names": ["height", "width", "channel"]} for c in CAMS}
    feats["state"] = {"dtype": "float32", "shape": (10,), "names": ["state"]}
    feats["actions"] = {"dtype": "float32", "shape": (10,), "names": ["actions"]}

    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=FPS, root=args.root,
                               robot_type="piper", features=feats,
                               use_videos=False, image_writer_threads=8)

    total = 0
    for ep in kept_eps:
        rows = [json.loads(l) for l in open(os.path.join(ep, "obs.jsonl"))]
        S = np.array([r["state"] for r in rows], dtype=np.float32)
        if len(S) < 3:
            print(f"  {os.path.basename(ep)}: too short, skipped")
            continue
        # distance-based resampling: keep a frame only after real motion
        keep = [0]
        for t in range(1, len(S)):
            moved = np.linalg.norm(S[t, :3] - S[keep[-1], :3]) >= MIN_TRANS_MM
            rotated = rot_dist_deg(S[t, 3:6], S[keep[-1], 3:6]) >= MIN_ROT_DEG
            gripped = S[t, 6] != S[keep[-1], 6]
            if moved or rotated or gripped:
                keep.append(t)
        S10 = np.array([state_to_10d(S[t]) for t in keep], dtype=np.float32)

        n = 0
        for i in range(len(keep) - 1):
            imgs = {}
            missing = False
            for c in CAMS:
                fp = os.path.join(ep, "frames", f"{c}_{keep[i]:06d}.jpg")
                im = cv2.imread(fp)
                if im is None:
                    missing = True
                    break
                imgs[f"{c}_image"] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if missing:
                continue
            ds.add_frame({**imgs, "state": S10[i], "actions": S10[i + 1], "task": args.prompt})
            n += 1
        ds.save_episode()
        total += n
        print(f"  {os.path.basename(ep)}: kept {n}/{len(S)-1} frames")

    print(f"\nDONE: {len(kept_eps)} episodes, {total} frames -> {args.root}")
    print(f"repo_id={args.repo_id}")
    print("next: symlink into ~/.cache/huggingface/lerobot/, then compute_norm_stats.py")


if __name__ == "__main__":
    main()
