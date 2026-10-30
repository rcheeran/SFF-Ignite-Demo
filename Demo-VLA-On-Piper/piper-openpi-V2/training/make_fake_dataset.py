"""Write a small synthetic LeRobot dataset that matches the dataset contract (piper/config.py, docs/CONTRACT.md).

python -m training.make_fake_dataset [--root data/fake_block_bowl] [--repo-id local/fake_block_bowl] [--episodes 3] [--frames 60]

This is the reference for collect/record.py. The LeRobotDataset (lerobot 0.6.1) facts
that matter when recording:

* features: images are declared with shape (H, W, C) and names ["height","width","channels"],
  dtype "video" (encoded to mp4 by pyav/libsvtav1 at save_episode time, no ffmpeg binary).
* add_frame(frame): frame must contain EVERY declared feature plus "task" (a str), and must
  NOT contain "timestamp"/"frame_index"/"episode_index"/"index" (computed from fps).
  - image values: np.uint8 array, HWC or CHW both accepted, RGB order (OpenCV gives BGR -> convert).
  - numeric values: np.ndarray with the exact declared dtype and shape (float32, (7,)).
  - frames accumulate in an in-memory episode buffer; images are written to PNG by
    image_writer_threads in the background, so add_frame is cheap enough for 15 Hz.
* save_episode(): encodes the buffered episode to video + parquet and resets the buffer.
  clear_episode_buffer() drops a bad episode instead.
* finalize(): REQUIRED once at the end, otherwise parquet footers/metadata are not written.
* push_to_hub(private=..., tags=[...]) after finalize() uploads to repo_id (needs HF_TOKEN).
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from piper import config as C

IMG_SHAPE = (C.IMG_H, C.IMG_W, 3)
IMG = {"dtype": "video", "shape": IMG_SHAPE, "names": ["height", "width", "channels"]}
FEATURES = {**{f"observation.images.{n}": IMG for n in C.CAMERA_NAMES},
            "observation.state": {"dtype": "float32", "shape": (7,), "names": C.STATE_NAMES},
            "action": {"dtype": "float32", "shape": (7,), "names": C.STATE_NAMES}}
CENTER = np.array([*C.READY_JOINTS_DEG, C.GRIP_OPEN_MM / 2], dtype=np.float32)   # sinusoids around READY, gripper 0..70


def fake_image(t: int, ep: int, cam: int) -> np.ndarray:
    h, w, _ = IMG_SHAPE
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    phase = 2 * np.pi * t / 60
    r = 127 + 127 * np.sin(xx / w * 2 * np.pi + phase)
    g = 127 + 127 * np.cos(yy / h * 2 * np.pi - phase)
    b = np.full_like(r, 40 + 60 * cam + 20 * ep)
    img = np.stack([r, g, b], axis=-1)
    cy, cx = int(h / 2 + 150 * np.sin(phase)), int(w / 2 + 250 * np.cos(phase))
    img[max(cy - 30, 0) : cy + 30, max(cx - 30, 0) : cx + 30] = 255
    return img.astype(np.uint8)


def fake_trajectory(n: int, ep: int) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, dtype=np.float32)[:, None]
    amp = np.array([20, 15, 15, 10, 10, 10, 35], dtype=np.float32) * (1 + 0.1 * ep)
    phase = np.arange(7, dtype=np.float32)[None, :] * 0.7
    return (CENTER + amp * np.sin(t + phase)).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/fake_block_bowl")
    p.add_argument("--repo-id", default="local/fake_block_bowl")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--frames", type=int, default=60)
    args = p.parse_args()

    root = Path(args.root)
    if root.exists():
        shutil.rmtree(root)

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=C.FPS,
        features=FEATURES,
        root=root,
        robot_type=C.ROBOT_TYPE,
        use_videos=True,
        image_writer_threads=4,
    )
    for ep in range(args.episodes):
        traj = fake_trajectory(args.frames + 1, ep)
        for t in range(args.frames):
            ds.add_frame(
                {
                    "observation.images.top": fake_image(t, ep, cam=0),
                    "observation.images.wrist": fake_image(t, ep, cam=1),
                    "observation.state": traj[t],
                    "action": traj[t + 1],
                    "task": C.TASK,
                }
            )
        ds.save_episode()
        print(f"saved episode {ep} ({args.frames} frames)")
    ds.finalize()

    ds = LeRobotDataset(args.repo_id, root=root)
    print(ds)
    print(f"frames={len(ds)} episodes={ds.num_episodes} fps={ds.fps}")
    item = ds[len(ds) // 2]
    for k in ("observation.images.top", "observation.images.wrist", "observation.state", "action"):
        v = item[k]
        print(f"{k}: shape={tuple(v.shape)} dtype={v.dtype} min={float(v.min()):.3f} max={float(v.max()):.3f}")
    print("task:", item["task"])
    print("stats keys:", list(ds.meta.stats["action"].keys()))


if __name__ == "__main__":
    main()
