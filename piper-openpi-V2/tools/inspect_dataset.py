#!/usr/bin/env python3
"""Judge a recorded dataset before paying for training: episode lengths, joint steps per frame, gripper range,
the one-frame-lag contract, timestamp gaps, and a contact sheet (first/middle/last frame of both cameras).

    python tools/inspect_dataset.py data/piper_block_bowl          # local root
    python tools/inspect_dataset.py royc1981/piper_block_bowl      # Hub repo id (downloads into the HF cache)
    python tools/inspect_dataset.py data/piper_block_bowl --strict # + contract assertions, non-zero exit on failure

Frames with a joint step above MAX_JOINT_STEP_DEG matter: inference clamps JointCtrl to that per frame.

--strict turns the summary above into a pass/fail gate, adding what the numbers alone cannot show:
the feature SCHEMA pi05 is handed via --rename_map; UNITS (degrees not radians, joints inside the arm's
limits, a gripper action the arm can actually reach - a radians mix-up or a hand-forced 92 mm jaw command
both look like a perfectly ordinary dataset); one decoded VIDEO frame per parquet row; and q01/q99 in
stats.json, because pi05 normalises on quantiles and their absence fails only once training starts.
ALWAYS look at the contact sheet as well - a dataset can pass every check here and still show the arm
grasping the bowl, which has happened.
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from piper import config as C  # noqa: E402
from piper.robot import JOINT_LIMITS_DEG  # noqa: E402

TILE = (320, 240)
MAX_SHEET_EPISODES = 6


def load(arg: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    path = Path(arg)
    if path.is_dir():
        return LeRobotDataset(f"local/{path.name}", root=arg, video_backend="pyav")
    # "data/test" has the same user/name shape as a Hub repo id, so a missing local directory used to be
    # taken for a Hub repo and fail 40 lines later with a 404 about huggingface.co. Say what is wrong.
    if path.parent.exists() or arg.startswith(("./", "../", "/")):
        raise SystemExit(f"ABORT: no dataset at {path.resolve()} - nothing recorded there yet?")
    return LeRobotDataset(arg, video_backend="pyav")


def tile(ds, i: int, cam: str, label: str) -> np.ndarray:
    img = (ds[i][f"observation.images.{cam}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), TILE)
    cv2.putText(img, f"{label} {cam}", (6, TILE[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    return img


def contact_sheet(ds, episode_rows: dict, path: Path):
    """One row per episode (up to MAX_SHEET_EPISODES): first / middle / last frame, both cameras."""
    rows = []
    for ep, idx in list(episode_rows.items())[:MAX_SHEET_EPISODES]:
        picks = (idx[0], idx[len(idx) // 2], idx[-1])
        rows.append(np.hstack([tile(ds, i, cam, f"ep{ep} f{i - idx[0]}") for i in picks for cam in C.CAMERA_NAMES]))
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])


TS_TOL_S = 1e-3          # float32 timestamps round at ~1e-6 s; only a real drift (>1 ms) matters


def verify(ds, A, S, E, T) -> bool:
    """Assert what MUST hold before training. Prints PASS/FAIL per check, returns False if any failed."""
    import glob
    import json
    fails = []

    def check(cond, label, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
        if not cond:
            fails.append(label)
        return cond

    print("\nSTRICT: format")
    info = json.loads((Path(ds.root) / "meta" / "info.json").read_text())
    feats = info["features"]
    check(info.get("codebase_version") == "v3.0", "codebase_version v3.0", str(info.get("codebase_version")))
    check(info.get("robot_type") == C.ROBOT_TYPE, "robot_type", str(info.get("robot_type")))
    check(info["fps"] == C.FPS, "fps matches the deploy rate", f"{info['fps']} vs {C.FPS}")
    for n in C.CAMERA_NAMES:
        f = feats.get(f"observation.images.{n}", {})
        check(f.get("dtype") == "video" and tuple(f.get("shape", ())) == (C.IMG_H, C.IMG_W, 3),
              f"observation.images.{n} schema", f"{f.get('dtype')} {tuple(f.get('shape', ()))}")
    for k in ("observation.state", "action"):
        f = feats.get(k, {})
        check(f.get("dtype") == "float32" and tuple(f.get("shape", ())) == (7,)
              and list(f.get("names", [])) == C.STATE_NAMES, f"{k} schema",
              f"{f.get('dtype')} {tuple(f.get('shape', ()))} {f.get('names')}")

    print("\nSTRICT: units")
    j, g, ag = S[:, :6], S[:, 6], A[:, 6]
    lo, hi = JOINT_LIMITS_DEG[:, 0], JOINT_LIMITS_DEG[:, 1]
    bad = ((j < lo - 0.5) | (j > hi + 0.5)).any(axis=1)
    check(not bad.any(), "joints inside the arm's limits",
          f"{bad.sum()} frames outside" if bad.any() else f"max |j| {np.abs(j).max():.1f} deg")
    check(np.abs(j).max() > 6.3, "joints are DEGREES not radians", f"max |j| {np.abs(j).max():.1f}")
    check(g.min() >= -5.0 and g.max() <= C.GRIP_OPEN_MM + 1.0, "gripper state in mm, in range",
          f"{g.min():.1f}..{g.max():.1f} mm")
    check(ag.min() >= C.GRIP_CLOSED_MM - 0.01 and ag.max() <= C.GRIP_OPEN_MM + 0.01,
          "gripper ACTION is commandable",
          f"{ag.min():.1f}..{ag.max():.1f} mm ({int((ag > C.GRIP_OPEN_MM).sum())} frames above {C.GRIP_OPEN_MM:g})")
    span = np.ptp(S, axis=0)
    check((span[:6] > 1.0).all(), "every joint moves", f"range {np.round(span[:6], 1).tolist()} deg")
    check(span[6] > 5.0, "gripper moves", f"range {span[6]:.1f} mm")

    print("\nSTRICT: timing")
    worst = max(float(np.abs(np.diff(T[E == ep]) - 1 / ds.fps).max()) for ep in np.unique(E) if (E == ep).sum() > 1)
    check(worst < TS_TOL_S, "timestamps 1/fps apart", f"worst deviation {worst:.2e} s (tol {TS_TOL_S:g})")
    lag = max(float(np.abs(A[E == ep][:-1, :6] - S[E == ep][1:, :6]).max())
              for ep in np.unique(E) if (E == ep).sum() > 1)
    check(lag < 1e-3, "action[t][:6] == state[t+1][:6]", f"max mismatch {lag:.2e} deg")
    check(len(np.unique(ds.meta.tasks.index)) == 1, "exactly one task string", str(list(ds.meta.tasks.index)))

    print("\nSTRICT: cameras")
    import av
    for n in C.CAMERA_NAMES:
        vids = sorted(glob.glob(str(Path(ds.root) / "videos" / f"observation.images.{n}" / "**" / "*.mp4"),
                                recursive=True))
        if not check(bool(vids), f"{n}: videos exist", f"{len(vids)} file(s)"):
            continue
        try:
            with av.open(vids[0]) as c:
                total = c.streams.video[0].frames or sum(1 for _ in c.decode(video=0))
            with av.open(vids[0]) as c:
                first = next(c.decode(video=0)).to_ndarray(format="rgb24")
        except Exception as e:
            check(False, f"{n}: video decodes", f"{type(e).__name__}: {e}")
            continue
        check(first.shape == (C.IMG_H, C.IMG_W, 3) and first.dtype == np.uint8,
              f"{n}: frame shape/dtype", f"{first.shape} {first.dtype}")
        check(int(first.max()) > 16, f"{n}: frames are not blank", f"max {first.max()}")
        n_rows = int((E == np.unique(E)[0]).sum())
        check(abs(total - n_rows) <= 1, f"{n}: one video frame per row", f"{total} frames vs {n_rows} rows")
    # Channel order is NOT asserted here: a global R-vs-B test is worthless on this scene (red AND blue
    # objects cancel: measured mean R 143 / B 143). The contact sheet is the check - it is written through
    # RGB2BGR before imwrite, so if the red cube looks red there, the stored frames really are RGB.
    print("  NOTE  channel order: confirm on the contact sheet that the red cube looks red (not blue)")

    print("\nSTRICT: training readiness")
    stats = json.loads((Path(ds.root) / "meta" / "stats.json").read_text())
    for k in ("observation.state", "action"):
        check(all(q in stats.get(k, {}) for q in ("q01", "q99")),
              f"{k} stats have q01/q99 (pi05 normalises on quantiles)", f"keys {sorted(stats.get(k, {}))}")
    for n in C.CAMERA_NAMES:
        check(f"observation.images.{n}" in stats, f"observation.images.{n} stats present")

    print(f"\n{'FAILED: ' + ', '.join(fails) if fails else 'STRICT: all checks passed'}")
    return not fails


def main(arg: str, strict=False):
    ds = load(arg)
    cols = ds.hf_dataset.with_format("numpy", columns=["action", "observation.state", "episode_index", "timestamp"])[:]
    A, S, E, T = cols["action"], cols["observation.state"], cols["episode_index"], cols["timestamp"]
    print(f"{ds.root}: {ds.num_episodes} episodes, {ds.num_frames} frames, {ds.fps} fps, "
          f"tasks {list(ds.meta.tasks.index)}, cameras {[k.split('.')[-1] for k in ds.meta.video_keys]}")
    print(f"{'ep':>4} {'frames':>6} {'s':>6} {'max step':>9} {'>limit':>6}  grip mm")
    rows, steps, lag, gaps, over = {}, [], [], [], 0
    for ep in np.unique(E):
        idx = np.flatnonzero(E == ep)
        a, s, t = A[idx], S[idx], T[idx]
        d = np.abs(np.diff(a[:, :6], axis=0))
        n_over = int((d > C.MAX_JOINT_STEP_DEG).any(axis=1).sum())
        rows[int(ep)], over = idx, over + n_over
        steps.append(d), lag.append(np.abs(a[:-1, :6] - s[1:, :6])), gaps.append(np.diff(t))
        print(f"{ep:>4} {len(idx):>6} {len(idx) / ds.fps:>6.1f} {d.max(initial=0):>9.2f} {n_over:>6}  "
              f"{a[:, 6].min():.0f}..{a[:, 6].max():.0f}")
    d, lag, gaps = np.concatenate(steps), np.concatenate(lag), np.concatenate(gaps)
    if not len(d):
        raise SystemExit("every episode is a single frame: no joint steps to judge")
    print("joint step deg/frame      " + "".join(f"{j:>7}" for j in C.STATE_NAMES[:6]))
    print("  max                     " + "".join(f"{v:7.2f}" for v in d.max(axis=0)))
    print("  p99                     " + "".join(f"{v:7.2f}" for v in np.percentile(d, 99, axis=0)))
    print(f"frames with a joint step > {C.MAX_JOINT_STEP_DEG:g} deg: {over} / {len(d)}  "
          f"(inference clamps there; teleop data should have 0)")
    print(f"gripper: state {S[:, 6].min():.1f}..{S[:, 6].max():.1f} mm, action {A[:, 6].min():.1f}..{A[:, 6].max():.1f} mm")
    print(f"lag check: mean |action[t][:6] - state[t+1][:6]| = {lag.mean():.4f} deg (expect 0), max {lag.max():.4f}")
    print(f"timestamps: mean gap {gaps.mean():.4f} s (1/fps = {1 / ds.fps:.4f}), "
          f"max |gap - 1/fps| = {np.abs(gaps - 1 / ds.fps).max():.4f} s")
    path = Path(ds.root).with_name(Path(ds.root).name + "_contact.jpg")
    contact_sheet(ds, rows, path)
    print(f"contact sheet: {path}")
    if strict and not verify(ds, A, S, E, T):
        sys.exit(1)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if len(args) != 1:
        raise SystemExit(__doc__)
    main(args[0], strict="--strict" in sys.argv)
