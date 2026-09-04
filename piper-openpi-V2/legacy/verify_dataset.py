#!/usr/bin/env python3
"""Verify a LeRobot dataset is complete and correct for pi0.5 fine-tuning.

Checks, in order of how badly each would waste a training run:
  1. dataset loads; episode/frame counts; fps
  2. every declared feature is present with the right shape/dtype
  3. images are real (not blank/constant) and the expected size
  4. state/actions are 10-D, finite, and the 6D rotation block is orthonormal
  5. actions[t] == state[t+1] within an episode (our absolute-action convention)
  6. the actual openpi training pipeline yields a model-ready batch
     (this is the one that catches config/transform mistakes)
  7. norm stats exist and look sane for delta training

Run in ~/pi05-venv:
    python verify_dataset.py --config pi05_piper_v3
    python verify_dataset.py --config pi05_piper          # the older 1-camera set
"""
import argparse
import json
import os
import sys

import numpy as np

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results = []


def check(name, ok, detail="", warn_only=False):
    tag = PASS if ok else (WARN if warn_only else FAIL)
    results.append((tag, name, detail))
    print(f"  [{tag}] {name}" + (f" - {detail}" if detail else ""))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pi05_piper_v3")
    ap.add_argument("--batches", type=int, default=2)
    args = ap.parse_args()

    import openpi.training.config as _config
    import openpi.training.data_loader as _data_loader

    cfg = _config.get_config(args.config)
    print(f"\n=== config {args.config}  (repo_id {cfg.data.repo_id}) ===\n")

    # ---------------------------------------------------------------- 1-2
    print("1) dataset & schema")
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    try:
        ds = LeRobotDataset(cfg.data.repo_id)
    except Exception as e:
        check("dataset loads", False, str(e)[:120])
        sys.exit(1)
    check("dataset loads", True, f"{ds.num_episodes} episodes, {len(ds)} frames, {ds.fps} fps")
    check("fps is 15", ds.fps == 15, f"fps={ds.fps}", warn_only=True)
    check("enough frames for fine-tuning", len(ds) >= 5000,
          f"{len(ds)} frames (pi guidance: 1-20h of data)", warn_only=len(ds) >= 2000)

    sample = ds[0]
    img_keys = [k for k in ds.features if k.endswith("_image")]
    check("image streams present", len(img_keys) >= 1, f"{img_keys}")
    check("three camera views", len(img_keys) == 3,
          f"{len(img_keys)} view(s) - pi0.5 has 3 slots; 1 view means no global reference",
          warn_only=True)
    for k in img_keys:
        t = sample[k]
        shp = tuple(t.shape)
        check(f"{k} shape", shp in ((3, 224, 224), (224, 224, 3)), f"{shp}")
        arr = np.asarray(t, dtype=np.float32)
        check(f"{k} is not blank", float(arr.std()) > 0.01, f"std={float(arr.std()):.4f}")

    for k in ("state", "actions"):
        check(f"{k} is 10-D", tuple(sample[k].shape) == (10,), f"{tuple(sample[k].shape)}")

    # ---------------------------------------------------------------- 4
    print("\n2) state / action content")
    N = min(len(ds), 3000)
    S = np.stack([np.asarray(ds[i]["state"]) for i in range(0, N, max(1, N // 400))])
    A = np.stack([np.asarray(ds[i]["actions"]) for i in range(0, N, max(1, N // 400))])
    check("state finite", bool(np.isfinite(S).all()))
    check("actions finite", bool(np.isfinite(A).all()))
    c1 = S[:, 3:6]
    c2 = S[:, 6:9]
    n1 = np.linalg.norm(c1, axis=1)
    n2 = np.linalg.norm(c2, axis=1)
    dot = np.abs(np.sum(c1 * c2, axis=1))
    check("6D rotation columns unit-norm", bool(np.allclose(n1, 1, atol=1e-3) and np.allclose(n2, 1, atol=1e-3)),
          f"|c1| in [{n1.min():.4f},{n1.max():.4f}]")
    check("6D rotation columns orthogonal", float(dot.max()) < 1e-3, f"max|c1.c2|={dot.max():.2e}")
    g = S[:, 9]
    check("gripper is binary-ish", set(np.round(np.unique(g), 3)).issubset({0.0, 0.07}),
          f"values {np.unique(np.round(g,3))[:6]}", warn_only=True)
    xyz = S[:, :3]
    print(f"       workspace: X {xyz[:,0].min():.0f}..{xyz[:,0].max():.0f}  "
          f"Y {xyz[:,1].min():.0f}..{xyz[:,1].max():.0f}  Z {xyz[:,2].min():.0f}..{xyz[:,2].max():.0f} mm")

    # ---------------------------------------------------------------- 5
    print("\n3) action convention (absolute: actions[t] == state[t+1])")
    ep_ends = set()
    try:
        ep_idx = np.asarray([ds[i]["episode_index"] for i in range(min(len(ds), 4000))])
        ep_ends = {int(i) for i in np.where(np.diff(ep_idx) != 0)[0]}
    except Exception:
        pass
    errs = []
    for i in range(0, min(len(ds), 4000) - 1, 37):
        if i in ep_ends:
            continue
        errs.append(float(np.abs(np.asarray(ds[i]["actions"]) - np.asarray(ds[i + 1]["state"])).max()))
    if errs:
        med = float(np.median(errs))
        check("actions[t] == state[t+1]", med < 1e-4,
              f"median |diff| = {med:.2e} over {len(errs)} samples")

    # ---------------------------------------------------------------- 7
    print("\n4) normalization stats")
    ns_path = cfg.assets_dirs / (cfg.data.repo_id or "")
    f = os.path.join(str(ns_path), "norm_stats.json")
    if not os.path.exists(f):
        check("norm_stats.json exists", False, f"missing at {f} - run compute_norm_stats.py")
    else:
        ns = json.load(open(f))["norm_stats"]
        check("norm_stats.json exists", True, f)
        for k in ("state", "actions"):
            m = np.array(ns[k]["mean"]); s = np.array(ns[k]["std"])
            check(f"{k} stats are 10-D", m.shape[0] == 10, f"{m.shape[0]}-D")
            check(f"{k} std non-degenerate", float(s.min()) > 1e-6, f"min std {s.min():.3g}")
        am = np.array(ns["actions"]["mean"]); asd = np.array(ns["actions"]["std"])
        sm = np.array(ns["state"]["mean"])
        # after DeltaActions the action mean must sit near zero, not at workspace scale
        near_zero = float(np.abs(am[:3]).max()) < 20.0
        check("actions look like DELTAS (mean ~0)", near_zero,
              f"action mean[:3]={np.round(am[:3],1)} vs state mean[:3]={np.round(sm[:3],1)}; "
              f"action std[:3]={np.round(asd[:3],1)}",
              warn_only=True)

    # ---------------------------------------------------------------- 6
    print("\n5) end-to-end: real training batch through openpi")
    try:
        dl = _data_loader.create_data_loader(cfg, num_batches=args.batches, shuffle=True, framework="pytorch")
        obs, act = next(iter(dl))
        B = cfg.batch_size
        check("batch state shape", tuple(np.asarray(obs.state).shape) == (B, cfg.model.action_dim),
              f"{tuple(np.asarray(obs.state).shape)} (want ({B},{cfg.model.action_dim}))")
        check("batch action shape",
              tuple(np.asarray(act).shape) == (B, cfg.model.action_horizon, cfg.model.action_dim),
              f"{tuple(np.asarray(act).shape)}")
        for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
            im = np.asarray(obs.images[k])
            check(f"image slot {k}", im.shape == (B, 3, 224, 224), f"{im.shape}")
        masks = {k: bool(np.asarray(v).flatten()[0]) for k, v in obs.image_masks.items()}
        check("image masks", True, str(masks))
        # distinct views? (duplicated views mean the extra slots add nothing)
        b = np.asarray(obs.images["base_0_rgb"])[0]
        w = np.asarray(obs.images["left_wrist_0_rgb"])[0]
        r = np.asarray(obs.images["right_wrist_0_rgb"])[0]
        check("base vs wrist are different views", float(np.abs(b - w).mean()) > 0.01,
              f"mean|diff|={np.abs(b-w).mean():.3f}")
        check("third slot is a distinct view", float(np.abs(r - w).mean()) > 0.01,
              f"mean|diff|={np.abs(r-w).mean():.3f}",
              warn_only=True)
        check("prompt tokenized", obs.tokenized_prompt is not None,
              f"{tuple(np.asarray(obs.tokenized_prompt).shape)}" if obs.tokenized_prompt is not None else "missing")
        st = np.asarray(obs.state)[:, :10]
        check("state normalized (|mean|<1.5)", float(np.abs(st.mean())) < 1.5,
              f"mean={st.mean():.2f} std={st.std():.2f}")
    except Exception as e:
        check("training batch builds", False, f"{type(e).__name__}: {str(e)[:160]}")

    # ---------------------------------------------------------------- summary
    nf = sum(1 for t, _, _ in results if t == FAIL)
    nw = sum(1 for t, _, _ in results if t == WARN)
    print(f"\n=== {len(results)-nf-nw} passed, {nw} warnings, {nf} failures ===")
    if nf:
        print("FAILURES:")
        for t, n, d in results:
            if t == FAIL:
                print(f"  - {n}: {d}")
    sys.exit(1 if nf else 0)


if __name__ == "__main__":
    main()
