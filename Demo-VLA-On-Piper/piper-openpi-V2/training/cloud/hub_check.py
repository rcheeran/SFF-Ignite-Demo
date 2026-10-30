"""Hub checks that fail fast, before GPU-hours are spent or a model is trusted. No hardware.

python -m training.cloud.hub_check dataset USER/piper_block_bowl [STEPS BATCH]   token, dataset contract, run plan + cost
python -m training.cloud.hub_check model USER/pi05_piper [model.safetensors]     weights on the Hub [+ local copy matches]

launch.sh runs `dataset` before touching Lambda; train.sh runs `model` after the upload (it is scp'd to the instance
as ~/hub_check.py, hence the lazy piper.config import); pull_checkpoint.sh runs `model` before and after the download.
Any problem: one-line reason, exit 1.
"""
import argparse
import hashlib
import json
import os

from huggingface_hub import HfApi, hf_hub_download

WEIGHTS = "model.safetensors"


def die(msg):
    raise SystemExit(f"ERROR: {msg}")


def first_line(e: Exception) -> str:
    return f"{e.__class__.__name__}: {str(e).splitlines()[0] if str(e) else ''}"


def contract_problems(info: dict, fps, names, cams, shape) -> list:
    """Mismatches between a dataset's meta/info.json and the contract (piper/config.py); [] = OK."""
    want = {f"observation.images.{c}": ("video", list(shape)) for c in cams}
    want |= {"observation.state": ("float32", [len(names)]), "action": ("float32", [len(names)])}
    feats = info.get("features", {})
    bad = [] if info.get("fps") == fps else [f"fps {info.get('fps')} != {fps}"]
    for key, (dtype, shp) in want.items():
        f = feats.get(key)
        if f is None:
            bad.append(f"missing feature {key}")
        elif (f.get("dtype"), list(f.get("shape", []))) != (dtype, shp):
            bad.append(f"{key}: {f.get('dtype')} {f.get('shape')} != {dtype} {shp}")
        elif dtype == "float32" and list(f.get("names") or []) != list(names):
            bad.append(f"{key} names {f.get('names')} != {list(names)}")
    if not info.get("total_frames"):
        bad.append("no frames")
    return bad


def plan(steps, batch, frames, step_s, overhead_min, usd_per_h):
    """(epochs, hours, usd) for a run; step_s is the estimated s/step at batch 16."""
    hours = (steps * step_s * batch / 16 + overhead_min * 60) / 3600
    return steps * batch / frames, hours, hours * usd_per_h


def check_dataset(repo, steps=None, batch=16):
    from piper import config as C  # lazy: the `model` check also runs on the Lambda instance, without the repo
    try:
        user = HfApi().whoami()["name"]
    except Exception as e:
        die(f"HF token rejected ({first_line(e)}) - check HF_TOKEN / {C.HF_TOKEN_FILE}")
    try:
        with open(hf_hub_download(repo, "meta/info.json", repo_type="dataset")) as f:
            info = json.load(f)
    except Exception as e:
        die(f"dataset {repo}: {first_line(e)} - not on the Hub, not yours, or not a LeRobot v3 dataset")
    bad = contract_problems(info, C.FPS, C.STATE_NAMES, C.CAMERA_NAMES, (C.IMG_H, C.IMG_W, 3))
    if bad:
        die(f"dataset {repo} violates the contract (docs/CONTRACT.md): " + "; ".join(bad))
    n = len(C.STATE_NAMES)
    print(f"token OK ({user}); dataset {repo}: {info['total_episodes']} episodes, {info['total_frames']} frames, "
          f"fps {info['fps']}, features OK ({', '.join(C.CAMERA_NAMES)}, state[{n}], action[{n}])")
    if steps:
        epochs, hours, usd = plan(steps, batch, info["total_frames"], C.H100_STEP_S, C.CLOUD_OVERHEAD_MIN, C.H100_USD_PER_H)
        print(f"plan: {steps} steps x batch {batch} = {epochs:.1f} epochs; ~{C.H100_STEP_S} s/step (H100 estimate) "
              f"+ {C.CLOUD_OVERHEAD_MIN} min setup/upload ~= {hours:.1f} h ~= ${usd:.2f} at ${C.H100_USD_PER_H}/h")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def check_model(repo, local=None):
    try:
        files = {s.rfilename: s for s in HfApi().model_info(repo, files_metadata=True).siblings}
    except Exception as e:
        die(f"model {repo}: {first_line(e)}")
    w = files.get(WEIGHTS)
    if w is None:
        die(f"{repo} has no {WEIGHTS} (training not finished or the upload failed); it holds {sorted(files)}")
    sha = w.lfs.sha256 if w.lfs else None
    print(f"{repo}: {WEIGHTS} {w.size / 1e9:.2f} GB on the Hub, sha256 {sha or '?'}")
    if local is None:
        return
    if not os.path.isfile(local):
        die(f"{local} missing")
    if (size := os.path.getsize(local)) != w.size:
        die(f"{local} is {size} bytes, the Hub lists {w.size} - incomplete copy")
    if sha and sha256(local) != sha:
        die(f"{local} sha256 differs from the Hub - corrupt copy")
    print(f"{local}: size and sha256 match the Hub")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dataset")
    d.add_argument("repo")
    d.add_argument("steps", nargs="?", type=int)
    d.add_argument("batch", nargs="?", type=int, default=16)
    m = sub.add_parser("model")
    m.add_argument("repo")
    m.add_argument("local", nargs="?")
    a = p.parse_args()
    check_dataset(a.repo, a.steps, a.batch) if a.cmd == "dataset" else check_model(a.repo, a.local)


if __name__ == "__main__":
    main()
