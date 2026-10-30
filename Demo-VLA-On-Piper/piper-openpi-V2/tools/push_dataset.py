#!/usr/bin/env python3
"""Push a finished dataset to the Hugging Face Hub, separately from recording it.

    python -m tools.push_dataset data/piper_block_bowl                 # summarise, confirm, push
    python -m tools.push_dataset data/piper_block_bowl --private
    python -m tools.push_dataset data/piper_block_bowl --repo-id someone/other_name
    python -m tools.push_dataset data/piper_block_bowl --yes           # no confirmation prompt
    python -m tools.push_dataset data/piper_block_bowl --dry-run       # show what would happen

Recording and publishing are separate decisions: a recording session ends when the arm is parked, but
whether the episodes are worth publishing is a judgement made afterwards, once inspect_dataset and the
contact sheet have been looked at. Pushing from inside the recorder forced that call in advance.

Uploading is public-facing and hard to take back, so this asks before it uploads unless --yes is given,
and it refuses data/test outright - that directory exists to hold recordings that are NOT trusted yet.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from piper import config as C   # noqa: E402


def read_token() -> str:
    """HF_TOKEN from the environment, else the hf_... string in HF_TOKEN_FILE."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    path = Path(C.HF_TOKEN_FILE).expanduser()
    if not path.exists():
        raise SystemExit(f"ABORT: no HF_TOKEN in the environment and no {path}")
    text = path.read_text()
    found = re.findall(r"hf_[A-Za-z0-9]+", text)      # the file holds prose around the token
    return found[-1] if found else text.strip()


def summarise(root: Path) -> dict:
    info = json.loads((root / "meta" / "info.json").read_text())
    size = subprocess.run(["du", "-sh", str(root)], capture_output=True, text=True).stdout.split()[0]
    tasks = []
    try:
        import pandas as pd
        tasks = list(pd.read_parquet(root / "meta" / "tasks.parquet").index)
    except Exception:
        pass
    return {"episodes": info["total_episodes"], "frames": info["total_frames"], "fps": info["fps"],
            "cameras": [k.split(".")[-1] for k in info["features"] if k.startswith("observation.images")],
            "size": size, "tasks": tasks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="dataset directory, e.g. data/piper_block_bowl")
    ap.add_argument("--repo-id", default=None, help=f"default {C.HF_USER}/<directory name>")
    ap.add_argument("--private", action="store_true", help="create/keep the Hub repo private")
    ap.add_argument("--yes", "-y", action="store_true", help="skip the confirmation prompt")
    ap.add_argument("--dry-run", action="store_true", help="show what would be pushed, upload nothing")
    ap.add_argument("--tags", nargs="*", default=C.HUB_TAGS)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not (root / "meta" / "info.json").exists():
        raise SystemExit(f"ABORT: {root} is not a LeRobot dataset (no meta/info.json)")
    if root.name == "test":
        raise SystemExit("ABORT: data/test holds recordings that have not been validated - that is the whole "
                         "point of it. Record into a real dataset directory, or pass --root elsewhere.")

    repo_id = args.repo_id or f"{C.HF_USER}/{root.name}"
    if "/" not in repo_id:
        repo_id = f"{C.HF_USER}/{repo_id}"
    s = summarise(root)
    print(f"\n  dataset   {root}")
    print(f"  episodes  {s['episodes']}   frames {s['frames']}   {s['frames'] / max(s['fps'], 1) / 60:.1f} min")
    print(f"  cameras   {', '.join(s['cameras'])}   fps {s['fps']}   size {s['size']}")
    print(f"  tasks     {s['tasks']}")
    print(f"  -> https://huggingface.co/datasets/{repo_id}  ({'private' if args.private else 'PUBLIC'})")

    if not s["episodes"]:
        raise SystemExit("ABORT: the dataset is empty - nothing to push")
    if args.dry_run:
        print("\n  --dry-run: nothing uploaded")
        return
    if not args.yes:
        print("\n  Uploading publishes these recordings, including everything the cameras saw.")
        if input("  push? [y/N] ").strip().lower() not in ("y", "yes"):
            raise SystemExit("  not pushed")

    os.environ["HF_TOKEN"] = read_token()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id, root=root)
    try:
        ds.push_to_hub(tags=args.tags, private=args.private)
    except KeyboardInterrupt:
        raise SystemExit("\nCtrl-C during push: the local dataset is intact - re-run to retry")
    print(f"\n  pushed -> https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
