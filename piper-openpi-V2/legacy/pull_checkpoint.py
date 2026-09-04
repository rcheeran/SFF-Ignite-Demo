#!/usr/bin/env python3
"""Pull a fine-tuned Piper checkpoint from HF onto the Thor for deployment.

Usage (in ~/pi05-venv):
    python pull_checkpoint.py piper_pick_v2
Then:
    python run_policy.py --checkpoint ~/Documents/openpi/checkpoints/pi05_piper/piper_pick_v2/hf --dry-run
"""
import os
import sys

from huggingface_hub import snapshot_download

REPO = "royc1981/pi05-piper-pick"
exp = sys.argv[1] if len(sys.argv) > 1 else "piper_pick_v2"
dst = os.path.expanduser(f"~/Documents/openpi/checkpoints/pi05_piper/{exp}/hf")
snapshot_download(REPO, allow_patterns=[f"{exp}/*"], local_dir=dst + "_tmp")
# strip the exp-name prefix folder so dst is a normal checkpoint dir
os.makedirs(dst, exist_ok=True)
src = os.path.join(dst + "_tmp", exp)
for item in os.listdir(src):
    os.replace(os.path.join(src, item), os.path.join(dst, item))
os.removedirs(src)
print(f"checkpoint ready: {dst}")
