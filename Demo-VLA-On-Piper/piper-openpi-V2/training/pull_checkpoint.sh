#!/bin/bash
# bash training/pull_checkpoint.sh USER/pi05_piper   (Jetson) -> outputs/pi05_piper/pretrained_model, verified against
# the Hub (size + sha256), 3 dry-run inference steps, then the deploy command.
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/lerobot-venv/bin/activate
export HF_TOKEN=${HF_TOKEN:-$(grep -o 'hf_[A-Za-z0-9]*' ~/Documents/HG-TOKEN.txt | tail -1)}
REPO=${1:?usage: $0 USER/pi05_piper}
DEST=outputs/$(basename "$REPO")/pretrained_model

python -m training.cloud.hub_check model "$REPO"                          # weights there at all? (before a 25 min download)
hf download "$REPO" --local-dir "$DEST"
python -m training.cloud.hub_check model "$REPO" "$DEST/model.safetensors" # complete and uncorrupted
python -m inference.run_policy --checkpoint "$DEST" --dry-run --steps 3
echo "deploy:  python -m inference.run_policy --checkpoint $DEST"
