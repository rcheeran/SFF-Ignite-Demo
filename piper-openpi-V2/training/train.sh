#!/bin/bash
# LOCAL FALLBACK on the Thor - the cloud run's exact lerobot-train flags (training/cloud/train.sh), no upload,
# no torch.compile. Real runs go to the cloud: training/cloud/README.md.
# bash training/train.sh smoke [dataset]    500 steps, batch 8   -> outputs/pi05_piper_smoke   (~45 min, 37 GB GPU)
# bash training/train.sh full  [dataset]    3000 steps, batch 16 -> outputs/pi05_piper_local   (~9 h)
# dataset = local dataset root (default data/fake_block_bowl) or an HF repo id. Log: ~/train.log
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_TOKEN=${HF_TOKEN:-$(grep -o 'hf_[A-Za-z0-9]*' ~/Documents/HG-TOKEN.txt | tail -1)}
case "${1:-smoke}" in
  smoke) STEPS=500;  BATCH=8;  OUT=outputs/pi05_piper_smoke ;;
  full)  STEPS=3000; BATCH=16; OUT=outputs/pi05_piper_local ;;
  *) echo "usage: $0 smoke|full [dataset]"; exit 1 ;;
esac
[ -e "$OUT" ] && { echo "$OUT exists (a previous run): move it away or rm -rf it first"; exit 1; }
OUT=$OUT COMPILE=false exec bash training/cloud/train.sh "${2:-data/fake_block_bowl}" - "$STEPS" "$BATCH"
