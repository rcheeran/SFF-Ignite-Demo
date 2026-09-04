#!/usr/bin/env bash
# Fine-tune pi0.5 on the Piper pick-soda-can dataset, on a Lambda Labs GPU VM.
#
# Usage (on a fresh Lambda instance, e.g. 1x H100 80GB):
#   export HF_TOKEN=hf_...           # write token
#   bash lambda_train.sh [exp_name]  # default exp: piper_pick_v2
#
# Produces: HF model repo royc1981/pi05-piper-pick/<exp_name>/
#   (model.safetensors + assets/ + metadata.pt - all the Thor needs to deploy)
set -euo pipefail

EXP="${1:-piper_pick_v2}"
OPENPI_COMMIT=15a9616a00943ada6c20a0f158e3adb39df2ccac   # match the Thor's checkout
HF_DATASET=royc1981/piper-pick-soda-can
HF_MODEL=royc1981/pi05-piper-pick
STEPS_LOG=~/train_${EXP}.log

[ -n "${HF_TOKEN:-}" ] || { echo "ERROR: export HF_TOKEN first"; exit 1; }

echo "=== 1/7 clone openpi @ ${OPENPI_COMMIT:0:12} + apply Piper patch ==="
cd ~
[ -d openpi ] || git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi
git checkout -q "$OPENPI_COMMIT"
git apply --check ~/openpi_piper.patch && git apply ~/openpi_piper.patch || echo "(patch already applied)"

echo "=== 2/7 python env (uv) ==="
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --extra dev 2>/dev/null || uv sync
# openpi requires its patched transformers modeling files
SITE=$(uv run python -c 'import transformers, os; print(os.path.dirname(transformers.__file__))')
cp -r src/openpi/models_pytorch/transformers_replace/* "$SITE/"

echo "=== 3/7 pull dataset from HF ==="
uv run python - <<EOF
import os
from huggingface_hub import snapshot_download
dst = os.path.expanduser("~/.cache/huggingface/lerobot/piper/pick_soda_can")
snapshot_download("${HF_DATASET}", repo_type="dataset", local_dir=dst, token="${HF_TOKEN}")
print("dataset ->", dst)
EOF

echo "=== 4/7 download pi05_base (JAX) + convert to PyTorch ==="
CKPT=~/.cache/openpi/openpi-assets/checkpoints/pi05_base
PT=~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch
if [ ! -f "$PT/model.safetensors" ]; then
  uv run python - <<'EOF'
import os, shutil, gcsfs
SRC = "openpi-assets/checkpoints/pi05_base"
DST = os.path.expanduser("~/.cache/openpi/openpi-assets/checkpoints/pi05_base")
os.makedirs(os.path.dirname(DST), exist_ok=True)
if not os.path.isdir(os.path.join(DST, "params")):
    fs = gcsfs.GCSFileSystem(token="anon")
    fs.get(SRC, DST, recursive=True)
    inner = os.path.join(DST, "pi05_base")   # gcsfs sometimes nests
    if os.path.isdir(inner):
        for x in os.listdir(inner):
            shutil.move(os.path.join(inner, x), os.path.join(DST, x))
        os.rmdir(inner)
print("base checkpoint ready")
EOF
  uv run python examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir "$CKPT" --config_name pi05_piper \
    --output_path "$PT" --precision bfloat16
fi

echo "=== 5/7 norm stats ==="
uv run python scripts/compute_norm_stats.py --config-name pi05_piper

echo "=== 6/7 train (logs: $STEPS_LOG) ==="
uv run python scripts/train_pytorch.py pi05_piper \
  --exp-name "$EXP" --no-wandb-enabled --overwrite --save-interval 1000 \
  2>&1 | tee "$STEPS_LOG"

echo "=== 7/7 push final checkpoint to HF ==="
FINAL=$(ls -d ~/openpi/checkpoints/pi05_piper/"$EXP"/* | grep -E '/[0-9]+$' | sort -t/ -k9 -n | tail -1)
echo "pushing $FINAL"
uv run python - <<EOF
from huggingface_hub import HfApi
api = HfApi(token="${HF_TOKEN}")
api.create_repo("${HF_MODEL}", private=True, exist_ok=True)
# model.safetensors + assets/ + metadata.pt; skip the 13GB optimizer.pt
import os
final = "${FINAL}"
for root, dirs, files in os.walk(final):
    for f in files:
        if f == "optimizer.pt":
            continue
        p = os.path.join(root, f)
        rel = os.path.relpath(p, final)
        print("upload", rel)
        api.upload_file(path_or_fileobj=p, path_in_repo="${EXP}/" + rel,
                        repo_id="${HF_MODEL}")
print("CHECKPOINT PUSHED -> ${HF_MODEL}/${EXP}")
EOF

echo "ALL DONE - terminate this instance to stop billing."
