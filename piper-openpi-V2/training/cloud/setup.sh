#!/bin/bash
# bash setup.sh   (on a fresh Lambda Ubuntu instance, x86_64; HF_TOKEN must be in the env; safe to re-run)
set -euo pipefail
: "${HF_TOKEN:?HF_TOKEN missing}"
cd "$(dirname "$0")"
VENV=~/lerobot-venv
export PATH=~/.local/bin:$PATH

command -v tmux >/dev/null || { sudo apt-get update -q && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q tmux; }
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
[ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" \
  --index https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
  -r requirements.txt
source "$VENV/bin/activate"
hf auth login --token "$HF_TOKEN"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False (NVIDIA driver too old for CUDA 12.8 wheels?)"
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))
EOF
lerobot-train --help > /dev/null
hf auth whoami
echo "setup OK: $VENV"
