# Restore this machine from scratch

Written 2026-09-04, before a planned reimage of the Jetson Thor. Follow in order. `SETUP.md` covers getting
the *arm* talking over CAN; this covers getting the *software and state* back.

Everything large is deliberately NOT in this repo — datasets and checkpoints live on the Hugging Face Hub and
are re-pullable. What IS here is everything that is small and irreplaceable.

## 0. What a reimage destroys, and where each thing comes back from

| lost | restored from |
|---|---|
| the code | this repo |
| the taught demonstrations (`piper-openpi-V2/data/trajectories/`, 61 files) | **this repo** — they are NOT on the Hub, and each is a hand-guided demo |
| camera/jaw calibration (`auto_data/calib.json`, `jaw_yaw.json`) | this repo |
| the `~/lerobot-venv` python environment | `piper-openpi-V2/requirements-lerobot.txt` |
| Claude Code's memory | `docs/agent-memory/` (step 5) |
| the dataset (463 MB) | Hub: `royc1981/piper_block_bowl` |
| the trained policy (9.35 GB) | Hub: `royc1981/lerobot-pi05-base-50` |
| the gs_usb CAN kernel module | `setup.sh` rebuilds it (Thor has no in-tree gs_usb) |
| **HF token, Lambda API key, Lambda SSH key** | **nothing — recreate by hand, step 4** |

## 1. System, CAN and the arm

```bash
git clone https://github.com/rcheeran/SFF-AgileX-Piper.git ~/Documents/SFF-AgileX-Piper
cd ~/Documents/SFF-AgileX-Piper && bash setup.sh
```

Then follow `SETUP.md` (arm out of Master mode, udev name `can_piper`, verify). Thor has no in-tree `gs_usb`,
so `setup.sh` builds and installs it — without that the USB-CAN adapter never enumerates as a `canX` interface.

## 2. The python environment the VLA stack runs on

`setup.sh` creates `~/venv` for the arm SDK. The VLA work uses a **separate** venv, `~/lerobot-venv`:

```bash
python3 -m venv ~/lerobot-venv && source ~/lerobot-venv/bin/activate
pip install -r ~/Documents/SFF-AgileX-Piper/piper-openpi-V2/requirements-lerobot.txt
```

That file carries `--extra-index-url https://download.pytorch.org/whl/cu130` because **`torch 2.9.1+cu130` is
not on PyPI**. It is the aarch64 SBSA CUDA-13 wheel and the only one whose `get_arch_list()` includes
`sm_110`. Thor is Blackwell-class, not Orin/Tegra — Orin or jetson-ai-lab wheels will not work. Verify:

```bash
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), 'sm_110' in torch.cuda.get_arch_list())"
# 2.9.1+cu130 True True
```

## 3. Prove the stack without hardware

```bash
cd ~/Documents/SFF-AgileX-Piper/piper-openpi-V2 && bash tools/selftest.sh
```

137 pytest tests plus replay and a fake recording. If this passes, the software is back.

## 4. Secrets — recreate by hand, never committed

**`~/Documents/HG-TOKEN.txt`** — plain text, read by several scripts via regex, so the format is forgiving.
It needs a Hugging Face **write** token (`hf_...`, from huggingface.co/settings/tokens; scripts take the LAST
`hf_` match in the file) and a Lambda Cloud API key (`secret_...`, from cloud.lambda.ai/api-keys).

**`~/.ssh/lambda_piper`** — the private key registered on Lambda as **`thor-piper`**. If lost, delete that key
in the Lambda console, generate a new one, register it, and update `LAMBDA_SSH_KEY_NAME` in
`piper-openpi-V2/training/cloud/lambda.env`:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/lambda_piper -C piper-training && chmod 600 ~/.ssh/lambda_piper
```

**`piper-openpi-V2/training/cloud/lambda.env`** — gitignored. Recreate:

```bash
cat > ~/Documents/SFF-AgileX-Piper/piper-openpi-V2/training/cloud/lambda.env <<'EOF'
LAMBDA_API_KEY=$(grep -o 'secret_[A-Za-z0-9_.-]*' ~/Documents/HG-TOKEN.txt | tail -1)
LAMBDA_KEY=$HOME/.ssh/lambda_piper
LAMBDA_SSH_KEY_NAME=thor-piper
AUTO_TERMINATE=1
EOF
```

It reads the key at source time so the secret stays in one file. `teleop/certs/` (HTTPS for the Quest) is also
gitignored and regenerable.

## 5. Claude Code memory

```bash
mkdir -p ~/.claude/projects/-home-jetson/memory
cp ~/Documents/SFF-AgileX-Piper/docs/agent-memory/*.md ~/.claude/projects/-home-jetson/memory/
rm ~/.claude/projects/-home-jetson/memory/README.md      # explains the snapshot; not a memory itself
```

See `docs/agent-memory/README.md`. Re-snapshot after a session that learns something worth keeping.

## 6. Data and the trained policy

```bash
source ~/lerobot-venv/bin/activate && cd ~/Documents/SFF-AgileX-Piper/piper-openpi-V2
bash training/pull_checkpoint.sh royc1981/lerobot-pi05-base-50    # 9.35 GB, sha256-verified
python tools/inspect_dataset.py royc1981/piper_block_bowl --strict  # pulls into the HF cache and gates it
```

The taught trajectories are already in the working tree from the clone — nothing to pull.

## 7. Confirm end to end

```bash
python -m inference.run_policy --checkpoint outputs/lerobot-pi05-base-50/pretrained_model --dry-run
```

Expect ~420 ms/call, `late 0`, GPU peak ~8.9 GB. Then `--shadow` with the cameras, then live with
`--n-action-steps 30` (see `piper-openpi-V2/docs/CONTRACT.md` for why that flag is not optional).

## Also worth keeping off-machine

`~/Documents/piper-openpi-V2-history.bundle` (348 KB) — the 18-commit git history of `piper-openpi-V2` from
before it was folded into this repo as plain files. Not needed to run anything; restore with
`git clone piper-openpi-V2-history.bundle`. Copy it somewhere safe or accept losing it.
