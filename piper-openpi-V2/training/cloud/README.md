# Cloud fine-tune on Lambda (copy-paste)

`launch.sh` runs on the Jetson and drives everything over ssh; `setup.sh`, `train.sh` and `hub_check.py` run on the
instance; `status.sh` / `terminate.sh` monitor and clean up; `../pull_checkpoint.sh` brings the model back.
`USER` = `hf auth whoami` (royc1981). Verified from the Jetson (2026-08-22) without an instance: the pre-flight against
the Hub, the shared `train.sh` (3 steps on the Thor), checkpoint verification (size + sha256) against `lerobot/pi05_base`,
the `status.sh` parser on a real log. NOT verified: `setup.sh` on a real Lambda image, `--policy.compile_model=true` on
x86, the H100 step time, the Lambda API path (no key).

## 1. Dataset to the Hub (Jetson)

```bash
source ~/lerobot-venv/bin/activate && cd ~/Documents/piper-openpi-V2
python -m collect.record --repo-id USER/piper_block_bowl --episodes 10 --push
```

Already recorded without `--push`:

```bash
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset as D; D('USER/piper_block_bowl', root='data/piper_block_bowl').push_to_hub(private=True)"
```

## 2. Instance

Either by hand: https://cloud.lambda.ai/instances → Launch instance → **1× H100 PCIe (80 GB, $3.29/h, 1 TiB disk)**, any
region, your SSH key, no filesystem; copy the IP once it is Running and write `training/cloud/lambda.env`:

```bash
printf 'LAMBDA_HOST=ubuntu@<instance-ip>\nLAMBDA_KEY=~/.ssh/<lambda-key>\n' > training/cloud/lambda.env
```

Or let `launch.sh` do it (https://cloud.lambda.ai/api-keys):

```bash
printf 'LAMBDA_API_KEY=<key>\nLAMBDA_KEY=~/.ssh/<lambda-key>\nAUTO_TERMINATE=1\n' > training/cloud/lambda.env
```

`LAMBDA_SSH_KEY_NAME=<name>` only if the account has several keys. `launch.sh` then picks the first region with
`gpu_1x_h100_pcie` capacity, waits for `active` (≤ 10 min), appends `LAMBDA_HOST` / `LAMBDA_INSTANCE_ID` to `lambda.env`;
`terminate.sh` removes them (if you terminate on the website instead, delete the `LAMBDA_HOST` line yourself).
`AUTO_TERMINATE=1` terminates only after a verified upload, and only instances `launch.sh` created.

- x86 only: not GH200 (arm64, the Jetson's wheel problems). 1× A100 40 GB ($1.99/h) only with `BATCH=8`, and it is
  tight (37 GB measured at batch 8); take the H100.
- 3000 steps at batch 16 on one H100: **~1–1.5 h, ~$4–6** (estimate; `launch.sh --check` prints the plan from
  `H100_STEP_S` / `H100_USD_PER_H` in `piper/config.py` — update `H100_STEP_S` after the first real run).
  Disk: one 23 GB checkpoint per 500 steps.

## 3. Launch (Jetson)

```bash
bash training/cloud/launch.sh --check USER/piper_block_bowl USER/pi05_piper   # pre-flight only, free
bash training/cloud/launch.sh USER/piper_block_bowl USER/pi05_piper           # [STEPS=3000] [BATCH=16]
```

Pre-flight (before any ssh, exit 1 on any problem): HF token (`whoami`), the dataset's `meta/info.json` against the
contract in `piper/config.py` (fps 15, `observation.images.top` + `wrist` 480×640×3 video, `observation.state` /
`action` float32[7] with the contract names, non-empty), and the run plan (epochs, hours, $). Then it scp's
`setup.sh train.sh hub_check.py requirements.txt`, runs `setup.sh` (~10 min, streams to your terminal), starts
`train.sh` in tmux and prints the monitoring commands:

```bash
bash training/cloud/status.sh [LINES]      # last log lines + "step N/M elapsed ETA loss", or the DONE / FAILED line
ssh -i ~/.ssh/<lambda-key> ubuntu@<ip> -t tmux attach -t train     # Ctrl-b d detaches
```

The log ends with `DONE (lerobot-train exit 0, upload verified): https://huggingface.co/USER/pi05_piper`. The checkpoint
goes up with `hf upload` (3 attempts, 60/120 s backoff), each attempt verified by `hub_check.py model` (size + sha256 on
the Hub), and `train.log` is uploaded next to it. On failure the log says `UPLOAD FAILED 3x ... do NOT terminate` with
the exact retry command; the checkpoint stays on the instance.

Manual equivalent (from the Jetson):

```bash
scp training/cloud/{setup.sh,train.sh,hub_check.py,requirements.txt} ubuntu@<instance-ip>:
ssh ubuntu@<instance-ip> "HF_TOKEN=$(grep -o 'hf_[A-Za-z0-9]*' ~/Documents/HG-TOKEN.txt | tail -1) bash setup.sh"
ssh ubuntu@<instance-ip> "tmux new -d -s train 'bash train.sh USER/piper_block_bowl USER/pi05_piper'"
```

Terminate as soon as the log says DONE (billing runs until then):

```bash
bash training/cloud/terminate.sh all            # or <INSTANCE-ID>; needs LAMBDA_API_KEY in lambda.env
```

or on https://cloud.lambda.ai/instances → Terminate. Raw API (Bearer auth, base URL `https://cloud.lambda.ai`):

```bash
curl -H "Authorization: Bearer $LAMBDA_API_KEY" https://cloud.lambda.ai/api/v1/instances
curl -H "Authorization: Bearer $LAMBDA_API_KEY" -X POST https://cloud.lambda.ai/api/v1/instance-operations/terminate -H "Content-Type: application/json" -d '{"instance_ids": ["<INSTANCE-ID>"]}'
```

## 4. Pull and deploy (Jetson)

```bash
bash training/pull_checkpoint.sh USER/pi05_piper   # checks the Hub, downloads to outputs/pi05_piper/pretrained_model (9.4 GB, ~25 min at 6 MB/s), verifies size + sha256, 3 dry-run steps, prints the deploy command
python -m inference.run_policy --checkpoint outputs/pi05_piper/pretrained_model --shadow      # no motion first
python -m inference.run_policy --checkpoint outputs/pi05_piper/pretrained_model
```

## Local fallback (Thor)

`bash training/train.sh smoke [dataset]` runs the cloud run's exact flags on the Thor (500 steps, batch 8, ~45 min,
37 GB GPU, one 23 GB checkpoint → `outputs/pi05_piper_smoke`, log `~/train.log`); `full` = 3000 steps, batch 16, ~9 h →
`outputs/pi05_piper_local`. Both wrap `training/cloud/train.sh` (one flag set for both machines; `DATASET` may be a
local dataset root, `MODEL` `-` = no upload).

## HF Jobs instead of Lambda (not run)

```bash
lerobot-train <the flags from train.sh> --job.target=a100-large --job.timeout=6h
```

`hf jobs hardware` lists flavors and prices; the job runs in `huggingface/lerobot-gpu:latest` with your token (license
accepted), pushes to `--policy.repo_id`; pull as in step 4.

## If it breaks

| symptom | fix |
|---|---|
| `launch.sh`: `ERROR: dataset ... violates the contract` / `HF token rejected` / `RepositoryNotFoundError` | fix before launching (nothing was billed) |
| `setup.sh`: `torch.cuda.is_available() is False` | driver older than CUDA 12.8 needs (≥ 525): pick another instance / image |
| torch.compile error or a >15 min stall before step 1 | `ssh ubuntu@<ip> "tmux new -d -s train 'COMPILE=false bash train.sh USER/piper_block_bowl USER/pi05_piper'"` |
| CUDA out of memory | `bash training/cloud/launch.sh USER/piper_block_bowl USER/pi05_piper 3000 8` |
| `duplicate session: train` | a run is already going: `ssh -t ubuntu@<ip> tmux attach -t train` |
| `UPLOAD FAILED 3x` in train.log | the checkpoint is on the instance: run the printed `hf upload ... && python ~/hub_check.py model ...` by hand; do not terminate |
| `status.sh`: ssh times out | instance terminated or `LAMBDA_HOST` stale: `bash training/cloud/terminate.sh all` cleans `lambda.env` |
| ssh: `REMOTE HOST IDENTIFICATION HAS CHANGED` | `ssh-keygen -R <ip>` (Lambda reuses IPs; `launch.sh` does this for instances it creates) |
| `pull_checkpoint.sh`: `has no model.safetensors` / `incomplete copy` | the upload was never verified (check `train.log` on the instance or in the model repo) / re-run the pull |
