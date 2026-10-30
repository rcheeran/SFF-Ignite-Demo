#!/bin/bash
# pi05 fine-tune - ONE lerobot-train flag set for the cloud and the Thor (training/train.sh wraps this locally).
# bash train.sh DATASET_REPO MODEL_REPO [STEPS=3000] [BATCH=16]   (on the Lambda instance, after setup.sh)
# bash train.sh DATASET      -          [STEPS] [BATCH]           (no upload; DATASET may also be a local dataset root)
# COMPILE=false bash train.sh ...   if torch.compile fails on the instance.   OUT=<dir> overrides ~/outputs/<run>.
# Log: ~/train.log. Prints DONE only once the Hub holds model.safetensors with the local size and sha256;
# on an upload failure the checkpoint stays on the instance and the log says what to run by hand.
set -euo pipefail
USAGE="usage: $0 DATASET_REPO MODEL_REPO|- [STEPS=3000] [BATCH=16]"
DATASET=${1:?$USAGE}
MODEL=${2:?$USAGE}
STEPS=${3:-3000}
BATCH=${4:-16}
source ~/lerobot-venv/bin/activate
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -f ~/train.env ]; then . ~/train.env; fi        # launch.sh: LAMBDA_API_KEY, LAMBDA_INSTANCE_ID, AUTO_TERMINATE

OUT=${OUT:-~/outputs/$(basename "$MODEL")_$(date +%Y%m%d-%H%M%S)}
RUN=$(basename "$OUT")
CKPT=$OUT/checkpoints/last/pretrained_model
LOG=~/train.log

if [ -f "$DATASET/meta/info.json" ]; then
  DS=(--dataset.repo_id="local/$(basename "$DATASET")" --dataset.root="$DATASET")
else
  DS=(--dataset.repo_id="$DATASET")
fi
ARGS=(
  "${DS[@]}"
  --dataset.video_backend=pyav
  --policy.path=lerobot/pi05_base
  --policy.device=cuda
  --policy.dtype=bfloat16
  --policy.gradient_checkpointing=true
  --policy.compile_model="${COMPILE:-true}"
  --policy.push_to_hub=false
  --rename_map='{"observation.images.top": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}'
  --num_workers=4
  --log_freq=10
  --wandb.enable=false
  --batch_size="$BATCH"
  --steps="$STEPS"
  --save_freq=500
  --output_dir="$OUT"
  --job_name="$RUN"
)

{
  echo "== $(date -Is) dataset=$DATASET model=$MODEL steps=$STEPS batch=$BATCH out=$OUT"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
  df -h ~ | tail -1
  printf '+ lerobot-train'; printf ' %q' "${ARGS[@]}"; echo
} | tee "$LOG"

set +e
lerobot-train "${ARGS[@]}" 2>&1 | tee -a "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

if [ ! -f "$CKPT/model.safetensors" ]; then
  echo "FAILED: lerobot-train exit $STATUS, no checkpoint under $OUT (see $LOG)" | tee -a "$LOG"
  exit 1
fi
if [ "$STATUS" != 0 ]; then       # never replace a good model with a partial one, never terminate a failed run
  cat <<MSG | tee -a "$LOG"
FAILED: lerobot-train exit $STATUS (see $LOG). Last checkpoint: $CKPT - not uploaded to $MODEL.
  resume:  lerobot-train --resume=true --config_path=$OUT/checkpoints/last/pretrained_model/train_config.json
  upload anyway:  hf upload $MODEL-partial $CKPT . --private
MSG
  exit "$STATUS"
fi
if [ "$MODEL" = - ]; then
  echo "DONE (lerobot-train exit $STATUS, not uploaded): $CKPT" | tee -a "$LOG"
  exit "$STATUS"
fi

UPLOADED=
for wait in 60 120 0; do      # 3 attempts, 1 min then 2 min backoff
  if hf upload "$MODEL" "$CKPT" . --private --commit-message "pi05 fine-tune: $DATASET, $STEPS steps, batch $BATCH" 2>&1 | tee -a "$LOG" \
     && python -u "$HERE/hub_check.py" model "$MODEL" "$CKPT/model.safetensors" 2>&1 | tee -a "$LOG"; then
    UPLOADED=1
    break
  fi
  if [ "$wait" -gt 0 ]; then
    echo "upload not verified, retrying in $wait s" | tee -a "$LOG"
    sleep "$wait"
  fi
done
if [ -z "$UPLOADED" ]; then
  cat <<MSG | tee -a "$LOG"
UPLOAD FAILED 3x. The checkpoint is safe on this instance: $CKPT - do NOT terminate it. Retry by hand:
  hf upload $MODEL $CKPT . --private && python $HERE/hub_check.py model $MODEL $CKPT/model.safetensors
MSG
  exit 1
fi
echo "DONE (lerobot-train exit $STATUS, upload verified): https://huggingface.co/$MODEL" | tee -a "$LOG"
hf upload "$MODEL" "$LOG" train.log --commit-message "train.log" >/dev/null 2>&1 || true   # keep the evidence with the model

if [ "${AUTO_TERMINATE:-0}" = 1 ] && [ -n "${LAMBDA_API_KEY:-}" ] && [ -n "${LAMBDA_INSTANCE_ID:-}" ]; then
  echo "AUTO_TERMINATE=1: terminating this instance ($LAMBDA_INSTANCE_ID)" | tee -a "$LOG"
  if curl -fsS --retry 5 --retry-all-errors --max-time 60 -X POST https://cloud.lambda.ai/api/v1/instance-operations/terminate \
       -H "Authorization: Bearer $LAMBDA_API_KEY" -H "Content-Type: application/json" \
       -d "{\"instance_ids\": [\"$LAMBDA_INSTANCE_ID\"]}" | tee -a "$LOG" | grep -q "$LAMBDA_INSTANCE_ID"; then
    echo "terminated $LAMBDA_INSTANCE_ID" | tee -a "$LOG"
  else
    echo "TERMINATE FAILED - billing continues: bash training/cloud/terminate.sh $LAMBDA_INSTANCE_ID" | tee -a "$LOG"
    exit 1
  fi
fi
exit "$STATUS"
