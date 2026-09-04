#!/bin/bash
# bash training/cloud/launch.sh [--check] [--wait SECONDS] DATASET_REPO MODEL_REPO [STEPS=3000] [BATCH=16]
# Pre-flight (HF token, dataset contract, run plan + cost) -> instance -> setup.sh -> train.sh in tmux.
# --check stops after the pre-flight (nothing launched, nothing billed).
# training/cloud/lambda.env, either:
#   LAMBDA_HOST=ubuntu@<ip>  LAMBDA_KEY=~/.ssh/<key>                       instance launched by hand on cloud.lambda.ai
#   LAMBDA_API_KEY=<key>     LAMBDA_KEY=~/.ssh/<key>  [LAMBDA_SSH_KEY_NAME=<name>]  [AUTO_TERMINATE=1]
#                            launch.sh creates a 1x H100 PCIe itself and appends LAMBDA_HOST/LAMBDA_INSTANCE_ID here;
#                            terminate.sh removes them. AUTO_TERMINATE=1: train.sh terminates after a verified upload.
set -euo pipefail
cd "$(dirname "$0")/../.."
source ~/lerobot-venv/bin/activate
export HF_TOKEN=${HF_TOKEN:-$(grep -o 'hf_[A-Za-z0-9]*' ~/Documents/HG-TOKEN.txt | tail -1)}
ENV=training/cloud/lambda.env
if [ -f $ENV ]; then set -a; . ./$ENV; set +a; fi

CHECK=; WAIT=
while :; do case "${1:-}" in
  --check) CHECK=1; shift ;;
  --wait)  WAIT=${2:-14400}; shift 2 ;;      # poll for capacity first (nothing billed while waiting)
  *) break ;;
esac; done
USAGE="usage: $0 [--check] DATASET_REPO MODEL_REPO [STEPS=3000] [BATCH=16]"
DATASET=${1:?$USAGE}
MODEL=${2:?$USAGE}
STEPS=${3:-3000}
BATCH=${4:-16}

python -m training.cloud.hub_check dataset "$DATASET" "$STEPS" "$BATCH"
HF_USER=$(python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])")
[[ $MODEL == "$HF_USER"/* ]] || { echo "ERROR: MODEL_REPO must be $HF_USER/<name> (the token can only write there)"; exit 1; }
echo "model: https://huggingface.co/$MODEL (private; overwritten if it exists)"
if [ -n "$CHECK" ]; then echo "--check: pre-flight OK, nothing launched"; exit 0; fi
if [ -n "$WAIT" ] && [ -n "${LAMBDA_API_KEY:-}" ] && [ -z "${LAMBDA_HOST:-}" ]; then
  python -m training.cloud.lambda_api wait "${INSTANCE_TYPE:-gpu_1x_h100_pcie}" "$WAIT" || exit 1
fi

INSTANCE_ID=${LAMBDA_INSTANCE_ID:-}
if [ -z "${LAMBDA_HOST:-}" ] && [ -n "$INSTANCE_ID" ]; then
  echo "LAMBDA_INSTANCE_ID=$INSTANCE_ID is in $ENV without LAMBDA_HOST: a launch did not finish booting."
  echo "  python -m training.cloud.lambda_api list   then add LAMBDA_HOST=ubuntu@<ip> or bash training/cloud/terminate.sh $INSTANCE_ID"
  exit 1
fi
if [ -z "${LAMBDA_HOST:-}" ] && [ -n "${LAMBDA_API_KEY:-}" ]; then
  exec 3< <(python -m training.cloud.lambda_api launch)     # line 1: instance id (billing started), line 2: ip when active
  read -r INSTANCE_ID <&3
  printf 'LAMBDA_INSTANCE_ID=%s\n' "$INSTANCE_ID" >> $ENV   # persisted before the boot poll: a failure below is not orphaned
  read -r IP <&3 || { echo "instance $INSTANCE_ID did not come up: bash training/cloud/terminate.sh $INSTANCE_ID"; exit 1; }
  exec 3<&-
  LAMBDA_HOST=ubuntu@$IP
  printf 'LAMBDA_HOST=%s\n' "$LAMBDA_HOST" >> $ENV
  ssh-keygen -R "$IP" >/dev/null 2>&1 || true      # Lambda reuses IPs; this machine is new
fi
: "${LAMBDA_HOST:?LAMBDA_HOST=ubuntu@<ip> or LAMBDA_API_KEY missing in $ENV (header of this script)}"
OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 ${LAMBDA_KEY:+-i "$LAMBDA_KEY"})
SSH="ssh ${LAMBDA_KEY:+-i $LAMBDA_KEY }$LAMBDA_HOST"

tries=0
until ssh "${OPTS[@]}" "$LAMBDA_HOST" true; do
  (( ++tries < 12 )) || { echo "ssh to $LAMBDA_HOST keeps failing (still booting? LAMBDA_KEY?)"; exit 1; }
  sleep 10
done
scp "${OPTS[@]}" training/cloud/{setup.sh,train.sh,hub_check.py,requirements.txt} "$LAMBDA_HOST:"
ssh "${OPTS[@]}" "$LAMBDA_HOST" 'HF_TOKEN=$(cat) bash ~/setup.sh' <<< "$HF_TOKEN"
TRAIN_ENV=                                         # always rewritten: no stale auto-terminate on a reused instance
if [ "${AUTO_TERMINATE:-0}" = 1 ] && [ -n "$INSTANCE_ID" ]; then
  TRAIN_ENV="LAMBDA_API_KEY=$LAMBDA_API_KEY"$'\n'"LAMBDA_INSTANCE_ID=$INSTANCE_ID"$'\n'"AUTO_TERMINATE=1"
fi
ssh "${OPTS[@]}" "$LAMBDA_HOST" 'umask 077; cat > ~/train.env' <<< "$TRAIN_ENV"
ssh "${OPTS[@]}" "$LAMBDA_HOST" "tmux new-session -d -s train \"bash ~/train.sh '$DATASET' '$MODEL' $STEPS $BATCH\""

cat <<MSG

training started on $LAMBDA_HOST (tmux session 'train', survives a dropped ssh)
  status:     bash training/cloud/status.sh
  attach:     $SSH -t tmux attach -t train        (Ctrl-b d detaches)
  finished:   the log ends with  DONE ... upload verified ... https://huggingface.co/$MODEL
  then:       bash training/pull_checkpoint.sh $MODEL
  terminate:  ${INSTANCE_ID:+bash training/cloud/terminate.sh $INSTANCE_ID   or  }https://cloud.lambda.ai/instances  (billing runs until you do${TRAIN_ENV:+; AUTO_TERMINATE=1 armed for $INSTANCE_ID})
${TRAIN_ENV:+  auto-terminate armed (LAMBDA_API_KEY sent to ~/train.env on the instance, umask 077)}${TRAIN_ENV:-$([ "${AUTO_TERMINATE:-0}" = 1 ] && echo "  AUTO_TERMINATE=1 ignored: no LAMBDA_INSTANCE_ID (instance not created by launch.sh)")}
MSG
