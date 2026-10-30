#!/bin/bash
# bash training/cloud/terminate.sh ID|all   (from the Jetson; LAMBDA_API_KEY in training/cloud/lambda.env)
# Terminates via the Lambda API and drops the LAMBDA_HOST/LAMBDA_INSTANCE_ID lines launch.sh appended to lambda.env.
set -euo pipefail
cd "$(dirname "$0")/../.."
ENV=training/cloud/lambda.env
if [ -f $ENV ]; then set -a; . ./$ENV; set +a; fi
python -m training.cloud.lambda_api terminate "${1:?usage: $0 ID|all}"
if [ -f $ENV ] && { [ "$1" = all ] || [ "$1" = "${LAMBDA_INSTANCE_ID:-}" ]; }; then
  sed -i '/^LAMBDA_HOST=\|^LAMBDA_INSTANCE_ID=/d' $ENV        # only the instance launch.sh recorded
fi
