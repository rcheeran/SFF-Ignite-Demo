#!/bin/bash
# bash training/cloud/status.sh [LINES=5]   (from the Jetson) - last lines of the cloud train.log + step / loss / ETA
set -euo pipefail
cd "$(dirname "$0")"
if [ -f lambda.env ]; then . ./lambda.env; fi
: "${LAMBDA_HOST:?LAMBDA_HOST missing (training/cloud/lambda.env)}"
ssh -o ConnectTimeout=15 ${LAMBDA_KEY:+-i "$LAMBDA_KEY"} "$LAMBDA_HOST" 'tail -c 300000 ~/train.log' | python3 -c '
import re, sys
text = sys.stdin.read()
lines = [l for l in text.replace("\r", "\n").splitlines() if l.strip()]
print(*lines[-int(sys.argv[1]):], sep="\n")
bars = re.findall(r"(\d+)/(\d+) \[([\d:]+)<([\d:?]+)", text)          # tqdm: 1234/3000 [12:34<15:20
loss = re.findall(r"loss:(\S+)", text)
for l in reversed(lines):
    if l.startswith(("DONE", "FAILED", "UPLOAD FAILED")):
        print("--", l)
        break
else:
    if bars:
        step, total, elapsed, eta = bars[-1]
        print(f"-- step {step}/{total}  elapsed {elapsed}  ETA {eta}" + (f"  loss {loss[-1]}" if loss else ""))
' "${1:-5}"
