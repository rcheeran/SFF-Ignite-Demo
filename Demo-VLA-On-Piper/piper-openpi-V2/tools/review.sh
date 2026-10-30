#!/bin/bash
# bash tools/review.sh <dataset root> [episode=0]   - replay one episode in rerun's web viewer, served from the Jetson
# Open the printed URL on the laptop (same WiFi). Ctrl-C stops the server.
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/lerobot-venv/bin/activate
ROOT=${1:?usage: $0 <dataset root> [episode]}
EP=${2:-0}
pkill -f "bin/lerobot-dataset-viz" 2>/dev/null && sleep 1 || true   # a stale viewer holding the ports = 'Load failed' in the browser
IP=$(python -c "from piper import config as C; print(C.TELEOP_URL.split('//')[1].split(':')[0])")
REPO=local/$(basename "$ROOT")
echo "open on the laptop:  http://$IP:9090/?url=rerun%2Bhttp://$IP:9876/proxy"
exec lerobot-dataset-viz --repo-id "$REPO" --root "$ROOT" --episode-index "$EP" --mode distant --web-port 9090 --grpc-port 9876
