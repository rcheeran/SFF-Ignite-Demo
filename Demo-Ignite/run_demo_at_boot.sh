#!/usr/bin/env bash
# Start the Ignite demo once the robot link and services are ready.
# Run by ignite-demo.service at boot; see EVENT_DAY.md ("Automatic start at boot").
#   --wait-only   perform the readiness waits, then exit without moving the arm
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${IGNITE_PYTHON:-/home/clouduser/venv-ignite-demo/bin/python}"
DEMO_ARGS=(--allow-motion --detector opencv --verbose --max-cycles 0)
REQUIRED_TIMEOUT_SECONDS=900   # camera, hand, CAN, and preflight
YOLO_TIMEOUT_SECONDS=300       # YOLO is optional for --detector opencv

cd "$REPO" || exit 1
mkdir -p Demo-Ignite/logs
LOG="Demo-Ignite/logs/run-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -i -a "$LOG") 2>&1

say() { echo "[boot $(date +%H:%M:%S)] $*"; }
healthy() { curl -fs -m 5 "http://127.0.0.1:$1/healthz" 2>/dev/null | grep -q '"status": "ok"'; }
can_up() { ip link show can_piper 2>/dev/null | grep -q "state UP"; }

say "waiting for CAN link, camera (8094), hand (8095), and YOLO detection (8096); log: $LOG"
start=$(date +%s)
yolo_warned=0
while :; do
    elapsed=$(( $(date +%s) - start ))
    missing=()
    can_up || missing+=("CAN can_piper")
    healthy 8094 || missing+=("camera 8094")
    healthy 8095 || missing+=("hand 8095")
    yolo_missing=0
    healthy 8096 || yolo_missing=1
    if [ ${#missing[@]} -eq 0 ] && { [ $yolo_missing -eq 0 ] || [ $elapsed -ge $YOLO_TIMEOUT_SECONDS ]; }; then
        if [ $yolo_missing -eq 1 ] && [ $yolo_warned -eq 0 ]; then
            say "YOLO detection (8096) still not healthy after ${YOLO_TIMEOUT_SECONDS}s; continuing because the demo uses --detector opencv"
            yolo_warned=1
        fi
        if "$PYTHON" Demo-Ignite/demo.py --check-only --detector opencv >/tmp/ignite-preflight.$$ 2>&1; then
            rm -f /tmp/ignite-preflight.$$
            break
        fi
        missing+=("preflight: $(tail -1 /tmp/ignite-preflight.$$)")
    fi
    [ $yolo_missing -eq 1 ] && missing+=("YOLO 8096")
    if [ $elapsed -ge $REQUIRED_TIMEOUT_SECONDS ]; then
        say "giving up after ${elapsed}s; not ready: ${missing[*]}"
        rm -f /tmp/ignite-preflight.$$
        exit 1
    fi
    [ $(( elapsed % 30 )) -lt 5 ] && say "still waiting (${elapsed}s): ${missing[*]}"
    sleep 5
done
say "all checks passed after $(( $(date +%s) - start ))s"

if [ "${1:-}" = "--wait-only" ]; then
    say "--wait-only: not starting the demo"
    exit 0
fi
say "starting demo: demo.py ${DEMO_ARGS[*]}"
exec "$PYTHON" -u Demo-Ignite/demo.py "${DEMO_ARGS[@]}"
