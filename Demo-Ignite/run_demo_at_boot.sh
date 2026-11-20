#!/usr/bin/env bash
# Start the Ignite demo once the robot link and services are ready.
# Run by ignite-demo.service at boot; see EVENT_DAY.md ("Automatic start at boot").
#   --wait-only   perform the readiness waits, then exit without moving the arm
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${IGNITE_PYTHON:-/home/clouduser/venv-ignite-demo/bin/python}"
DEMO_ARGS=(--allow-motion --detector opencv --verbose --max-cycles 0)
#DEMO_ARGS=(--check-only --detector opencv --verbose --max-cycles 0)
REQUIRED_TIMEOUT_SECONDS=900   # camera, hand, CAN, and preflight

cd "$REPO" || exit 1
mkdir -p Demo-Ignite/logs
LOG="Demo-Ignite/logs/run-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -i -a "$LOG") 2>&1

say() { echo "[boot $(date +%H:%M:%S)] $*"; }
healthy() { curl -fs -m 5 "http://127.0.0.1:$1/healthz" 2>/dev/null | grep -q '"status": "ok"'; }
can_up() { ip link show can_piper 2>/dev/null | grep -q "state UP"; }
diagnostic() {
    local status
    say "timeout diagnostic: $*"
    timeout --kill-after=5s 20s "$@"
    status=$?
    say "diagnostic exit status: $status (124 means timed out)"
}

timeout_diagnostics() {
    diagnostic sudo -n systemctl status k3s --no-pager --full
    diagnostic sudo -n k3s kubectl --request-timeout=10s get pods -A -o wide
    diagnostic curl -v --max-time 5 http://127.0.0.1:8094/healthz
    diagnostic curl -v --max-time 5 http://127.0.0.1:8095/healthz
}

uptime_seconds() {
    local uptime idle
    if ! read -r uptime idle </proc/uptime || [[ ! "$uptime" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        say "cannot read system uptime; refusing to start demo" >&2
        return 1
    fi
    echo "${uptime%%.*}"
}

say "waiting for CAN link, camera (8094), and hand (8095); log: $LOG"
start=$(uptime_seconds) || exit 1
while :; do
    now=$(uptime_seconds) || exit 1
    elapsed=$(( now - start ))
    missing=()
    can_up || missing+=("CAN can_piper")
    healthy 8094 || missing+=("camera 8094")
    healthy 8095 || missing+=("hand 8095")
    if [ ${#missing[@]} -eq 0 ]; then
        if "$PYTHON" Demo-Ignite/demo.py --check-only --detector opencv >/tmp/ignite-preflight.$$ 2>&1; then
            rm -f /tmp/ignite-preflight.$$
            break
        fi
        missing+=("preflight: $(tail -1 /tmp/ignite-preflight.$$)")
    fi
    if [ $elapsed -ge $REQUIRED_TIMEOUT_SECONDS ]; then
        say "giving up after ${elapsed}s; not ready: ${missing[*]}"
        timeout_diagnostics
        rm -f /tmp/ignite-preflight.$$
        exit 1
    fi
    [ $(( elapsed % 30 )) -lt 5 ] && say "still waiting (${elapsed}s): ${missing[*]}"
    sleep 5
done
now=$(uptime_seconds) || exit 1
say "all checks passed after $(( now - start ))s"

if [ "${1:-}" = "--wait-only" ]; then
    say "--wait-only: not starting the demo"
    exit 0
fi
say "starting demo: demo.py ${DEMO_ARGS[*]}"
exec "$PYTHON" -u Demo-Ignite/demo.py "${DEMO_ARGS[@]}"
