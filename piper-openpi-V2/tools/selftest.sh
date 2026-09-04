#!/bin/bash
# Prove the whole stack without the robot (run before a hardware session, after any edit):
#   bash tools/selftest.sh                       pytest + teleop replay + fake recording + dataset inspection   (~3 min)
#   bash tools/selftest.sh --checkpoint <dir>    ... + policy dry-run on that checkpoint (fake arm, synthetic frames, ~4 min more)
# Full output goes to data/logs/selftest.log; the terminal gets one PASS/FAIL line per step and a summary.
cd "$(dirname "$0")/.."
source ~/lerobot-venv/bin/activate
CKPT=""; [ "${1:-}" = "--checkpoint" ] && CKPT=$2
mkdir -p data/logs
LOG=data/logs/selftest.log; : > "$LOG"
ROOT=data/selftest_fake_rec
fail=0

step() {   # step NAME CMD...  -> runs the command, tees to the log, prints one line
  local name=$1; shift
  printf '%-28s' "$name"
  local t0=$SECONDS
  if "$@" >> "$LOG" 2>&1; then echo "PASS  ($((SECONDS - t0)) s)"; else echo "FAIL  ($((SECONDS - t0)) s)  -> see $LOG"; fail=1; fi
}

step "pytest (all)"        python -m pytest tests -q
step "teleop replay"       python -m teleop.quest --replay
step "fake recording"      python -m collect.record --fake --episodes 2 --overwrite --root "$ROOT" --repo-id local/selftest_fake_rec
step "fake recording resume" python -m collect.record --fake --episodes 3 --root "$ROOT" --repo-id local/selftest_fake_rec
step "inspect dataset"     python tools/inspect_dataset.py "$ROOT"
step "cloud scripts"       bash -c 'for f in training/train.sh training/pull_checkpoint.sh training/cloud/*.sh; do bash -n "$f" || exit 1; done'
if [ -n "$CKPT" ]; then
  step "policy dry-run"    python -m inference.run_policy --checkpoint "$CKPT" --dry-run --max-steps 45
fi
rm -rf "$ROOT" "${ROOT}_contact.jpg"
echo "----"
if [ $fail = 0 ]; then echo "SELFTEST PASS  (log: $LOG)"; else echo "SELFTEST FAIL  (log: $LOG)"; fi
exit $fail
