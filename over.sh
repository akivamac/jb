#!/bin/bash
# Overnight expert training (Mac). Waits for any currently-running experts
# (lock files) to finish, then runs the queued batches sequentially,
# max 3 parallel, pushing every 500 steps. Safe to re-run.
set -u

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO" || exit 1
LOG="$REPO/overnight_train.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

run_expert() {
  local name="$1" steps="$2"
  log "START $name ($steps steps, resume)"
  python3 training/train_expert.py --name "$name" --resume --steps "$steps" \
      --push 500 --log 100 --sample 100 \
      > "data/experts/$name/overnight.log" 2>&1 &
  local pid=$!
  log "  pid $pid"
  pids+=("$pid")
}

# Wait for batch 1 experts (launched earlier) to finish before starting batch 2.
for name in tree reptiles fish; do
  lock="data/experts/$name/training.lock"
  while [ -f "$lock" ]; do
    log "WAIT $name still running (lock present), sleeping 60s"
    sleep 60
  done
done
log "Batch 1 (tree/reptiles/fish) finished."

run_batch() {
  log "=== Batch: $* ==="
  pids=()
  for entry in "$@"; do
    run_expert "${entry%%:*}" "${entry##*:}"
  done
  local ok=0
  for pid in "${pids[@]}"; do
    if wait "$pid"; then ok=1; else log "  FAILED pid $pid"; fi
  done
  log "=== Batch done ==="
}

run_batch emotion:1000 cot:1000 horse:500
run_batch greeting:500 knowledge:500 coding:500
run_batch python:500

log "All batches complete. Final commit+push."
git add -A
git commit -m "overnight: all 10 experts fine-tuned on new data" >/dev/null 2>&1 \
  || log "nothing to commit"
git push >/dev/null 2>&1 && log "pushed" || log "push failed - push manually"
log "DONE."





