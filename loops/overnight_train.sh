#!/bin/bash
# Overnight expert training (Mac). Runs all 10 experts via resume,
# max 2 parallel given ~2.2GB RSS per 512-seq process on 8GB RAM.
# Pushes snapshots every 1000 steps. Safe to re-run. Driven by nohup.
set -u

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO" || exit 1
LOG="$REPO/overnight_train.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

run_expert() {
  local name="$1" steps="$2"
  log "START $name ($steps steps, resume)"
  nohup python3 training/train_expert.py --name "$name" --resume --steps "$steps" \
      --push 1000 --log 100 --sample 100 \
      > "data/experts/$name/overnight.log" 2>&1 &
  local pid=$!
  log "  pid $pid"
  pids+=("$pid")
}

run_batch() {
  log "=== Batch: $* ==="
  pids=()
  for entry in "$@"; do
    run_expert "${entry%%:*}" "${entry##*:}"
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  log "=== Batch done ==="
}

run_batch tree:1500 reptiles:1500
run_batch fish:1500 emotion:1500
run_batch cot:1500 horse:1500
run_batch greeting:1500 knowledge:1500
run_batch coding:1500 python:1500

log "All batches complete. Final commit+push."
git add -A
git commit -m "overnight: all 10 experts fine-tuned on doubled data" >/dev/null 2>&1 \
  || log "nothing to commit"
git push >/dev/null 2>&1 && log "pushed" || log "push failed - push manually"
log "DONE."