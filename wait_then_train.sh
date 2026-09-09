#!/bin/bash
# wait_then_train.sh P1 P2 P3  (Mac side)
# Waits for three training PIDs to exit, then runs the queued batches
# (max 3 parallel, --resume, --push every 1000). Deferred doubling.
# Usage:  nohup bash wait_then_train.sh <PID1> <PID2> <PID3> > wait_then_train.log 2>&1 &
#         ./wait_then_train.sh PIDS=(...)   or pass pids positionally
set -u

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO" || exit 1
LOG="$REPO/wait_then_train.log"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# initial splash
echo "START wait_then_train at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
echo "given args: $*" >> "$LOG"

log "waiting for pids: $*"
for p in "$@"; do
  log "  wait pid $p"
  while kill -0 "$p" 2>/dev/null; do sleep 60; done
  log "  pid $p done"
done
log "ALL PIDS DONE"

# ---------- queued batches (first = finish stranded trio) ----------
BATCH1="emotion:1500 cot:1500 horse:1500"   # resume 500 -> 2000
BATCH2="greeting:500 knowledge:500 python:500"  # top-up with fresh data

run_batch() {
  log "=== Batch: $* ==="
  pids=()
  for entry in "$@"; do
    name="${entry%%:*}"; steps="${entry##*:}"
    log "START $name (+$steps steps, resume)"
    python3 training/train_expert.py --name "$name" --resume --steps "$steps" \
        --push 1000 --log 100 --sample 100 \
        > "data/experts/$name/wait_then.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || log "  FAILED pid $pid"
  done
  log "=== Batch done ==="
}

run_batch $BATCH1
run_batch $BATCH2

log "All batches done. Final commit+push."
git add -A >/dev/null 2>&1
git -c user.name="akivamac" -c user.email="akivamac@k4r.org" \
    commit -m "chore: finished stranded emotion/cot/horse to 2000, topped up greeting/knowledge/python" >/dev/null 2>&1 \
  || log "nothing to commit"
if git remote | grep -qx jb; then R=jb; else R=origin; fi
git push -q "$R" HEAD:main 2>/dev/null && log "pushed to $R" || log "push failed - push manually: git push $R HEAD:main"
log "DONE."