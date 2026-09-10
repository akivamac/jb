#!/bin/bash
# overnight_loop.sh — keep training ALL NIGHT after overnight_train.sh finishes.
# Waits for the current training pass to complete, then runs continuous passes
# over all 10 experts (~1000 steps each/pass), max 2 parallel (RAM-safe on 8GB),
# pushing a checkpoint at the end of every pass. Safe to re-run.
# Run:  nohup bash overnight_loop.sh >/dev/null 2>&1 &
set -u

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO" || exit 1
LOG="$REPO/overnight_loop.log"
ALL="tree reptiles fish emotion cot horse greeting knowledge coding python"
STEPS=1000

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

STOP_FILE="$REPO/.stop_overnight"

run_pass() {
  local pids=() n=0 name
  for name in $ALL; do
    [ -f "$STOP_FILE" ] && { log "stop requested - aborting pass"; return 1; }
    python3 training/train_expert.py --name "$name" --resume --steps "$STEPS" \
        --log 100 --sample 100 \
        > "data/experts/$name/overnight_loop.log" 2>&1 &
    pids+=("$!")
    n=$((n+1))
    if [ "$n" -ge 2 ]; then
      for p in "${pids[@]}"; do wait "$p" || log "FAILED subset pid $p"; done
      pids=(); n=0
    fi
  done
  for p in "${pids[@]}"; do wait "$p" || log "FAILED pid $p"; done
  local ok=$?
  git add -A >/dev/null 2>&1
  git commit -m "overnight_loop: pass done - all 10 experts +1000 steps" >/dev/null 2>&1 \
    || log "nothing to commit"
  git push >/dev/null 2>&1 && log "pass pushed" || log "push failed"
  return $ok
}

log "waiting for current training pass to finish..."
while pgrep -f 'train_expert.py' >/dev/null 2>&1; do sleep 60; done
log "clear. starting all-night training loop (${STEPS} steps/expert/pass)"

PASS=0
while true; do
  [ -f "$STOP_FILE" ] && { log "stop file present - exiting"; break; }
  PASS=$((PASS+1))
  log "=== PASS $PASS ==="
  run_pass || { log "aborting on error"; break; }
done
log "DONE."