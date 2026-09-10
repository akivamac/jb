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
ERRS="$REPO/overnight_errors.log"
ALL="tree reptiles fish emotion cot horse greeting knowledge coding python"
STEPS=1000
STOP_FILE="$REPO/.stop_overnight"
export GIT_TERMINAL_PROMPT=0

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
fail() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" | tee -a "$LOG" >>"$ERRS"; }

push_pass() {
  # Commit + push the finished pass. The commit ALWAYS happens locally (weights
  # are safe); a failed push must NOT kill the loop — report it and continue.
  local msg="$1" pid pid_child
  git add -A >/dev/null 2>&1
  if git diff --cached --quiet 2>/dev/null; then
    log "nothing new staged - not committing"
    return 0
  fi
  git commit -m "$msg" >/dev/null 2>&1 || { fail "git commit: $msg"; return 0; }
  local attempt
  for attempt in 1 2 3; do
    ( git push -q origin main 2>/dev/null & pid=$!; ( sleep 120; kill "$pid" 2>/dev/null ) & pid_child=$!; wait "$pid"; kill "$pid_child" 2>/dev/null ) \
      && { log "pass pushed (attempt $attempt)"; return 0; }
    fail "push attempt $attempt failed (no wifi? committing was OK)" 
    [ "$attempt" -lt 3 ] && sleep 30
  done
  return 0
}

pull_latest() {
  git fetch --no-tags -q origin 2>/dev/null || return 0
  if git rev-list --count HEAD..origin/main 2>/dev/null | grep -q .; then
    if git status --porcelain | grep -q .; then
      log "origin has updates but working tree is dirty - rebase later"
    else
      git pull --rebase -q origin main 2>/dev/null \
        && log "pulled latest from origin" || fail "pull --rebase failed"
    fi
  fi
}

run_pass() {
  local pids=() n=0 name p
  for name in $ALL; do
    [ -f "$STOP_FILE" ] && { log "stop requested - aborting pass"; return 1; }
    python3 training/train_expert.py --name "$name" --resume --steps "$STEPS" \
        --log 100 --sample 100 \
        > "data/experts/$name/overnight_loop.log" 2>&1 &
    pids+=("$!")
    n=$((n+1))
    if [ "$n" -ge 2 ]; then
      for p in "${pids[@]}"; do wait "$p" || fail "train failed (pid $p)"; done
      pids=(); n=0
    fi
  done
  for p in "${pids[@]}"; do wait "$p" || fail "train failed (pid $p)"; done
  push_pass "$1"
  return 0
}

log "waiting for current training pass to finish..."
while pgrep -f 'train_expert.py' >/dev/null 2>&1; do sleep 60; done
log "clear. starting all-night training loop (${STEPS} steps/expert/pass)"

PASS=0
while true; do
  [ -f "$STOP_FILE" ] && { log "stop file present - exiting"; break; }
  PASS=$((PASS+1))
  log "=== PASS $PASS ==="
  pull_latest
  run_pass "overnight_loop: PASS $PASS - all 10 experts +1000 steps"
  [ -f "$STOP_FILE" ] && { log "stop file present - exiting"; break; }
done
log "DONE."