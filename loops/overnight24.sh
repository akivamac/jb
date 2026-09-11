#!/bin/bash
# overnight24.sh — ~24h train on current data, then swap to DOUBLED data.
# Watches for the "doubled data" commit from the tablet agent (signal token
# in .data_double_signal), pulls it, then keeps training on the new data.
# Run:  nohup bash overnight24.sh > /dev/null 2>&1 &
# Safe to re-run. Logs to overnight24.log.

set -u
REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO" || exit 1
LOG="$REPO/overnight24.log"

# The Mac's training stack pushes via the remote named "jb"
# (see git_push in training/train_expert.py). Use that remote if present.
if git remote | grep -qx jb; then REMOTE=jb; else REMOTE=origin; fi
echo "using remote: $REMOTE" >> "$LOG"

# ---- signal setup (also saved in .data_double_signal) ----
SIGNAL_FILE="$REPO/.data_double_signal"
SIGNAL_TOKEN="DOUBLE-DONE"
[ -f "$SIGNAL_FILE" ] && SIGNAL_TOKEN="$(head -1 "$SIGNAL_FILE" | tr -d '[:space:]')"
[ -z "$SIGNAL_TOKEN" ] && SIGNAL_TOKEN="DOUBLE-DONE"
echo "signal token: $SIGNAL_TOKEN" >> "$LOG"

PHASE1_HOURS=${1:-24}          # time spent on current data (default 24h)
PHASE2_HOURS=${2:-24}          # time spent on doubled data (default 24h)
BR="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

ALL_EXPERTS="greeting emotion knowledge coding cot python horse fish reptiles tree"
STEPS_PER_PASS="1000"

# Keep the local tree clean before any merge.
commit_wip() {
  git add -A >/dev/null 2>&1
  git -c user.name="akivamac" -c user.email="akivamac@k4r.org" \
    commit -m "overnight24: wip logs" >/dev/null 2>&1 || true
}

signal_present() {
  timeout 60 git fetch "$REMOTE" "$BR" 2>/dev/null || return 1
  git log --oneline -30 "$REMOTE/$BR" | grep -q "$SIGNAL_TOKEN"
}

# Run one pass: all experts, max 3 parallel, --push 1000.
run_pass() {
  local pids=() n=0 entry name rc
  for name in $ALL_EXPERTS; do
    python3 training/train_expert.py --name "$name" --resume --steps "$STEPS_PER_PASS" \
        --push 1000 --log 100 --sample 100 \
        > "data/experts/$name/overnight24.log" 2>&1 &
    pids+=("$!")
    n=$((n+1))
    if [ "$n" -ge 3 ]; then
      for p in "${pids[@]}"; do wait "$p" || log "FAILED subset pid $p"; done
      pids=(); n=0
    fi
  done
  for p in "${pids[@]}"; do wait "$p" || log "FAILED pid $p"; done
}

case "$1" in
  --train-new|phase2)
    # Jump straight to doubled-data training.
    for name in $ALL_EXPERTS; do
      [ -f "data/experts/$name/training.lock" ] && {
        log "waiting lock for $name"; while [ -f "data/experts/$name/training.lock" ]; do sleep 60; done; }
    done
    commit_wip
    log "PHASE 2 on doubled data (${PHASE2_HOURS}h x ${STEPS_PER_PASS}/pass)"
    PH1_END=$(( $(date +%s) + PHASE2_HOURS*3600 ))
    while [ "$(date +%s)" -lt "$PH1_END" ]; do run_pass; log "pass finished"; done
    commit_wip
    git push -q "$REMOTE" HEAD:main 2>/dev/null && log "final push ok" || log "final push failed"
    log "DONE."
    exit 0
    ;;
esac

# ============================ PHASE 1 ======================================
# Wait for any in-flight training (trees etc) to finish.
for name in $ALL_EXPERTS; do
  if [ -f "data/experts/$name/training.lock" ]; then
    log "WAIT lock: $name"; while [ -f "data/experts/$name/training.lock" ]; do sleep 60; done
  fi
done
commit_wip

PH1_END=$(( $(date +%s) + PHASE1_HOURS*3600 ))
log "PHASE 1 on current data (${PHASE1_HOURS}h, ${STEPS_PER_PASS}/pass/expert)"
while [ "$(date +%s)" -lt "$PH1_END" ]; do
  run_pass
  log "pass finished"
  commit_wip
  if signal_present; then
    log "SIGNAL FOUND ($SIGNAL_TOKEN) — doubled data is on the remote."
    break
  fi
done

# If still no signal, wait for it (checks every 5 min).
while ! signal_present; do
  log "no doubled-data commit yet — checking again in 5min"
  sleep 300
done

log "doubled-data commit confirmed; pulling it in..."
git -c user.name="akivamac" -c user.email="akivamac@k4r.org" pull --rebase "$REMOTE" "$BR" \
    >> "$LOG" 2>&1 || { log "rebased? if this failed, resolve and re-run 'bash $0 phase2'"; exit 1; }

# ============================ PHASE 2 ======================================
exec bash "$0" --train-new "$PHASE2_HOURS"