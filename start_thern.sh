#!/bin/bash
# Robust thern.py launcher using exec to avoid duplicate processes
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCK="/tmp/thern.lock"
trap 'rm -f /tmp/thern.lock; exit' SIGTERM SIGINT
if [ -f "$LOCK" ]; then
  OLD=$(cat "$LOCK")
  if kill -0 "$OLD" 2>/dev/null; then
    echo "thern.py already running (PID $OLD)"
    exit 1
  fi
  rm -f "$LOCK"
fi
cd "$REPO_DIR"
echo $$ > "$LOCK"
exec python3 -u thern.py > /tmp/thern_overnight.log 2>&1
