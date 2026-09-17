#!/bin/bash
# Robust thern.py launcher using exec to avoid duplicate processes
LOCK="/tmp/thern.pid"
if [ -f "$LOCK" ]; then
  OLD=$(cat "$LOCK")
  if kill -0 "$OLD" 2>/dev/null; then
    echo "thern.py already running (PID $OLD)"
    exit 1
  fi
  rm -f "$LOCK"
fi
cd /Users/dev/github-projects/joe-brain
echo $$ > "$LOCK"
exec python3 thern.py > /tmp/thern_overnight.log 2>&1
