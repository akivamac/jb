#!/bin/bash
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCK="/tmp/ssh_ui.lock"
trap 'rm -f /tmp/ssh_ui.lock; exit' SIGTERM SIGINT
if [ -f "$LOCK" ]; then
  OLD=$(cat "$LOCK")
  if kill -0 "$OLD" 2>/dev/null; then
    echo "ssh_ui_server already running (PID $OLD)"
    exit 1
  fi
  rm -f "$LOCK"
fi
cd "$REPO_DIR"
echo $$ > "$LOCK"
exec python3 -u ssh_ui_server.py > /tmp/ssh_ui.log 2>&1
