#!/bin/bash
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"
rm -f /tmp/thern36.lock
END=$(($(date +%s) + 36*3600))
echo "Training until $(date -r $END 2>/dev/null || date)"
while [ $(date +%s) -lt $END ]; do
  python3 -u thern.py
  echo "thern.py exited (code $?), restarting in 5s..."
  sleep 5
done
echo "36 hours done"
