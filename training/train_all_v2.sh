#!/bin/bash
set -e
REPO=/Users/dev/github-projects/joe-brain
cd "$REPO"
EXPERTS="greeting emotion knowledge coding cot python horse fish reptiles tree"
for e in $EXPERTS; do
  echo "=== $e 4000 steps ==="
  python3 training/train_expert.py --name "$e" --steps 4000 --resume --push 1000 --log 100 --backend mlx --window 30 2>&1 | tail -15 &
done
wait
