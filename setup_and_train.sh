#!/bin/bash
# Joe Brain — Mac setup + train script
# Run this once on a new machine:
#   bash setup_and_train.sh

set -e

REPO="https://github.com/akivamac/joe-brain.git"
DIR="$HOME/joe-brain"
BRANCH="new-monkey"

echo "=== Joe Brain Setup ==="

# Clone if not already there
if [ ! -d "$DIR" ]; then
  echo "Cloning repo..."
  git clone -b "$BRANCH" "$REPO" "$DIR"
else
  echo "Repo already exists, pulling latest..."
  git -C "$DIR" pull origin "$BRANCH"
fi

cd "$DIR"

# Install Python deps (numpy only)
echo "Installing numpy..."
pip3 install numpy

# Regenerate conversation data
echo "Generating conversation training data..."
python3 training/make_conversations.py

# Prepare full training text
echo "Preparing training data..."
python3 training/prepare_data.py

# Train — resume from saved model, push every 2000 steps
echo "Starting training..."
python3 training/train.py --steps 50000 --resume --push 2000

echo "=== Done! ==="
