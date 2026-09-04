# Mac Mini (M2, 8GB) Setup + Run

How to clone JoeBrain onto the Mac Mini and train/run it. All code paths are now
portable (resolved from `__file__`), so no editing is needed after clone.

## 1. Prerequisites
- macOS with Python 3 (12+). Apple Silicon only — numpy runs natively via
  Apple's Accelerate; install CPython not an x86 build.
- `git`, and SSH keys set up on the Mac (for cloning from GitHub).

## 2. Clone
```bash
cd ~
git clone https://github.com/akivamac/jb.git joe-brain
cd joe-brain
```

Everything (servers, trainers, data scripts) auto-locates the repo root, so the
folder name and location don't matter.

## 3. Python + numpy
```bash
python3 -m pip install --user numpy
python3 -c "import numpy; print(numpy.__version__)"   # want 2.x
```

That's the only dependency — the model is pure numpy (no torch/tf).

## 4. External brain data (only if rebuilding train.txt)
The legacy corpus builders reference `~/github-projects/Mj.ai/brain` for
`prepare_data.py`. Point it at wherever the brain JSON lives on the Mac:
```bash
JOE_BRAIN=/path/to/Mj.ai/brain python3 training/prepare_data.py
```
If you don't rebuild data, this is unnecessary — `data/train.txt` is committed.

## 5. Run the servers
```bash
# Chat server (SSE streaming) — port 9090
python3 server.py

# Training dashboard — port 9091  (separate terminal)
python3 training_server.py
```
- Chat UI:  http://localhost:9090
- Training: http://localhost:9091

## 6. Train experts (dashboard)
From the dashboard, pick an expert and set step count, then launch. Training
subprocesses are spawned automatically; `--resume` continues from the saved
`.npz`. All checkpoints live in `data/experts/{name}/`.

## Things that deliberately still reference Termux
- `training_server.py` has a `PYTHON`/site-packages fallback chain listing
  `/data/data/com.termux/...`. Those entries are only consulted if
  `/usr/bin/python3` or the default numpy import fails, so they're harmless on
  the Mac. No action needed.

## Notes for a fresh Mac (8GB RAM)
- Keep concurrent training low (`max_concurrent`, default 2) — seq_len 256 runs
  are memory-hungry in pure numpy.
- Each expert is ~<1M params; the whole ensemble fits trivially. The bottleneck
  is numpy speed, so the M2 will train notably faster than the tablet.
