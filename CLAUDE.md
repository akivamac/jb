# JoeBrain — Claude Instructions

## REPO TOPOLOGY (IMPORTANT)
- The NEW repo is **`akivamac/jb`** on GitHub. WORK HERE.
- Tablet clone: **`/data/data/com.termux/files/home/github-projects/jb`** — always use this on the tablet.
- Mac clone: **`~/github-projects/joe-brain`** — this is a CLONE of `akivamac/jb` (renamed dir, NOT the old Mj.ai repo). The Mac's remote is named `jb` → `akivamac/jb.git`. Training pushes on the Mac go through remote `jb`.
- The OLD repo `akivamac/Mj.ai` and the tablet's old `~/github-projects/joe-brain` dir are DEAD — ignore them.
- On the tablet, sync with GitHub by running `git pull --rebase origin main` (you are already cloned; do NOT `git clone` again).

## THE TWO MACHINES (who's who)
- **Tablet** = a Termux Android device, user `akiva`. Clone at `/data/data/com.termux/files/home/github-projects/jb`, pushes via `origin`.
- **Mac** = Apple silicon machine, user `dev`. Clone at `~/github-projects/joe-brain`, remote named `jb` → `akivamac/jb.git`.
- They never talk directly; they sync through GitHub. Either machine can generate data or train.

## Context Window Strategy
- Currently using **sliding window trimming** — oldest turns dropped when context exceeds seq_len
- **Future**: swap to **compaction** (summarize old turns) once Joe is fluent enough to generate coherent summaries
- Do not change this without asking first

## Communication
- Never say "we can't" or "that's not possible" without first exhausting alternatives. If something seems impossible, find a workaround or at minimum present the tradeoffs of possible approaches.
- Never suggest retraining or destructive changes without asking first. Training takes time and progress is valuable.
- Keep responses short and direct.

## Project Structure
- `server.py` - Chat HTTP server on port 9090 (SSE streaming, router auto-select or manual expert)
- `training_server.py` - Training Dashboard server on port 9091 (launch/stop/queue/dashboard)
- `training_ui/index.html` - Training dashboard UI (cards, live logs, loss charts, test chat, test suite)
- `index.html` - Chat UI with temperature, max tokens, top-k sliders
- `training/train.py` - Legacy main-model training script (`--resume`, `--steps`, `--sample`, `--push`, `--log`)
- `training/train_expert.py` - Train a specialist expert (`--name`, `--resume`, `--save`, `--push`, `--sample`)
- `training/model.py` - JoeBrain transformer (pure numpy, KV cache, Adam optimizer) — **THE core engine**. Every expert, the chat server, and the dashboard run through it. Everything else in the repo is plumbing around this file.
- `training/tokenizer.py` - BPE tokenizer (2000-token vocab), saves to `data/tokenizer.json`
- `training/expert.py` - ExpertRegistry: load all experts, quality filter, confidence-blend logits
- `training/router.py` + `training/train_router.py` - RouterNet classifier (message → expert probs)
- `training/make_conversations.py` - Generates Q&A training pairs
- `training/prepare_data.py` - Combines conversations + brain knowledge → `data/train.txt`
- `data/experts.json` - Expert registry config (10 experts + quality filter thresholds)
- `data/experts/{name}/` - Per-expert dirs: `{name}.npz`, `{name}_train.txt`, `{name}_test.txt`, `training.log`
- `grow_model.py` / `grow_expert.py` - Expand embed_dim/layers without retraining (model vs expert)
- `extend_seq_len.py` - Expand seq_len without retraining
- `check_model.py` / `check_models.py` - Quick scripts to test model checkpoint(s)
- `build_tokenizer.py` - Rebuild shared tokenizer from corpus
- `fix_data.py` - Dedupe/format training data
- `add_*.py` (add_math, add_coding, add_science, add_code_execution, etc.) - One-off corpus builders

## Current State (Sep 2026)
- **Active dev area is the expert ensemble + training dashboard.**
- All old overnight/master/chain shell watcher scripts (`master.sh`, `master_train.py`, `overnight*.sh`,
  `chain.sh`, `chain2.sh`, `chain3.sh`, `queue_manager.sh`, `safe_queue.sh`, `train_queue.sh`)
  have been **deleted** — the dashboard (`training_server.py`) replaces them.
- **Numpy**: `/usr/bin/python3` (Debian Python 3.14.4) has numpy 2.3.5. The Termux python
  (`/data/data/com.termux/files/usr/bin/python3`) also has numpy installed since Sep 2026. Use
  `python3` which resolves to the one with numpy.
- Training is driven from the dashboard at `http://localhost:9091`.

## Model / Experts
- **Expert architecture**: BPE vocab 2000, default embed_dim=128, n_heads=4, n_layers=3, seq_len=128 (~869K params). Some grown larger.
- **Experts (10)**: greeting, emotion, knowledge, coding, cot, python, horse, fish, reptiles, tree
- **Chat server routing**: `server.py` loads all enabled experts from `data/experts.json`, plus a
  RouterNet at `data/router/router.npz` (trained on 7 experts) for auto-selection. Manual expert
  override via `?expert=` query / dropdown.
- **ExpertRegistry ensemble** (`training/expert.py`): no router — all experts run; quality filter
  suppresses junk; survivors blended weighted by confidence. Used by tests/tools, not by server.py.
- **Quality filter** (data/experts.json): enabled, max_logit_threshold -1.0, entropy_ratio 0.70, min_top_prob 0.15
- **LR schedule**: constant LR on `--resume`; cosine decay (min 1e-4) on fresh runs
- Brain data historically sourced from `~/github-projects/Mj.ai/brain/*.json`

## Training Dashboard (port 9091)
- Start: `python3 training_server.py` → open `http://localhost:9091`
- Start/stop training per expert, queue with max_concurrent (default 2), live log streaming
  via SSE, loss curve windows (All/1W/1D/Run), model metadata, test chat, and keyword-match test suite.
- Spawns `training/train_expert.py` subprocesses with per-expert lock files
  (`data/experts/{name}/training.lock`) and per-expert logs.

## Git
- Repo: `akivamac/jb`, branch `main` (this repo was cloned from `jb`, formerly `new-monkey` on `Mj.ai`)
- `data/tokenizer.json`, `data/experts.json` tracked in git
- Expert `.npz` files tracked in git (per-expert)
- Remote URL has PAT embedded for push auth
- Mac pushes via remote named `jb`; tablet pushes via `origin` (= same PAT URL)