# Agent Profile: JoeBrain Assistant

## Role
AI pair programmer for **JoeBrain** — a lightweight, NumPy-based transformer ensemble trained
to chat like Monkey Joe. I keep training logs tidy, maintain data quality, and help iterate on
expert models and the training dashboard.

## Project Summary
- **Goal**: A small, learned conversational agent built from an **expert ensemble** of narrow
  specialists, using only in-project data (no external datasets).
- **Stack**: Pure NumPy transformer, BPE tokenizer (2000 vocab), Adam optimizer,
  `User:/Joe:` turn-structured corpus.
- **Core engine**: `training/model.py` is the transformer itself (forward/backward/Adam/attention).
  Every expert, `server.py`, and the dashboard build on it — treat it as the heart of the repo.
- **Current model**: 10 specialist experts — greeting, emotion, knowledge, coding, cot, python,
  horse, fish, reptiles, tree. All grown as of Sep 2026 to **256 embed_dim, 4 layers, 4 heads,
  512 seq_len (~3.8M params)** for the 2-4x expanded datasets.
- **Key scripts**: `training/train_expert.py`, `training/expert.py`, `server.py`,
  `training_server.py`, `grow_expert.py`, `training/router.py`.

## Quick Commands
```bash
# Train an expert (or use the dashboard at http://localhost:9091)
python3 training/train_expert.py --name greeting --steps 2000 --resume --log 50 --sample 50

# Chat UI (port 9090)
python3 server.py

# Training dashboard (port 9091)
python3 training_server.py
```

## Workflow Guidelines
- **Log updates**: Add concise entries to `log.txt` after major milestones (a few sentences only).
- **Expert count**: Always refer to the ensemble as "existing experts" — never "the 10 experts"
  (the count may change, e.g. a vision expert may be added).
- **Data hygiene**: Per-expert training text lives in `data/experts/{name}/{name}_train.txt`
  with `User:/Joe:` prefixes; test sets in `{name}_test.txt`.
- **Expert growth**: Use `grow_expert.py` to expand embed_dim/layers; preserve weights,
  zero-init new params.
- **Commit discipline**: Commit `data/experts.json`, per-expert `.npz` files, and
  `data/tokenizer.json`; do not commit generated corpus files or `training.log`s.
- **Testing**: Dashboard test chat + keyword-match test suite per expert; `check_models.py` for a CLI sweep.

## Training Dashboard (replaces old watch scripts)
- `training_server.py` on port 9091 → `training_ui/index.html`.
- Launch/stop/queue expert training, live SSE logs, loss curves, test chat, test suite.
- Old overnight/master/chain/queue shell watchers were **deleted Sep 2026** — do not recreate them.
- Dashboard passes `--push N` (default every 500 steps) to `train_expert.py`.

## Remote Training on Mac Mini (Sep 2026)
- Tablet trains too slowly for grown models (~0.02 steps/s at seq 512); the **Mac Mini (M2) is
  the training box** (~0.3-0.8 steps/s, up to ~5 parallel on 8GB).
- Mac clone lives at `~/github-projects/joe-brain` and mirrors the `jb` git repo
  (https://github.com/akivamac/jb.git). Trainers auto-push checkpoints with `--push 500`.
- `git_push` in `training/train_expert.py` is **adaptive**: on the tablet (jb remote exists) it
  builds small `commit-tree` snapshot commits parented on `refs/remotes/jb/main` and pushes only
  those to `jb main` (never the huge `new-monkey` history). On the Mac (no jb remote) it commits
  and pushes `origin`. Serialized via `.git/push.lock` (fcntl).
- **Chain queue script `chain_batch.py`**: waits until no `train_expert.py` processes run, then
  launches the next batch (emotion, knowledge, tree) under nohup; logs to `/tmp/chain_batch.log`.
- After Mac training, tablet syncs via `git fetch jb` then `git checkout jb/main -- data/experts/*/`.

## Recent Activity (Sep 2026)
- Expert ensemble grew from 7 to 10 experts (added fish, reptiles, tree).
- RouterNet retrained on **10 experts** at `data/router/router.npz` (98% val) for chat auto-routing.
- Deleted legacy master/overnight/chain watch scripts; all training now driven by the dashboard.
- Expanded all 10 experts' training data 2-4x (reviewed + deduped); grown all to 256/4L/512.
- **NaN fix in `grow_expert.py`**: Adam m/v paddings must be zero (not tiny) — tiny-randn v caused
  `sqrt(v)` → NaN during resume. Regrow corrupted checkpoints regenerated; doctored npz at e.g.
  commit `29ed53b`. Do not regress the `fill='zero'` padding.
- **Vision (queued)**: dedicated vision expert that passes image description text to the right
  domain expert; existing experts stay untouched. Backbone (ViT-lite vs numpy CNN) and vocab
  bump (2000→4000) undecided. Revisit after current retrain round.

## Known Issues & Notes
- Chat server routing (`server.py`) auto-routes the 10 experts via RouterNet now; `?expert=` still
  overrides manually. (Fish/reptiles/tree folded into the router in the retrain.)
- Two blend paths coexist: `server.py` probability-averages; `training/expert.py` uses
  quality filter + confidence weighting.
- Quality filter (data/experts.json): max_logit_threshold -1.0, entropy_ratio 0.70, min_top_prob 0.15.
- Greedy argmax decoding loops on grown seq-512 models — use temperature sampling (~0.8) in chat/tests.
- BPE tokenizer; `id_to_token` uses int keys after load (json round-trip).
- `speechSynthesis` is not available in Termux terminal (speaker button hidden).

## Files & Links
- `log.txt` — Project log (append concise updates).
- `training/` — Model, tokenizer, expert training, router.
- `data/experts/{name}/` — Per-expert checkpoints, train/test text, logs.
- `server.py` + `index.html` — Web chat UI.
- `training_server.py` + `training_ui/` — Training dashboard.

--
*Last updated: 2026-09-06*