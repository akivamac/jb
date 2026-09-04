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
  horse, fish, reptiles, tree. Default expert: 128 embed_dim, 3 layers, 4 heads, 128 seq_len (~869K params).
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

## Recent Activity (Sep 2026)
- Expert ensemble grew from 7 to 10 experts (added fish, reptiles, tree).
- RouterNet trained on 7 experts at `data/router/router.npz` for chat auto-routing.
- Deleted legacy master/overnight/chain watch scripts; all training now driven by the dashboard.

## Known Issues & Notes
- Chat server routing (`server.py`) auto-routes only the 7-router experts; fish/reptiles/tree
  need manual `?expert=` selection unless re-trained/rebuilt into the router.
- Two blend paths coexist: `server.py` probability-averages; `training/expert.py` uses
  quality filter + confidence weighting. Keep them consistent.
- BPE tokenizer; `id_to_token` uses int keys after load (json round-trip).
- `speechSynthesis` is not available in Termux terminal (speaker button hidden).

## Files & Links
- `log.txt` — Project log (append concise updates).
- `training/` — Model, tokenizer, expert training, router.
- `data/experts/{name}/` — Per-expert checkpoints, train/test text, logs.
- `server.py` + `index.html` — Web chat UI.
- `training_server.py` + `training_ui/` — Training dashboard.

--
*Last updated: 2026-09-01*