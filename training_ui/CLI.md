# Training Dashboard CLI

`training_cli.py` is an interactive terminal UI for the JoeBrain training dashboard. It uses the existing `/api/experts`, `/api/train`, `/api/stop`, `/api/dequeue`, and `/api/pause` endpoints, so training behavior remains in `training_server.py`.

## Launch

Start the training server:

```bash
python3 training_server.py
```

Launch the interactive dashboard in another terminal:

```bash
python3 training_cli.py
```

A terminal of at least 40 columns and 20 rows is recommended. The interface redraws automatically when the terminal is resized.

Set another server URL with `--url` or `TRAINING_URL`:

```bash
python3 training_cli.py --url http://localhost:9091
```

For automation or terminals without curses support:

```bash
python3 training_cli.py --once
python3 training_cli.py --plain
```

## Controls

- Arrow keys or `j` / `k` — select an expert.
- `Enter` — start an idle expert, stop a running expert, or remove a queued expert.
- `P` — pause or resume the selected expert.
- `N` / Right arrow — next card page.
- `B` / Left arrow — previous card page.
- `R` — refresh immediately.
- `L` — show the live-log command for the selected expert.
- `C` — enter a test-chat message for the selected expert.
- `S` — show the test-suite command for the selected expert.
- `Q` or `Esc` — exit.

Idle, running, paused, and queued states are shown in real bordered cards. Wide screens use up to four columns, medium screens use three or two, and narrow screens use one. Resource totals, queue size, running count, total steps, and average loss remain visible in the header.

The browser dashboard remains available at `http://localhost:9091` and needs no hover-only interactions to operate the existing training server.
