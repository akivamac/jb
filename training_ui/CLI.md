# Training Dashboard CLI

The CLI is an overflow-safe curses dashboard with bordered expert cards, live status, a selected-card loss sparkline, and real footer controls.

## Launch

```bash
python3 training_server.py
python3 training_cli.py
```

Use `--url URL` or `TRAINING_URL` for another API server. Use `--once` for plain status output. The layout recalculates from the actual terminal width and height after resize.

## Controls

- `1` or Enter: start, stop, or dequeue the selected expert.
- `2` or P: pause/resume selected training.
- `3` or L: show run log/status data.
- `4` or C: test chat.
- `5` or S: test suite.
- `6` or A: queue-all.
- Arrow keys: select and page.
- R: refresh; Q/Esc: quit; mouse: select cards or click buttons.

On narrow terminals the layout falls back to one column and truncates names, summaries, and button labels instead of writing beyond the screen. On short terminals only the highest available card rows are rendered. The full HTML dashboard remains available at `http://localhost:9091`.
