# Training Dashboard CLI

The CLI is a keyboard- and mouse-operable terminal dashboard with bordered expert cards and live footer buttons. It uses the existing `training_server.py` API, so training continues to run through the same server and safeguards as the HTML dashboard.

## Launch

```bash
python3 training_server.py
python3 training_cli.py
```

Use `--url URL` or `TRAINING_URL` for another API server. Use `--once` for a plain, one-time status list. The terminal UI redraws after resize and polls status every three seconds.

## Controls

- Arrow keys or `j` / `k`: move between expert cards.
- `Enter` or `[1]`: start, stop, or dequeue the selected expert.
- `[2]` or `P`: pause/resume training.
- `[3]` or `L`: view the selected run's events and recent loss, learning-rate, and speed points.
- `[4]` or `C`: run test chat and open the streamed reply.
- `[5]` or `S`: run the test suite and open a scrollable result view.
- `[6]` or `A`: queue all experts needing training.
- Left/right or `N` / `B`: page through cards.
- `R`: refresh immediately.
- Mouse: select a card or click any footer button.
- `Q` or Esc: quit.

The selected card includes an ASCII loss sparkline loaded from the existing `/api/logdata` endpoint. Wide screens use up to four columns, medium screens use three or two, and narrow screens use one. Browser-only hover effects and graphical charts are omitted, while the card layout, status, resources, live refresh, and training controls remain available. The full HTML dashboard stays at `http://localhost:9091`.
