# Training Dashboard CLI

The CLI is a keyboard- and mouse-operable terminal UI with bordered expert cards and action buttons. It uses the existing `training_server.py` API, so training remains managed by the same server as the HTML dashboard.

## Launch

```bash
python3 training_server.py
```

In another terminal:

```bash
python3 training_cli.py
```

Use `--url URL` or `TRAINING_URL` to select another API server. Run `python3 training_cli.py --once` for a plain one-time status list. A full-screen interactive terminal is required; a wider/taller terminal displays more cards at once. The UI recalculates its grid after terminal resize.

## Dashboard controls

- Arrow keys or `j` / `k`: select an expert (up/down moves by grid row).
- `Enter`: start an idle expert (prompts for steps), stop a running expert, or dequeue a queued expert.
- `P`: pause or resume the selected running expert.
- `L`: display the selected expert's current-run events and recent loss/step log data.
- `C`: prompt for a test-chat question and show Joe's streamed answer.
- `S`: run the selected expert's test suite and browse results.
- `A`: queue all experts that need training.
- Left/right or `N` / `B`: move by a visible page of cards.
- `R`: refresh now; `Q` or Esc: quit.
- Mouse: click a card to select it and click a footer action button to activate it.

The responsive layout uses up to four columns on wide terminals, fewer columns on smaller terminals, and pages cards that do not fit. The browser-only hover effects and graphical charts are omitted; the interface keeps the dashboard's light background, card layout, live status refresh, and training controls. Use the HTML dashboard at `http://localhost:9091` for the full graphical loss charts.
