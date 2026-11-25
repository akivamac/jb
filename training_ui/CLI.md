# Training Dashboard CLI

`training_cli.py` is a responsive terminal version of the JoeBrain training dashboard. It shares the dashboard's HTTP API and server-side training behavior with `training_ui/index.html`, while presenting status and controls as a compact, white-background card layout. Terminal width and height are detected each time the dashboard renders: wide terminals show multiple expert cards per row; narrow terminals switch to one column, and `next` / `prev` browse cards that do not fit vertically.

## Start

Start the API server from the repository root:

```bash
python3 training_server.py
```

In another terminal, launch the CLI:

```bash
python3 training_cli.py
```

It connects to `http://localhost:9091` by default and needs no additional Python packages. Configure another server with `--url URL` or `TRAINING_URL`. Use `--plain` or `NO_COLOR` to disable ANSI white-background styling; `--once` prints a single status view and exits.

## Commands

- `refresh` — reload status.
- `watch [seconds]` — refresh repeatedly (default 5 seconds; Ctrl-C stops).
- `next` / `prev` — page through expert cards when the screen is too small to show them all.
- `start NAME [steps]`, `stop NAME`, `queue-all`, `dequeue NAME` — manage training and queue.
- `pause NAME` / `resume NAME`, `pause-all` / `resume-all` — pause or resume training.
- `logs NAME` — stream training logs (Ctrl-C returns).
- `chat NAME QUESTION` — stream a test-chat response.
- `suite NAME` — run an expert test suite.
- `help` / `quit` — show commands or exit.

Training remains managed by `training_server.py`, including queue limits, subprocess lifecycle, and existing safeguards. The CLI omits browser-only hover effects and graphical charts; open `http://localhost:9091` for the full HTML dashboard.
