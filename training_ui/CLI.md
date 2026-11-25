# Training Dashboard CLI

`training_cli.py` is a small terminal client for the existing JoeBrain training dashboard. It uses the same local HTTP API as `training_ui/index.html`, so training behavior and state remain in `training_server.py`; the CLI only changes how you view and control them.

## Start

From the repository root, start the dashboard server in one terminal:

```bash
python3 training_server.py
```

In another terminal, launch the CLI:

```bash
python3 training_cli.py
```

It connects to `http://localhost:9091` by default and needs no additional Python packages. To use a different server URL:

```bash
python3 training_cli.py --url http://localhost:9091
```

Set `TRAINING_URL` to change the default URL. Add `--plain` to disable ANSI styling or `--once` to print the dashboard and exit.

## Commands

- `refresh` — reload training status and summary.
- `watch [seconds]` — refresh repeatedly (default: 5 seconds; press Ctrl-C to stop).
- `start NAME [steps]` — start or queue training for an enabled expert; defaults to 1000 steps and resumes by default.
- `stop NAME` — stop training or remove the expert from the queue.
- `queue-all` — queue experts needing more training, following the server's existing criteria.
- `dequeue NAME` — remove a queued expert.
- `pause NAME` / `resume NAME` — pause or resume one running expert.
- `pause-all` / `resume-all` — pause or resume the active training fleet.
- `logs NAME` — stream that expert's training log; press Ctrl-C to return.
- `chat NAME QUESTION` — send a test prompt to one expert and stream Joe's response.
- `suite NAME` — run and display an expert's test suite.
- `help` / `quit` — show commands or exit.

The CLI also refreshes the dashboard after control commands. Training remains managed by `training_server.py`, including its queue limit, subprocess lifecycle, and existing safeguards.

## Appearance and limitations

The display is a compact, card-like text layout with fleet totals and per-expert status, steps, loss, and speed. On terminals that support ANSI colors it uses black text on a white background; `--plain` or `NO_COLOR` disables styling. There are no hover effects, animations, or graphical loss charts. The browser dashboard remains available and unchanged at `http://localhost:9091`.
