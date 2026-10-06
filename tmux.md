# Tmux Snapshot

## Save

Run before shutting down:

    tmux list-panes -a -F '#{session_name} | #{window_index} | #{window_name} | #{pane_current_path} | #{pane_current_command}' > ~/tmux.txt

## Restore

After reboot, open `~/tmux.txt`. You'll see lines like:

    dev | 0 | zsh | ~/projects/app | vim
    dev | 1 | logs | ~/projects/app | tail

For each **unique session name** (here: `dev`), run once:

    tmux new-session -d -s dev -c ~/projects/app

Then for each **window** (line) in that session, run:

    tmux new-window -t dev:1 -n logs -c ~/projects/app
    tmux send-keys -t dev:1.0 tail C-m

The first window (index 0) already exists from `new-session`, so just send its command:

    tmux send-keys -t dev:0.0 vim C-m

Finally, attach:

    tmux attach -t dev

### Column → command mapping

| Column | Used in |
|--------|---------|
| session_name | `-s <session>` in `new-session`, and in `-t <session>:<index>` everywhere |
| window_index | `-t <session>:<index>` |
| window_name | `-n <name>` in `new-window` |
| pane_current_path | `-c <path>` |
| pane_current_command | argument to `send-keys` (skip if it's `bash`/`zsh`) |

## FAQ

### What does each column mean?

| Column | Meaning |
|--------|---------|
| session_name | tmux session name |
| window_index | window number within the session |
| window_name | window title |
| pane_current_path | working directory of the pane |
| pane_current_command | name of the foreground process in the pane |

### I have multiple panes in one window

Only the active pane's command is captured. The other panes' commands won't appear — you'll need to remember or check them manually.

### The command column just says "bash" or "zsh"

That window was just a plain shell with nothing running in it. Skip it or recreate it without `send-keys`.

### I want to skip shell-only windows when saving

Filter them out at save time:

    tmux list-panes -a -F '#{session_name} | #{window_index} | #{window_name} | #{pane_current_path} | #{pane_current_command}' | grep -vE '\| (bash|zsh|sh)$' > ~/tmux.txt

### Can I save to a different location?

Change the output path: `> ~/Desktop/tmux.txt`, `> /tmp/tmux.txt`, etc.

### What if I have multiple sessions?

All sessions are captured in one file. When restoring, run `new-session` once per unique session name, then add windows to it.

### What does this NOT save?

- Unsaved editor buffers
- Running processes (compilations, servers, etc.)
- Scrollback history
- Pane splits/layout (only one pane per window is assumed)
