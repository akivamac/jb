#!/usr/bin/env python3
"""Responsive terminal client for JoeBrain's training dashboard API."""

import argparse
import json
import os
import shlex
import shutil
import sys
import textwrap
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_URL = os.environ.get("TRAINING_URL", "http://localhost:9091")
WHITE = "\033[47;30m"
RESET = "\033[0m"


class ApiError(Exception):
    pass


class TrainingCLI:
    def __init__(self, base_url, plain=False):
        self.base_url = base_url.rstrip("/")
        self.color = sys.stdout.isatty() and not plain and os.environ.get("NO_COLOR") is None
        self.page = 0
        self.last_data = None

    def terminal_size(self):
        size = shutil.get_terminal_size(fallback=(80, 24))
        return max(40, size.columns), max(12, size.lines)

    def request(self, path, data=None):
        body = json.dumps(data).encode() if data is not None else None
        headers = {"Content-Type": "application/json"} if body is not None else {}
        try:
            with urlopen(Request(self.base_url + path, data=body, headers=headers), timeout=30) as response:
                payload = response.read()
        except HTTPError as exc:
            try:
                detail = json.loads(exc.read()).get("error", str(exc))
            except (ValueError, AttributeError):
                detail = str(exc)
            raise ApiError(str(detail)) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ApiError(f"Cannot reach {self.base_url}: {exc}") from exc
        try:
            result = json.loads(payload) if payload else {}
        except ValueError as exc:
            raise ApiError("Server returned invalid JSON") from exc
        if isinstance(result, dict) and result.get("error"):
            raise ApiError(result["error"])
        return result

    def stream(self, path, data=None, on_event=None):
        body = json.dumps(data).encode() if data is not None else None
        headers = {"Content-Type": "application/json"} if body is not None else {}
        try:
            with urlopen(Request(self.base_url + path, data=body, headers=headers), timeout=None) as response:
                for raw in response:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    try:
                        event = json.loads(line[5:].strip())
                    except ValueError:
                        continue
                    if on_event:
                        on_event(event)
                    if event.get("done"):
                        if event.get("error"):
                            raise ApiError(event["error"])
                        return event
        except HTTPError as exc:
            try:
                detail = json.loads(exc.read()).get("error", str(exc))
            except (ValueError, AttributeError):
                detail = str(exc)
            raise ApiError(str(detail)) from exc
        except (URLError, OSError) as exc:
            raise ApiError(f"Stream failed: {exc}") from exc
        return {}

    @staticmethod
    def status(expert):
        if expert.get("running"):
            return "PAUSED" if expert.get("paused") else "RUNNING"
        if expert.get("queued"):
            pos = expert.get("queue_pos")
            return f"QUEUED #{pos}" if pos else "QUEUED"
        return "IDLE"

    @staticmethod
    def card(expert, width):
        name = expert.get("name", "?")
        status = TrainingCLI.status(expert)
        step, target = expert.get("step", 0), expert.get("target")
        step_text = f"{step:,}" + (f" / {target:,}" if isinstance(target, int) else "")
        loss = expert.get("loss")
        loss_text = f"{loss:.4f}" if isinstance(loss, (int, float)) else "—"
        speed = expert.get("speed")
        speed_text = f"{speed:.2f} steps/s" if isinstance(speed, (int, float)) else "—"
        inner = max(10, width - 4)
        lines = [name[:inner], f"Status  {status}", f"Steps   {step_text}",
                 f"Loss    {loss_text}   Speed {speed_text}"]
        if expert.get("eta") and expert["eta"].get("hours") is not None:
            lines.append(f"ETA     {expert['eta']['hours']} h")
        else:
            lines.append(f"Goal    {expert.get('loss_goal', '—')}")
        return ["+" + "-" * (width - 2) + "+"] + [
            "| " + line[:inner].ljust(inner) + " |" for line in lines
        ] + ["+" + "-" * (width - 2) + "+"]

    def draw(self, data=None, clear=True):
        if data is None:
            data = self.request("/api/experts")
        self.last_data = data
        width, height = self.terminal_size()
        experts = data.get("experts", [])
        resources = data.get("resources") or {}

        # Match the browser dashboard's card grid while adapting to the actual
        # terminal width and height: narrow screens use one column.
        cols = max(1, min(4, width // 38))
        gap = 2
        card_width = max(24, (width - 2 - gap * (cols - 1)) // cols)
        cols = max(1, min(cols, (width - 2 + gap) // (card_width + gap)))
        card_width = max(24, (width - 2 - gap * (cols - 1)) // cols)
        card_height = 6
        available_rows = max(1, (height - 12) // card_height)
        page_size = cols * available_rows
        pages = max(1, (len(experts) + page_size - 1) // page_size)
        self.page = min(self.page, pages - 1)
        start = self.page * page_size
        visible = experts[start:start + page_size]

        lines = [
            "JOEBRAIN  /  TRAINING DASHBOARD",
            "=" * min(width, 72),
            (f"Running {data.get('running_count', 0)}/{data.get('max_concurrent', '?')}   "
             f"Queue {len(data.get('queue', []))}   Total steps {data.get('total_step', 0):,}   "
             f"Avg loss {data.get('avg_loss') if data.get('avg_loss') is not None else '—'}"),
        ]
        if resources:
            lines.append(f"CPU {resources.get('cpu', '—')}%   RAM {resources.get('ram', '—')}%   "
                         f"Disk free {resources.get('disk_free_gb', '—')} GB")
        lines += ["-" * min(width, 72), "EXPERTS"]

        for offset in range(0, len(visible), cols):
            row = visible[offset:offset + cols]
            cards = [self.card(expert, card_width) for expert in row]
            for card_line in range(card_height):
                parts = [card[card_line] for card in cards]
                lines.append((" " * gap).join(parts)[:width])

        if not experts:
            lines.append("No enabled experts found.")
        if len(experts) > page_size:
            lines.append(f"Page {self.page + 1}/{pages} — use next / prev to browse experts.")
        lines += ["-" * min(width, 72),
                  "Commands: start NAME [steps] | stop NAME | queue-all | logs NAME | chat NAME QUESTION",
                  "          suite NAME | pause/resume NAME | refresh | watch [seconds] | help | quit"]
        if clear and sys.stdout.isatty():
            sys.stdout.write("\033[H\033[2J")
        for line in lines[:height]:
            line = line[:width]
            if self.color:
                print(f"{WHITE}{line.ljust(width)}{RESET}")
            else:
                print(line)
        if self.color:
            for _ in range(max(0, height - len(lines))):
                print(f"{WHITE}{' ' * width}{RESET}")
        return data

    def post(self, path, payload=None):
        result = self.request(path, payload or {})
        print(json.dumps(result, ensure_ascii=False))

    def command(self, line):
        try:
            parts = shlex.split(line)
        except ValueError as exc:
            print(f"Input error: {exc}")
            return True
        if not parts:
            return True
        cmd, args = parts[0].lower(), parts[1:]
        if cmd in ("quit", "exit", "q"):
            return False
        if cmd in ("help", "?"):
            print("start NAME [steps] | stop NAME | queue-all | dequeue NAME")
            print("pause/resume NAME | pause-all | resume-all | logs NAME")
            print("chat NAME QUESTION | suite NAME | refresh | watch [seconds]")
            print("next / prev browse dashboard cards; quit exits.")
            return True
        if cmd in ("refresh", "status"):
            self.draw()
            return True
        if cmd in ("next", "prev"):
            self.page += 1 if cmd == "next" else -1
            self.page = max(0, self.page)
            self.draw()
            return True
        if cmd == "start" and args:
            payload = {"name": args[0], "resume": True}
            if len(args) > 1:
                payload["steps"] = int(args[1])
            self.post("/api/train", payload)
        elif cmd == "stop" and args:
            self.post("/api/stop", {"name": args[0]})
        elif cmd == "dequeue" and args:
            self.post("/api/dequeue", {"name": args[0]})
        elif cmd == "queue-all":
            self.post("/api/queue-all")
        elif cmd in ("pause", "resume") and args:
            self.post("/api/pause", {"name": args[0], "paused": cmd == "pause"})
        elif cmd in ("pause-all", "resume-all"):
            self.post("/api/pause-fleet", {"paused": cmd == "pause-all"})
        elif cmd == "logs" and args:
            print(f"Live log: {args[0]} (Ctrl-C to return)")
            self.stream("/api/logs?" + urlencode({"name": args[0]}), on_event=lambda e: print(e.get("line", "")))
        elif cmd == "chat" and len(args) >= 2:
            expert, message = args[0], " ".join(args[1:])
            print("Joe: ", end="", flush=True)
            self.stream("/api/test", {"expert": expert, "msg": message},
                        on_event=lambda e: print(e.get("char", ""), end="", flush=True))
            print()
        elif cmd == "suite" and args:
            print(f"Running test suite for {args[0]} (Ctrl-C to return)")
            path = "/api/test-stream?" + urlencode({"name": args[0]})
            def show_suite(event):
                if event.get("done"):
                    print(f"Result: {event.get('correct', 0)}/{event.get('total', 0)} ({event.get('accuracy', 0)}%)")
                elif "question" in event:
                    mark = "PASS" if event.get("correct") else "FAIL"
                    print(f"[{mark}] {event['question']}\n  Joe: {event.get('actual', '')}")
            self.stream(path, on_event=show_suite)
        elif cmd == "watch":
            interval = max(1, int(args[0])) if args else 5
            print(f"Refreshing every {interval}s (Ctrl-C to stop)")
            while True:
                self.draw()
                time.sleep(interval)
        else:
            print("Unknown command or missing arguments. Type 'help'.")
            return True
        self.draw()
        return True


def main():
    parser = argparse.ArgumentParser(description="Responsive JoeBrain terminal training dashboard")
    parser.add_argument("--url", default=DEFAULT_URL, help="training server URL (default: %(default)s)")
    parser.add_argument("--plain", action="store_true", help="disable terminal colors")
    parser.add_argument("--once", action="store_true", help="show status and exit")
    args = parser.parse_args()
    cli = TrainingCLI(args.url, plain=args.plain)
    try:
        cli.draw(clear=not args.once)
        if args.once:
            return 0
        while True:
            try:
                if not cli.command(input("jb> ")):
                    break
            except KeyboardInterrupt:
                print("\nInterrupted.")
            except (ApiError, ValueError) as exc:
                print(f"Error: {exc}")
        return 0
    except ApiError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
