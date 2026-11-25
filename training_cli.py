#!/usr/bin/env python3
"""Lightweight terminal client for JoeBrain's training dashboard API."""

import argparse
import json
import os
import shlex
import sys
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

    def request(self, path, data=None):
        body = json.dumps(data).encode() if data is not None else None
        headers = {"Content-Type": "application/json"} if body is not None else {}
        req = Request(self.base_url + path, data=body, headers=headers)
        try:
            with urlopen(req, timeout=30) as response:
                payload = response.read()
        except HTTPError as exc:
            payload = exc.read()
            try:
                detail = json.loads(payload).get("error", str(exc))
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
        req = Request(self.base_url + path, data=body, headers=headers)
        try:
            with urlopen(req, timeout=None) as response:
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

    def display(self, text=""):
        if self.color:
            print(f"{WHITE}{text:<{max(1, min(120, len(text)))}}{RESET}")
        else:
            print(text)

    @staticmethod
    def status(expert):
        if expert.get("running"):
            return "PAUSED" if expert.get("paused") else "RUNNING"
        if expert.get("queued"):
            pos = expert.get("queue_pos")
            return f"QUEUED #{pos}" if pos else "QUEUED"
        return "IDLE"

    def dashboard(self, data=None):
        if data is None:
            data = self.request("/api/experts")
        experts = data.get("experts", [])
        resources = data.get("resources") or {}
        self.display("JoeBrain  /  TRAINING DASHBOARD")
        self.display("=" * 72)
        summary = (f"Running {data.get('running_count', 0)}/{data.get('max_concurrent', '?')}"
                   f"   Queue {len(data.get('queue', []))}"
                   f"   Total steps {data.get('total_step', 0):,}"
                   f"   Avg loss {data.get('avg_loss') if data.get('avg_loss') is not None else '—'}")
        self.display(summary)
        if resources:
            self.display(f"CPU {resources.get('cpu', '—')}%   RAM {resources.get('ram', '—')}%"
                         f"   Disk free {resources.get('disk_free_gb', '—')} GB")
        self.display("-" * 72)
        if not experts:
            self.display("No enabled experts found.")
        for expert in experts:
            name = expert.get("name", "?")
            step, target = expert.get("step", 0), expert.get("target")
            progress = f"{step:,}/{target:,}" if isinstance(target, int) else f"{step:,} steps"
            loss = expert.get("loss")
            loss_text = f"{loss:.4f}" if isinstance(loss, (int, float)) else "—"
            speed = expert.get("speed")
            speed_text = f"{speed:.2f} steps/s" if isinstance(speed, (int, float)) else "—"
            self.display(f"[{self.status(expert):9}] {name}")
            self.display(f"  Step {progress}   Loss {loss_text}   Speed {speed_text}")
        self.display("-" * 72)
        self.display("Commands: help | start NAME [steps] | stop NAME | queue-all | logs NAME | quit")
        return data

    def post(self, path, payload=None):
        result = self.request(path, payload or {})
        self.display(json.dumps(result, ensure_ascii=False))

    def command(self, line):
        try:
            parts = shlex.split(line)
        except ValueError as exc:
            self.display(f"Input error: {exc}")
            return True
        if not parts:
            return True
        cmd, args = parts[0].lower(), parts[1:]
        if cmd in ("quit", "exit", "q"):
            return False
        if cmd in ("help", "?"):
            self.display("start NAME [steps] | stop NAME | dequeue NAME | queue-all")
            self.display("pause NAME | resume NAME | pause-all | resume-all | logs NAME")
            self.display("chat NAME QUESTION | suite NAME | refresh | watch [seconds] | quit")
            return True
        if cmd in ("refresh", "status"):
            self.dashboard()
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
            self.display(f"Live log: {args[0]} (Ctrl-C to return)")
            self.stream("/api/logs?" + urlencode({"name": args[0]}), on_event=lambda e: print(e.get("line", "")))
        elif cmd == "chat" and len(args) >= 2:
            expert, message = args[0], " ".join(args[1:])
            self.display("Joe: ")
            self.stream("/api/test", {"expert": expert, "msg": message},
                        on_event=lambda e: print(e.get("char", ""), end="", flush=True))
            print()
        elif cmd == "suite" and args:
            self.display(f"Running test suite for {args[0]} (Ctrl-C to return)")
            path = "/api/test-stream?" + urlencode({"name": args[0]})
            def show_suite(event):
                if event.get("done"):
                    self.display(f"Result: {event.get('correct', 0)}/{event.get('total', 0)}"
                                 f" ({event.get('accuracy', 0)}%)")
                elif "question" in event:
                    mark = "PASS" if event.get("correct") else "FAIL"
                    self.display(f"[{mark}] {event['question']}\n  Joe: {event.get('actual', '')}")
            self.stream(path, on_event=show_suite)
        elif cmd == "watch":
            interval = max(1, int(args[0])) if args else 5
            self.display(f"Refreshing every {interval}s (Ctrl-C to stop)")
            while True:
                self.dashboard()
                time.sleep(interval)
        else:
            self.display("Unknown command or missing arguments. Type 'help'.")
            return True
        self.dashboard()
        return True


def main():
    parser = argparse.ArgumentParser(description="JoeBrain terminal training dashboard")
    parser.add_argument("--url", default=DEFAULT_URL, help="training server URL (default: %(default)s)")
    parser.add_argument("--plain", action="store_true", help="disable terminal colors")
    parser.add_argument("--once", action="store_true", help="show status and exit")
    args = parser.parse_args()
    cli = TrainingCLI(args.url, plain=args.plain)
    try:
        cli.dashboard()
        if args.once:
            return 0
        while True:
            try:
                if not cli.command(input("jb> ")):
                    break
            except KeyboardInterrupt:
                print("\nInterrupted.")
            except (ApiError, ValueError) as exc:
                cli.display(f"Error: {exc}")
        return 0
    except ApiError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
