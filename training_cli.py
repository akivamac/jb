#!/usr/bin/env python3
"""Responsive, keyboard/mouse-driven curses UI for JoeBrain training."""

import argparse
import curses
import json
import os
import sys
import textwrap
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_URL = os.environ.get("TRAINING_URL", "http://localhost:9091")


class ApiError(Exception):
    pass


class Api:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")

    def request(self, path, payload=None, on_event=None):
        body = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"} if body is not None else {}
        try:
            with urlopen(Request(self.base_url + path, data=body, headers=headers), timeout=None if on_event else 30) as response:
                if on_event:
                    for raw in response:
                        line = raw.decode("utf-8", errors="replace").strip()
                        if line.startswith("data:"):
                            try:
                                event = json.loads(line[5:].strip())
                            except ValueError:
                                continue
                            on_event(event)
                            if event.get("done"):
                                if event.get("error"):
                                    raise ApiError(event["error"])
                                return event
                    return {}
                raw = response.read()
        except HTTPError as exc:
            try:
                detail = json.loads(exc.read()).get("error", str(exc))
            except (ValueError, AttributeError):
                detail = str(exc)
            raise ApiError(str(detail)) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ApiError(f"Cannot reach {self.base_url}: {exc}") from exc
        try:
            result = json.loads(raw) if raw else {}
        except ValueError as exc:
            raise ApiError("Server returned invalid JSON") from exc
        if isinstance(result, dict) and result.get("error"):
            raise ApiError(result["error"])
        return result

    def get(self, path):
        return self.request(path)

    def post(self, path, payload=None, on_event=None):
        return self.request(path, payload or {}, on_event=on_event)


class TrainingUI:
    REFRESH_SECONDS = 3
    CARD_HEIGHT = 8
    GAP = 1
    BUTTONS = ("Start / Stop", "Pause", "Logs", "Test chat", "Test suite", "Queue all")

    def __init__(self, screen, api):
        self.screen = screen
        self.api = api
        self.data = {"experts": [], "queue": [], "resources": None}
        self.selected = 0
        self.offset = 0
        self.message = "Loading dashboard…"
        self.last_refresh = 0
        self.button_hits = []
        self.running = True
        try:
            curses.curs_set(0)
            curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
            curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLUE)
            curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_WHITE)
            curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_WHITE)
            curses.init_pair(5, curses.COLOR_RED, curses.COLOR_WHITE)
            curses.init_pair(6, curses.COLOR_CYAN, curses.COLOR_WHITE)
            screen.bkgd(" ", curses.color_pair(1))
        except curses.error:
            pass
        screen.keypad(True)
        screen.timeout(250)

    @staticmethod
    def status(expert):
        if expert.get("running"):
            return "PAUSED" if expert.get("paused") else "RUNNING"
        if expert.get("queued"):
            pos = expert.get("queue_pos")
            return f"QUEUED #{pos}" if pos else "QUEUED"
        return "IDLE"

    def dimensions(self):
        height, width = self.screen.getmaxyx()
        # Reserve a header and footer; never assume a particular terminal size.
        cols = 4 if width >= 132 else 3 if width >= 98 else 2 if width >= 64 else 1
        card_width = max(24, (width - 2 - self.GAP * (cols - 1)) // cols)
        if cols > 1 and card_width < 27:
            cols -= 1
            card_width = max(24, (width - 2 - self.GAP * (cols - 1)) // cols)
        grid_top, footer_top = 4, max(8, height - 4)
        rows = max(1, (footer_top - grid_top) // self.CARD_HEIGHT)
        return height, width, cols, card_width, rows, rows * cols, grid_top, footer_top

    def safe_add(self, y, x, text, attr=None):
        height, width = self.screen.getmaxyx()
        if y < 0 or y >= height or x < 0 or x >= width:
            return
        text = str(text)
        # Avoid writing into the terminal's bottom-right cell, which raises in curses.
        text = text[:max(0, width - x - (1 if y == height - 1 else 0))]
        if not text:
            return
        try:
            if attr is None:
                self.screen.addstr(y, x, text)
            else:
                self.screen.addstr(y, x, text, attr)
        except curses.error:
            pass

    def refresh_data(self):
        try:
            self.data = self.api.get("/api/experts")
            self.last_refresh = time.monotonic()
            self.message = "Connected"
        except ApiError as exc:
            self.message = f"Connection error: {exc}"

    def draw_box(self, y, x, height, width, title, selected=False):
        if height < 3 or width < 4:
            return
        attr = curses.A_BOLD | (curses.color_pair(6) if selected else curses.color_pair(1))
        try:
            win = self.screen.derwin(height, width, y, x)
            win.attron(attr)
            win.box()
            win.attroff(attr)
            label = f" {title} "[:max(1, width - 4)]
            win.addstr(0, 2, label, attr | (curses.A_REVERSE if selected else 0))
        except curses.error:
            pass

    def draw_button(self, y, x, label, action, selected=False):
        text = f"[ {label} ]"
        width = self.screen.getmaxyx()[1]
        if x >= width:
            return x
        attr = curses.A_BOLD | (curses.color_pair(2) if selected else curses.color_pair(1))
        self.safe_add(y, x, text, attr)
        self.button_hits.append((y, x, min(width, x + len(text)), action))
        return x + len(text) + 1

    def draw(self):
        self.screen.erase()
        height, width, cols, card_w, rows, page_size, grid_top, footer_top = self.dimensions()
        experts = self.data.get("experts", [])
        self.selected = min(self.selected, max(0, len(experts) - 1))
        pages = max(1, (len(experts) + page_size - 1) // page_size)
        current_page = self.selected // page_size
        self.offset = current_page * page_size

        try:
            self.screen.addstr(0, 0, " JOEBRAIN  /  TRAINING DASHBOARD ".ljust(width),
                              curses.color_pair(2) | curses.A_BOLD)
        except curses.error:
            pass
        res = self.data.get("resources") or {}
        summary = (f"Running {self.data.get('running_count', 0)}/{self.data.get('max_concurrent', '?')}"
                   f"   Queue {len(self.data.get('queue', []))}"
                   f"   Steps {self.data.get('total_step', 0):,}"
                   f"   Avg loss {self.data.get('avg_loss') if self.data.get('avg_loss') is not None else '—'}")
        if res:
            summary += f"   CPU {res.get('cpu', '—')}%   RAM {res.get('ram', '—')}%"
        self.safe_add(1, 1, summary, curses.A_BOLD)
        self.safe_add(2, 1, "EXPERTS", curses.A_BOLD)

        for i, expert in enumerate(experts[self.offset:self.offset + page_size]):
            row, col = divmod(i, cols)
            y = grid_top + row * self.CARD_HEIGHT
            x = 1 + col * (card_w + self.GAP)
            if y + self.CARD_HEIGHT > footer_top:
                break
            idx = self.offset + i
            selected = idx == self.selected
            name = expert.get("name", "?")
            self.draw_box(y, x, self.CARD_HEIGHT, card_w, name, selected)
            inner = card_w - 4
            status = self.status(expert)
            status_attr = {"RUNNING": curses.color_pair(3) | curses.A_BOLD,
                           "PAUSED": curses.color_pair(4) | curses.A_BOLD}.get(status)
            if status.startswith("QUEUED"):
                status_attr = curses.color_pair(6) | curses.A_BOLD
            step, target = expert.get("step", 0), expert.get("target")
            step_text = f"{step:,}" + (f" / {target:,}" if isinstance(target, int) else "")
            loss, speed = expert.get("loss"), expert.get("speed")
            loss_text = f"{loss:.4f}" if isinstance(loss, (int, float)) else "—"
            speed_text = f"{speed:.2f} steps/s" if isinstance(speed, (int, float)) else "—"
            eta = expert.get("eta") or {}
            extra = f"ETA {eta['hours']}h" if eta.get("hours") is not None else f"Goal {expert.get('loss_goal', '—')}"
            self.safe_add(y + 1, x + 2, f"Status  {status}"[:inner], status_attr)
            self.safe_add(y + 2, x + 2, f"Steps   {step_text}"[:inner])
            self.safe_add(y + 3, x + 2, f"Loss    {loss_text}"[:inner])
            self.safe_add(y + 4, x + 2, f"Speed   {speed_text}"[:inner])
            self.safe_add(y + 5, x + 2, extra[:inner])
            self.safe_add(y + 6, x + 2, "Selected" if selected else "", curses.A_DIM)

        if not experts:
            self.safe_add(grid_top + 1, 2, "No enabled experts found.")
        if pages > 1:
            self.safe_add(footer_top - 1, 1, f"Page {current_page + 1}/{pages}   ←/→ to page")

        self.button_hits = []
        x, y = 1, height - 3
        chosen = experts[self.selected] if experts else {}
        running = chosen.get("running")
        queued = chosen.get("queued")
        start_label = "Stop" if running else "Remove" if queued else "Start"
        pause_label = "Resume" if chosen.get("paused") else "Pause"
        for label, action in ((start_label, "start_stop"), (pause_label, "pause"),
                              ("Logs", "logs"), ("Test chat", "chat"),
                              ("Test suite", "suite"), ("Queue all", "queue_all")):
            x = self.draw_button(y, x, label, action)
        self.safe_add(height - 2, 1,
                      "↑/↓ select  Enter start/stop  P pause  L logs  C chat  S suite  ←/→ page  R refresh  Q quit",
                      curses.A_DIM)
        if self.message:
            attr = curses.color_pair(5) if self.message.startswith("Error") else curses.A_NORMAL
            self.safe_add(height - 1, 1, self.message, attr)
        self.screen.refresh()

    def prompt(self, label, default=""):
        height, width = self.screen.getmaxyx()
        y = max(0, height - 1)
        try:
            curses.echo()
            curses.curs_set(1)
            self.screen.move(y, 0)
            self.screen.clrtoeol()
            self.screen.addstr(y, 0, f"{label}: ")
            value = self.screen.getstr(y, min(width - 1, len(label) + 2), max(1, width - len(label) - 3))
            result = value.decode("utf-8", errors="replace").strip()
            return result or default
        except curses.error:
            return default
        finally:
            curses.noecho()
            try:
                curses.curs_set(0)
            except curses.error:
                pass

    def show_text(self, title, content):
        height, width = self.screen.getmaxyx()
        wrapped = []
        for line in content.splitlines() or [""]:
            wrapped.extend(textwrap.wrap(line, max(10, width - 4)) or [""])
        start = 0
        while True:
            self.screen.erase()
            self.safe_add(0, 0, f" {title} ".ljust(width), curses.color_pair(2) | curses.A_BOLD)
            visible = max(1, height - 3)
            for i, line in enumerate(wrapped[start:start + visible], 1):
                self.safe_add(i, 2, line)
            self.safe_add(height - 1, 1, "↑/↓ scroll   Q/Enter return", curses.A_DIM)
            self.screen.refresh()
            key = self.screen.getch()
            if key in (ord("q"), ord("Q"), 10, 13, 27):
                return
            if key in (curses.KEY_DOWN, ord("j")):
                start = min(max(0, len(wrapped) - visible), start + 1)
            elif key in (curses.KEY_UP, ord("k")):
                start = max(0, start - 1)

    def run_action(self, action):
        experts = self.data.get("experts", [])
        expert = experts[self.selected] if experts else None
        try:
            if action == "queue_all":
                result = self.api.post("/api/queue-all")
                self.message = f"Queued {len(result.get('queued', []))} experts"
            elif expert is None:
                self.message = "Select an expert first"
            elif action == "start_stop":
                name = expert["name"]
                if expert.get("running"):
                    self.api.post("/api/stop", {"name": name})
                    self.message = f"Stopped {name}"
                elif expert.get("queued"):
                    self.api.post("/api/dequeue", {"name": name})
                    self.message = f"Removed {name} from queue"
                else:
                    steps = self.prompt("Training steps", "1000")
                    if steps and not steps.isdigit():
                        raise ApiError("Steps must be a positive whole number")
                    self.api.post("/api/train", {"name": name, "steps": int(steps), "resume": True})
                    self.message = f"Started or queued {name}"
            elif action == "pause":
                paused = not bool(expert.get("paused"))
                self.api.post("/api/pause", {"name": expert["name"], "paused": paused})
                self.message = f"{'Paused' if paused else 'Resumed'} {expert['name']}"
            elif action == "logs":
                self.logs(expert["name"])
            elif action == "chat":
                question = self.prompt(f"Ask {expert['name']}")
                if question:
                    chunks = []
                    self.api.post("/api/test", {"expert": expert["name"], "msg": question},
                                  on_event=lambda event: chunks.append(event.get("char", "")))
                    self.show_text(f"{expert['name']} test chat", "You: " + question + "\n\nJoe: " + "".join(chunks))
            elif action == "suite":
                events = []
                path = "/api/test-stream?" + urlencode({"name": expert["name"]})
                self.api.request(path, on_event=events.append)
                lines = []
                for event in events:
                    if event.get("question"):
                        mark = "PASS" if event.get("correct") else "FAIL"
                        lines.append(f"[{mark}] {event['question']}\n  Expected: {event.get('expected', '')}\n  Actual: {event.get('actual', '')}")
                    if event.get("done"):
                        lines.append(f"\nResult: {event.get('correct', 0)}/{event.get('total', 0)} ({event.get('accuracy', 0)}%)")
                self.show_text(f"{expert['name']} test suite", "\n\n".join(lines) or "No test results returned.")
        except (ApiError, ValueError) as exc:
            self.message = f"Error: {exc}"
        self.refresh_data()

    def logs(self, name):
        try:
            result = self.api.get("/api/logdata?" + urlencode({"name": name, "window": "run"}))
            lines = [entry.get("raw", "") for entry in result.get("history", [])]
            for point in result.get("data", [])[-30:]:
                lines.append(f"step {point['step']}: loss {point['loss']:.5f}, lr {point['lr']:.3g}, speed {point['speed']:.2f}/s")
            self.show_text(f"{name} training log / current run", "\n".join(lines) or "No training log data yet.")
        except ApiError as exc:
            self.message = f"Error: {exc}"

    def click_action(self, y, x):
        for by, left, right, action in self.button_hits:
            if y == by and left <= x < right:
                self.run_action(action)
                return True
        height, width, cols, card_w, rows, page_size, top, _ = self.dimensions()
        if top <= y < top + rows * self.CARD_HEIGHT and x >= 1:
            row, col = divmod(y - top, self.CARD_HEIGHT)
            col = min(cols - 1, max(0, (x - 1) // (card_w + self.GAP)))
            index = self.offset + row * cols + col
            if index < len(self.data.get("experts", [])):
                self.selected = index
                return True
        return False

    def run(self):
        self.refresh_data()
        while self.running:
            self.draw()
            key = self.screen.getch()
            experts = self.data.get("experts", [])
            height, width, cols, card_w, rows, page_size, _, _ = self.dimensions()
            if key == -1:
                if time.monotonic() - self.last_refresh >= self.REFRESH_SECONDS:
                    self.refresh_data()
                continue
            if key in (ord("q"), ord("Q"), 27):
                self.running = False
            elif key in (curses.KEY_DOWN, ord("j")):
                self.selected = min(max(0, len(experts) - 1), self.selected + cols)
            elif key in (curses.KEY_UP, ord("k")):
                self.selected = max(0, self.selected - cols)
            elif key in (curses.KEY_RIGHT, ord("n"), ord("N")):
                self.selected = min(max(0, len(experts) - 1), self.selected + page_size)
            elif key in (curses.KEY_LEFT, ord("b"), ord("B")):
                self.selected = max(0, self.selected - page_size)
            elif key in (10, 13, curses.KEY_ENTER):
                self.run_action("start_stop")
            elif key in (ord("p"), ord("P")):
                self.run_action("pause")
            elif key in (ord("l"), ord("L")):
                self.run_action("logs")
            elif key in (ord("c"), ord("C")):
                self.run_action("chat")
            elif key in (ord("s"), ord("S")):
                self.run_action("suite")
            elif key in (ord("a"), ord("A")):
                self.run_action("queue_all")
            elif key in (ord("r"), ord("R"), curses.KEY_RESIZE):
                self.refresh_data()
            elif key == curses.KEY_MOUSE:
                try:
                    _id, mx, my, _z, state = curses.getmouse()
                    if state & (curses.BUTTON1_CLICKED | curses.BUTTON1_PRESSED):
                        self.click_action(my, mx)
                except curses.error:
                    pass


def main_curses(screen, url):
    TrainingUI(screen, Api(url)).run()


def main():
    parser = argparse.ArgumentParser(description="Interactive JoeBrain training dashboard")
    parser.add_argument("--url", default=DEFAULT_URL, help="training API URL")
    parser.add_argument("--once", action="store_true", help="print a one-time plain status list")
    args = parser.parse_args()
    if args.once:
        try:
            data = Api(args.url).get("/api/experts")
        except ApiError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        for expert in data.get("experts", []):
            print(f"[{TrainingUI.status(expert)}] {expert.get('name')}: step={expert.get('step', 0)} loss={expert.get('loss', '—')}")
        return 0
    try:
        curses.wrapper(main_curses, args.url)
    except curses.error as exc:
        print(f"Could not start terminal UI: {exc}. Try a larger, interactive terminal.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
