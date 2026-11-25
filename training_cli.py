#!/usr/bin/env python3
"""Interactive curses training dashboard for JoeBrain."""

import argparse
import curses
import json
import os
import sys
import threading
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

    def get(self, path):
        return self.request(path)

    def post(self, path, payload=None):
        return self.request(path, payload or {})

    def request(self, path, payload=None):
        body = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"} if body is not None else {}
        try:
            with urlopen(Request(self.base_url + path, data=body, headers=headers), timeout=30) as response:
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


class JoeBrainCurses:
    def __init__(self, screen, api):
        self.screen = screen
        self.api = api
        self.data = {"experts": [], "running_count": 0, "max_concurrent": 2,
                     "queue": [], "resources": None, "total_step": 0,
                     "avg_loss": None}
        self.selected = 0
        self.scroll = 0
        self.message = ""
        self.running = True
        curses.curs_set(0)
        screen.nodelay(True)
        screen.timeout(200)
        curses.start_color()
        try:
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)
            curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)
            curses.init_pair(3, curses.COLOR_GREEN, -1)
            curses.init_pair(4, curses.COLOR_YELLOW, -1)
            curses.init_pair(5, curses.COLOR_RED, -1)
            curses.init_pair(6, curses.COLOR_CYAN, -1)
        except curses.error:
            pass

    def status(self, expert):
        if expert.get("running"):
            return "PAUSED" if expert.get("paused") else "RUNNING"
        if expert.get("queued"):
            pos = expert.get("queue_pos")
            return f"QUEUED #{pos}" if pos else "QUEUED"
        return "IDLE"

    def refresh(self):
        try:
            self.data = self.api.get("/api/experts")
            self.message = ""
        except ApiError as exc:
            self.message = f"Error: {exc}"

    def refresh_loop(self):
        while self.running:
            self.refresh()
            time.sleep(3)

    def safe_addstr(self, y, x, text, attr=0):
        height, width = self.screen.getmaxyx()
        if y < 0 or y >= height or x >= width:
            return
        text = str(text)
        if x + len(text) > width:
            text = text[:max(0, width - x)]
        if text:
            try:
                self.screen.addstr(y, x, text, attr)
            except curses.error:
                pass

    def draw_box(self, y, x, height, width, title=""):
        if height < 2 or width < 2:
            return
        try:
            self.screen.attron(curses.A_NORMAL)
            self.screen.addch(y, x, curses.ACS_ULCORNER)
            self.screen.addch(y, x + width - 1, curses.ACS_URCORNER)
            self.screen.addch(y + height - 1, x, curses.ACS_LLCORNER)
            self.screen.addch(y + height - 1, x + width - 1, curses.ACS_LRCORNER)
            for ix in range(x + 1, x + width - 1):
                self.screen.addch(y, ix, curses.ACS_HLINE)
                self.screen.addch(y + height - 1, ix, curses.ACS_HLINE)
            for iy in range(y + 1, y + height - 1):
                self.screen.addch(iy, x, curses.ACS_VLINE)
                self.screen.addch(iy, x + width - 1, curses.ACS_VLINE)
        except curses.error:
            pass
        if title:
            self.safe_addstr(y, x + 2, f" {title[:max(1, width - 4)]} ", curses.A_BOLD)

    def draw(self):
        self.screen.erase()
        height, width = self.screen.getmaxyx()

        header_attr = curses.color_pair(1) | curses.A_BOLD
        self.safe_addstr(0, 0, " JOEBRAIN / TRAINING DASHBOARD " + " " * max(0, width - 36), header_attr)

        resources = self.data.get("resources") or {}
        summary = (f" Running {self.data.get('running_count', 0)}/{self.data.get('max_concurrent', '?')}"
                   f"  Queue {len(self.data.get('queue', []))}"
                   f"  Steps {self.data.get('total_step', 0):,}"
                   f"  Avg loss {self.data.get('avg_loss') if self.data.get('avg_loss') is not None else '—'}")
        if resources:
            summary += (f"  CPU {resources.get('cpu', '—')}%"
                        f"  RAM {resources.get('ram', '—')}%"
                        f"  Disk {resources.get('disk_free_gb', '—')} GB")
        self.safe_addstr(1, 1, summary)

        experts = self.data.get("experts", [])
        if self.selected >= len(experts):
            self.selected = max(0, len(experts) - 1)

        top = 3
        footer_height = 4
        grid_height = max(8, height - top - footer_height)

        # Responsive card grid; screen resize changes the number of columns.
        if width >= 125:
            cols = 4
        elif width >= 90:
            cols = 3
        elif width >= 58:
            cols = 2
        else:
            cols = 1
        gap = 1
        card_width = max(24, (width - 2 - gap * (cols - 1)) // cols)
        card_height = 8
        visible_rows = max(1, grid_height // card_height)
        page_size = cols * visible_rows
        max_scroll = max(0, (len(experts) + page_size - 1) // page_size - 1)
        self.scroll = min(self.scroll, max_scroll)
        first = self.scroll * page_size
        visible = experts[first:first + page_size]

        for index, expert in enumerate(visible):
            row, col = divmod(index, cols)
            y = top + row * card_height
            x = 1 + col * (card_width + gap)
            if y + card_height > height - footer_height:
                break
            actual_index = first + index
            selected_attr = curses.color_pair(2) | curses.A_BOLD if actual_index == self.selected else curses.A_NORMAL
            status_attr = curses.A_NORMAL
            status = self.status(expert)
            if status == "RUNNING":
                status_attr = curses.color_pair(3) | curses.A_BOLD
            elif status == "PAUSED":
                status_attr = curses.color_pair(4) | curses.A_BOLD
            elif status.startswith("QUEUED"):
                status_attr = curses.color_pair(6) | curses.A_BOLD

            self.draw_box(y, x, card_height, card_width, expert.get("name", "?"))
            inner = card_width - 3
            step, target = expert.get("step", 0), expert.get("target")
            progress = f"{step:,}" + (f" / {target:,}" if isinstance(target, int) else "")
            loss = expert.get("loss")
            loss_text = f"{loss:.4f}" if isinstance(loss, (int, float)) else "—"
            speed = expert.get("speed")
            speed_text = f"{speed:.2f}/s" if isinstance(speed, (int, float)) else "—"

            if actual_index == self.selected:
                self.safe_addstr(y + 1, x + 2, "▶ " + expert.get("name", "?")[:inner - 2], selected_attr)
            self.safe_addstr(y + 2, x + 2, f"Status: {status}", status_attr)
            self.safe_addstr(y + 3, x + 2, f"Steps: {progress}"[:inner])
            self.safe_addstr(y + 4, x + 2, f"Loss: {loss_text}  Speed: {speed_text}"[:inner])
            eta = expert.get("eta")
            eta_text = f"ETA: {eta['hours']} h" if eta and eta.get("hours") is not None else f"Goal: {expert.get('loss_goal', '—')}"
            self.safe_addstr(y + 5, x + 2, eta_text[:inner])
            if actual_index == self.selected:
                action_y = y + 6
                running = bool(expert.get("running"))
                queued = bool(expert.get("queued"))
                if running:
                    label = "[ Pause ]" if not expert.get("paused") else "[ Resume ]"
                elif queued:
                    label = "[ Remove ]"
                else:
                    label = "[ Start ]"
                self.safe_addstr(action_y, x + 2, label, curses.color_pair(2) | curses.A_BOLD)

        if not experts:
            self.safe_addstr(top + 1, 2, "No enabled experts found.")

        footer_y = height - 3
        self.safe_addstr(footer_y, 1, "[ENTER] Start/Stop  [P] Pause  [L] Logs  [C] Chat  [S] Suite  [N/→] Next  [B/←] Prev  [R] Refresh  [Q] Quit")
        if self.message:
            self.safe_addstr(height - 2, 1, self.message, curses.color_pair(5) | curses.A_BOLD)
        self.screen.refresh()

    def prompt(self, label):
        height, width = self.screen.getmaxyx()
        curses.echo()
        curses.curs_set(1)
        prompt = f" {label}: "
        self.safe_addstr(height - 2, 1, prompt)
        try:
            value = self.screen.getstr(height - 2, min(width - 2, 1 + len(prompt)), max(1, width - len(prompt) - 3))
            result = value.decode("utf-8", errors="replace").strip()
        except (curses.error, UnicodeDecodeError):
            result = ""
        curses.noecho()
        curses.curs_set(0)
        return result

    def selected_expert(self):
        experts = self.data.get("experts", [])
        if not experts:
            return None
        index = self.scroll * self.page_size + (self.selected - self.scroll * self.page_size)
        if 0 <= self.selected < len(experts):
            return experts[self.selected]
        return None

    def start_stop(self):
        expert = self.selected_expert()
        if not expert:
            return
        name = expert["name"]
        try:
            if expert.get("running"):
                self.api.post("/api/stop", {"name": name})
                self.message = f"Stopped {name}"
            elif expert.get("queued"):
                self.api.post("/api/dequeue", {"name": name})
                self.message = f"Removed {name} from queue"
            else:
                steps = self.prompt("Steps (Enter=1000)")
                payload = {"name": name, "resume": True}
                if steps:
                    payload["steps"] = int(steps)
                self.api.post("/api/train", payload)
                self.message = f"Started or queued {name}"
        except (ApiError, ValueError) as exc:
            self.message = f"Error: {exc}"
        self.refresh()

    def pause_toggle(self):
        expert = self.selected_expert()
        if not expert:
            return
        try:
            paused = not bool(expert.get("paused", False))
            self.api.post("/api/pause", {"name": expert["name"], "paused": paused})
            self.message = "Paused" if paused else "Resumed"
        except ApiError as exc:
            self.message = f"Error: {exc}"
        self.refresh()

    def logs(self):
        expert = self.selected_expert()
        if expert:
            self.message = f"Live logs for {expert['name']} use: python3 training_cli.py logs {expert['name']}"

    def chat(self):
        expert = self.selected_expert()
        if not expert:
            return
        message = self.prompt("Message")
        if message:
            self.message = f"Chat command: POST /api/test expert={expert['name']}"

    def suite(self):
        expert = self.selected_expert()
        if expert:
            self.message = f"Run test suite from a terminal: python3 training_cli.py suite {expert['name']}"

    def run(self):
        self.refresh()
        while self.running:
            try:
                key = self.screen.getch()
                experts = self.data.get("experts", [])
                if key != -1:
                    if key in (ord("q"), ord("Q"), 27):
                        self.running = False
                    elif key in (curses.KEY_DOWN, ord("j")):
                        self.selected = min(len(experts) - 1, self.selected + 1)
                    elif key in (curses.KEY_UP, ord("k")):
                        self.selected = max(0, self.selected - 1)
                    elif key in (curses.KEY_RIGHT, ord("n"), ord("N"), ord(" ")):
                        height, width = self.screen.getmaxyx()
                        cols = 4 if width >= 125 else 3 if width >= 90 else 2 if width >= 58 else 1
                        page_size = cols * max(1, (height - 7 - 4) // 8)
                        if self.scroll * page_size < len(experts) - page_size:
                            self.scroll += 1
                            self.selected = self.scroll * page_size
                    elif key in (curses.KEY_LEFT, ord("b"), ord("B")):
                        if self.scroll > 0:
                            height, width = self.screen.getmaxyx()
                            cols = 4 if width >= 125 else 3 if width >= 90 else 2 if width >= 58 else 1
                            page_size = cols * max(1, (height - 7 - 4) // 8)
                            self.scroll -= 1
                            self.selected = self.scroll * page_size
                    elif key in (10, 13, curses.KEY_ENTER):
                        self.start_stop()
                    elif key in (ord("p"), ord("P")):
                        self.pause_toggle()
                    elif key in (ord("l"), ord("L")):
                        self.logs()
                    elif key in (ord("c"), ord("C")):
                        self.chat()
                    elif key in (ord("s"), ord("S")):
                        self.suite()
                    elif key in (ord("r"), ord("R")):
                        self.refresh()
                    elif key == curses.KEY_RESIZE:
                        curses.update_lines_cols()
                self.draw()
                time.sleep(0.05)
            except KeyboardInterrupt:
                self.running = False


def main_curses(screen, url):
    ui = JoeBrainCurses(screen, Api(url))
    thread = threading.Thread(target=ui.refresh_loop, daemon=True)
    thread.start()
    ui.run()
    ui.running = False


def main():
    parser = argparse.ArgumentParser(description="Interactive JoeBrain training dashboard")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--once", action="store_true", help="print status once as a plain dashboard")
    parser.add_argument("--plain", action="store_true", help="disable interactive curses and use plain text")
    args = parser.parse_args()

    if args.once or args.plain:
        api = Api(args.url)
        try:
            data = api.get("/api/experts")
        except ApiError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        for expert in data.get("experts", []):
            status = "RUNNING" if expert.get("running") else "QUEUED" if expert.get("queued") else "IDLE"
            print(f"[{status}] {expert.get('name')}: step={expert.get('step', 0)} loss={expert.get('loss', '—')}")
        return 0

    try:
        curses.wrapper(main_curses, args.url)
    except curses.error:
        print("Terminal is too small or curses is unavailable.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
