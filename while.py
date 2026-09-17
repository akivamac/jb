import subprocess
import time
import datetime
import sys

DEADLINE = datetime.datetime(2026, 9, 15, 12, 0, 0)  # Monday midday

def main():
    print(f"[{datetime.datetime.now()}] while.py starting — will run until {DEADLINE}", flush=True)
    cycle = 0
    while datetime.datetime.now() < DEADLINE:
        cycle += 1
        print(f"[{datetime.datetime.now()}] Cycle {cycle}: starting thern.py...", flush=True)
        proc = subprocess.Popen(
            [sys.executable, 'thern.py'],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        while proc.poll() is None and datetime.datetime.now() < DEADLINE:
            time.sleep(10)
        ret = proc.poll()
        print(f"[{datetime.datetime.now()}] Cycle {cycle}: thern.py exited (code {ret})", flush=True)
        if ret is not None and ret != 0:
            print(f"[{datetime.datetime.now()}] WARNING: thern.py crashed (exit code {ret})", flush=True)
        if datetime.datetime.now() >= DEADLINE:
            break
    print(f"[{datetime.datetime.now()}] Deadline reached. Done.", flush=True)

if __name__ == "__main__":
    main()
