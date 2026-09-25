import subprocess
import os
import sys
import time
import fcntl
import signal
from console_watch import mom_on_console

def launch(cmd: list) -> subprocess.Popen:
    return subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

def acquire_lock():
    lock_path = '/tmp/thern.lock'
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock_fd
    except BlockingIOError:
        os.close(lock_fd)
        return None

experts = ["coding", "cot", "emotion", "fish", "greeting",
           "horse", "knowledge", "python", "reptiles", "tree"]

# --- CONFIG: edit these before relaunching ---
MAX_PARALLEL   = 2   # experts running at once (Mac 8GB: keep 1-2 at seq=1024)
BATCH          = 4   # per-expert batch size
STEPS          = 4000
PUSH_EVERY     = 500
LOG_EVERY      = 25
SAMPLE_EVERY   = 500
BACKEND        = 'mlx'  # 'mlx'=Apple GPU, 'numpy'=CPU-only
# seq_len is NOT set here on purpose: --resume forces seq_len = checkpoint's
# model.T (1024 for all current experts). Do not add a fixed --seq-len.
# --- END CONFIG ---

running = []  # list of (proc, expert) — reset and restart all experts after they finish
paused_for_mom = False
away_checks = 0

while True:
    lock_fd = acquire_lock()
    if lock_fd is None:
        print("Another thern.py instance is running. Waiting...", flush=True)
        time.sleep(30)
        continue
    try:
        i = 0
        running = []
        while i < len(experts) or running:
            # Mom-pause check: stop spawning (and signal running experts) while she has the console.
            if mom_on_console():
                if not paused_for_mom:
                    print("PAUSED for ellenfriedman (console). Signaling experts to save & stop...", flush=True)
                    for proc, expert in running:
                        try:
                            os.kill(proc.pid, signal.SIGUSR1)
                        except Exception:
                            pass
                    paused_for_mom = True
                away_checks = 0
            else:
                if paused_for_mom:
                    away_checks += 1
                    if away_checks >= 2:
                        print("RESUMED: console free for 2 checks. Continuing.", flush=True)
                        paused_for_mom = False
                        away_checks = 0
                else:
                    away_checks = 0

            if not paused_for_mom:
                while len(running) < MAX_PARALLEL and i < len(experts):
                    expert = experts[i]
                    proc = launch(['python3', 'training/train_expert.py',
                                   '--name', expert, '--steps', str(STEPS),
                                   '--resume', '--push', str(PUSH_EVERY),
                                   '--batch', str(BATCH),
                                   '--log', str(LOG_EVERY), '--sample', str(SAMPLE_EVERY),
                                   '--backend', BACKEND])
                    running.append((proc, expert))
                    print(f"Started {expert} (PID {proc.pid})", flush=True)
                    i += 1

            time.sleep(60)
            sys.stdout.flush()
            still = []
            for proc, expert in running:
                ret = proc.poll()
                if ret is None:
                    still.append((proc, expert))
                elif ret == 0:
                    print(f"Finished {expert}", flush=True)
                else:
                    print(f"CRASHED {expert} (exit code {ret})", flush=True)
            running = still
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    time.sleep(30)
