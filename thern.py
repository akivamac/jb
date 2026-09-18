import subprocess
import os
import sys
import time
import fcntl

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
BATCH          = 8   # per-expert batch size (16 caused swap thrash on the Mac)
STEPS          = 4000
PUSH_EVERY     = 500
LOG_EVERY      = 25
SAMPLE_EVERY   = 500
BACKEND        = 'mlx'  # 'mlx'=Apple GPU, 'numpy'=CPU-only
# seq_len is NOT set here on purpose: --resume forces seq_len = checkpoint's
# model.T (1024 for all current experts). Do not add a fixed --seq-len.
# --- END CONFIG ---

running = []  # list of (proc, expert)

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
