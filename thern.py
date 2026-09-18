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

MAX_PARALLEL = 4
running = []  # list of (proc, expert)

while True:
    lock_fd = acquire_lock()
    if lock_fd is None:
        print("Another thern.py instance is running. Waiting...", flush=True)
        time.sleep(30)
        continue
    i = 0
    running = []
    while i < len(experts) or running:
        while len(running) < MAX_PARALLEL and i < len(experts):
            expert = experts[i]
            proc = launch(['python3', 'training/train_expert.py',
                           '--name', expert, '--steps', '4000',
                           '--resume', '--push', '500',
                            '--batch', '16',
                           '--log', '25', '--sample', '25'])
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

    print("All done. Restarting in 30s...", flush=True)
    fcntl.flock(lock_fd, fcntl.LOCK_UN)
    os.close(lock_fd)
    time.sleep(30)
