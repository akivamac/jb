import subprocess
import os
import time

def launch(cmd: list) -> subprocess.Popen:
    return subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

experts = ["coding", "cot", "emotion", "fish", "greeting",
           "horse", "knowledge", "python", "reptiles", "tree"]

MAX_PARALLEL = 2
running = []  # list of (proc, expert)

i = 0
while i < len(experts) or running:
    while len(running) < MAX_PARALLEL and i < len(experts):
        expert = experts[i]
        proc = launch(['python3', 'training/train_expert.py',
                       '--name', expert, '--steps', '2000',
                       '--resume', '--push', '100',
                       '--log', '25', '--sample', '25'])
        running.append((proc, expert))
        print(f"Started {expert} (PID {proc.pid})")
        i += 1

    time.sleep(60)
    still = []
    for proc, expert in running:
        ret = proc.poll()
        if ret is None:
            still.append((proc, expert))
        elif ret == 0:
            print(f"Finished {expert}")
        else:
            print(f"CRASHED {expert} (exit code {ret})")
    running = still

print("All done.")   
