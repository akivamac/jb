import subprocess, time, sys

def launch(cmd):
    return subprocess.Popen(cmd, stdin=subprocess.DEVNULL, 
                           stdout=open('/dev/null','w'),
                           stderr=subprocess.STDOUT,
                           start_new_session=True)

# Skip coding/cot (reverted from overfitting, await RT3 data merge)
experts = ["emotion", "fish", "greeting", "horse", "knowledge",
           "python", "reptiles", "tree"]

MAX_PARALLEL = 2
running = []
i = 0

with open('train_8.log', 'w') as log:
    while i < len(experts) or running:
        while len(running) < MAX_PARALLEL and i < len(experts):
            expert = experts[i]
            proc = launch(['python3', 'training/train_expert.py',
                           '--name', expert, '--steps', '4000',
                           '--resume', '--push', '100',
                           '--log', '25', '--sample', '25'])
            msg = f"Started {expert} (PID {proc.pid})"
            print(msg); log.write(msg + '\n'); log.flush()
            running.append((proc, expert))
            i += 1

        time.sleep(60)
        still = []
        for proc, expert in running:
            ret = proc.poll()
            if ret is None:
                still.append((proc, expert))
            else:
                msg = f"Finished {expert} (exit={ret})"
                print(msg); log.write(msg + '\n'); log.flush()
        running = still

    msg = "All 8 experts done."
    print(msg); log.write(msg + '\n')
