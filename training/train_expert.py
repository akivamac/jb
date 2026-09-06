"""
Train a specialist expert model.

Usage:
  python3 train_expert.py --name greeting --steps 2000
  python3 train_expert.py --name greeting --steps 2000 --layers 3 --dim 128
  python3 train_expert.py --name greeting --steps 1000 --resume
  python3 train_expert.py --name greeting --steps 5000 --push 1000 --sample 500

Looks for training data at: data/experts/{name}/{name}_train.txt
Saves model to:             data/experts/{name}/{name}.npz
"""

import numpy as np
import json
import os
import sys
import argparse
import time
import math
import signal
import fcntl
import subprocess
import traceback

sys.path.insert(0, os.path.dirname(__file__))
from tokenizer import Tokenizer
from model import JoeBrain

BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
TOK_PATH = os.path.join(BASE, 'tokenizer.json')
def _lock_path(name):
    return os.path.join(BASE, 'experts', name, "training.lock")
REPO = os.path.dirname(os.path.dirname(__file__))


def is_pid_running(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as cf:
            cmd = cf.read().decode(errors="ignore")
        return "train_expert" in cmd
    except (OSError, FileNotFoundError):
        return False


def get_running_train_pid(name):
    lock_path = _lock_path(name)
    if not os.path.exists(lock_path):
        return None
    try:
        with open(lock_path) as f:
            pid = int(f.read().strip())
        if is_pid_running(pid):
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as cf:
                    cmd = cf.read().decode(errors="ignore")
                if "train_expert" in cmd:
                    return pid
            except Exception:
                return pid
    except Exception:
        pass
    return None


def write_lock(name, pid):
    try:
        with open(_lock_path(name), "w") as f:
            f.write(str(pid))
    except Exception:
        pass


def remove_lock(name):
    try:
        os.remove(_lock_path(name))
    except Exception:
        pass


def cosine_lr(step, total_steps, lr_max, lr_min=1e-4, warmup=100):
    if step < warmup:
        return lr_max * step / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def _remote_exists(name):
    r = subprocess.run(['git', '-C', REPO, 'remote'], check=True,
                       capture_output=True, text=True)
    return name in r.stdout.split()


def git_push(name, step):
    """Commit the expert checkpoint and push.

    Tablet (this repo): remote 'jb' exists -> build a small incremental
    snapshot commit parented off refs/remotes/jb/main and push to jb main,
    avoiding shipping the large new-monkey history.
    Mac (jb clone): no 'jb' remote -> normal commit on the current branch
    and push it to origin (which IS the jb repo).
    """
    try:
        expert_file = f'data/experts/{name}/{name}.npz'
        msg = f'chore: auto-save {name} expert at step {step}'
        lock_path = os.path.join(REPO, '.git', 'push.lock')
        with open(lock_path, 'a+') as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)
            try:
                subprocess.run(['git', '-C', REPO, 'add', expert_file], check=True)
                if _remote_exists('jb'):
                    subprocess.run(
                        ['git', '-C', REPO, 'commit', '--no-verify', '-m', msg],
                        check=True)
                    tree = subprocess.run(
                        ['git', '-C', REPO, 'write-tree'], check=True,
                        capture_output=True, text=True).stdout.strip()
                    parent = subprocess.run(
                        ['git', '-C', REPO, 'rev-parse', 'refs/remotes/jb/main'],
                        check=True, capture_output=True, text=True).stdout.strip()
                    commit = subprocess.run(
                        ['git', '-C', REPO, 'commit-tree', tree, '-p', parent, '-m', msg],
                        check=True, capture_output=True, text=True).stdout.strip()
                    subprocess.run(
                        ['git', '-C', REPO, 'push', 'jb', f'{commit}:main'], check=True)
                    subprocess.run(['git', '-C', REPO, 'update-ref',
                                    'refs/remotes/jb/main', commit], check=True)
                else:
                    branch = subprocess.run(
                        ['git', '-C', REPO, 'rev-parse', '--abbrev-ref', 'HEAD'],
                        check=True, capture_output=True, text=True).stdout.strip()
                    subprocess.run(
                        ['git', '-C', REPO, 'commit', '--no-verify', '-m', msg],
                        check=True)
                    subprocess.run(
                        ['git', '-C', REPO, 'push', 'origin', f'HEAD:{branch}'],
                        check=True)
                print(f"  [pushed {name} expert at step {step}]")
            finally:
                fcntl.flock(lockf, fcntl.LOCK_UN)
    except Exception as e:
        print(f"  [git push failed: {e}]")


def train_expert(name, steps=2000, lr=3e-4, seq_len=128, batch_size=8,
                 embed_dim=128, n_heads=4, n_layers=3, log_every=100,
                 sample_every=0, resume=False, push_every=0, save_every=0):

    expert_dir = os.path.join(BASE, 'experts', name)
    data_path = os.path.join(expert_dir, f'{name}_train.txt')
    model_path = os.path.join(expert_dir, f'{name}.npz')

    if not os.path.exists(data_path):
        print(f"ERROR: Training data not found at {data_path}")
        print(f"Create {name}_train.txt with your specialist training text.")
        sys.exit(1)

    os.makedirs(expert_dir, exist_ok=True)

    # Load shared tokenizer
    if not os.path.exists(TOK_PATH):
        print(f"ERROR: Shared tokenizer not found at {TOK_PATH}")
        print("Run training/train.py first to build the tokenizer.")
        sys.exit(1)

    tok = Tokenizer()
    tok.load(TOK_PATH)
    print(f"Tokenizer loaded (vocab {tok.size})")

    # Load training data
    with open(data_path) as f:
        raw = f.read()
    print(f"Training text: {len(raw):,} characters from {data_path}")

    data = np.array(tok.encode(raw), dtype=np.int32)
    print(f"Encoded: {len(data):,} tokens")

    if len(data) < seq_len + 10:
        print(f"ERROR: Not enough data ({len(data)} tokens). Need at least {seq_len + 10}.")
        sys.exit(1)

    # Create or resume model
    if os.path.exists(model_path):
        model = JoeBrain.load(model_path)
        seq_len = model.T
        if resume:
            print(f"Resumed expert '{name}' from {model_path} (Adam step {model.t})")
        else:
            print(f"Loaded existing expert '{name}' from {model_path} (fresh optimizer)")
            model.t = 0
            model.m = {k: np.zeros_like(v) for k, v in model.p.items()}
            model.v = {k: np.zeros_like(v) for k, v in model.p.items()}
    else:
        model = JoeBrain(
            vocab_size=tok.size,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            seq_len=seq_len,
        )

    total_params = sum(v.size for v in model.p.values())
    print(f"Expert '{name}': {total_params:,} params | {model.L} layers | dim {model.C}")
    print(f"Training for {steps} steps | lr={lr} | seq={seq_len} | batch={batch_size}\n")

    losses = []
    start = time.time()
    last_log_time = start
    last_log_step = 0

    # Setup training log
    cmd = " ".join(sys.argv)
    log_path = os.path.join(expert_dir, "training.log")

    # Concurrency check
    prev_pid = get_running_train_pid(name)
    if prev_pid is not None and prev_pid != os.getpid():
        print(f"[INFO] Found existing training (PID {prev_pid}). Signaling to save and stop...")
        with open(log_path, "a") as lf:
            lf.write(f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] STOP_PREV: signaling PID {prev_pid} to stop\n")
        try:
            os.kill(prev_pid, signal.SIGUSR1)
            for _ in range(180):
                if not is_pid_running(prev_pid):
                    break
                time.sleep(1)
            else:
                print("[WARN] Previous did not exit gracefully; forcing...")
                os.kill(prev_pid, signal.SIGTERM)
                time.sleep(2)
        except Exception as e:
            print(f"[WARN] Could not signal previous: {e}")

    if resume and os.path.exists(log_path):
        with open(log_path, "a") as lf:
            lf.write(f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] RESUME: {cmd}\n")
    else:
        with open(log_path, "w") as lf:
            lf.write(f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] START: {cmd}\n")
            lf.write("# step,loss,lr,steps_per_sec,timestamp\n")

    # Write lock
    write_lock(name, os.getpid())

    # Graceful stop
    should_stop = False
    def handle_stop(signum, frame):
        nonlocal should_stop
        should_stop = True
    signal.signal(signal.SIGUSR1, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    completed = False
    step = 0
    try:
        for step in range(1, steps + 1):
            if should_stop:
                print("\n[INFO] Stop signal received. Saving and exiting...")
                with open(log_path, "a") as lf:
                    lf.write(f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] STOPPED by signal at step {step}\n")
                break

            model.zero_grad()

            starts = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)
            x_batch = np.stack([data[s:s + seq_len] for s in starts])
            y_batch = np.stack([data[s + 1:s + seq_len + 1] for s in starts])

            logits, cache = model.forward(x_batch)
            batch_loss, dlogits = model.loss(logits, y_batch)
            model.backward(dlogits, cache)

            eff_lr = lr if resume else cosine_lr(step, steps, lr)
            model.step(eff_lr)
            losses.append(batch_loss)

            if step % log_every == 0:
                avg = np.mean(losses[-log_every:])
                now = time.time()
                sps = (step - last_log_step) / (now - last_log_time)
                last_log_time = now
                last_log_step = step
                print(f"  step {step:5d}/{steps} | loss {avg:.4f} | lr {eff_lr:.2e} | {sps:.2f} steps/s")

                with open(log_path, "a") as lf:
                    lf.write(f"{step},{avg:.4f},{eff_lr:.2e},{sps:.2f},{time.time()}\n")
                    lf.flush()

            if sample_every and step % sample_every == 0:
                sample = model.generate(tok, "\n", max_new=80, temperature=0.8)
                print(f"  Sample: {repr(sample[:100])}\n")
                with open(log_path, "a") as lf:
                    lf.write(f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] SAMPLE step {step}:\n")
                    for line in sample[:200].splitlines():
                        lf.write(f"# {line}\n")
                    lf.flush()

            if push_every and step % push_every == 0:
                model.save(model_path)
                git_push(name, step)

            if save_every and step % save_every == 0:
                model.save(model_path)

    except KeyboardInterrupt:
        with open(log_path, "a") as lf:
            lf.write(f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] INTERRUPT at step {step}\n")
        raise
    else:
        completed = True
    finally:
        try:
            model.save(model_path)
        except Exception as e:
            try:
                with open(log_path, "a") as lf:
                    lf.write(f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] SAVE FAILED: {e}\n")
            except Exception:
                pass
            print(f"[ERROR] Failed to save model: {e}")
        remove_lock(name)
        if completed:
            with open(log_path, "a") as lf:
                lf.write(f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] FINISH at step {step}\n")

    elapsed = time.time() - start
    print(f"\nDone. Expert '{name}' saved to {model_path}")
    print(f"Time: {elapsed:.1f}s | Final loss: {np.mean(losses[-100:]):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a specialist expert model')
    parser.add_argument('--name', required=True, help='Expert name (e.g. greeting, emotion)')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--log', type=int, default=100)
    parser.add_argument('--sample', type=int, default=0, help='Print sample every N steps (0=off)')
    parser.add_argument('--resume', action='store_true', help='Continue from saved expert model')
    parser.add_argument('--push', type=int, default=0, help='Push to github every N steps')
    parser.add_argument('--save', type=int, default=0, help='Save checkpoint every N steps (0=only at end)')
    args = parser.parse_args()

    try:
        train_expert(
            name=args.name,
            steps=args.steps,
            lr=args.lr,
            seq_len=args.seq_len,
            batch_size=args.batch,
            embed_dim=args.dim,
            n_heads=args.heads,
            n_layers=args.layers,
            log_every=args.log,
            sample_every=args.sample,
            resume=args.resume,
            push_every=args.push,
            save_every=args.save,
        )
    except Exception:
        expert_dir = os.path.join(BASE, 'experts', args.name)
        log_path = os.path.join(expert_dir, "training.log")
        try:
            with open(log_path, "a") as lf:
                lf.write(f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: training crashed\n")
                traceback.print_exc(file=lf)
        except Exception:
            pass
        raise
