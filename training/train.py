"""
Train JoeBrain on the prepared text data.
Saves model weights + tokenizer as JSON (ready for browser inference).

Usage:
  python3 train.py
  python3 train.py --steps 5000 --lr 3e-4
  python3 train.py --resume --steps 5000
"""

import numpy as np
import json
import os
import sys
import argparse
import time
import math
import signal
import traceback

sys.path.insert(0, os.path.dirname(__file__))
from tokenizer import Tokenizer
from model import JoeBrain
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, 'data', 'train.txt')
OUT_DIR = os.path.join(REPO, 'data')

# Training concurrency control
LOCK_PATH = os.path.join(OUT_DIR, "training.lock")

def is_pid_running(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def get_running_train_pid():
    if not os.path.exists(LOCK_PATH):
        return None
    try:
        with open(LOCK_PATH) as f:
            pid = int(f.read().strip())
        if is_pid_running(pid):
            # quick cmdline check (best-effort)
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as cf:
                    cmd = cf.read().decode(errors="ignore")
                if "train.py" in cmd:
                    return pid
            except Exception:
                return pid  # fallback if can't read cmdline
    except Exception:
        pass
    return None

def write_lock(pid):
    try:
        with open(LOCK_PATH, "w") as f:
            f.write(str(pid))
    except Exception:
        pass

def remove_lock():
    try:
        os.remove(LOCK_PATH)
    except Exception:
        pass


def get_batch(data, seq_len, batch_size):
    starts = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)
    x = np.stack([data[s:s + seq_len] for s in starts])
    y = np.stack([data[s + 1:s + seq_len + 1] for s in starts])
    return x, y


def cosine_lr(step, total_steps, lr_max, lr_min=1e-4, warmup=200):
    if step < warmup:
        return lr_max * step / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def git_push(step):
    try:
        repo = REPO
        subprocess.run(['git', '-C', repo, 'add', 'data/model.npz', 'data/tokenizer.json'], check=True)
        subprocess.run(['git', '-C', repo, 'commit', '-m', f'chore: auto-save model at step {step}'], check=True)
        subprocess.run(['git', '-C', repo, 'push', 'origin', 'new-monkey'], check=True)
        print(f"  [pushed to github at step {step}]")
    except Exception as e:
        print(f"  [git push failed: {e}]")


def train(steps=5000, lr=3e-4, seq_len=128, batch_size=8,
          embed_dim=128, n_heads=4, n_layers=3, log_every=100, resume=False, push_every=0, sample_every=0):

    # Load data
    print ("Loading model and other data...")
    with open(DATA) as f:
        raw = f.read()
    print(f"Training text: {len(raw):,} characters")

    # Tokenizer
    tok = Tokenizer()
    tok_path = os.path.join(OUT_DIR, 'tokenizer.json')
    model_path = os.path.join(OUT_DIR, 'model.npz')
    model_path_legacy = os.path.join(OUT_DIR, 'model.json')

    if resume and os.path.exists(tok_path) and (os.path.exists(model_path) or os.path.exists(model_path_legacy)):
        if not os.path.exists(model_path) and os.path.exists(model_path_legacy):
            model_path = model_path_legacy
        tok.load(tok_path)
        data = np.array(tok.encode(raw), dtype=np.int32)
        model = JoeBrain.load(model_path)
        seq_len = model.T  # use the model's actual seq_len
        print(f"Resumed from saved model (vocab {tok.size}, Adam step {model.t})")
    else:
        # Load pre-built tokenizer if available, otherwise build from scratch
        if os.path.exists(tok_path):
            tok.load(tok_path)
            print(f"Loaded tokenizer (vocab {tok.size})")
        else:
            tok.build(raw)
        data = np.array(tok.encode(raw), dtype=np.int32)
        model = JoeBrain(
            vocab_size=tok.size,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            seq_len=seq_len,
        )

    total_params = sum(v.size for v in model.p.values())
    print(f"Parameters: {total_params:,}")
    print(f"Training for {steps} steps | lr={lr} | seq={seq_len} | batch={batch_size}\n")

    losses = []
    start = time.time()
    last_log_time = start
    last_log_step = 0
    
    # Setup training log
    cmd = " ".join(sys.argv)
    log_path = os.path.join(OUT_DIR, "training.log")
    
    # Concurrency check: stop existing training gracefully
    prev_pid = get_running_train_pid()
    if prev_pid is not None:
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
        mode = "a"
        with open(log_path, "a") as lf:
            lf.write(f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] RESUME: {cmd}\n")
    else:
        mode = "w"
        with open(log_path, "w") as lf:
            lf.write(f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] START: {cmd}\n")
            lf.write("# step,loss,lr,steps_per_sec,timestamp\n")
    
    # Write lock for current process
    write_lock(os.getpid())
    
    # Graceful stop flag and signal handler
    should_stop = False
    def handle_stop(signum, frame):
        global should_stop
        should_stop = True
    signal.signal(signal.SIGUSR1, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    
    completed = False
    try:
        for step in range(1, steps + 1):
            if should_stop:
                print("\n[INFO] Stop signal received. Saving and exiting...")
                with open(log_path, "a") as lf:
                    lf.write(f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] STOPPED by signal at step {step}\n")
                break
            
            model.zero_grad()

            # Batched forward+backward: single call with (B, T) batch
            starts = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)
            x_batch = np.stack([data[s:s + seq_len] for s in starts])        # (B, T)
            y_batch = np.stack([data[s + 1:s + seq_len + 1] for s in starts])  # (B, T)
            logits, cache = model.forward(x_batch)   # (B, T, V)
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
                print(f"step {step:5d}/{steps} | loss {avg:.4f} | lr {eff_lr:.2e} | {sps:.2f} steps/s")

                with open(log_path, "a") as lf:
                    lf.write(f"{step},{avg:.4f},{eff_lr:.2e},{sps:.2f},{time.time()}\n")

            if sample_every and step % sample_every == 0:
                sample = model.generate(tok, "\n", max_new=80, temperature=0.8)
                print(f"  Sample: {repr(sample[:100])}\n")

            if push_every and step % push_every == 0:
                model.save(os.path.join(OUT_DIR, "model.npz"))
                tok.save(os.path.join(OUT_DIR, "tokenizer.json"))
                git_push(step)
    except KeyboardInterrupt:
        with open(log_path, "a") as lf:
            lf.write(f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] INTERRUPT at step {step}\n")
        raise
    finally:
        # Save current model state on exit
        try:
            model.save(os.path.join(OUT_DIR, "model.npz"))
            tok.save(os.path.join(OUT_DIR, "tokenizer.json"))
        except Exception as e:
            try:
                with open(log_path, "a") as lf:
                    lf.write(f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] SAVE FAILED: {e}\n")
            except Exception:
                pass
            print(f"[ERROR] Failed to save model: {e}")
        remove_lock()
        if completed:
            with open(log_path, "a") as lf:
                lf.write(f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] FINISH at step {step}\n")
    
    completed = True

    # Save
    model.save(os.path.join(OUT_DIR, 'model.npz'))
    tok.save(os.path.join(OUT_DIR, 'tokenizer.json'))
    print("\nDone. Files saved to data/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--log', type=int, default=100)
    parser.add_argument('--resume', action='store_true', help='Continue from saved model')
    parser.add_argument('--push', type=int, default=0, help='Push to github every N steps')
    parser.add_argument('--sample', type=int, default=0, help='Print a sample every N steps (0=off)')
    args = parser.parse_args()

    try:
        train(
            steps=args.steps,
            lr=args.lr,
            seq_len=args.seq_len,
            batch_size=args.batch,
            embed_dim=args.dim,
            n_heads=args.heads,
            n_layers=args.layers,
            log_every=args.log,
            resume=args.resume,
            push_every=args.push,
            sample_every=args.sample,
        )
    except Exception:
        # Record crash details in training.log so errors are visible without pasting
        try:
            with open(os.path.join(OUT_DIR, "training.log"), "a") as lf:
                lf.write(f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: training crashed\n")
                traceback.print_exc(file=lf)
        except Exception:
            pass
        raise
