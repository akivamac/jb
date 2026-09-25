"""
Train a specialist expert model.

Usage:
  python3 train_expert.py --name greeting --steps 2000
  python3 train_expert.py --name greeting --steps 2000 --layers 3 --dim 128
  python3 train_expert.py --name greeting --steps 1000 --resume
  python3 train_expert.py --name greeting --steps 5000 --push 1000 --sample 500
  python3 train_expert.py --name tree --steps 2000 --window 4 --conversations
  python3 train_expert.py --name tree --steps 2000 --stop-at-loss 0.05 --stop-sigma 3.0 --tag "v2 run"

Looks for training data at: data/experts/{name}/{name}_train.txt
Saves model to:             data/experts/{name}/{name}.npz

--backend selects the engine: mlx=Apple GPU via training/model_mlx.py (default),
numpy=CPU via training/model.py (always available).
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
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from tokenizer import Tokenizer
from console_watch import mom_on_console

BACKEND = 'mlx'

def set_backend(backend):
    global BACKEND
    BACKEND = backend

def get_model_class():
    """Return the JoeBrain class for the active backend (mlx=GPU or numpy)."""
    if BACKEND == 'numpy':
        from model import JoeBrain
    else:
        from model_mlx import JoeBrain
    return JoeBrain

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
        result = subprocess.run(['ps', '-p', str(pid), '-o', 'args='],
                                capture_output=True, text=True, timeout=2)
        return 'train_expert' in result.stdout
    except Exception:
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
                result = subprocess.run(['ps', '-p', str(pid), '-o', 'args='],
                               capture_output=True, text=True, timeout=2)
                if "train_expert" in result.stdout:
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
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    progress = max(0.0, progress)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


RESUME_LR = 1e-4  # constant LR on --resume: keeps models parked at their floor (no sawtooth)

EMA_ALPHA = 1.0 - 1.0 / 40.0  # goal-tracker EMA horizon (~40 steps)


def split_blocks(raw):
    """Split raw text into blank-line-delimited blocks (list of stripped lines)."""
    blocks = []
    cur = []
    for line in raw.splitlines():
        if not line.strip():
            if cur:
                blocks.append(cur)
                cur = []
        else:
            cur.append(line.strip())
    if cur:
        blocks.append(cur)
    return blocks


def detect_conversation_format(blocks, min_multi_blocks=2):
    """True if the file is conversation-formatted: >=2 blank-delimited blocks each
    containing >=2 User: turns (multiple genuine back-and-forths). Guards against
    flat single-Q/A files (including an all-alternating single giant block)."""
    multi = 0
    for b in blocks:
        turns = sum(1 for ln in b if ln.startswith('User: '))
        if turns >= 2:
            multi += 1
            if multi >= min_multi_blocks:
                return True
    return False


def build_conversation_groups(tok, blocks, window):
    """Tokenize blocks and group consecutive blocks into window-sized conversation
    bundles. Returns a list of token arrays; each bundle is sampled independently
    so windows never straddle unrelated conversations."""
    block_ids = [np.array(tok.encode('\n\n'.join(b)), dtype=np.int32) for b in blocks]
    groups = []
    for i in range(0, len(block_ids), window):
        groups.append(np.concatenate(block_ids[i:i + window]))
    return groups


def _remote_exists(name):
    r = subprocess.run(['git', '-C', REPO, 'remote'], check=True,
                       capture_output=True, text=True)
    return name in r.stdout.split()


def _run_git(cmd, retries=3, timeout=120):
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0')
    for attempt in range(1, retries + 1):
        try:
            subprocess.run(cmd, check=True, timeout=timeout, env=env)
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"  [git cmd failed (attempt {attempt}/{retries}): {e}]")
            if attempt < retries:
                time.sleep(15)
    return False


def git_push(name, step):
    try:
        expert_file = f'data/experts/{name}/{name}.npz'
        log_file = f'data/experts/{name}/training.log'
        msg = f'chore: auto-save {name} expert at step {step}'
        lock_path = os.path.join(REPO, '.git', 'push.lock')
        with open(lock_path, 'a+') as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)
            try:
                subprocess.run(['git', '-C', REPO, 'add', expert_file], check=True)
                if os.path.exists(os.path.join(REPO, log_file)):
                    subprocess.run(['git', '-C', REPO, 'add', log_file], check=True)
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
                    if not _run_git(['git', '-C', REPO, 'push', 'jb', f'{commit}:main']):
                        raise RuntimeError(f"push to jb failed after retries ({name} step {step})")
                    subprocess.run(['git', '-C', REPO, 'update-ref',
                                    'refs/remotes/jb/main', commit], check=True)
                else:
                    branch = subprocess.run(
                        ['git', '-C', REPO, 'rev-parse', '--abbrev-ref', 'HEAD'],
                        check=True, capture_output=True, text=True).stdout.strip()
                    subprocess.run(
                        ['git', '-C', REPO, 'commit', '--no-verify', '-m', msg],
                        check=True)
                    if not _run_git(['git', '-C', REPO, 'push', 'origin', f'HEAD:{branch}']):
                        raise RuntimeError(f"push to origin failed after retries ({name} step {step})")
                print(f"  [pushed {name} expert at step {step}]")
            finally:
                fcntl.flock(lockf, fcntl.LOCK_UN)
    except Exception as e:
        print(f"  [git push failed: {e}]")


def remove_stale_locks():
    expert_dir = os.path.join(BASE, 'experts')
    if not os.path.isdir(expert_dir):
        return
    for fname in os.listdir(expert_dir):
        fpath = os.path.join(expert_dir, fname, 'training.lock')
        if not os.path.exists(fpath):
            continue
        try:
            with open(fpath) as f:
                pid = int(f.read().strip())
            if not is_pid_running(pid):
                os.remove(fpath)
        except (ValueError, OSError):
            os.remove(fpath)


def train_expert(name, steps=2000, lr=3e-4, seq_len=128, batch_size=8,
                 embed_dim=128, n_heads=4, n_layers=3, log_every=100,
                 sample_every=0, resume=False, push_every=0, save_every=0,
                 tag=None, window=4, force_conversations=False,
                 stop_at_loss=None, stop_sigma=None):

    remove_stale_locks()

    expert_dir = os.path.join(BASE, 'experts', name)
    data_path = os.path.join(expert_dir, f'{name}_train.txt')
    model_path = os.path.join(expert_dir, f'{name}.npz')

    if not os.path.exists(data_path):
        print(f"ERROR: Training data not found at {data_path}")
        print(f"Create {name}_train.txt with your specialist training text.")
        sys.exit(1)

    os.makedirs(expert_dir, exist_ok=True)

    if not os.path.exists(TOK_PATH):
        print(f"ERROR: Shared tokenizer not found at {TOK_PATH}")
        print("Run training/train.py first to build the tokenizer.")
        sys.exit(1)

    tok = Tokenizer()
    tok.load(TOK_PATH)
    print(f"Tokenizer loaded (vocab {tok.size})")

    with open(data_path) as f:
        raw = f.read()
    print(f"Training text: {len(raw):,} characters from {data_path}")

    data = np.array(tok.encode(raw), dtype=np.int32)
    print(f"Encoded: {len(data):,} tokens")

    JB = get_model_class()
    if os.path.exists(model_path):
        model = JB.load(model_path)
        seq_len = model.T
        if resume:
            print(f"Resumed expert '{name}' from {model_path} (Adam step {model.t})")
        else:
            print(f"Loaded existing expert '{name}' from {model_path} (fresh optimizer)")
            model.t = 0
            model.m = {k: np.zeros_like(v) for k, v in model.p.items()}
            model.v = {k: np.zeros_like(v) for k, v in model.p.items()}
    else:
        model = JB(
            vocab_size=tok.size,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            seq_len=seq_len,
        )

    if len(data) < seq_len + 10:
        print(f"ERROR: Not enough data ({len(data)} tokens). Need at least {seq_len + 10}.")
        sys.exit(1)

    total_params = sum(v.size for v in model.p.values())
    print(f"Expert '{name}': {total_params:,} params | {model.L} layers | dim {model.C}")
    print(f"Training for {steps} steps | lr={lr} | seq={seq_len} | batch={batch_size}\n")

    blocks = split_blocks(raw)
    conv_mode = force_conversations or detect_conversation_format(blocks)
    conv_groups = None
    if conv_mode:
        conv_groups = build_conversation_groups(tok, blocks, window)
        conv_groups = [g for g in conv_groups if len(g) >= seq_len + 1]
        if not conv_groups:
            conv_groups = None
            print("  [conversations] No group long enough for seq_len; falling back to flat sampling")
        else:
            print(f"  [conversations] {len(blocks)} blocks in {len(conv_groups)} window groups (window={window})")

    effective_lr = RESUME_LR if resume else lr
    print(f"# EFFECTIVE [{time.strftime('%Y-%m-%d %H:%M:%S')}] seq_len={seq_len} lr={effective_lr} backend={BACKEND}", file=sys.stderr, flush=True)

    losses = []
    start = time.time()
    last_log_time = start
    last_log_step = 0

    cmd = " ".join(sys.argv)
    log_path = os.path.join(expert_dir, "training.log")

    prev_pid = get_running_train_pid(name)
    if prev_pid is not None and prev_pid != os.getpid():
        print(f"[INFO] Found existing training (PID {prev_pid}). Signaling to save and stop...")
        with open(log_path, "a") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            lf.write(f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] STOP_PREV: signaling PID {prev_pid} to stop\n")
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
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
            if tag:
                lf.write(f"# TAG [{time.strftime('%Y-%m-%d %H:%M:%S')}] {tag}\n")
            lf.write(f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] RESUME: {cmd}\n")
    else:
        with open(log_path, "w") as lf:
            if tag:
                lf.write(f"# TAG [{time.strftime('%Y-%m-%d %H:%M:%S')}] {tag}\n")
            lf.write(f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] START: {cmd}\n")
            lf.write("# step,loss,lr,steps_per_sec,timestamp\n")

    with open(log_path, "a") as lf:
        if conv_mode and conv_groups is not None:
            lf.write(f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] CONVERSATIONS: {len(blocks)} blocks, window={window}\n")
        elif window >= 2:
            lf.write(f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] WINDOW ignored (flat data)\n")

    write_lock(name, os.getpid())

    should_stop = [False]
    def handle_stop(signum, frame):
        should_stop[0] = True
    signal.signal(signal.SIGUSR1, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    completed = False
    interrupted = False
    step = 0
    nan_steps = 0
    stop_window = []
    ema_loss = None
    mom_away_checks = 0
    last_mom_check = 0.0
    try:
        for step in range(1, steps + 1):
            if should_stop[0]:
                interrupted = True
                break

            # Mom-pause: save & wait while she holds the console (time-gated check).
            now = time.time()
            if now - last_mom_check >= 10.0:
                last_mom_check = now
                if mom_on_console():
                    model.save(model_path)
                    with open(log_path, "a") as lf:
                        lf.write(f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] PAUSE: ellenfriedman on console, saved & waiting\n")
                        lf.flush()
                    print(f"  [{time.strftime('%H:%M:%S')}] PAUSE: ellenfriedman on console — waiting...", flush=True)
                    while True:
                        time.sleep(10)
                        last_mom_check = time.time()
                        if should_stop[0]:
                            interrupted = True
                            break
                        if not mom_on_console():
                            mom_away_checks += 1
                            if mom_away_checks >= 2:
                                mom_away_checks = 0
                                with open(log_path, "a") as lf:
                                    lf.write(f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] RESUME: console free\n")
                                    lf.flush()
                                print(f"  [{time.strftime('%H:%M:%S')}] RESUME: console free — continuing", flush=True)
                                break
                        else:
                            mom_away_checks = 0
                else:
                    mom_away_checks = 0

            if interrupted:
                break

            model.zero_grad()

            if conv_groups is not None:
                gidx = np.random.randint(0, len(conv_groups), size=batch_size)
                starts = [np.random.randint(0, len(conv_groups[g]) - seq_len) for g in gidx]
                x_batch = np.stack([conv_groups[g][s:s + seq_len] for g, s in zip(gidx, starts)])
                y_batch = np.stack([conv_groups[g][s + 1:s + seq_len + 1] for g, s in zip(gidx, starts)])
            else:
                starts = np.random.randint(0, len(data) - seq_len, size=batch_size)
                x_batch = np.stack([data[s:s + seq_len] for s in starts])
                y_batch = np.stack([data[s + 1:s + seq_len + 1] for s in starts])

            logits, cache = model.forward(x_batch)
            batch_loss, dlogits = model.loss(logits, y_batch)
            if not np.isfinite(batch_loss):
                nan_steps += 1
                with open(log_path, "a") as lf:
                    lf.write(f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] NAN at step {step} (loss={batch_loss})\n")
                    lf.flush()
                model.zero_grad()
                continue
            model.backward(dlogits, cache)

            eff_lr = RESUME_LR if resume else cosine_lr(step, steps, lr)
            model.step(eff_lr)
            losses.append(batch_loss)

            stop_window.append(float(batch_loss))
            if len(stop_window) > 40:
                stop_window.pop(0)
            if ema_loss is None:
                ema_loss = float(batch_loss)
            else:
                ema_loss = EMA_ALPHA * ema_loss + (1.0 - EMA_ALPHA) * float(batch_loss)

            if stop_at_loss is not None and ema_loss <= stop_at_loss:
                with open(log_path, "a") as lf:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                    lf.write(f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] GOAL_MET at step {step}\n")
                    lf.flush()
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
                print(f"  [goal] EMA loss {ema_loss:.4f} <= {stop_at_loss:.4f} at step {step}; stopping")
                model.save(model_path)
                break
            if stop_sigma is not None and len(stop_window) >= 10:
                wmean = sum(stop_window) / len(stop_window)
                wvar = sum((x - wmean) ** 2 for x in stop_window) / len(stop_window)
                wstd = wvar ** 0.5
                if wstd > 0 and float(batch_loss) > wmean + stop_sigma * wstd:
                    with open(log_path, "a") as lf:
                        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                        lf.write(f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] DIVERGED at step {step} (loss={batch_loss})\n")
                        lf.flush()
                        fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
                    print(f"  [diverged] loss {batch_loss:.4f} > {wmean:.4f} + {stop_sigma}σ at step {step}; saving and stopping")
                    model.save(model_path)
                    break

            if step % log_every == 0:
                avg = np.mean(losses[-log_every:])
                now = time.time()
                sps = (step - last_log_step) / (now - last_log_time)
                last_log_time = now
                last_log_step = step
                print(f"  step {step:5d}/{steps} | loss {avg:.4f} | lr {eff_lr:.2e} | {sps:.2f} steps/s")

                with open(log_path, "a") as lf:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                    lf.write(f"{step},{avg:.4f},{eff_lr:.2e},{sps:.2f},{time.time()}\n")
                    lf.flush()
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

            if sample_every and step % sample_every == 0:
                sample = model.generate_fast(tok, "\n", max_new=20, temperature=0.8)
                print(f"  Sample: {repr(sample[:100])}\n")
                with open(log_path, "a") as lf:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                    lf.write(f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] SAMPLE step {step}:\n")
                    for line in sample[:200].splitlines():
                        lf.write(f"# {line}\n")
                    lf.flush()
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

            if push_every and step % push_every == 0:
                print(f"# PUSH at step {step} (next in {push_every} steps)", file=sys.stderr, flush=True)
                model.save(model_path)
                git_push(name, step)

            if save_every and step % save_every == 0:
                model.save(model_path)

    except (KeyboardInterrupt, SystemExit):
        if not completed:
            with open(log_path, "a") as lf:
                lf.write(f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] STOPPED by signal at step {step}\n")
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
        if interrupted or completed:
            with open(log_path, "a") as lf:
                tail = f" (NaN steps: {nan_steps})" if nan_steps else ""
                lf.write(f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] FINISH at step {step}{tail}\n")
        if completed:
            git_push(name, step)

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
    parser.add_argument('--push', type=int, default=1000, help='Push to github every N steps (default 1000)')
    parser.add_argument('--save', type=int, default=0, help='Save checkpoint every N steps (0=only at end)')
    parser.add_argument('--backend', type=str, default='mlx', choices=['mlx', 'numpy'],
                        help='Compute backend: mlx=Apple GPU (default), numpy=CPU')
    parser.add_argument('--tag', type=str, default=None,
                        help='Run tag; written as first line of training.log')
    parser.add_argument('--window', type=int, default=4,
                        help='Conversation-block grouping window (only used if data is conversation-formatted)')
    parser.add_argument('--conversations', action='store_true',
                        help='Force multi-turn conversation mode (auto-detected otherwise)')
    parser.add_argument('--stop-at-loss', type=float, default=None,
                        help='Auto-stop when EMA loss <= X (save + git push + exit cleanly)')
    parser.add_argument('--stop-sigma', type=float, default=None,
                        help='Auto-stop on divergence: loss > rolling mean + S*std (default None; use 3.0 convention)')
    args = parser.parse_args()

    set_backend(args.backend)

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
            tag=args.tag,
            window=args.window,
            force_conversations=args.conversations,
            stop_at_loss=args.stop_at_loss,
            stop_sigma=args.stop_sigma,
        )
    except Exception:
        expert_dir = os.path.join(BASE, 'experts', args.name)
        log_path = os.path.join(expert_dir, "training.log")
        try:
            with open(log_path, "a") as lf:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                lf.write(f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: training crashed\n")
                traceback.print_exc(file=lf)
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        raise
