#!/usr/bin/env python3
"""
panel_wizard.py — pretrain-then-seed wizard (dashboard-invoked via subprocess).

Chains, in order, against the repo checkout:
  0. ensure data/train.txt exists (run training/prepare_data.py if missing)
  1. pretrain on the full corpus:      python3 training/train.py --steps N
  2. grow model capacity:              python3 grow_model.py --new_embed_dim 256 --new_layers 4
  3. seed the new expert:              copy data/model.npz -> data/experts/{name}/{name}.npz
                                       copy data/train.txt   -> data/experts/{name}/{name}_train.txt

Refuses to clobber an existing expert unless --force. Emits one JSON step-status
object per line for the dashboard:
  {"step": "<id>", "cmd": "<shell cmd>", "ok": true/false, "error": "..."}

Usage:
  python3 training/panel_wizard.py --new-name math --steps 5000
  python3 training/panel_wizard.py --new-name math --steps 5000 --force
"""

import argparse
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, 'data')
EXPERTS = os.path.join(DATA, 'experts')
MODEL_NPZ = os.path.join(DATA, 'model.npz')
TRAIN_TXT = os.path.join(DATA, 'train.txt')

GROW_DIM = 256   # grown embedded dim per AGENTS.md grown-expert arch
GROW_LAYERS = 4  # grown layer count

PY = sys.executable


def emit(step, cmd, ok, error=None):
    payload = {'step': step, 'cmd': cmd, 'ok': bool(ok)}
    if error:
        payload['error'] = str(error)
    print(json_dumps(payload), flush=True)


def json_dumps(obj):
    import json
    return json.dumps(obj, default=str)


def run(cmd, cwd=REPO, timeout=None):
    """Run a command, return (returncode, tail_of_output)."""
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              timeout=timeout, env=dict(os.environ, PYTHONUNBUFFERED='1'))
    except subprocess.TimeoutExpired as e:
        return None, f'timed out (>{timeout}s)'
    except Exception as e:
        return None, str(e)
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-5:]
    return proc.returncode, '\n'.join(tail)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--new-name', required=True, help='New expert name (dir under data/experts/)')
    ap.add_argument('--steps', type=int, default=5000, help='Pretrain steps for training/train.py')
    ap.add_argument('--force', action='store_true', help='Allow overwriting an existing expert dir')
    args = ap.parse_args()

    name = args.new_name
    if not name or name != name.strip() or '/' in name or '..' in name:
        emit('check-name', f'--new-name {name!r}', False, 'invalid expert name (escape chars)')
        sys.exit(1)

    expert_dir = os.path.join(EXPERTS, name)
    if os.path.exists(expert_dir) and not args.force:
        emit('check-name', f'--new-name {name}', False,
             f'data/experts/{name} already exists; pass --force to overwrite')
        sys.exit(1)

    # Step 0: corpus (only if missing)
    if not os.path.exists(TRAIN_TXT):
        cmd = [PY, os.path.join(REPO, 'training', 'prepare_data.py')]
        rc, tail = run(cmd)
        emit('prepare_data', ' '.join(cmd), rc == 0, None if rc == 0 else tail)
        if rc != 0:
            sys.exit(1)
    else:
        emit('prepare_data', 'skip (data/train.txt present)', True)

    # Step 1: pretrain main model on full corpus
    cmd = [PY, os.path.join(REPO, 'training', 'train.py'),
           '--steps', str(args.steps), '--push', '0', '--log', '100']
    rc, tail = run(cmd)
    emit('pretrain', ' '.join(cmd), rc == 0, None if rc == 0 else f'{tail} (tail)')
    if rc != 0:
        sys.exit(1)
    if not os.path.exists(MODEL_NPZ):
        emit('pretrain', 'check ' + MODEL_NPZ, False, 'data/model.npz missing after train.py')
        sys.exit(1)

    # Step 2: grow capacity (preserves knowledge, zeroed new weights/Adam)
    cmd = [PY, os.path.join(REPO, 'grow_model.py'),
           '--new_embed_dim', str(GROW_DIM), '--new_layers', str(GROW_LAYERS)]
    rc, tail = run(cmd)
    emit('grow', ' '.join(cmd), rc == 0, None if rc == 0 else f'{tail} (tail)')
    if rc != 0:
        sys.exit(1)

    # Step 3: seed the new expert dir
    os.makedirs(expert_dir, exist_ok=True)
    errors = []
    dst_model = os.path.join(expert_dir, f'{name}.npz')
    dst_train = os.path.join(expert_dir, f'{name}_train.txt')
    if not os.path.exists(MODEL_NPZ):
        errors.append(f'{MODEL_NPZ} missing')
    else:
        shutil.copy2(MODEL_NPZ, dst_model)
    if os.path.exists(TRAIN_TXT):
        shutil.copy2(TRAIN_TXT, dst_train)
    else:
        errors.append(f'{TRAIN_TXT} missing')
    if errors:
        emit('seed', 'copy model.npz + train.txt', False, '; '.join(errors))
        sys.exit(1)
    emit('seed', f'cp model.npz -> {dst_model}; cp train.txt -> {dst_train}', True)
    print(json_dumps({'step': 'done', 'expert': name, 'model': dst_model,
                      'train': dst_train, 'ok': True}), flush=True)


if __name__ == '__main__':
    main()