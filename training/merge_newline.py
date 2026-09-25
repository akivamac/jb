"""merge_newline.py — merge add_newline blocks into expert train files.
Keeps whole blocks, dedups against train (using the block's first question)
and within the new set, appends to {name}_train.txt, logs to
data/_gen/add_newline/merge.log.
Usage: python3 training/merge_newline.py [expert ...]  (default: all)
"""
import os, re, sys
from difflib import SequenceMatcher

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(REPO, 'data', '_gen', 'add_newline')
BASE = os.path.join(REPO, 'data', 'experts')
EXPERTS = ['tree','reptiles','fish','knowledge','greeting','emotion','coding','python','cot','horse']

def norm(s):
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def load_blocks(path):
    blocks = []
    if not os.path.exists(path):
        return blocks
    cur = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line.strip():
                if cur:
                    blocks.append(cur)
                    cur = []
            else:
                cur.append(line.strip())
    if cur:
        blocks.append(cur)
    return blocks

def first_q(block):
    for line in block:
        if line.startswith('User: '):
            return line[6:]
    return ''

def sim(a, b):
    return SequenceMatcher(None, a, b).ratio()

names = sys.argv[1:] or EXPERTS
for name in names:
    cand_path = os.path.join(GEN, f'{name}.txt')
    train_path = os.path.join(BASE, name, f'{name}_train.txt')
    with open(train_path) as _tf:
        train_qs = set(norm(q) for q in (l[6:] for l in _tf if l.startswith('User: ')))

    kept, removed_train, removed_self = [], 0, 0
    seen = set()
    for block in load_blocks(cand_path):
        nq = norm(first_q(block))
        if nq in train_qs or (kept and any(sim(nq, norm(first_q(b))) > 0.82 for b in kept)):
            removed_train += 1
            continue
        if nq in seen:
            removed_self += 1
            continue
        seen.add(nq)
        kept.append(block)

    added = 0
    with open(train_path, 'a') as tf:
        for block in kept:
            tf.write('\n'.join(block) + '\n\n')
            added += 1

    log_path = os.path.join(GEN, 'merge.log')
    with open(log_path, 'a') as lf:
        lf.write(f"[{name}] cands={len(load_blocks(cand_path))} "
                 f"dropped(train)={removed_train} dropped(intra)={removed_self} MERGED={added}\n")
    if os.path.exists(log_path):
        with open(log_path) as f:
            log_lines = f.readlines()
        if len(log_lines) > 200:
            with open(log_path, 'w') as f:
                f.writelines(log_lines[-100:])
    print(f"{name}: cands={len(load_blocks(cand_path))} -> merged={added} "
          f"(dup vs {removed_train}+{removed_self})")
