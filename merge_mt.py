"""merge_mt.py — merge multi-turn training blocks.
Keeps whole conversation blocks ACCEPTED by BOTH reviewers, dedups against train
(using the block's first question) + within new set, appends to {name}_train.txt.
Logs to data/_gen/{name}/merge_mt.log.
Usage: python3 merge_mt.py [expert ...]  (default: all)
"""
import os, re, sys
from difflib import SequenceMatcher

REPO = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(REPO, 'data', '_gen')
BASE = os.path.join(REPO, 'data', 'experts')
EXPERTS = ['tree','reptiles','fish','knowledge','greeting','emotion','coding','python','cot','horse']

def norm(s):
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def load_blocks(path):
    """Return list of blocks (each a list of stripped lines), split by blank lines."""
    blocks = []
    if not os.path.exists(path):
        return blocks
    cur = []
    for line in open(path):
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

def read_verdict(path):
    idx = set()
    if not os.path.exists(path):
        return set()
    for line in open(path):
        line = line.strip()
        if re.match(r'^\d+:(ACCEPT|REJECT)', line):
            num, rest = line.split(':', 1)
            if rest.startswith('ACCEPT'):
                idx.add(int(num))
    return idx

def sim(a, b):
    return SequenceMatcher(None, a, b).ratio()

names = sys.argv[1:] or EXPERTS
for name in names:
    edir = os.path.join(GEN, name)
    if not os.path.isdir(edir):
        continue
    train_path = os.path.join(BASE, name, f'{name}_train.txt')
    train_qs = set(norm(q) for q in (l[6:] for l in open(train_path) if l.startswith('User: ')))

    accepted = []  # (chunk, idx, block)
    for f in sorted(os.listdir(edir)):
        if not (f.endswith('_cand.txt') and '_mt_' in f):
            continue
        chunk = f[:-len('_cand.txt')]
        v1 = read_verdict(os.path.join(edir, f'{chunk}_v1.txt'))
        v2 = read_verdict(os.path.join(edir, f'{chunk}_v2.txt'))
        ok = v1 & v2
        for i, block in enumerate(load_blocks(os.path.join(edir, f))):
            if i in ok:
                accepted.append((chunk, i, block))

    kept, removed_train, removed_self = [], 0, 0
    seen = set()
    for chunk, i, block in accepted:
        nq = norm(first_q(block))
        if nq in train_qs or (kept and any(sim(nq, norm(first_q(b))) > 0.82 for _, _, b in kept)):
            removed_train += 1
            continue
        if nq in seen or (kept and any(sim(nq, norm(first_q(b))) > 0.82 for _, _, b in kept)):
            removed_self += 1
            continue
        seen.add(nq)
        kept.append((chunk, i, block))

    added = 0
    with open(train_path, 'a') as tf:
        for chunk, i, block in kept:
            tf.write('\n'.join(block) + '\n\n')
            added += 1

    with open(os.path.join(edir, 'merge_mt.log'), 'a') as lf:
        lf.write(f"[{name}] dual-accepted={len(accepted)} dropped(train)={removed_train} "
                 f"dropped(intra)={removed_self} MERGED={added}\n")
    print(f"{name}: accepted={len(accepted)} -> merged={added} (dup vs {removed_train}+{removed_self})")