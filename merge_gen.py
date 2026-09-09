"""merge_gen.py — consume author chunks + dual-review verdicts.
Keeps pairs ACCEPTED by BOTH reviewers, dedups q against train + within new set,
appends to {name}_train.txt. Logs to data/_gen/{name}/merge.log.
Usage: python3 merge_gen.py [expert ...]  (default: all)
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

def load_pairs(path):
    pairs = []
    if not os.path.exists(path):
        return pairs
    lines = open(path).read().splitlines()
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].strip()
        if line.startswith('User: '):
            q = line[6:]
            j = i + 1
            while j < n and not lines[j].strip().startswith('Joe: '):
                j += 1
            if j < n:
                a = lines[j].strip()[5:]
                pairs.append((q, a))
            i = j
        i += 1
    return pairs

def read_verdict(path):
    idx = set()
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
    train_pairs = load_pairs(train_path)
    train_norm = [norm(q) for q, _ in train_pairs]

    # gather accepted pairs across all chunks (need BOTH verdict files to say accept)
    accepted = []   # (chunk, index, q, a)
    for f in sorted(os.listdir(edir)):
        if not f.endswith('_cand.txt'):
            continue
        chunk = f[:-len('_cand.txt')]
        cand_path = os.path.join(edir, f)
        v1 = read_verdict(os.path.join(edir, f'{chunk}_v1.txt'))
        v2 = read_verdict(os.path.join(edir, f'{chunk}_v2.txt'))
        ok = v1 & v2  # dual accept
        cands = load_pairs(cand_path)
        for i, (q, a) in enumerate(cands):
            if i in ok:
                accepted.append((chunk, i, q, a))

    # dedup against train
    train_norms = set(train_norm)
    kept = []
    removed_train = removed_self = 0
    seen = set()
    for chunk, i, q, a in accepted:
        nq = norm(q)
        if nq in train_norms or (kept and any(sim(nq, norm(q2)) > 0.82 for q2, _, _ in kept)):
            removed_train += 1
            continue
        if nq in seen or (kept and any(sim(nq, norm(q2)) > 0.82 for q2, _, _ in kept)):
            removed_self += 1
            continue
        seen.add(nq)
        kept.append((q, a, chunk))

    # append merge
    added = 0
    with open(train_path, 'a') as tf:
        for q, a, chunk in kept:
            tf.write(f"User: {q}\nJoe: {a}\n\n")
            added += 1

    with open(os.path.join(edir, 'merge.log'), 'a') as lf:
        lf.write(f"[{name}] dual-accepted={len(accepted)} dropped(train)={removed_train} "
                 f"dropped(intra)={removed_self} MERGED={added}\n")
    print(f"{name}: accepted={len(accepted)} -> merged={added} "
          f"(dup vs {removed_train}+{removed_self})")