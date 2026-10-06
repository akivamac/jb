"""merge_st.py — merge screened single-turn data into train.txt.
Takes cleaned ST files from data/_gen/cleaned/ (*_st_*_clean.txt),
dedups against train + intra-set, appends to {name}_train.txt.
Usage: python3 merge_st.py [expert ...]  (default: all)
"""
import os, re, sys
from difflib import SequenceMatcher

REPO = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(REPO, 'data', 'experts')
CLEAN = os.path.join(REPO, 'data', '_gen', 'cleaned')
EXPERTS = ['tree','reptiles','fish','knowledge','greeting','emotion','coding','python','cot','horse']

def norm(s):
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def load_pairs(path):
    pairs = []
    with open(path) as f:
        lines = f.read().splitlines()
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].strip()
        if line.startswith('User: '):
            q = line[6:]
            if not q.strip():
                i += 1; continue
            j = i + 1
            while j < n and not lines[j].strip().startswith('User: '):
                j += 1
            answer_parts = []
            for k in range(i + 1, j):
                l = lines[k].strip()
                if l.startswith('Joe: '):
                    answer_parts.append(l[5:])
                elif l:
                    answer_parts.append(l)
            a = '\n'.join(answer_parts)
            if q.strip() and a.strip():
                pairs.append((q.strip(), a.strip()))
            i = j
        else:
            i += 1
    return pairs

def sim(a, b):
    return SequenceMatcher(None, a, b).ratio()

def merge(name):
    train_path = os.path.join(BASE, name, f'{name}_train.txt')
    train_pairs = load_pairs(train_path)
    train_norm = set(norm(q) for q, _ in train_pairs)

    new_pairs = []
    for i in range(3):
        fpath = os.path.join(CLEAN, f'{name}_st_{i}_clean.txt')
        if os.path.exists(fpath):
            new_pairs.extend(load_pairs(fpath))

    kept = []
    removed_train = removed_self = 0
    seen = set()
    for q, a in new_pairs:
        nq = norm(q)
        if nq in train_norm or (kept and any(sim(nq, norm(kq)) > 0.82 for kq, _ in kept)):
            removed_train += 1
            continue
        if nq in seen:
            removed_self += 1
            continue
        seen.add(nq)
        kept.append((q, a))

    with open(train_path, 'a') as tf:
        for q, a in kept:
            tf.write(f'User: {q}\n')
            a_lines = a.split('\n')
            tf.write(f'Joe: {a_lines[0]}\n')
            for line in a_lines[1:]:
                tf.write(f'{line}\n')
            tf.write('\n')

    print(f'{name}: total_new={len(new_pairs)} merged={len(kept)} dropped_train={removed_train} dropped_intra={removed_self}')

if __name__ == '__main__':
    names = sys.argv[1:] or EXPERTS
    for name in names:
        merge(name)