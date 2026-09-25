"""merge_unmerged.py — merge reviewed {name}/unmerged_data.txt into {name}_train.txt.
Dedups against train (normalized first question) + within new set.
Usage: python3 merge_unmerged.py [expert ...]  (default: all)
"""
import os, re, sys
from difflib import SequenceMatcher

REPO = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(REPO, 'data', 'experts')
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
            j = i + 1
            while j < n and not lines[j].strip().startswith('User: '):
                j += 1
            answer_lines = []
            k = i + 1
            while k < j and k < n:
                l = lines[k].strip()
                if l.startswith('Joe: '):
                    answer_lines.append(l[5:])
                elif l:
                    answer_lines.append(l)
                k += 1
            a = '\n'.join(answer_lines)
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
    unmerged_path = os.path.join(BASE, name, 'unmerged_data.txt')
    if not os.path.exists(unmerged_path):
        print(f'{name}: no unmerged_data.txt, skip')
        return
    with open(train_path) as f:
        train_lines = f.read().splitlines()
    train_pairs = load_pairs(train_path)
    train_norm = set(norm(q) for q, _ in train_pairs)

    new_pairs = load_pairs(unmerged_path)
    kept = []
    removed_train = removed_self = 0
    seen = set()
    for q, a in new_pairs:
        nq = norm(q)
        if nq in train_norm or (kept and any(sim(nq, norm(q2)) > 0.82 for q2, _ in kept)):
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

    # log
    log_dir = os.path.join(BASE, name)
    log_path = os.path.join(log_dir, 'merge_unmerged.log')
    with open(log_path, 'a') as lf:
        lf.write(f'merged={len(kept)} skipped_train={removed_train} skipped_intra={removed_self} '
                 f'new_total={len(train_pairs)+len(kept)}\n')
    print(f'{name}: merged={len(kept)} (skipped train={removed_train}, intra={removed_self})')

names = sys.argv[1:] or EXPERTS
for name in names:
    merge(name)
