"""Deduplicate each *_new.txt against its *_train.txt and itself.
Keeps only Q&A pairs whose question is not already answered (exact or ~similar) in train.
Also removes intra-file duplicates within _new.txt.
"""
import os, re, sys
from difflib import SequenceMatcher

REPO = os.path.dirname(os.path.abspath(__file__))
EXPERTS = ['tree','reptiles','fish','knowledge','greeting','emotion','coding','python','cot','horse']

def norm(s):
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def load_pairs(path):
    """Parse 'User: <q>\nJoe: <a>\n\n' blocks into [(q,a)]."""
    pairs = []
    if not os.path.exists(path):
        return pairs
    lines = open(path).read().splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if line.startswith('User: '):
            q = line[6:]
            # find the Joe: line
            j = i + 1
            while j < n and not lines[j].strip().startswith('Joe: '):
                j += 1
            if j < n:
                a = lines[j].strip()[5:]
                pairs.append((q, a))
                i = j + 1
                continue
        i += 1
    return pairs

def write_pairs(path, pairs):
    with open(path, 'w') as f:
        for q, a in pairs:
            f.write(f"User: {q}\nJoe: {a}\n\n")

def sim(a, b):
    return SequenceMatcher(None, a, b).ratio()

for name in EXPERTS:
    new_path = os.path.join(REPO, 'data', 'experts', name, f'{name}_new.txt')
    train_path = os.path.join(REPO, 'data', 'experts', name, f'{name}_train.txt')
    if not os.path.exists(new_path):
        continue

    new_pairs = load_pairs(new_path)
    train_pairs = load_pairs(train_path)

    # Normalized question sets from train (also keep full answer text pairs)
    train_qs = [(norm(q), q, a) for q, a in train_pairs]

    kept = []
    removed_against_train = []
    removed_dup = []

    for qi, (q, a) in enumerate(new_pairs):
        nq = norm(q)
        # 1) exact match in train
        if any(nq == tq for tq, _, _ in train_qs):
            removed_against_train.append((q, 'exact-train'))
            continue
        # 2) fuzzy match in train (ratio > 0.8)
        if any(sim(nq, tq) > 0.8 for tq, _, _ in train_qs):
            removed_against_train.append((q, 'fuzzy-train'))
            continue
        # 3) intra-file exact
        if any(norm(q2) == nq for q2, _ in kept):
            removed_dup.append((q, 'exact-new'))
            continue
        # 4) intra-file fuzzy
        if any(sim(norm(q2), nq) > 0.8 for q2, _ in kept):
            removed_dup.append((q, 'fuzzy-new'))
            continue
        kept.append((q, a))

    orig = len(new_pairs)
    print(f"\n=== {name}: {orig} -> {len(kept)} ===")
    for q, why in removed_against_train:
        print(f"  REMOVE (dup w/ train): {q[:70]}")
    for q, why in removed_dup:
        print(f"  REMOVE (dup in new):   {q[:70]}")

    write_pairs(new_path, kept)
    print(f"  wrote {len(kept)} pairs to {new_path}")

print("\nDone deduplicating. Errors relevant to merge:")
print("NOTE: file sizes were reduced; old counts were in the review summaries.")