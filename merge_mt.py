"""merge_mt.py — merge multi-turn training blocks.
Keeps whole conversation blocks ACCEPTED by BOTH reviewers, dedups against train
(use the block's first question) + within new set, appends to {name}_train.txt.
Logs to data/_gen/add_newline/merge_mt.log.
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

def read_verdict(path):
    idx = set()
    reject_count = 0
    if not os.path.exists(path):
        return set(), reject_count
    with open(path) as f:
        for line in f:
            line = line.strip()
            if re.match(r'^\d+:(ACCEPT|REJECT)', line):
                num, rest = line.split(':', 1)
                if rest.startswith('ACCEPT'):
                    idx.add(int(num))
                elif rest.startswith('REJECT'):
                    reject_count += 1
    return idx, reject_count

def sim(a, b):
    return SequenceMatcher(None, a, b).ratio()

def generate_review_prompt(chunk_text, chunk_idx):
    return (f"Review chunk {chunk_idx} for multi-turn training.\n"
            f"Chunk content:\n{chunk_text}\n\n"
            f"Criteria:\n"
            f"1. Factual accuracy\n"
            f"2. Multi-turn quality\n"
            f"3. Directness\n"
            f"4. Relevance\n"
            f"5. Format\n\n"
            f"Verdict: ACCEPT or REJECT with reason.")

def retry_reviews(edir, review_dir):
    """For chunks without v1/v2 reviews, generate prompts and log them."""
    missing = []
    for f in sorted(os.listdir(edir)):
        if not f.endswith('.txt'):
            continue
        chunk = f[:-len('.txt')]
        v1_path = os.path.join(review_dir, f'{chunk}_v1.txt')
        v2_path = os.path.join(review_dir, f'{chunk}_v2.txt')
        if not os.path.exists(v1_path) or not os.path.exists(v2_path):
            chunk_path = os.path.join(edir, f)
            with open(chunk_path) as cf:
                chunk_text = cf.read()
            prompt = generate_review_prompt(chunk_text, chunk)
            missing.append((chunk, prompt))
    if missing:
        print(f"Missing reviews for chunks: {[c for c, _ in missing]}")
        prompt_path = os.path.join(review_dir, '_pending_reviews.txt')
        with open(prompt_path, 'w') as pf:
            for chunk, prompt in missing:
                pf.write(f"### {chunk}\n{prompt}\n\n")
        print(f"Generated review prompts for {len(missing)} chunks in {prompt_path}")
    return missing

names = sys.argv[1:] or EXPERTS
for name in names:
    edir = os.path.join(GEN, 'prompts_mt')
    if not os.path.isdir(edir):
        continue
    train_path = os.path.join(BASE, name, f'{name}_train.txt')
    with open(train_path) as _tf:
        train_qs = set(norm(q) for q in (l[6:] for l in _tf if l.startswith('User: ')))
    review_dir = os.path.join(GEN, 'prompts_review_mt')

    retry_reviews(edir, review_dir)

    missing_chunks = []
    accepted = []
    for f in sorted(os.listdir(edir)):
        if not (f.startswith(name) and f.endswith('.txt')):
            continue
        chunk = f[:-len('.txt')]
        v1_path = os.path.join(review_dir, f'{chunk}_v1.txt')
        v2_path = os.path.join(review_dir, f'{chunk}_v2.txt')
        if not os.path.exists(v1_path) or not os.path.exists(v2_path):
            missing_chunks.append(chunk)
            continue
        v1, v1_rejects = read_verdict(v1_path)
        v2, v2_rejects = read_verdict(v2_path)
        ok = v1 & v2
        for i, block in enumerate(load_blocks(os.path.join(edir, f))):
            if i in ok:
                accepted.append((chunk, i, block))

    if missing_chunks:
        print(f"{name}: missing reviews for chunks: {missing_chunks}")

    kept, removed_train, removed_self = [], 0, 0
    seen = set()
    for chunk, i, block in accepted:
        nq = norm(first_q(block))
        if nq in train_qs or (kept and any(sim(nq, norm(first_q(b))) > 0.82 for _, _, b in kept)):
            removed_train += 1
            continue
        if nq in seen:
            removed_self += 1
            continue
        seen.add(nq)
        kept.append((chunk, i, block))

    added = 0
    with open(train_path, 'a') as tf:
        for chunk, i, block in kept:
            tf.write('\n'.join(block) + '\n\n')
            added += 1

    add_newline_gen = os.path.join(GEN, 'add_newline')
    os.makedirs(add_newline_gen, exist_ok=True)
    log_path = os.path.join(add_newline_gen, 'merge_mt.log')
    total_v1_rejects = 0
    total_v2_rejects = 0
    for f_name in sorted(os.listdir(edir)):
        if not (f_name.startswith(name) and f_name.endswith('.txt')):
            continue
        chunk = f_name[:-len('.txt')]
        v1_path = os.path.join(review_dir, f'{chunk}_v1.txt')
        v2_path = os.path.join(review_dir, f'{chunk}_v2.txt')
        if os.path.exists(v1_path):
            _, r1 = read_verdict(v1_path)
            total_v1_rejects += r1
        if os.path.exists(v2_path):
            _, r2 = read_verdict(v2_path)
            total_v2_rejects += r2
    with open(log_path, 'a') as lf:
        lf.write(f"[{name}] dual-accepted={len(accepted)} dropped(train)={removed_train} "
                 f"dropped(intra)={removed_self} MERGED={added} v1_rejects={total_v1_rejects} "
                 f"v2_rejects={total_v2_rejects}\n")
    print(f"{name}: accepted={len(accepted)} -> merged={added} (dup vs {removed_train}+{removed_self})")
