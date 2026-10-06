"""pipeline.py — screen + dedup-account + review-verdict + merge for one expert.

Usage:
  python3 pipeline.py screen <expert>     # validate + dedup vs train + intra-set, write *_clean.txt + report
  python3 pipeline.py stats <expert>      # print report only
  python3 pipeline.py merge <expert>      # merge accepted (clean + AC-verified) into train.txt

Screens every data/_gen/{expert}/gen_*.txt, in sorted order, accumulating a
seen-set so cross-file and intra-file duplicates are caught. Every drop is
counted by reason so dups are accounted for.
"""
import os, re, sys, json
from difflib import SequenceMatcher

REPO = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(REPO, 'data', '_gen')
BASE = os.path.join(REPO, 'data', 'experts')
EXPERTS = ['tree','reptiles','fish','knowledge','greeting','emotion',
           'coding','python','cot','horse']

SIM_THRESHOLD = 0.82

def norm(s):
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def load_blocks(path):
    """Split on blank lines -> list of blocks (list of stripped lines)."""
    blocks, cur = [], []
    if not os.path.exists(path):
        return blocks
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

def has_id_suffix(s):
    return bool(re.search(r'\([0-9A-Za-z]{1,12}\)\s*$', s))

def check_block(b, train_qs, seen_qs):
    """Return (ok, reason). seen_qs is mutated on accept."""
    if not b:
        return False, 'empty'
    if not b[0].startswith('User: '):
        return False, 'no_user_start'
    for i, line in enumerate(b):
        if not (line.startswith('User: ') or line.startswith('Joe: ')):
            return False, f'bad_prefix@{i}'
        if has_id_suffix(line):
            return False, f'id_suffix@{i}'
        if i % 2 == 0 and not line.startswith('User: '):
            return False, f'non_alternating@{i}'
        if i % 2 == 1 and not line.startswith('Joe: '):
            return False, f'non_alternating@{i}'
    uc = sum(1 for x in b if x.startswith('User: '))
    jc = sum(1 for x in b if x.startswith('Joe: '))
    if uc != jc:
        return False, 'uneven_turns'
    if uc < 1:
        return False, 'no_turns'
    if uc > 40:
        return False, 'too_long'
    block_qs = []
    for i, line in enumerate(b):
        if line.startswith('User: '):
            q = line[6:].strip()
            if not q:
                return False, f'empty_user@{i}'
            block_qs.append(norm(q))
    if len(set(block_qs)) != len(block_qs):
        return False, 'dup_q_in_block'
    fq = block_qs[0]
    if fq in train_qs:
        return False, 'dup_vs_train'
    if fq in seen_qs:
        return False, 'dup_vs_new'
    return True, 'ok'

def build(name):
    edir = os.path.join(GEN, name)
    train_path = os.path.join(BASE, name, f'{name}_train.txt')
    train_qs = set()
    if os.path.exists(train_path):
        with open(train_path) as f:
            for line in f:
                if line.startswith('User: '):
                    train_qs.add(norm(line[6:]))

    files = sorted(f for f in os.listdir(edir)
                   if f.startswith('gen_') and f.endswith('.txt'))
    kept, seen = [], set()
    reasons = {}
    per_file = {}
    for fn in files:
        path = os.path.join(edir, fn)
        blocks = load_blocks(path)
        fk = 0
        for b in blocks:
            ok, why = check_block(b, train_qs, seen)
            if ok:
                seen.add(norm(next(x[6:] for x in b if x.startswith('User: '))))
                kept.append(b)
                fk += 1
            else:
                reasons[why] = reasons.get(why, 0) + 1
        per_file[fn] = {'total': len(blocks), 'kept': fk,
                        'dropped': len(blocks) - fk}

    outdir = os.path.join(edir, 'clean')
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f'{name}_clean.txt')
    with open(out, 'w') as f:
        for b in kept:
            f.write('\n'.join(b) + '\n\n')

    mt = sum(1 for b in kept if sum(1 for x in b if x.startswith('User: ')) >= 2)
    st = len(kept) - mt
    report = {
        'expert': name,
        'gen_files': len(files),
        'gen_blocks_in': sum(v['total'] for v in per_file.values()),
        'kept_blocks': len(kept),
        'kept_ST': st,
        'kept_MT': mt,
        'kept_exchanges': sum(sum(1 for x in b if x.startswith('User: ')) for b in kept),
        'dropped_total': sum(reasons.values()),
        'drop_reasons': reasons,
        'train_qs': len(train_qs),
        'per_file': per_file,
        'clean_path': out,
    }
    rp = os.path.join(edir, 'report.json')
    with open(rp, 'w') as f:
        json.dump(report, f, indent=2)
    return report, kept

def merge(name):
    edir = os.path.join(GEN, name)
    clean = os.path.join(edir, 'clean', f'{name}_clean.txt')
    if not os.path.exists(clean):
        print(f'{name}: no clean file, run screen first')
        return
    train_path = os.path.join(BASE, name, f'{name}_train.txt')
    blocks = load_blocks(clean)
    # final safety: re-dedup vs current train (train may have grown since screen)
    train_qs = set()
    with open(train_path) as f:
        for line in f:
            if line.startswith('User: '):
                train_qs.add(norm(line[6:]))
    added = dtrain = 0
    with open(train_path, 'a') as tf:
        for b in blocks:
            fq = norm(next(x[6:] for x in b if x.startswith('User: ')))
            if fq in train_qs:
                dtrain += 1
                continue
            train_qs.add(fq)
            tf.write('\n'.join(b) + '\n\n')
            added += 1
    with open(os.path.join(edir, 'merge.log'), 'a') as lf:
        lf.write(f'added={added} dropped_dup_vs_train={dtrain}\n')
    print(f'{name}: MERGED {added} blocks (dropped {dtrain} as dup vs train)')

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    cmd, names = sys.argv[1], sys.argv[2:] or EXPERTS
    for name in names:
        if cmd == 'screen':
            rep, _ = build(name)
            print(f"[{name}] in={rep['gen_blocks_in']} kept={rep['kept_blocks']} "
                  f"(ST={rep['kept_ST']} MT={rep['kept_MT']}) "
                  f"exchanges={rep['kept_exchanges']} dropped={rep['dropped_total']}")
            for k, v in sorted(rep['drop_reasons'].items(), key=lambda x: -x[1]):
                print(f"    drop {k}: {v}")
        elif cmd == 'stats':
            rp = os.path.join(GEN, name, 'report.json')
            if os.path.exists(rp):
                print(json.dumps(json.load(open(rp)), indent=2))
            else:
                print(f'{name}: no report')
        elif cmd == 'merge':
            merge(name)

if __name__ == '__main__':
    main()