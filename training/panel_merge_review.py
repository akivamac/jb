#!/usr/bin/env python3
"""
panel_merge_review.py — review & merge multi-turn training blocks.

Ports the exact dedup/merge logic from merge_mt.py (blocks split on blank
lines; a block is ACCEPTED only when its index is accepted in BOTH v1 and v2;
dedup on the block's first question against the expert's existing train.txt
plus within the new set using SequenceMatcher ratio > 0.82). Do not reinvent:
this file mirrors merge_mt.py line-for-line where behavior overlaps.

Paths:
  chunks:    data/_gen/prompts_mt/{chunk}.txt
  verdicts:  data/_gen/prompts_review_mt/{chunk}_v1.txt / {chunk}_v2.txt
  output:    data/experts/{expert}/{expert}_train.txt (appended)

Actions (mutually exclusive):
  --list                      list reviewable chunks + v1/v2 verdict status
  --show CHUNK                show per-index blocks + current v1/v2 verdicts
  --verdict CHUNK INDEX V     record VERDICT (ACCEPT|REJECT) for a block index
  --merge EXPERT              merge dual-accepted blocks into {expert}_train.txt

Flags:
  --reviewer {v1,v2}   verdict file to write via --verdict (default v1)
  --json               machine-readable output (default human text)

Usage:
  python3 training/panel_merge_review.py --list
  python3 training/panel_merge_review.py --show reptiles_mt_07
  python3 training/panel_merge_review.py --verdict reptiles_mt_07 12 ACCEPT
  python3 training/panel_merge_review.py --verdict reptiles_mt_07 12 ACCEPT --reviewer v2
  python3 training/panel_merge_review.py --merge reptiles --json
"""

import argparse
import json
import os
import re
import sys
from difflib import SequenceMatcher

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(REPO, 'data', '_gen')
BASE = os.path.join(REPO, 'data', 'experts')
CHUNK_DIR = os.path.join(GEN, 'prompts_mt')
REVIEW_DIR = os.path.join(GEN, 'prompts_review_mt')
EXPERTS = ['tree', 'reptiles', 'fish', 'knowledge', 'greeting', 'emotion',
           'coding', 'python', 'cot', 'horse']
VERDICT_LINE = re.compile(r'^(\d+):(ACCEPT|REJECT)(?::(.*))?$')


# --- ported from merge_mt.py (verbatim where behavior matters) ---------------

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


# --- panel helpers ------------------------------------------------------------

def index_verdicts(path):
    """Return {index: 'ACCEPT'|'REJECT'} for a verdict file (latest line wins)."""
    v = {}
    if not os.path.exists(path):
        return v
    with open(path) as f:
        for line in f:
            m = VERDICT_LINE.match(line.strip())
            if m:
                v[int(m.group(1))] = m.group(2)
    return v


def chunk_path(chunk):
    return os.path.join(CHUNK_DIR, f'{chunk}.txt')


def verdict_path(chunk, reviewer):
    return os.path.join(REVIEW_DIR, f'{chunk}_{reviewer}.txt')


def list_chunks():
    """[{chunk, blocks, v1_exists, v2_exists, v1_accept, v1_reject, v2_accept, v2_reject}]"""
    out = []
    if not os.path.isdir(CHUNK_DIR):
        return out
    for f in sorted(os.listdir(CHUNK_DIR)):
        if not f.endswith('.txt'):
            continue
        chunk = f[:-len('.txt')]
        v1p, v2p = verdict_path(chunk, 'v1'), verdict_path(chunk, 'v2')
        a1, r1 = read_verdict(v1p)
        a2, r2 = read_verdict(v2p)
        out.append({
            'chunk': chunk,
            'blocks': len(load_blocks(chunk_path(chunk))),
            'v1_exists': os.path.exists(v1p),
            'v2_exists': os.path.exists(v2p),
            'v1_accept': len(a1), 'v1_reject': r1,
            'v2_accept': len(a2), 'v2_reject': r2,
        })
    return out


def show_chunk(chunk):
    """[{index, first_question, turns, lines, v1, v2}]"""
    blocks = load_blocks(chunk_path(chunk))
    iv1 = index_verdicts(verdict_path(chunk, 'v1'))
    iv2 = index_verdicts(verdict_path(chunk, 'v2'))
    out = []
    for i, b in enumerate(blocks):
        out.append({
            'index': i,
            'first_question': first_q(b),
            'turns': sum(1 for ln in b if ln.startswith('User: ')),
            'lines': len(b),
            'v1': iv1.get(i, 'CLEAR'),
            'v2': iv2.get(i, 'CLEAR'),
        })
    return out


def record_verdict(chunk, index, verdict, reviewer):
    """Append (or overwrite) `index:VERDICT` in the chosen reviewer file."""
    if verdict not in ('ACCEPT', 'REJECT'):
        return None
    blocks = load_blocks(chunk_path(chunk))
    if index < 0 or index >= len(blocks):
        return None
    vp = verdict_path(chunk, reviewer)
    os.makedirs(os.path.dirname(vp), exist_ok=True)
    kept = []
    if os.path.exists(vp):
        with open(vp) as f:
            kept = [line for line in f if not line.strip().startswith(f'{index}:')]
    kept.append(f'{index}:{verdict}\n')
    with open(vp, 'w') as f:
        f.write(''.join(kept))
    return {'chunk': chunk, 'index': index, 'verdict': verdict,
            'reviewer': reviewer, 'file': vp}


def merge_expert(name):
    """Port of merge_mt.py's merge loop for one expert."""
    if not os.path.isdir(CHUNK_DIR):
        return {'expert': name, 'error': f'no chunk dir {CHUNK_DIR}'}
    train_path = os.path.join(BASE, name, f'{name}_train.txt')
    if not os.path.exists(train_path):
        return {'expert': name, 'error': f'no train file {train_path}'}
    with open(train_path) as _tf:
        train_qs = set(norm(q) for q in (l[6:] for l in _tf if l.startswith('User: ')))

    missing_chunks = []
    accepted = []
    for f in sorted(os.listdir(CHUNK_DIR)):
        if not (f.startswith(name) and f.endswith('.txt')):
            continue
        chunk = f[:-len('.txt')]
        v1p = verdict_path(chunk, 'v1')
        v2p = verdict_path(chunk, 'v2')
        if not os.path.exists(v1p) or not os.path.exists(v2p):
            missing_chunks.append(chunk)
            continue
        v1, _r1 = read_verdict(v1p)
        v2, _r2 = read_verdict(v2p)
        ok = v1 & v2
        for i, block in enumerate(load_blocks(chunk_path(chunk))):
            if i in ok:
                accepted.append((chunk, i, block))

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

    return {
        'expert': name,
        'dual_accepted': len(accepted),
        'merged': added,
        'removed_train': removed_train,
        'removed_self': removed_self,
        'missing_chunks': missing_chunks,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--list', action='store_true', help='List reviewable chunks')
    ap.add_argument('--show', metavar='CHUNK', help='Show chunk blocks + verdict status')
    ap.add_argument('--verdict', metavar=('CHUNK', 'INDEX', 'VERDICT'), nargs=3,
                    help='Record a verdict: CHUNK INDEX ACCEPT|REJECT')
    ap.add_argument('--merge', metavar='EXPERT', help='Merge accepted blocks into expert train.txt')
    ap.add_argument('--reviewer', choices=['v1', 'v2'], default='v1',
                    help='Reviewer file to write for --verdict (default v1)')
    ap.add_argument('--json', action='store_true', help='Machine-readable JSON output')
    args = ap.parse_args()

    selected = sum(1 for x in (args.list, args.show, args.verdict, args.merge) if x)
    if selected != 1:
        ap.error('exactly one of --list / --show / --verdict / --merge required')

    if args.list:
        data = list_chunks()
        if args.json:
            json.dump(data, sys.stdout)
            sys.stdout.write('\n')
        else:
            for c in data:
                status = 'v1' if c['v1_exists'] else '  '
                status += '/'
                status += 'v2' if c['v2_exists'] else '  '
                print(f"{c['chunk']:<40} blocks={c['blocks']:<4} [{status}] "
                      f"v1 {c['v1_accept']}+{c['v1_reject']}- v2 {c['v2_accept']}+{c['v2_reject']}-")
        return

    if args.show:
        data = show_chunk(args.show)
        if args.json:
            json.dump(data, sys.stdout)
            sys.stdout.write('\n')
        else:
            for b in data:
                print(f"{b['index']:>3}  {b['v1']:<6} {b['v2']:<6} turns={b['turns']:<2} lines={b['lines']:<3} {b['first_question'][:70]}")
        return

    if args.verdict:
        chunk, idx_s, verdict = args.verdict
        try:
            index = int(idx_s)
        except ValueError:
            print(f"index must be an integer: {idx_s!r}")
            sys.exit(1)
        res = record_verdict(chunk, index, verdict, args.reviewer)
        if res is None:
            if args.json:
                json.dump({'error': 'cannot record verdict (check chunk file & index range)'}, sys.stdout)
                sys.stdout.write('\n')
            else:
                print(f"cannot record verdict (check chunk file & index range, chunk={chunk!r})")
            sys.exit(0)
        if args.json:
            json.dump(res, sys.stdout)
            sys.stdout.write('\n')
        else:
            print(f"{chunk}[{index}] {verdict} written to {res['file']}")
        return

    if args.merge:
        res = merge_expert(args.merge)
        if args.json:
            json.dump(res, sys.stdout)
            sys.stdout.write('\n')
        else:
            if 'error' in res:
                print(f"ERROR: {res['error']}")
                sys.exit(1)
            print(f"expert {res['expert']}: dual_accepted={res['dual_accepted']} -> "
                  f"merged={res['merged']} (dropped {res['removed_train']} train / "
                  f"{res['removed_self']} self)")
            if res['missing_chunks']:
                print(f"missing reviews: {res['missing_chunks']}")
        return


if __name__ == '__main__':
    main()