import argparse, os, re, sys
from difflib import SequenceMatcher

def norm(s):
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def clean_line(s):
    return s.rstrip()

def has_id_suffix(s):
    # forbid (...) suffix like (5970)
    return bool(re.search(r'\([0-9A-Za-z]{1,12}\)\s*$', s))

class Block(list):
    pass

def load_blocks(path):
    blocks = []
    if not os.path.exists(path):
        return blocks
    cur = []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\r\n')
            # treat blank as separator
            if not line.strip():
                if cur:
                    blocks.append(cur[:])
                    cur = []
                # skip multiple blanks
            else:
                cur.append(line)
    if cur:
        blocks.append(cur[:])
    return blocks

def read_train_qs(train_path):
    qs = set()
    if train_path and os.path.exists(train_path):
        with open(train_path) as f:
            for line in f:
                if line.startswith('User: '):
                    qs.add(norm(line[6:]))
    return qs

def valid_block(b, forbid_against_train=set()):
    if not b:
        return False, "empty"
    # must start with User:
    if not b[0].strip().startswith('User:'):
        return False, "does_not_start_user"
    turns = 0
    last_user = None
    last_joe = None
    seen_q = set()
    for i, raw in enumerate(b):
        s = raw.strip()
        if not s.startswith('User:') and not s.startswith('Joe:'):
            return False, f"bad_prefix_line_{i}"
        if has_id_suffix(s):
            return False, f"id_suffix_line_{i}"
        if s.startswith('User:'):
            q = s[6:]
            if q.strip() == '':
                return False, f"empty_user_{i}"
            n = norm(q)
            if n in seen_q:
                return False, f"dup_q_in_block_{i}"
            seen_q.add(n)
            if n in forbid_against_train:
                return False, f"train_dup_{i}"
            last_user = n
            turns += 1
        else:  # Joe
            a = s[5:]
            if a.strip() == '':
                return False, f"empty_joe_{i}"
            if has_id_suffix(a):
                return False, f"id_suffix_line_{i}"
            last_joe = a
        # alternation sanity: after User comes Joe
    # even? pairs
    # count: turns = number of User lines
    user_count = sum(1 for x in b if x.strip().startswith('User:'))
    joe_count = sum(1 for x in b if x.strip().startswith('Joe:'))
    if user_count != joe_count:
        return False, "uneven_turns"
    # strict alternation
    for i in range(len(b)):
        if i % 2 == 0:
            if not b[i].strip().startswith('User:'):
                return False, "non_alternating"
        else:
            if not b[i].strip().startswith('Joe:'):
                return False, "non_alternating"
    # reasonable size
    if user_count < 1:
        return False, "no_turns"
    if user_count > 40:
        return False, "too_long"
    return True, "ok"

def sim(a,b):
    return SequenceMatcher(None,a,b).ratio()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--train', default=None)
    ap.add_argument('--out_dir', default='data/_gen/cleaned')
    ap.add_argument('--repair', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.basename(args.input)
    blocks = load_blocks(args.input)
    train_qs = read_train_qs(args.train)
    good = []
    bad = []
    for i,b in enumerate(blocks):
        ok, why = valid_block(b, forbid_against_train=train_qs)
        if ok:
            good.append(b)
        else:
            bad.append((i, b, why))
    out_clean = os.path.join(args.out_dir, base.replace('.txt','') + '_clean.txt')
    out_bad = os.path.join(args.out_dir, base.replace('.txt','') + '_bad.txt')
    with open(out_clean,'w') as f:
        for b in good:
            f.write('\n'.join(b)+'\n\n')
    with open(out_bad,'w') as f:
        for i,b,why in bad:
            f.write(f'# block_{i} REJECT: {why}\n')
            f.write('\n'.join(b)+'\n\n')
    print(f"[screen] {base}: good={len(good)} bad={len(bad)} -> {out_clean}")
if __name__=='__main__':
    main()
