"""prep_data.py — prepare raw Brave Search data for review.
Reads a raw file, dedupes internally, checks format, dedupes against train.txt.
Output: data/experts/{name}/unmerged_data.txt (ready for review only).
Usage: python3 prep_data.py --name emotion --input raw_emotion.txt
"""
import os, re, sys, argparse
from difflib import SequenceMatcher

REPO = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(REPO, 'data', 'experts')

def norm(s):
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def parse_pairs(path):
    """Parse a file into (question, answer) pairs. Supports multi-line answers.
    Skips lines that don't follow User:/Joe: format."""
    pairs = []
    skipped_format = 0
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
            else:
                skipped_format += 1
            i = j
        elif line.startswith('#') or line == '':
            i += 1
        else:
            skipped_format += 1
            i += 1
    return pairs, skipped_format

STOPWORDS = {'what', 'is', 'a', 'an', 'the', 'do', 'does', 'can', 'how', 'why', 'when',
             'where', 'who', 'which', 'this', 'that', 'these', 'those', 'it', 'they',
             'we', 'you', 'i', 'define', 'tell', 'explain', 'describe', 'make', 'got',
             'get', 'may', 'might', 'will', 'shall', 'could', 'would', 'should'}

def content_words(s):
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return set(w for w in s.split() if w not in STOPWORDS and len(w) > 1)

def dedupe_pairs(pairs):
    """Remove duplicates within the set. Returns deduped list and count of removed."""
    kept = []
    removed = 0
    seen_content = []
    for q, a in pairs:
        cw = content_words(q)
        if not cw:
            removed += 1
            continue
        is_dup = False
        for sc in seen_content:
            if cw and sc:
                overlap = len(cw & sc) / max(len(cw), len(sc))
                if overlap > 0.70:
                    is_dup = True
                    break
        if is_dup:
            removed += 1
            continue
        seen_content.append(cw)
        kept.append((q, a))
    return kept, removed

def sim(a, b):
    return SequenceMatcher(None, a, b).ratio()

def load_train_pairs(train_path):
    pairs, _ = parse_pairs(train_path)
    return pairs, set(norm(q) for q, _ in pairs)

def prep(name, input_path):
    train_path = os.path.join(BASE, name, f'{name}_train.txt')
    unmerged_path = os.path.join(BASE, name, 'unmerged_data.txt')

    if not os.path.exists(train_path):
        print(f'ERROR: {train_path} not found')
        return

    # Parse raw input
    raw_pairs, skipped_format = parse_pairs(input_path)
    print(f'Raw pairs parsed: {len(raw_pairs)} (skipped format: {skipped_format})')

    # Internal dedupe
    deduped, intra_removed = dedupe_pairs(raw_pairs)
    print(f'After internal dedupe: {len(deduped)} (removed {intra_removed} duplicates)')

    # Load train.txt for dedupe
    train_pairs, train_norm = load_train_pairs(train_path)
    print(f'Train.txt has {len(train_pairs)} pairs')

    # Dedupe against train.txt
    kept = []
    removed_train = 0
    removed_intra = 0
    seen = set()
    for q, a in deduped:
        nq = norm(q)
        if nq in train_norm:
            removed_train += 1
            continue
        if kept and any(sim(nq, norm(q2)) > 0.82 for q2, _ in kept):
            removed_train += 1
            continue
        if nq in seen:
            removed_intra += 1
            continue
        seen.add(nq)
        kept.append((q, a))

    # Write unmerged_data.txt
    with open(unmerged_path, 'w') as f:
        f.write(f'# unmerged training data for the {name} expert\n')
        f.write(f'# Generated from {os.path.basename(input_path)} via prep_data.py\n')
        f.write(f'# {len(kept)} pairs after dedupe\n')
        f.write('# Format: "User: <question>" / "Joe: <answer>" pairs.\n')
        for q, a in kept:
            f.write(f'User: {q}\n')
            a_lines = a.split('\n')
            f.write(f'Joe: {a_lines[0]}\n')
            for line in a_lines[1:]:
                f.write(f'{line}\n')
            f.write('\n')

    # Stats
    print(f'\n=== {name} ===')
    print(f'Raw input:           {len(raw_pairs)} pairs')
    print(f'Format skipped:      {skipped_format}')
    print(f'Internal dedupe:     -{intra_removed}')
    print(f'Train dedupe:        -{removed_train}')
    print(f'Intra new dedupe:    -{removed_intra}')
    print(f'Output:              {len(kept)} pairs → {unmerged_path}')
    print(f'Ready for review.    Run: python3 merge_unmerged.py --name {name}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare raw data for expert review')
    parser.add_argument('--name', required=True, help='Expert name (emotion, knowledge, etc.)')
    parser.add_argument('--input', required=True, help='Path to raw data file')
    args = parser.parse_args()
    prep(args.name, args.input)
