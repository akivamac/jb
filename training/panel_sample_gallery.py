#!/usr/bin/env python3
"""
panel_sample_gallery.py — extract SAMPLE blocks from an expert's training.log.

Parses `# [ts] SAMPLE step N:` blocks (newer logs prefix each sample line with
`# `, older logs write bare text lines) using the same log conventions as
training_server.py (shared-flock read; sample blocks stop at data lines or
timestamped `# [...]` headers). Outputs a JSON array of {step, ts, text}.

Usage:
  python3 training/panel_sample_gallery.py --name coding
  python3 training/panel_sample_gallery.py --name coding --limit 5 --pretty
"""

import argparse
import fcntl
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(REPO, 'data')

SAMPLE_RE = re.compile(r'#\s*\[([^\]]+)\]\s*SAMPLE\s+step\s+(\d+):')


def parse_samples(log_path, limit=None):
    """Return list of {'step': int, 'ts': str, 'text': str} most-recent-first."""
    if not os.path.exists(log_path):
        return []
    samples = []
    with open(log_path, 'rb') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        try:
            for raw in f:
                line = raw.decode('utf-8', errors='replace').rstrip('\n')
                s = line.strip()
                if not s:
                    continue
                m = SAMPLE_RE.search(s)
                if m and s.startswith('# ['):
                    samples.append({'step': int(m.group(2)), 'ts': m.group(1).strip(), 'lines': []})
                    continue
                if not samples:
                    continue
                cur = samples[-1]
                if s.startswith('# ['):
                    continue
                if s.startswith('# TAG ') or s.startswith('# step,loss'):
                    continue
                if not s.startswith('#') and ',' in s and s.split(',', 1)[0].strip().isdigit():
                    continue
                if s.startswith('# '):
                    cur['lines'].append(s[2:])
                elif not s.startswith('#'):
                    cur['lines'].append(s)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    out = [{'step': sm['step'], 'ts': sm['ts'], 'text': '\n'.join(sm['lines'])} for sm in samples]
    if limit is not None and limit > 0:
        out = out[-limit:]
    return list(reversed(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name', required=True, help='Expert name (e.g. coding)')
    ap.add_argument('--limit', type=int, default=None, help='Return at most N most-recent samples (default: all)')
    ap.add_argument('--pretty', action='store_true', help='Pretty-print JSON')
    args = ap.parse_args()

    log_path = os.path.join(BASE, 'experts', args.name, 'training.log')
    samples = parse_samples(log_path, limit=args.limit)
    if args.pretty:
        json.dump(samples, sys.stdout, indent=2)
        sys.stdout.write('\n')
    else:
        json.dump(samples, sys.stdout)
        sys.stdout.write('\n')


if __name__ == '__main__':
    main()