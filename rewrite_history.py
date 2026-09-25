#!/usr/bin/env python3
"""Thin historical .npz blobs:
- in each commit, keep the npz file-change only if that version is one of
  "every 15th from the beginning" or one of the 10 most recent;
- empty the content of every other npz blob so the export stream stays tiny;
- final gc --prune=now removes the now-unreferenced old blobs entirely.
Usage: thin_npz.py REPO
"""
import os
import subprocess
import sys
from git_filter_repo import RepoFilter, FilteringOptions

REPO = os.path.realpath(sys.argv[1])
EVERY = 50
RECENT = 3

def git(*args):
    return subprocess.check_output(['git', '-C', REPO] + list(args)).rstrip(b'\n')

paths = [l.decode() for l in
         git('ls-files', 'data/experts/*/*.npz').split(b'\n') if l.endswith(b'.npz')]

# 1) per path: which commits keep the npz file-change?
KEEP_COMMITS = {}
# 2) blob hashes whose content must be preserved
KEEP_BLOBS = set()
# 3) every npz blob hash ever (so we only empty these)
ALL_NPZ_BLOBS = set()

for p in paths:
    commits = list(reversed(git('log', '--format=%H', '--', p).split(b'\n')))
    n = len(commits)
    keep_idx = sorted(set(range(0, n, EVERY)) | set(range(max(0, n - RECENT), n)))
    KEEP_COMMITS[p] = set(commits[i] for i in keep_idx)
    for i in range(n):
        try:
            blob = git('rev-parse', f'{commits[i].decode()}:{p}').strip()
        except subprocess.CalledProcessError:
            continue
        if blob:
            ALL_NPZ_BLOBS.add(blob)
            if i in keep_idx:
                KEEP_BLOBS.add(blob)
    print(f"{p}: {n} versions -> keep {len(keep_idx)} blob-versions", file=sys.stderr)

print(f"Total npz blob-hashes in history: {len(ALL_NPZ_BLOBS)}", file=sys.stderr)
print(f"Keeping {len(KEEP_BLOBS)} npz blobs", file=sys.stderr)

removed_changes = 0
emptied_blobs = 0

def commit_callback(commit, metadata):
    global removed_changes
    if not commit.file_changes:
        return
    kept = []
    for fc in commit.file_changes:
        fn = fc.filename
        fn_s = fn.decode('utf-8', 'replace') if isinstance(fn, bytes) else fn
        if fn_s.startswith('data/experts/') and fn_s.endswith('.npz'):
            keepset = KEEP_COMMITS.get(fn_s)
            if keepset is not None and commit.original_id not in keepset:
                removed_changes += 1
                continue
        kept.append(fc)
    commit.file_changes = kept

def blob_callback(blob, metadata):
    global emptied_blobs
    if blob.original_id in ALL_NPZ_BLOBS and blob.original_id not in KEEP_BLOBS:
        blob.data = b''
        emptied_blobs += 1

os.chdir(REPO)
args = FilteringOptions.parse_args(['--force'])
filter = RepoFilter(args, commit_callback=commit_callback, blob_callback=blob_callback)
filter.run()
print(f"Dropped {removed_changes} non-kept npz file-changes", file=sys.stderr)
print(f"Emptied {emptied_blobs} old npz blobs", file=sys.stderr)