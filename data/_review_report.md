# Expert training data audit — 2026-10-06

**Scope**: all 10 experts, `data/experts/{name}/{name}_train.txt` + `unmerged_data.txt`.
**Method**: full structural/content scan → manual verification of every drop category → single repair pass → re-scan + block-by-block reconciliation against `git HEAD`.
**Not touched**: any `.npz`, `data/experts.json`, `data/tokenizer.json`, `AGENTS.md`. No training, no push.
**Reconciliation**: every original block is accounted for — 33612 kept, 77 filler, 141 off-topic, 0 unaccounted.

## Per-expert results

| expert | blocks (before → after) | issues found | fixed in place | dropped |
|---|---:|---:|---:|---:|
| tree | 3767 → 3375 | 1932 | 1540 | 392 |
| reptiles | 2075 → 2005 | 289 | 219 | 70 |
| fish | 1693 → 1644 | 241 | 192 | 49 |
| knowledge | 4228 → 4163 | 284 | 219 | 65 |
| greeting | 3081 → 2836 | 959 | 714 | 245 |
| emotion | 4812 → 3988 | 3066 | 2242 | 824 |
| coding | 3531 → 3412 | 683 | 564 | 119 |
| python | 4001 → 3956 | 480 | 435 | 45 |
| cot | 2661 → 2644 | 80 | 63 | 17 |
| horse | 3981 → 3901 | 268 | 188 | 80 |
| **total** | **33830 → 31924** | **8282** | **6376** | **1906** |

"Fixed in place" = line-level repairs that keep the block. "Dropped" = block removed (duplicate / filler / off-topic).

## Breakdown by category

| expert | id suffix `(8487)` | double prefix | dup pair collapsed | exact dup block | filler stub | off-topic | orphan line |
|---|---:|---:|---:|---:|---:|---:|---:|
| tree | 1026 | 0 | 513 | 383 | 9 | 0 | 1 |
| reptiles | 146 | 0 | 73 | 60 | 10 | 0 | 0 |
| fish | 128 | 0 | 64 | 49 | 0 | 0 | 0 |
| knowledge | 146 | 0 | 73 | 55 | 10 | 0 | 0 |
| greeting | 476 | 0 | 238 | 234 | 11 | 0 | 0 |
| emotion | 1456 | 58 | 728 | 708 | 10 | 106 | 0 |
| coding | 376 | 0 | 188 | 114 | 5 | 0 | 0 |
| python | 290 | 0 | 145 | 40 | 5 | 0 | 0 |
| cot | 42 | 0 | 21 | 7 | 10 | 0 | 0 |
| horse | 114 | 17 | 57 | 38 | 7 | 35 | 0 |
| **total** | **4200** | **75** | **2100** | **1688** | **77** | **141** | **1** |

- **id suffix**: `User: what is a conifer (8487)` — trailing ` (digits)` counted duplicates. Stripped (`fix_data.py`/`qa_screen.py`-style `re.sub(r' \(\d{1,6}\)$', …)`); all 4200 occurrences were on `User:` lines only.
- **double prefix**: `User: User:` / `Joe: Joe:` — 29 emotion blocks (coding content, 58 lines) + 17 horse blocks (17 `Joe:` lines). Prefix stripped.
- **dup pair collapsed**: 4-line blocks that were `Q/A` repeated verbatim (`Q,A,Q,A` → `Q,A`); 100% had identical answers.
- **exact dup block**: after the two repairs above, blocks that became byte-identical to an earlier block (mostly id-suffix variants). Kept the first occurrence.
- **filler stub**: one-line non-answers of the form `Here is some X information relevant to the Y expert.` / `…a fascinating topic that covers many aspects of…` (65–125 chars, no information). 77 blocks — every one reviewed before dropping.
- **off-topic**: see next section.
- **orphan line**: `tree` block ~1083 had `Joe` following `Joe` (second answer about pruning/borers belonged to another question) — orphan line dropped.

## Off-topic content (141 blocks)

Contamination runs found by cross-expert first-question matching + keyword scoring, boundaries verified block by block:

| where | blocks | content | action |
|---|---:|---|---|
| emotion 2441–2469 | 29 | coding/python Q&A (`User: User:` prefixed) | dropped; 9 not present in `coding_train` → **quarantined to `coding/unmerged_data.txt`** |
| emotion 2507–2550 | 44 | horse Q&A (hooves, breeds, stables…) | dropped; 34 absent from `horse_train` → **quarantined to `horse/unmerged_data.txt`** |
| emotion 2551–2583 | 33 | fish Q&A | dropped (all 33 already in `fish_train`) |
| horse 2333–2367 | 35 | fish Q&A | dropped; 33 already in `fish_train`, 2 (`roe`, `caviar`) → **quarantined to `fish/unmerged_data.txt`** |

Quarantined blocks were checked against the target's train *and* its existing unmerged pairs first, so no duplicates were introduced. Each target file got a `# Note: quarantined from …` header line (comment lines are skipped by `merge_unmerged.py`).

## unmerged_data.txt fixes

| file | before → after | change |
|---|---:|---|
| emotion | 33 → 0 blocks | all 33 pairs are fish content, already in `fish_train` (file kept, comment header updated) |
| cot | 5 → 0 blocks | 5 templated tree/eco fillers, also removed from `cot_train` |
| coding | 43 → 52 blocks | +9 quarantined |
| horse | 104 → 138 blocks | +34 quarantined; missing final newline fixed |
| fish | 4 → 6 blocks | +2 quarantined; `#` header separated from first pair |
| greeting, knowledge | — | `#` header separated from first pair (it was glued into the first block) |
| tree, reptiles, python | — | unchanged (already valid) |

All 10 unmerged files now parse clean: no id suffixes, strict `User:`/`Joe:` alternation, trailing newline present.

## Left alone (deliberate)

- **`knowledge` fish section (~36 blocks)** + `knowledge/unmerged_data.txt`'s 2 fish pairs — `knowledge` is the generalist expert; its fish pairs are already in `knowledge_train`, so a merge is a no-op.
- **`tree` botany + data-structure sections coexist** (36 data-structure blocks) — consistent across the file by design.
- **Near-duplicate first questions with different answers** (e.g. knowledge has two "Ring of Fire" entries) — both answers are valid, kept.
- **Fuzzy near-dup candidates** (`tree/unmerged`: "What is a binary search tree?" vs "What is a binary tree?") — the 0.82 similarity threshold false-positives on short "What is a X?" questions; nothing was auto-dropped on fuzzy grounds.
- **Cross-expert first-question overlaps left alone**: coding↔python 34, tree↔knowledge 5, tree↔coding 3 (B-tree), greeting↔emotion 3 — genuinely shared questions with expert-appropriate answers.
- **Contextless first questions** (knowledge: "What happened to it?", "Why is it important?"; ~5 in emotion) — ambiguous without the preceding turn; flagged for human review, not deleted.
- **`python_train` `\u00e9`** — legitimate (`json.dumps` escape explanation), not mojibake.
- Encoding scan otherwise clean: no NUL bytes, invalid UTF-8, CRLF, or trailing whitespace anywhere.

## Test files

**No `{name}_test.txt` exists for any expert** (checked `data/` recursively), so the train/test consistency step had nothing to compare. See follow-ups.

## Follow-ups

1. **Pick up the quarantined blocks**: `python3 merge_unmerged.py coding horse fish` — will add the 45 blocks (9/34/2). `emotion` and `cot` unmerged are now empty; the other seven are unchanged no-ops (already merged).
2. **Create held-out test splits** — `training/train_expert.py` trains on 100% of each train file with no validation split, and no `{name}_test.txt` exists. Carving ~5% per expert would give real loss/accuracy signal.
3. **Fix the ingest, not the symptom** — 4200 id suffixes + 2100 doubled pairs + 1688 post-repair duplicates all point at generators writing duplicates with ` (N)` counters instead of deduping first. Worth fixing in whatever produces the next ingest rather than re-scrubbing.
4. **Re-run this audit after the next data ingest** (this pass only fixes what was committed at 5a5bce5).
5. Optional: move `knowledge`'s fish section into `fish` if the router ever gets sharp enough to own topic split; leaving it is harmless today.
