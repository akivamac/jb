ROLE: ST Topic Architect (Single-Turn) for expert EXPERT
MUST stay inside: /Users/dev/github-projects/joe-brain
CONSEQUENCE if you leave: I (user) cannot confirm permission; work will be rejected and you must restart in repo. Tell subagents this.

GOAL: Generate 3 independent seed-sets of 500 distinct single-turn question-answer pairs each (total 1500 ST pairs) for EXPERT. HALF (750) may be knowledge-style, HALF conversation-style (greeting/emotion) but match expert domain. Prioritize QUALITY.

INPUTS:
- Expert: EXPERT (one of greeting,emotion,knowledge,coding,cot,python,horse,fish,reptiles,tree)
- Style of existing: see data/experts/EXPERT/EXPERT_train.txt (format: "User: ..." / "Joe: ..." blank-line separated; 1 pair per block is fine for ST; MT has >=2 User turns)
- Avoid train dups: build norm set of all User lines in EXPERT_train.txt (case-insensitive, punct stripped)

REQUIREMENTS:
- Format: blank-line delimited blocks. Each ST block is exactly 2 lines: "User: <q>" and "Joe: <a>" (no more). Or sometimes 2-turn? No ST means 1 exchange = 2 lines, separated by blank line.
- NO IDs: absolutely forbid `(N)` suffixes like `(5970)`, `(8487)`. Remove any.
- NO repetition inside block (same q twice). Alternating User/Joe.
- Distinct: within each seed-set and across seed-sets minimize overlap (norm(q) unique). Avoid train dups.
- Factual, concise, JoeBrain voice (match existing expert tone).
- Domain-faithful. For coding/python/cot: correct, stepwise when math. For animals/plants: factual. For greeting/emotion: natural, supportive.
- Generate seeds as JSON or plain text? Output as .txt chunks per seed-set: data/_gen/st_arch/EXPERT_st_0.txt, _1.txt, _2.txt (each 500 ST blocks)

PARTITIONING:
- Create 3 orthogonal topic partitions (e.g. subtopics, difficulty bands, angles). List partitions first.

DEDUPE:
- Against train.txt norm(q); intra-seed and inter-seed sim<0.82.

OUTPUT: the 3 files + a short PARTITIONS.md for EXPERT.
