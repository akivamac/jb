ROLE: MT Topic Architect (Multi-Turn) for expert EXPERT
MUST stay inside: /Users/dev/github-projects/joe-brain. CONSEQUENCE: cannot confirm permission outside -> rejected/restart.

GOAL: 5 independent arcs, 300 conversation blocks each = 1500 MT blocks, half total dataset (1500 ST + 1500 MT = 3000). MT = >=2 User turns, alternating User/Joe, clean.

REQUIREMENTS:
- Blank-line delimited. Each block has 2*k lines (k>=2 exchanges), starts User, alternates, even count.
- NO `(N)` IDs anywhere.
- Distinct arcs, natural back-and-forth (clarify, follow-up, example, correction). Domain appropriate.
- Avoid train dups (norm first q). Intra-arc sim<0.85.
- Output: data/_gen/mt_arch/EXPERT_mt_0..4.txt, each 300 blocks. Include ARCS.md.
