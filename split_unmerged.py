#!/usr/bin/env python3
"""Split each expert's unmerged_data.txt into ~200-block chunks in data/_gen/prompts_mt/."""
import os

CHUNKS_DIR = "data/_gen/prompts_mt"
os.makedirs(CHUNKS_DIR, exist_ok=True)

experts = ["greeting", "emotion", "knowledge", "coding", "cot", "python", "horse", "fish", "reptiles", "tree"]

def load_blocks(path):
    with open(path) as f:
        content = f.read()
    blocks = [b for b in content.strip().split('\n\n') if b.strip()]
    return blocks

for expert in experts:
    data_path = f"data/experts/{expert}/unmerged_data.txt"
    if not os.path.exists(data_path):
        print(f"Skipping {expert}: no unmerged_data.txt")
        continue
    blocks = load_blocks(data_path)
    print(f"{expert}: {len(blocks)} blocks")
    # Create chunk files, ~200 blocks each
    for i in range(0, len(blocks), 200):
        chunk = blocks[i:i+200]
        chunk_num = i // 200
        chunk_file = f"{expert}_{chunk_num:03d}.txt"
        chunk_path = os.path.join(CHUNKS_DIR, chunk_file)
        with open(chunk_path, 'w') as f:
            for b in chunk:
                f.write(b + '\n\n')
        print(f"Wrote {chunk_file} with {len(chunk)} blocks")
