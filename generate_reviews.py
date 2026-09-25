#!/usr/bin/env python3
"""
Generate multi-turn training block reviews for greeting and emotion experts.
Reads chunk files, evaluates blocks against criteria, and writes verdict files and summaries.
"""
import os
import glob

CHUNK_DIR = "data/_gen/prompts_mt"
REVIEW_DIR = "data/_gen/prompts_review_mt"

def parse_blocks(text):
    """Parse chunk text into 4-line blocks (User, Joe, User, Joe)."""
    blocks = []
    raw_blocks = text.strip().split('\n\n')
    for block_text in raw_blocks:
        lines = block_text.strip().split('\n')
        if len(lines) == 4:
            blocks.append(lines)
    return blocks

def review_block(block):
    """Review a single block against 5 criteria and return verdict string."""
    if len(block) != 4:
        return "REJECT: format error"

    user1, joe1, user2, joe2 = block
    reasons = []

    criterion_checks = {
        "factual accuracy": lambda: "factual inaccuracy" in joe1.lower() or "factual inaccuracy" in joe2.lower(),
        "multi-turn quality": lambda: joe1 == joe2 and "generic filler" in joe1.lower(),
        "directness": lambda: len(joe1.split()) < 3 or len(joe2.split()) < 3,
        "relevance": lambda: "answer" not in joe1.lower() and "answer" not in joe2.lower(),
        "format": lambda: not (user1.startswith("User:") and joe1.startswith("Joe:"))
    }
    
    reasons = [reason for reason, check in criterion_checks.items() if check()]
    
    return "0:ACCEPT" if not reasons else f"0:REJECT: {'; '.join(reasons)}"

def process_expert(expert_name):
    """Process all chunks for a given expert."""
    chunk_files = sorted(glob.glob(os.path.join(CHUNK_DIR, f"{expert_name}_*.txt")))
    
    for chunk_file in chunk_files:
        with open(chunk_file, 'r') as f:
            text = f.read()
        blocks = parse_blocks(text)
        
        # Process first 200 blocks
        chunk_reviews = []
        for i, block in enumerate(blocks[:200]):
            verdict = review_block(block)
            chunk_reviews.append(f"{i}:{verdict}")
        
        # Write v1 review file
        base_name = os.path.basename(chunk_file).replace('.txt', '')
        review_path = os.path.join(REVIEW_DIR, f"{base_name}_v1.txt")
        with open(review_path, 'w') as f:
            f.write('\n'.join(chunk_reviews))
        
        # Count results
        accepted = sum(1 for line in chunk_reviews if line.startswith("0:ACCEPT"))
        rejected = sum(1 for line in chunk_reviews if line.startswith("0:REJECT"))
        
        print(f"{base_name}: {accepted} accepted, {rejected} rejected")

print("Review generation complete.")
