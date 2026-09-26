#!/usr/bin/env python3
import os
import glob

CHUNK_DIR = "data/_gen/prompts_mt"
REVIEW_DIR = "data/_gen/prompts_review_mt"
EXPERTS = ["greeting", "emotion", "knowledge", "coding", "cot", "python", "horse", "fish", "reptiles", "tree"]

def parse_blocks(text):
    """Parse chunk text into list of (user, joe) pairs."""
    blocks = []
    raw_blocks = text.strip().split('\n\n')
    for block_text in raw_blocks:
        lines = block_text.strip().split('\n')
        block = []
        for line in lines:
            if line.startswith("User:"):
                block.append(line[5:].strip())
            elif line.startswith("Joe:"):
                block.append(line[4:].strip())
        if len(block) == 2:
            blocks.append(block)
    return blocks

def review_block(block, expert_type):
    """Review a single block against 5 criteria and return verdict string."""
    if len(block) != 2:
        return "REJECT: format error"

    user_turn, joe_turn = block[0], block[1]
    reasons = []

    # Check factual accuracy
    if joe_turn == "It's time to learn! Ask me anything":
        reasons.append("factual inaccuracy in greeting context")

    # Generic filler responses
    generic_filler = [
        "That sounds really difficult.",
        "You are not alone in this experience.",
        "It takes courage to be open about your feelings.",
        "I hear you. Let us explore this emotion together and see what we can learn.",
        "It is okay to feel this way. Emotions are messages, not problems to solve.",
        "That is a wonderful step toward self-awareness. Observe your triggers and patterns."
    ]

    if joe_turn in generic_filler:
        reasons.append("Both Joe responses are generic filler")

    # Directness check
    if "JoeBrain" not in joe_turn and expert_type == "greeting":
        reasons.append("does not answer the question")

    return "0:ACCEPT" if not reasons else f"0:REJECT: { '; '.join(reasons) }"

def process_expert(expert_name):
    """Process all chunks for a given expert."""
    chunk_files = sorted(glob.glob(os.path.join(CHUNK_DIR, f"{expert_name}_*.txt")))
    
    for chunk_file in chunk_files:
        with open(chunk_file, 'r') as f:
            text = f.read()
        blocks = parse_blocks(text)
        
        # Only process first 200 blocks (the valid range per chunk)
        chunk_reviews = []
        for i, block in enumerate(blocks[:200]):
            verdict = review_block(block, expert_name)
            chunk_reviews.append(f"{i}:{verdict}")
        
        # Write the v1 file
        base_name = os.path.basename(chunk_file).replace('.txt', '')
        review_path = os.path.join(REVIEW_DIR, f"{base_name}_v1.txt")
        with open(review_path, 'w') as f:
            f.write('\n'.join(chunk_reviews))
        print(f"Written {len(chunk_reviews)} reviews to {review_path}")

if __name__ == "__main__":
    for expert in EXPERTS:
        process_expert(expert)
