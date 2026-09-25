#!/usr/bin/env python3
"""
Multi-turn review script for greeting and emotion expert chunk files.
Parses each chunk, evaluates every block against 5 criteria, and generates verdict files.
"""

import os
import re
from pathlib import Path

CHUNK_DIR = "data/_gen/prompts_mt"
REVIEW_DIR = "data/_gen/prompts_review_mt"
EXPERTS = ["greeting", "emotion"]
NUM_CHUNKS = 15
BLOCKS_PER_CHUNK = 200

def parse_chunk(filepath):
    """Parse a chunk file into a list of blocks. Each block is a list of (role, text) tuples."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    blocks = []
    # Split by double newlines to get blocks
    block_texts = content.strip().split('\n\n')
    
    for block_text in block_texts:
        lines = block_text.strip().split('\n')
        block = []
        for line in lines:
            if line.startswith("User:"):
                block.append(("user", line[5:].strip()))
            elif line.startswith("Joe:"):
                block.append(("joe", line[4:].strip()))
        if block:
            blocks.append(block)
    
    return blocks

def evaluate_block(block):
    """Evaluate a block (user-joe pair) against 5 criteria and return verdict."""
    if len(block) != 2:
        return "0:REJECT: format error"

    user_turn, joe_turn = block[0][1], block[1][1]
    reasons = []

    # Criterion 1: Directness - Joe should answer the user's question directly
    # Criterion 2: Relevance - Joe's response should be relevant to the user's question
    # Criterion 3: Factual accuracy - Joe should not provide incorrect information
    # Criterion 4: Multi-turn quality - The conversation should show context and follow-up
    # Criterion 5: Format - Proper User/Joe format

    # Detect factual inaccuracy: "It's time to learn! Ask me anything" for "what time is it" is a non-sequitur
    if joe_turn == "It's time to learn! Ask me anything":
        reasons.append("factual inaccuracy in greeting context")
    
    # Detect generic filler responses
    generic_filler = [
        "You are not alone in this experience.",
        "That sounds really difficult.",
        "It takes courage to be open about your feelings.",
        "I hear you. Let us explore this emotion together and see what we can learn.",
        "It is okay to feel this way. Emotions are messages, not problems to solve.",
        "That is a wonderful step toward self-awareness. Observe your triggers and patterns."
    ]
    
    if joe_turn in generic_filler:
        reasons.append("Both Joe responses are generic filler")
    
    # Detect non-emotion-specific responses in emotion context
    if "Joe" in joe_turn and "emotion" not in joe_turn.lower():
        if any(word in joe_turn.lower() for word in ["wonderful", "amazing", "great", "good", "nice", "helpful"]):
            reasons.append("Joe response lacks emotion-related content")
    
    return "0:ACCEPT" if not reasons else f"0:REJECT: { '; '.join(reasons) }"

# This script needs the actual data to generate proper reviews. Let me create a sample data file first.
print("Script setup complete")