"""
Script to fix training data format.
1. Reads existing convos from training/make_conversations.py
2. Formats them as "User: Q\nJoe: A"
3. Appends them to data/train.txt
"""

import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'train.txt')
SCRIPT_PATH = os.path.join(BASE_DIR, 'training', 'make_conversations.py')
CONVOS_PATH = os.path.join(BASE_DIR, 'data', 'conversations.txt')

def extract_convos(script_path):
    """Extracts the convos list from make_conversations.py"""
    if not os.path.exists(script_path):
        print(f"❌ Error: {script_path} not found.")
        return []

    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to find the convos list: convos = [ ... ]
    # This handles multi-line tuples and comments
    match = re.search(r'convos\s*=\s*\[(.*?)\]', content, re.DOTALL)
    
    if not match:
        print("❌ Error: Could not find 'convos = [...] ' in make_conversations.py")
        return []

    list_str = match.group(1)
    
    # Parse the list manually to handle comments and newlines
    # We look for ("...", "...") patterns
    pairs = re.findall(r'\(\s*"([^"]*)"\s*,\s*"([^"]*)"\s*\)', list_str)
    
    return pairs

def format_and_append(pairs, train_path, convos_path):
    """Formats pairs and appends to train.txt"""
    
    if not pairs:
        print("❌ Error: No conversation pairs found.")
        return

    lines = []
    for q, a in pairs:
        # Clean up any internal newlines in the answer if needed, 
        # but usually keep them for CoT.
        # Ensure we don't add extra newlines if the answer already ends with one.
        line = f"User: {q}\nJoe: {a}\n"
        lines.append(line)
    
    new_content = "\n".join(lines)
    
    # Save to a temporary file first to verify
    with open(convos_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ Created {convos_path} with {len(pairs)} pairs.")

    # Append to train.txt
    if os.path.exists(train_path):
        with open(train_path, 'r', encoding='utf-8') as f:
            existing = f.read()
        
        # Ensure there's a newline before appending
        if not existing.endswith('\n'):
            existing += '\n'
        
        with open(train_path, 'w', encoding='utf-8') as f:
            f.write(existing + "\n" + new_content)
        
        print(f"✅ Appended {len(pairs)} pairs to {train_path}")
        print(f"   Total characters in train.txt: {os.path.getsize(train_path):,}")
    else:
        print(f"⚠️ Warning: {train_path} not found. Only created {convos_path}")

if __name__ == "__main__":
    print("🔍 Scanning make_conversations.py for conversation pairs...")
    pairs = extract_convos(SCRIPT_PATH)
    
    if pairs:
        print(f"📝 Found {len(pairs)} pairs. Formatting...")
        format_and_append(pairs, TRAIN_PATH, CONVOS_PATH)
        print("✅ Done! Your training data is now formatted correctly.")
    else:
        print("❌ Failed to extract pairs. Check the script content.")
