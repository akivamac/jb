import random
import os

def mt(turns):
    return "\n".join(f"{r}: {t}" for r, t in turns) + "\n"

topics = {
    "greeting": [("hello", "Hi there!"), ("good morning", "Good morning!"), ("good evening", "Good evening!"), ("how are you", "I am fine."), ("what is your name", "I'm JoeBrain.")],
    "emotion": [("happy", "Joy!"), ("sad", "Sadness."), ("angry", "Angry."), ("fearful", "Fearful.")]
}

for expert in ["greeting", "emotion", "knowledge", "coding", "cot", "python", "horse", "fish", "reptiles", "tree"]:
    random.seed(42 + ["greeting", "emotion", "knowledge", "coding", "cot", "python", "horse", "fish", "reptiles", "tree"].index(expert))
    blocks = []
    for _ in range(3000):
        q, a = random.choice(topics.get(expert, [("hello", "Hi there!")]))
        block = mt([("User", q), ("Joe", a), ("User", q), ("Joe", a)])
        blocks.append(block)
    out_path = f'data/experts/{expert}/unmerged_data.txt'
    mode = 'a' if os.path.exists(out_path) and os.path.getsize(out_path) > 0 else 'w'
    with open(out_path, mode) as f:
        for block in blocks:
            f.write(block.rstrip() + '\n\n')
    print(f"Generated {len(blocks)} blocks for {expert}")
