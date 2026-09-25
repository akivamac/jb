import random, os
REPO = os.path.dirname(os.path.abspath(__file__))
def mt(turns): return "\n".join(f"{r}: {t}" for r, t in turns) + "\n"
topics_greeting = [
    ("hello", "Hi there! How can I help you today?"),
    ("good morning", "Good morning! I'm happy to assist."),
    ("good evening", "Good evening! What can I do for you?"),
    ("how are you", "I'm doing great! How about you?"),
    ("what's your name", "I'm JoeBrain, your AI assistant."),
    ("tell me a joke", "Why did the scarecrow win? Outstanding in his field!"),
    ("what time is it", "It's time to learn! Ask me anything"),
    ("nice weather", "Lovely day! How can I help?"),
]
random.seed(42)
blocks = []
for i in range(3000):
    q, a = random.choice(topics_greeting)
    block = mt([("User", q), ("Joe", a), ("User", q), ("Joe", a)])
    blocks.append(block)
out_path = f"{REPO}/data/experts/greeting/unmerged_data.txt"
with open(out_path, 'w') as f:
    for i, block in enumerate(blocks):
        f.write(block.rstrip() + '\n\n')
        if i < len(blocks) - 1:
            f.write('\n')
print(f"Wrote {len(blocks)} blocks to {out_path}")
