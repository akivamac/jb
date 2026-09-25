import random, os

def mt(turns):
    return "\n".join(f"{r}: {t}" for r, t in turns) + "\n"

experts = {
    'greeting': {'seed': 42, 'topics': [
        ("hello", "Hi there! Nice to meet you."),
        ("good morning", "Good morning! How can I help?"),
        ("good evening", "Good evening! What can I do for you?"),
        ("how are you", "I am doing great! Thanks for asking."),
        ("what is your name", "I am JoeBrain, your AI assistant."),
        ("what can you do", "I can help with many things!"),
        ("tell me a joke", "Why did the scarecrow win? Outstanding in his field!"),
        ("how is the weather", "Lovely day! Ask me anything."),
        ("thank you", "You are welcome!"),
        ("goodbye", "Take care! See you later."),
    ]}
}

for expert in ['greeting', 'emotion', 'knowledge', 'coding', 'cot', 'python', 'horse', 'fish', 'reptiles', 'tree']:
    random.seed(42 + ['greeting','emotion','knowledge','coding','cot','python','horse','fish','reptiles','tree'].index(expert))
    blocks = []
    for _ in range(3000):
        q, a = random.choice(experts[expert]['topics'])
        block = mt([("User", q), ("Joe", a), ("User", q), ("Joe", a)])
        blocks.append(block)
    out_path = os.path.join("data", "experts", expert, "unmerged_data.txt")
    with open(out_path, "w") as f:
        for block in blocks:
            f.write(block.rstrip() + "\n\n")
    print(f"Generated {len(blocks)} blocks for {expert}")
