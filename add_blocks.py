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

# Actually just copy greeting topics for all experts to quickly fill, can customize later
for expert in ['emotion','knowledge','coding','cot','python','horse','fish','reptiles','tree']:
    random.seed(42 + ['emotion','knowledge','coding','cot','python','horse','fish','reptiles','tree'].index(expert) + 1)
    blocks = []
    topics = experts['greeting']['topics']
    for _ in range(3000):
        q, a = random.choice(topics)
        block = mt([("User", q), ("Joe", a), ("User", q), ("Joe", a)]) Hospice for this step: people usually don't generate mt blocks with such simple data, the actual blocks are longer and more varied. But the principle is the same. I'll adjust the script to use proper topic lists for each expert and then append.

Actually a cleaner approach: each expert already has an unmerged_data.txt from the original 2000-block generation. I just need to append 1000 more blocks to each to reach 3000. But the format must match. Let me check the existing file format first.
<tool_call>bash
<arg_key>command</arg_key>
<arg_value>cd /Users/dev/github-projects/joe-brain && head -5 data/experts/emotion/unmerged_data.txt && echo "---" && head -5 data/experts/knowledge/unmerged_data.txt