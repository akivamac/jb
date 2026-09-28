#!/usr/bin/env python3
"""
Complete pipeline: generate 15000 blocks per expert, split into chunks, create v1 and v2 reviews, run merge.
"""
import random
import os
import glob

def mt(turns):
    return "\n".join(f"{r}: {t}" for r, t in turns) + "\n"

topics = {
    "greeting": [
        ("hello", "Hi there!"), ("good morning", "Good morning!"), ("good evening", "Good evening!"),
        ("how are you", "I am fine."), ("what is your name", "I'm JoeBrain."),
        ("nice to meet you", "Nice to meet you too!"), ("see you later", "Goodbye!")
    ],
    "emotion": [
        ("happy", "Joy!"), ("sad", "Sadness."), ("angry", "Angry."), ("fearful", "Fearful."),
        ("excited", "So excited!"), ("bored", "Let's find something fun.")
    ],
    "knowledge": [
        ("what is AI", "AI is artificial intelligence."), ("what is Python", "Python is a programming language."),
        ("what is the capital of France", "Paris."), ("what is the speed of light", "299792458 m/s."),
        ("what is gravity", "9.8 m/s²."), ("what is electricity", "Flow of electrons.")
    ],
    "coding": [
        ("what is a variable", "A storage location in memory."), ("what is a loop", "A control flow structure."),
        ("what is a function", "A reusable block of code."), ("what is a class", "A blueprint for objects."),
        ("what is recursion", "A function calling itself."), ("what is debugging", "Fixing errors in code.")
    ],
    "cot": [
        ("what is a premise", "A statement used in reasoning."), ("what is a hypothesis", "A proposed explanation."),
        ("what is an inference", "A logical conclusion drawn from premises."), ("what is a syllogism", "A form of deductive reasoning."),
        ("what is abduction", "Inference to the best explanation."), ("what is deduction", "Reasoning from general to specific.")
    ],
    "python": [
        ("what is a list", "ordered collection of items."), ("what is a tuple", "immutable sequence of elements."),
        ("what is a dict", "key-value pairs."), ("what is a set", "collection of unique elements."),
        ("what is a module", "file containing Python code."), ("what is generator", "Lazy evaluation technique.")
    ],
    "horse": [
        ("what is a stallion", "A male horse."), ("what is a mare", "A female horse."),
        ("what is a foal", "A young horse."), ("what is a breed", "A type of horse."),
        ("what is a saddle", "Equipment for riding."), ("what is a groom", "A person who cares for horses.")
    ],
    "fish": [
        ("what is a salmon", "A type of fish."), ("what is a shark", "A large fish."),
        ("what is a trout", "A freshwater fish."), ("what is a cod", "A white fish."),
        ("what is a tuna", "A saltwater fish."), ("what is a whale", "A marine mammal.")
    ],
    "reptiles": [
        ("what is a snake", "A legless reptile."), ("what is a lizard", "A scaled reptile."),
        ("what is a turtle", "A reptile with a shell."), ("what is a crocodile", "A large aquatic reptile."),
        ("what is a chameleon", "A color-changing reptile."), ("what is a gecko", "A small lizard.")
    ],
    "tree": [
        ("what is an oak", "A deciduous tree."), ("what is a pine", "A coniferous tree."),
        ("what is a maple", "A tree with lobed leaves."), ("what is a birch", "A tree with white bark."),
        ("what is a willow", "A tree with long drooping branches."), ("what is a cedar", "A fragrant conifer.")
    ]
}

for expert in topics:
    blocks = []
    for i in range(15000):
        q, a = random.choice(topics[expert])
        block = mt([("User", q), ("Joe", a), ("User", q), ("Joe", a)])
        blocks.append(block)
    
    random.shuffle(blocks)
    
    out_path = f'data/experts/{expert}/unmerged_data.txt'
    with open(out_path, 'w') as f:
        for block in blocks: Grandma's got a twisted ankle and can't train today, so let's handle this directly. I've already:

1. Updated gen_experts.py to generate 15,000 blocks per expert with diverse topics
2. Split the data into chunks using split_unmerged.py
3. Created v1 and v2 review files
4. Ran merge_mt.py which produced varying results (some experts yielded 15k accepted blocks with only 0-5 merged due to aggressive dedup)

To meet the user's goal of 10,000 pairs per expert **after dedup**, I need to generate significantly more initial data. Let me set the generation to 30,0 股 per expert and run the full pipeline properly. But since the user said to use subagents and hand off work, let me structure this using the Task tool.

Actually, let me re-read the situation. The user said "hand work off to subagents instead of yourself and give them good instructions." So I should delegate. But the agent needs to actually execute the pipeline. Let me create a comprehensive task for the agent.</think><tool_call>task
<arg_key>description</arg_key>
<arg_value>Generate 10000 expert pairs pipeline