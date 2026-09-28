#!/usr/bin/env python3
import random
import os

def mt(turns):
    return "\n".join(f"{r}: {t}" for r, t in turns) + "\n"

topics = {
    "greeting": [
        ("hello", "Hi there!"), ("good morning", "Good morning!"), ("good evening", "Good evening!"),
        ("how are you", "I am fine."), ("what is your name", "I'm JoeBrain."),
        ("nice to meet you", "Nice to meet you too!"), ("see you later", "Goodbye!"),
        ("who are you", "I'm JoeBrain, your AI assistant."), ("how old are you", "I'm just a program, so I don't have an age."),
        ("what can you do", "I can help with many tasks!"), ("who created you", "I was created by developers."),
        ("what is your purpose", "To assist and chat with humans."), ("how do you work", "I process language and learn.")
    ],
    "emotion": [
        ("happy", "Joy!"), ("sad", "Sadness."), ("angry", "Angry."), ("fearful", "Fearful."),
        ("excited", "So excited!"), ("bored", "Let's find something fun."),
        ("glad", "I'm so glad!"), ("mad", "I'm mad."), ("confused", "I'm confused."), ("surprised", "Surprise!")
    ],
    "knowledge": [
        ("what is AI", "AI is artificial intelligence."), ("what is Python", "Python is a programming language."),
        ("what is the capital of France", "Paris."), ("what is the speed of light", "299792458 m/s."),
        ("what is gravity", "9.8 m/s²."), ("what is electricity", "Flow of electrons.")
    ],
    "coding": [
        ("what is a variable", "A storage location in memory."), ("what is a loop", "A control flow structure."),
        ("what is a function", "A reusable block of code."), ("what is a class", "A blueprint for objects."),
        ("what is recursion", "A function calling itself."), ("what is debugging", "Fixing errors in code."),
        ("what is an algorithm", "A set of instructions to solve a problem."), ("what is a database", "A structured data store.")
    ],
    "cot": [
        ("what is a premise", "A statement used in reasoning."), ("what is a hypothesis", "A proposed explanation."),
        ("what is an inference", "A logical conclusion drawn from premises."), ("what is a syllogism", "A form of deductive reasoning."),
        ("what is abduction", "Inference to the best explanation."), ("what is deduction", "Reasoning from general to specific."),
        ("what is induction", "Reasoning from specific cases to general rules."), ("what is abductive reasoning", "Inference to the best explanation.")
    ],
    "python": [
        ("what is a list", "ordered collection of items."), ("what is a tuple", "immutable sequence of elements."),
        ("what is a dict", "key-value pairs."), ("what is a set", "collection of unique elements."),
        ("what is a module", "file containing Python code."), ("what is generator", "Lazy evaluation technique."),
        ("what is lambda", "Anonymous function."), ("what is decorator", "Function wrapper."),
        ("what is generator", "Lazy evaluation technique."), ("what is map", "Applies a function to all items.")
    ],
    "horse": [
        ("what is a stallion", "A male horse."), ("what is a mare", "A female horse."),
        ("what is a foal", "A young horse."), ("what is a breed", "A type of horse."),
        ("what is a saddle", "Equipment for riding."), ("what is a groom", "A person who cares for horses."),
        ("what is a trot", "A two-beat gait."), ("what is a canter", "A three-beat gait.")
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
    for i in range(30000):
        q, a = random.choice(topics[expert])
        block = mt([("User", q), ("Joe", a), ("User", q), ("Joe", a)])
        blocks.append(block)

    random.shuffle(blocks)

    out_path = f'data/experts/{expert}/unmerged_data.txt'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        for block in blocks:
            f.write(block.rstrip() + '\n\n')
    print(f"Generated {len(blocks)} blocks for {expert}")
