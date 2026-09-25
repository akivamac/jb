#!/usr/bin/env python3
"""Test all expert models with sample prompts (doubled-data era)."""
import sys, os, json, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training'))
from tokenizer import Tokenizer
from model import JoeBrain

BASE = os.path.join(os.path.dirname(__file__), 'data')
TOK_PATH = os.path.join(BASE, 'tokenizer.json')

tok = Tokenizer()
tok.load(TOK_PATH)

experts = ['greeting', 'emotion', 'knowledge', 'coding', 'cot', 'python',
           'horse', 'fish', 'reptiles', 'tree']

prompts = {
    'greeting':  'User: hey how is your day going\nJoe:',
    'emotion':   'User: I feel nervous\nJoe:',
    'knowledge': 'User: why does the moon affect the tides\nJoe:',
    'coding':    'User: what is the select statement in Go\nJoe:',
    'cot':       'User: If 2/3 of a class of 30 are girls, how many boys are there?\nJoe:',
    'python':    'User: how do i get the match positions from re.findall\nJoe:',
    'horse':     'User: what is a close contact saddle used for\nJoe:',
    'fish':      'User: why do fish lay so many eggs\nJoe:',
    'reptiles':  'User: how do you set up a leopard gecko tank\nJoe:',
    'tree':      'User: what is the tallest tree species\nJoe:',
}

for name in experts:
    path = os.path.join(BASE, 'experts', name, f'{name}.npz')
    if not os.path.exists(path):
        print(f'\n=== {name}: NO CHECKPOINT ===')
        continue

    model = JoeBrain.load(path)
    prompt = prompts[name]

    for temp in [0.7, 1.0]:
        random.seed(42)
        sys.modules['random'].seed(42)
        out = model.generate(tok, prompt, max_new=150, temperature=temp)
        response = out[len(prompt):]
        for stop in ['\nUser:', '\n\n']:
            if stop in response:
                response = response[:response.index(stop)]
        response = response.strip()

        print(f'\n=== {name} (temp={temp}) ===')
        print(f'Prompt: {prompt.strip()}')
        print(f'Response: {response[:220]}')