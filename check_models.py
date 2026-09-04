#!/usr/bin/env python3
"""Test all expert models with sample prompts."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training'))
from tokenizer import Tokenizer
from model import JoeBrain

BASE = os.path.join(os.path.dirname(__file__), 'data')
TOK_PATH = os.path.join(BASE, 'tokenizer.json')

tok = Tokenizer()
tok.load(TOK_PATH)

experts = ['greeting', 'emotion', 'knowledge', 'coding', 'cot', 'python', 'horse', 'fish', 'reptiles']

prompts = {
    'greeting': 'User: hey how are you\nJoe:',
    'emotion': 'User: I feel so anxious today\nJoe:',
    'knowledge': 'User: what causes earthquakes\nJoe:',
    'coding': 'User: what is a for loop\nJoe:',
    'cot': 'User: if I have 5 apples and give 3 away, how many do I have\nJoe:',
    'python': 'User: how do I reverse a list in python\nJoe:',
    'horse': 'User: how do you take care of a horse\nJoe:',
    'fish': 'User: what is a betta fish\nJoe:',
    'reptiles': 'User: what is a bearded dragon\nJoe:',
}

for name in experts:
    path = os.path.join(BASE, 'experts', name, f'{name}.npz')
    if not os.path.exists(path):
        print(f'\n=== {name}: NO CHECKPOINT ===')
        continue
    
    model = JoeBrain.load(path)
    prompt = prompts[name]
    
    # Generate at different temperatures
    for temp in [0.7, 1.0]:
        out = model.generate(tok, prompt, max_new=150, temperature=temp)
        # Extract just the Joe response
        response = out[len(prompt):]
        # Cut at next User: or newline
        for stop in ['\nUser:', '\n\n']:
            if stop in response:
                response = response[:response.index(stop)]
        response = response.strip()
        
        print(f'\n=== {name} (temp={temp}) ===')
        print(f'Prompt: {prompt.strip()}')
        print(f'Response: {response[:200]}')
