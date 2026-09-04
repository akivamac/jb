"""Build BPE tokenizer from training data and save it."""
import sys
import time
sys.path.insert(0, 'training')
from tokenizer import Tokenizer

with open('data/train.txt') as f:
    text = f.read()

print(f"Training text: {len(text):,} characters")
start = time.time()
tok = Tokenizer()
tok.build(text, vocab_size=2000)
elapsed = time.time() - start
print(f"Build time: {elapsed:.1f}s")
tok.save('data/tokenizer.json')
print(f"Sample encode: 'Hello, how are you?' -> {tok.encode('Hello, how are you?')}")
print(f"Sample decode: {tok.decode(tok.encode('Hello, how are you?'))}")
