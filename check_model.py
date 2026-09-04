import sys
sys.path.insert(0, 'training')
from model import JoeBrain
from tokenizer import Tokenizer
m = JoeBrain.load(sys.argv[1])
t = Tokenizer()
t.load('data/tokenizer.json')
print(m.generate(t, '\nHi\n', max_new=80, temperature=0.8))
