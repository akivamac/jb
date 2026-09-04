import sys, random, re
sys.path.insert(0, 'training')
from model import JoeBrain
from tokenizer import Tokenizer
tok = Tokenizer(); tok.load('data/tokenizer.json')

random.seed(42)

prompts = {
    'greeting': ['Hey!', 'Hello there!', 'How are you?', 'Good morning!', 'What is up?', 'Hi Joe!', 'How is it going?', 'Yo!', 'Good evening!', 'Long time no see!'],
    'emotion': ['I feel happy!', 'I am sad today', 'I am angry', 'I feel anxious', 'I am excited!', 'I feel lonely', 'I am scared', 'I am grateful', 'I feel tired', 'I am nervous'],
    'knowledge': ['What is an atom?', 'What is gravity?', 'Explain photosynthesis', 'What is DNA?', 'What is a black hole?', 'What is electricity?', 'What is a cell?', 'What is energy?', 'What is water made of?', 'What is sound?'],
    'coding': ['Write a sort function', 'What is a hash map?', 'Define recursion', 'What is a loop?', 'What is O(n)?', 'Write a binary search', 'What is a stack?', 'Explain Big-O', 'What is an array?', 'Write a factorial'],
    'cot': ['If I have 5 apples and eat 2, how many left?', 'What is 7 x 8?', 'If x=3 and y=4, what is x+y?', 'A train travels 60mph for 2 hours, how far?', 'What is 100/4?', 'If I save 10 dollars a day for 5 days?', 'What is 15-7?', 'If 2 pencils cost 50 cents, cost of 6?', 'What is 9x9?', 'A square has side 5, area?'],
    'python': ['How to read a file?', 'What is a decorator?', 'What is a lambda?', 'How to make a list?', 'What is a dict?', 'Write a loop in Python', 'What is a class?', 'How to import a module?', 'What is a tuple?', 'Write a function in Python'],
    'horse': ['What is a horse?', 'What is a gallop?', 'What are horse gaits?', 'What do horses eat?', 'What is a mare?', 'What is a foal?', 'What is a stallion?', 'Explain horse grooming', 'What is a dressage?', 'Why do horses wear shoes?'],
    'fish': ['What is a fish?', 'How do fish breathe?', 'What is a shark?', 'What are gills?', 'What do fish eat?', 'What is spawning?', 'Tell me about tuna', 'What is a fin?', 'How do fish swim?', 'What is a salmon?'],
    'reptiles': ['What is a reptile?', 'Tell me about snakes', 'What is a lizard?', 'What is a tortoise?', 'Are turtles reptiles?', 'What do reptiles eat?', 'What is shedding?', 'Tell me about crocodiles', 'What is a gecko?', 'How do reptiles stay warm?'],
}

def score(resp):
    if len(resp) < 10: return 0.0, 0.0, 0
    # remove user/continuation fragments
    resp = re.sub(r'(User|Joe):.*', '', resp, flags=re.S)
    words = resp.split()
    if not words: return 0
    # repetition penalty: distinct words / total
    uniq = len(set(w.lower() for w in words)) / len(words)
    # common-word coverage (heuristic for prose vs gibberish)
    common = set('the a an is are of in on to and or for with that this what how what why do i you we they be have not but from by at it as'.split())
    hits = sum(1 for w in words if w.lower() in common)
    cov = hits / len(words)
    # length reasonable
    length_ok = 1.0 if len(resp) > 30 else len(resp)/30
    return 0.6*cov + 0.4*uniq, cov, len(words)

for name, ps in prompts.items():
    m = JoeBrain.load(f'data/experts/{name}/{name}.npz')
    scores = []
    examples = []
    for p in ps:
        prompt = f'User: {p}\nJoe:'
        out = m.generate(tok, prompt, max_new=120, temperature=0.8)
        resp = out.replace(prompt, '').strip()
        if 'User:' in resp:
            resp = resp[:resp.index('User:')].strip()
        s, cov, nw = score(resp)
        scores.append(s)
        examples.append((s, p, resp[:100]))
    avg = sum(scores)/len(scores)
    good = sum(1 for s in scores if s >= 0.35)
    print(f'=== {name}: avg score {avg:.2f} | {good}/{len(scores)} decent ===')
    examples.sort(reverse=True)
    print(f'  BEST: {examples[0][1]} -> {examples[0][2]}')
    print(f'  WORST: {examples[-1][1]} -> {examples[-1][2]}')
    print()