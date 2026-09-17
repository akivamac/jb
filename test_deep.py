import sys, random, re, json, time
sys.path.insert(0, 'training')
from model import JoeBrain
from tokenizer import Tokenizer
tok = Tokenizer(); tok.load('data/tokenizer.json')

random.seed(42)

EXPERTS = ['greeting','emotion','knowledge','coding','cot','python','horse','fish','reptiles','tree']

SINGLE = {
    'greeting': ['Hey!', 'Hello there!', 'How are you?', 'Good morning!', 'What is up?', 'Hi Joe!', 'How is it going?', 'Yo!', 'Good evening!', 'Long time no see!', 'Whats good?', 'Hey whats happening?'],
    'emotion': ['I feel happy!', 'I am sad today', 'I am angry', 'I feel anxious', 'I am excited!', 'I feel lonely', 'I am scared', 'I am grateful', 'I feel tired', 'I am nervous', 'Why am I sad?', 'How do I cheer up?'],
    'knowledge': ['What is an atom?', 'What is gravity?', 'Explain photosynthesis', 'What is DNA?', 'What is a black hole?', 'What is electricity?', 'What is a cell?', 'What is energy?', 'What is water made of?', 'What is sound?', 'What is a magnet?', 'How do magnets work?'],
    'coding': ['Write a sort function', 'What is a hash map?', 'Define recursion', 'What is a loop?', 'What is O(n)?', 'Write a binary search', 'What is a stack?', 'Explain Big-O', 'What is an array?', 'Write a factorial', 'What is polymorphism?', 'Explain encapsulation'],
    'cot': ['If I have 5 apples and eat 2, how many left?', 'What is 7 x 8?', 'If x=3 and y=4, what is x+y?', 'A train travels 60mph for 2 hours, how far?', 'What is 100/4?', 'If I save 10 dollars a day for 5 days?', 'What is 15-7?', 'If 2 pencils cost 50 cents, cost of 6?', 'What is 9x9?', 'A square has side 5, area?', 'If a rectangle has area 24 and width 4, what is the length?', 'How many seconds in a day?'],
    'python': ['How to read a file?', 'What is a decorator?', 'What is a lambda?', 'How to make a list?', 'What is a dict?', 'Write a loop in Python', 'What is a class?', 'How to import a module?', 'What is a tuple?', 'Write a function in Python', 'What is a generator?', 'Explain threading in Python'],
    'horse': ['What is a horse?', 'What is a gallop?', 'What are horse gaits?', 'What do horses eat?', 'What is a mare?', 'What is a foal?', 'What is a stallion?', 'Explain horse grooming', 'What is a dressage?', 'Why do horses wear shoes?', 'What is a Mustang?', 'How tall can a horse get?'],
    'fish': ['What is a fish?', 'How do fish breathe?', 'What is a shark?', 'What are gills?', 'What do fish eat?', 'What is spawning?', 'Tell me about tuna', 'What is a fin?', 'How do fish swim?', 'What is a salmon?', 'What is coral?', 'How deep can fish swim?'],
    'reptiles': ['What is a reptile?', 'Tell me about snakes', 'What is a lizard?', 'What is a tortoise?', 'Are turtles reptiles?', 'What do reptiles eat?', 'What is shedding?', 'Tell me about crocodiles', 'What is a gecko?', 'How do reptiles stay warm?', 'What is a chameleon?', 'How many snake species exist?'],
    'tree': ['What is a tree?', 'How do trees make food?', 'What is photosynthesis in a tree?', 'Why do leaves change color in autumn?', 'What is the trunk of a tree?', 'How can you tell a trees age?', 'What is a growth ring?', 'Do all trees grow one ring per year?', 'What is the cambium?', 'What is heartwood?', 'Why do trees have bark?', 'What are roots for?'],
}

MULTI_TURN = {
    'greeting': [
        ('User: Hey!', 'Joe:', 'User: How are you?'),
        ('User: Hello there!', 'Joe:', 'User: What is up?'),
        ('User: Good morning!', 'Joe:', 'User: How is it going?'),
        ('User: Hi Joe!', 'Joe:', 'User: Whats good?'),
        ('User: Good evening!', 'Joe:', 'User: Long time no see!'),
    ],
    'emotion': [
        ('User: I feel happy!', 'Joe:', 'User: Why am I happy?'),
        ('User: I am sad today', 'Joe:', 'User: Why am I sad?'),
        ('User: I feel anxious', 'Joe:', 'User: How do I cheer up?'),
        ('User: I am scared', 'Joe:', 'User: What should I do?'),
        ('User: I am grateful', 'Joe:', 'User: Why am I grateful?'),
    ],
    'knowledge': [
        ('User: What is an atom?', 'Joe:', 'User: What is it made of?'),
        ('User: What is gravity?', 'Joe:', 'User: Why does it matter?'),
        ('User: What is DNA?', 'Joe:', 'User: How does it work?'),
        ('User: What is a black hole?', 'Joe:', 'User: What happens inside?'),
        ('User: What is electricity?', 'Joe:', 'User: How is it generated?'),
    ],
    'coding': [
        ('User: Write a sort function', 'Joe:', 'User: What is the time complexity?'),
        ('User: What is a hash map?', 'Joe:', 'User: When should I use it?'),
        ('User: Define recursion', 'Joe:', 'User: What is an example?'),
        ('User: What is a loop?', 'Joe:', 'User: What is a for loop?'),
        ('User: Explain Big-O', 'Joe:', 'User: Why does it matter?'),
    ],
    'cot': [
        ('User: If I have 5 apples and eat 2, how many left?', 'Joe:', 'User: What if I eat 3 instead?'),
        ('User: What is 7 x 8?', 'Joe:', 'User: What is 8 x 7?'),
        ('User: If x=3 and y=4, what is x+y?', 'Joe:', 'User: What if x=5?'),
        ('User: A train travels 60mph for 2 hours, how far?', 'Joe:', 'User: What if it goes 3 hours?'),
        ('User: What is 100/4?', 'Joe:', 'User: What is 100/5?'),
    ],
    'python': [
        ('User: How to read a file?', 'Joe:', 'User: How do I write to a file?'),
        ('User: What is a decorator?', 'Joe:', 'User: When should I use one?'),
        ('User: What is a lambda?', 'Joe:', 'User: What is the difference from def?'),
        ('User: What is a class?', 'Joe:', 'User: What is inheritance?'),
        ('User: Write a loop in Python', 'Joe:', 'User: What is a list comprehension?'),
    ],
    'horse': [
        ('User: What is a horse?', 'Joe:', 'User: What are horse gaits?'),
        ('User: What do horses eat?', 'Joe:', 'User: How much do they eat?'),
        ('User: What is a mare?', 'Joe:', 'User: What is a foal?'),
        ('User: What is a stallion?', 'Joe:', 'User: What is a gelding?'),
        ('User: Explain horse grooming', 'Joe:', 'User: Why is grooming important?'),
    ],
    'fish': [
        ('User: What is a fish?', 'Joe:', 'User: How do fish breathe?'),
        ('User: What are gills?', 'Joe:', 'User: Do all fish have gills?'),
        ('User: What is spawning?', 'Joe:', 'Joe: How many eggs do fish lay?'),
        ('User: What is a shark?', 'Joe:', 'User: What is a great white?'),
        ('User: How do fish swim?', 'Joe:', 'User: How fast can fish swim?'),
    ],
    'reptiles': [
        ('User: What is a reptile?', 'Joe:', 'User: What is a lizard?'),
        ('User: What is shedding?', 'Joe:', 'User: How often do they shed?'),
        ('User: Tell me about snakes', 'Joe:', 'User: What is a python?'),
        ('User: Are turtles reptiles?', 'Joe:', 'User: What is a tortoise?'),
        ('User: How do reptiles stay warm?', 'Joe:', 'User: What is basking?'),
    ],
    'tree': [
        ('User: What is a tree?', 'Joe:', 'User: How do trees make food?'),
        ('User: What is photosynthesis in a tree?', 'Joe:', 'User: Why do leaves change color?'),
        ('User: What is the trunk of a tree?', 'Joe:', 'User: What is bark for?'),
        ('User: How can you tell a trees age?', 'Joe:', 'User: What is a growth ring?'),
        ('User: What are roots for?', 'Joe:', 'User: How deep do roots go?'),
    ],
}

def score(resp):
    if len(resp) < 10: return 0.0, 0.0, 0
    resp = re.sub(r'(User|Joe):.*', '', resp, flags=re.S)
    words = resp.split()
    if not words: return 0
    uniq = len(set(w.lower() for w in words)) / len(words)
    common = set('the a an is are of in on to and or for with that this what how what why do i you we they be have not but from by at it as'.split())
    hits = sum(1 for w in words if w.lower() in common)
    cov = hits / len(words)
    length_ok = 1.0 if len(resp) > 30 else len(resp)/30
    return 0.6*cov + 0.4*uniq, cov, len(words)

def score_multi_turn(pairs):
    """Score each turn in a multi-turn conversation."""
    results = []
    for user_q, joe_a, next_q in pairs:
        # Score the Joe response
        s, cov, nw = score(joe_a)
        results.append({'question': user_q, 'response': joe_a[:80], 'score': s, 'coverage': cov, 'words': nw})
    return results

results = {}
for name in EXPERTS:
    m = JoeBrain.load(f'data/experts/{name}/{name}.npz')
    results[name] = {'single': [], 'multi': []}

    # Single-turn
    for p in SINGLE[name]:
        prompt = f'User: {p}\nJoe:'
        out = m.generate(tok, prompt, max_new=120, temperature=0.8)
        resp = out.replace(prompt, '').strip()
        if 'User:' in resp:
            resp = resp[:resp.index('User:')].strip()
        s, cov, nw = score(resp)
        results[name]['single'].append({'q': p, 'resp': resp[:80], 'score': s, 'cov': cov, 'nw': nw})

    # Multi-turn
    for user1, joe_prefix, user2 in MULTI_TURN[name]:
        prompt1 = f'{user1}\n{joe_prefix}'
        out1 = m.generate(tok, prompt1, max_new=120, temperature=0.8)
        joe1 = out1.replace(prompt1, '').strip()
        if 'User:' in joe1:
            joe1 = joe1[:joe1.index('User:')].strip()
        prompt2 = f'{user1}\n{joe_prefix}{joe1}\n{user2}\nJoe:'
        out2 = m.generate(tok, prompt2, max_new=120, temperature=0.8)
        resp2 = out2.replace(prompt2, '').strip()
        if 'User:' in resp2:
            resp2 = resp2[:resp2.index('User:')].strip()
        s, cov, nw = score(resp2)
        results[name]['multi'].append({'q1': user1, 'a1': joe1[:60], 'q2': user2, 'a2': resp2[:60], 'score': s})

# Report
print("=" * 80)
print("DEEP TEST REPORT — Single-Turn & Multi-Turn")
print("=" * 80)

all_single_scores = []
all_multi_scores = []

for name in EXPERTS:
    single = results[name]['single']
    multi = results[name]['multi']
    single_scores = [r['score'] for r in single]
    multi_scores = [r['score'] for r in multi]
    avg_s = sum(single_scores)/len(single_scores)
    avg_m = sum(multi_scores)/len(multi_scores)
    good_s = sum(1 for s in single_scores if s >= 0.35)
    good_m = sum(1 for s in multi_scores if s >= 0.35)
    all_single_scores.extend(single_scores)
    all_multi_scores.extend(multi_scores)

    print(f"\n{'='*60}")
    print(f"  {name.upper()}")
    print(f"{'='*60}")
    print(f"  Single-turn: avg={avg_s:.2f} | good={good_s}/{len(single)} | worst={min(single_scores):.2f}")
    worst_s = min(single, key=lambda x: x['score'])
    best_s = max(single, key=lambda x: x['score'])
    print(f"    BEST q: {best_s['q']}")
    print(f"      -> {best_s['resp']}")
    print(f"    WORST q: {worst_s['q']}")
    print(f"      -> {worst_s['resp']}")
    print(f"  Multi-turn:  avg={avg_m:.2f} | good={good_m}/{len(multi)}")
    worst_m = min(multi, key=lambda x: x['score'])
    best_m = max(multi, key=lambda x: x['score'])
    print(f"    BEST turn: {best_m['q2']}")
    print(f"      -> {best_m['a2']}")
    print(f"    WORST turn: {worst_m['q2']}")
    print(f"      -> {worst_m['a2']}")

# Summary
avg_all_s = sum(all_single_scores)/len(all_single_scores)
avg_all_m = sum(all_multi_scores)/len(all_multi_scores)
good_all_s = sum(1 for s in all_single_scores if s >= 0.35)
good_all_m = sum(1 for s in all_multi_scores if s >= 0.35)
total_s = len(all_single_scores)
total_m = len(all_multi_scores)

print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
print(f"  All single-turn: avg={avg_all_s:.2f} | {good_all_s}/{total_s} good")
print(f"  All multi-turn:  avg={avg_all_m:.2f} | {good_all_m}/{total_m} good")
print(f"  Grand avg (all): {(sum(all_single_scores)+sum(all_multi_scores))/(total_s+total_m):.2f}")

# Save to file
with open('data/test_deep_report.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Report saved to data/test_deep_report.json")
