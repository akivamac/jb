import sys, re, json, random
sys.path.insert(0, 'training')
from model import JoeBrain
from tokenizer import Tokenizer

tok = Tokenizer(); tok.load('data/tokenizer.json')

EXPERTS = ['greeting','emotion','knowledge','coding','cot','python','horse','fish','reptiles','tree']

def load_questions(expert):
    path = f'data/experts/{expert}/{expert}_train.txt'
    questions = []
    with open(path) as f:
        for line in f:
            if line.startswith('User: '):
                questions.append(line[6:].strip())
    return questions

def score(resp):
    if len(resp) < 10: return 0.0, 0.0, 0
    resp = re.sub(r'(User|Joe):.*', '', resp, flags=re.S)
    words = resp.split()
    if not words: return 0, 0, 0
    uniq = len(set(w.lower() for w in words)) / len(words)
    common = set('the a an is are of in on to and or for with that this what how why do i you we they be have not but from by at it as'.split())
    hits = sum(1 for w in words if w.lower() in common)
    cov = hits / len(words)
    return 0.6*cov + 0.4*uniq, cov, len(words)

results = {}
for name in EXPERTS:
    m = JoeBrain.load(f'data/experts/{name}/{name}.npz')
    questions = load_questions(name)
    random.seed(42)
    sample_qs = random.sample(questions, min(12, len(questions)))
    single_results = []
    for q in sample_qs:
        prompt = f'User: {q}\nJoe:'
        out = m.generate(tok, prompt, max_new=120, temperature=0.8)
        resp = out.replace(prompt, '').strip()
        if 'User:' in resp:
            resp = resp[:resp.index('User:')].strip()
        s, cov, nw = score(resp)
        single_results.append({'q': q, 'resp': resp[:80], 'score': s, 'cov': cov, 'nw': nw})
    results[name] = {'single': single_results}
    avg = sum(r['score'] for r in single_results) / len(single_results) if single_results else 0
    good = sum(1 for r in single_results if r['score'] >= 0.35)
    print(f"{name}: avg={avg:.3f} | good={good}/{len(single_results)}", flush=True)

all_scores = [r['score'] for name in EXPERTS for r in results[name]['single']]
grand = sum(all_scores)/len(all_scores) if all_scores else 0
print(f"\nGrand avg: {grand:.3f}", flush=True)

with open('data/test_deep_report.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Report saved to data/test_deep_report.json", flush=True)