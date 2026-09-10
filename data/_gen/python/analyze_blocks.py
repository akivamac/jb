import sys, re

with open('/Users/dev/github-projects/joe-brain/data/_gen/python/python_mt_03_cand.txt') as f:
    cand_lines = f.readlines()

# Parse blocks
blocks = []
i = 0
while i < len(cand_lines):
    if cand_lines[i].strip() == '':
        i += 1
        continue
    if i + 3 < len(cand_lines):
        blocks.append(cand_lines[i:i+4])
        i += 4
    else:
        i += 1

print(f"Found {len(blocks)} blocks", file=sys.stderr)

# Read existing training
with open('/Users/dev/github-projects/joe-brain/data/experts/python/python_train.txt') as f:
    existing_text = f.read()

# Existing first-turn fingerprints
existing_first = set()
for eb in existing_text.strip().split('\n\n'):
    lines = eb.strip().split('\n')
    if not lines:
        continue
    fp = ''
    for l in lines:
        fp += l.strip()
        if l.startswith('Joe:'):
            break
    existing_first.add(re.sub(r'[^a-z0-9]', '', fp.lower()))

print(f"Existing first-turn fingerprints: {len(existing_first)}", file=sys.stderr)

STOPWORDS = {'the','a','an','is','are','was','were','be','been','has','have',
             'had','do','does','did','will','would','could','should','may','might',
             'can','shall','to','of','in','for','on','with','at','by','from','as',
             'into','through','during','before','after','above','below','between',
             'out','off','over','under','again','further','then','once','here',
             'there','when','where','why','how','all','each','every','both','few',
             'more','most','other','some','such','no','nor','not','only','own',
             'same','so','than','too','very','just','because','but','and','or',
             'if','while','that','this','it','its','you','your','i','my','me',
             'we','our','they','them','their','he','she','him','her','his','what',
             'which','who','whom','whose','about','up','down','back','well','also',
             'much','still','already','yet','now','even','ever','never','then',
             'here','there','do','does','did','doing','done','get','gets','got',
             'going','go','goes','went','gone','make','makes','made','making',
             'like','want','use','used','using','need','needs','needed','know',
             'knows','knew','known','take','takes','took','taken','see','sees',
             'saw','seen','come','comes','came','think','thinks','thought','say',
             'says','said','one','two','three','first','second','third','last',
             'next','previous','new','old','good','bad','big','small','long',
             'short','high','low','top','bottom','right','left','another','each',
             'every','both','same','different','own','very','just','also','even',
             'well','back','still','already','yet','now','then','here','there',
             'way','thing','things','something','anything','nothing','everything',
             'kind','types','type','part','parts','number','numbers','name',
             'names','example','examples','case','cases','line','lines','value',
             'values','item','items','key','keys','list','lists','set','sets',
             'dict','dicts','string','strings','tuple','tuples','function',
             'functions','method','methods','class','classes','file','files',
             'data','text','word','words','character','characters','result',
             'results','output','input','call','calls','called','work','works',
             'worked','working','want','wants','wanted','need','needs','needed',
             'use','uses','used','using','get','gets','got','getting','make',
             'makes','made','making','put','puts','putting','take','takes',
             'took','taking','give','gives','gave','giving','return','returns',
             'returned','print','prints','printed','printing','find','finds',
             'found','finding','create','creates','created','creating','add',
             'adds','added','adding','remove','removes','removed','removing',
             'check','checks','checked','checking','run','runs','ran','running',
             'write','writes','wrote','writing','read','reads','reading','open',
             'opens','opened','opening','close','closes','closed','closing',
             'start','starts','started','starting','stop','stops','stopped',
             'stopping','convert','converts','converted','converting','change',
             'changes','changed','changing','set','sets','setting','build',
             'builds','built','building','handle','handles','handled','handling',
             'process','processes','processed','processing','pass','passes',
             'passed','passing','show','shows','showed','showing','look','looks',
             'looked','looking'}

def get_kw(s):
    return set(re.findall(r'\b[a-z]+\b', s.lower())) - STOPWORDS

def count_sents(s):
    return max(1, len(re.findall(r'[.!?]', s)))

def has_nonascii(s):
    return any(ord(c) > 127 for c in s)

def norm(s):
    return re.sub(r'[^a-z0-9\s]', '', s.lower())

verdicts = []
accepted_norms = []

for idx, block in enumerate(blocks):
    reasons = []
    lns = [l.rstrip('\n') for l in block]
    
    if len(lns) != 4:
        verdicts.append(f"{idx}:REJECT: wrong line count {len(lns)}")
        continue
    if not (lns[0].startswith('User: ') and lns[1].startswith('Joe: ') and
            lns[2].startswith('User: ') and lns[3].startswith('Joe: ')):
        verdicts.append(f"{idx}:REJECT: format")
        continue
    
    u1 = lns[0][6:]
    a1 = lns[1][5:]
    u2 = lns[2][6:]
    a2 = lns[3][5:]
    
    alltxt = ''.join(lns)
    if has_nonascii(alltxt):
        verdicts.append(f"{idx}:REJECT: non-ASCII")
        continue
    
    # Sentence count
    for ans_i, ans in enumerate([a1, a2], 1):
        sc = count_sents(ans)
        if sc > 3:
            reasons.append(f"ans{ans_i} {sc}sents")
    
    # Multi-turn: u2 must reference a1
    kw1 = get_kw(a1)
    kw2 = get_kw(u2)
    overlap = kw1 & kw2
    ref_words = {'that','this','it','they','them','those','these','instead',
                 'too','also','there','then','one','do','does','did','can',
                 'could','would'}
    has_ref = bool(overlap) or any(w in u2.lower() for w in ref_words)
    if not has_ref and len(u2.split()) > 3:
        reasons.append("no ref to prior answer")
    
    # Directness: no filler starts
    fillers = ('well,', 'actually,', 'basically,', 'essentially,', 'so,', 'now,', 'okay,', 'ok,')
    for ans_i, ans in enumerate([a1, a2], 1):
        if ans.strip().lower().startswith(fillers):
            reasons.append(f"ans{ans_i} filler")
    
    # Within-file dup
    bn = norm(alltxt)
    for pn in accepted_norms:
        pt = set(pn[j:j+30] for j in range(0, len(pn), 15))
        bt = set(bn[j:j+30] for j in range(0, len(bn), 15))
        if pt and bt:
            r = len(pt & bt) / min(len(pt), len(bt))
            if r > 0.6:
                reasons.append("dup within file")
                break
    
    # Existing dup check
    fp = re.sub(r'[^a-z0-9]', '', (lns[0] + lns[1]).lower())
    if fp in existing_first:
        reasons.append("dup of existing")
    
    if reasons:
        verdicts.append(f"{idx}:REJECT: {'; '.join(reasons)}")
    else:
        verdicts.append(f"{idx}:ACCEPT")
        accepted_norms.append(bn)

with open('/Users/dev/github-projects/joe-brain/data/_gen/python/python_mt_03_v1.txt', 'w') as f:
    for v in verdicts:
        f.write(v + '\n')

ac = sum(1 for v in verdicts if v.endswith(':ACCEPT'))
print(f"VERDICT {ac}/{len(verdicts)}")
for v in verdicts:
    if 'REJECT' in v:
        print(v)
