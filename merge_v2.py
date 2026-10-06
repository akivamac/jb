"""merge_v2.py: merge by verdicts
Verdicts: write per-chunk or per-block? Better per chunk file containing lines like
0:ACCEPT
1:REJECT: reason
...
VERDICT 23/30

Convention: for each generated chunk file {name}_{i}.txt, verdicts in
data/_gen/verdicts/{name}_{i}_v1.txt and _v2.txt (dual-review). Merge requires BOTH ACCEPT.

Also dedup against train: first question norm must not exist in train_qs; intra-dedup by sim>0.82.
"""
import os, re, sys
from difflib import SequenceMatcher

REPO = os.path.dirname(os.path.abspath(__file__))
BASE_EX = os.path.join(REPO, 'data', 'experts')
GEN = os.path.join(REPO, 'data', '_gen')
VER = os.path.join(GEN, 'verdicts')
os.makedirs(VER, exist_ok=True)

EXPERTS = ['tree','reptiles','fish','knowledge','greeting','emotion','coding','python','cot','horse']

def norm(s):
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def load_blocks(path):
    blocks=[]; cur=[]
    if not os.path.exists(path): return blocks
    with open(path) as f:
        for line in f:
            line=line.rstrip()
            if not line.strip():
                if cur: blocks.append(cur); cur=[]
            else: cur.append(line.strip())
    if cur: blocks.append(cur)
    return blocks

def first_q(b):
    for x in b:
        if x.startswith('User: '): return x[6:]
    return ''

def read_verdict(p):
    ok=set(); rej=0
    if not os.path.exists(p): return ok,rej
    with open(p) as f:
        for line in f:
            line=line.strip()
            m=re.match(r'^(\d+):(ACCEPT|REJECT)', line)
            if m:
                i=int(m.group(1)); t=m.group(2)
                if t=='ACCEPT': ok.add(i)
                else: rej+=1
    return ok,rej

def sim(a,b): return SequenceMatcher(None,a,b).ratio()

def merge_for(name):
    edir = os.path.join(GEN, 'cleaned')
    train_path = os.path.join(BASE_EX, name, f'{name}_train.txt')
    train_qs=set()
    if os.path.exists(train_path):
        with open(train_path) as f:
            for l in f:
                if l.startswith('User: '): train_qs.add(norm(l[6:]))
    # collect candidate cleaned files for this expert? chunks named name_*.txt_clean.txt or name_*.txt
    accepted=[]; missing=[]
    # scan cleaned dir
    cdir = edir
    for f in sorted(os.listdir(cdir)):
        if not f.startswith(name+'_') or not f.endswith('.txt'): continue
        # original chunk stem?
        stem = f.replace('_clean.txt','.txt') if '_clean.txt' in f else f
        chunk = stem[:-4]
        v1=os.path.join(VER,f'{chunk}_v1.txt')
        v2=os.path.join(VER,f'{chunk}_v2.txt')
        if not os.path.exists(v1) or not os.path.exists(v2):
            missing.append(chunk); continue
        v1o,v1r=read_verdict(v1); v2o,v2r=read_verdict(v2)
        ok=v1o&v2o
        path_block=os.path.join(cdir,f)
        for i,b in enumerate(load_blocks(path_block)):
            if i in ok: accepted.append(b)
    # dedup
    kept=[]; seen=set(); dtrain=dself=0
    for b in accepted:
        nq=norm(first_q(b))
        if nq in train_qs: dtrain+=1; continue
        # intra
        if any(sim(nq,norm(first_q(kb)))>0.82 for kb in kept): dself+=1; continue
        if nq in seen: dself+=1; continue
        seen.add(nq); kept.append(b)
    # append
    with open(train_path,'a') as tf:
        for b in kept:
            tf.write('\n'.join(b)+'\n\n')
    print(f"{name}: accepted_blocks={len(accepted)} merged={len(kept)} drop_train={dtrain} drop_self={dself} missing={len(missing)}")
    return len(kept), missing

if __name__=='__main__':
    names=sys.argv[1:] or EXPERTS
    allm=[]
    for n in names:
        _,m=merge_for(n); allm.extend(m)
    if allm: print('MISSING_VERDICTS:', len(set(allm)))
