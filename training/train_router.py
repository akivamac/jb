"""
Train the RouterNet classifier — v4.
Uses actual expert training data + balanced augmentation.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from router import RouterNet
from tokenizer import Tokenizer

EXPERTS = ['greeting', 'emotion', 'knowledge', 'coding', 'cot', 'python', 'horse', 'fish', 'reptiles', 'tree']
EXPERT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'experts')
ROUTER_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'router', 'router.npz')
SEQ_LEN = 64
TARGET_PER_CLASS = 300


def load_expert_messages(expert):
    path = os.path.join(EXPERT_DIR, expert, f'{expert}_train.txt')
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [l.strip()[6:].strip().lower() for l in f
                if l.strip().startswith('User: ') and l.strip()[6:].strip()]


def augment(msg, expert):
    msgs = set()
    low = msg.lower()

    if expert == 'greeting':
        names = ['', ' joe', ' monkey']
        base = low.replace('joe', '').replace('monkey', '').strip()
        suffixes = ['', ' how are you', ' whats up', ' how is it going',
                     ' how have you been', ' whats new', ' how are things',
                     ' nice to see you', ' long time no see',
                     ' how are you doing', ' good to see you']
        for n in names:
            for s in suffixes:
                for p in ['', 'hey ', 'hi ', 'hello ', 'good morning ']:
                    c = f'{p}{base}{n}{s}'.strip()
                    if c and len(c) < 55:
                        msgs.add(c)

    elif expert == 'emotion':
        feelings = [
            'i feel {x}', 'i am feeling {x}', 'i am {x}',
            'ive been feeling {x}', 'feeling {x} today',
            'i feel so {x}', 'i feel very {x}', 'today i feel {x}',
            'i am kind of {x}', 'a bit {x}', 'really {x}',
        ]
        base = low.replace('i am ', '').replace('i feel ', '').replace('i am feeling ', '').strip()
        for f in feelings:
            msgs.add(f.format(x=base))
        for f in ['hey joe i feel {x}', 'hello joe i am {x}', 'joe i feel {x}',
                   'feeling {x} joe', 'good morning i feel {x}']:
            msgs.add(f.format(x=base))

    elif expert == 'knowledge':
        topic = low.replace('what is ', '').replace('what are ', '').replace('how does ', '').strip()
        patterns = [
            'what is {t}', 'what are {t}', 'tell me about {t}',
            'explain {t}', 'how does {t} work', 'what do you know about {t}',
            'can you tell me about {t}', 'teach me about {t}',
        ]
        for p in patterns:
            msgs.add(p.format(t=topic))

    elif expert == 'coding':
        topic = low.replace('what is ', '').replace('what are ', '').replace('how do i ', '').strip()
        patterns = [
            'what is a {t}', 'what is an {t}', 'what are {t}',
            'explain {t}', 'how do i use {t}', 'tell me about {t}',
            'how does {t} work', 'help me with {t}',
            'code a {t}', 'write a {t}', 'build a {t}', 'create a {t}',
        ]
        for p in patterns:
            msgs.add(p.format(t=topic))

    elif expert == 'cot':
        import re
        nums = re.findall(r'\d+', low)
        if nums:
            n1, n2 = nums[0], nums[1] if len(nums) > 1 else '0'
            patterns = [
                'what is {n1} + {n2}', 'what is {n1} - {n2}',
                'what is {n1} * {n2}', 'what is {n1} / {n2}',
                '{n1} plus {n2}', '{n1} times {n2}',
                '{n1} minus {n2}', '{n1} divided by {n2}',
                'calculate {n1} + {n2}', 'compute {n1} * {n2}',
            ]
            for p in patterns:
                msgs.add(p.format(n1=n1, n2=n2))

    elif expert == 'python':
        topic = low.replace('what is a ', '').replace('what is an ', '').replace('how do i ', '').replace(' in python', '').strip()
        patterns = [
            'what is a {t} in python', 'what is an {t} in python',
            'how do i use {t} in python', 'explain {t} in python',
            'python {t}', 'tell me about {t} in python',
            'what is {t} in python', 'teach me about {t} in python',
        ]
        for p in patterns:
            msgs.add(p.format(t=topic))

    elif expert == 'horse':
        topic = low.replace('what is ', '').replace('how do ', '').replace('how does a ', '').replace(' how ', '').strip()
        patterns = [
            'what is a {t}', 'tell me about {t}',
            'how do horses {t}', 'how does a horse {t}',
            'explain {t}', 'my horse {t}', 'my horse is {t}',
            'horse {t}', 'teach me about {t}',
        ]
        for p in patterns:
            msgs.add(p.format(t=topic))

    elif expert == 'fish':
        topic = low.replace('what is a ', '').replace('what is an ', '').replace('what are ', '').replace('how do ', '').strip()
        patterns = [
            'what is a {t} fish', 'what is an {t}',
            'what are {t}', 'tell me about {t}',
            'how do fish {t}', 'how do {t}',
            'explain {t}', 'my fish {t}', 'my fish is {t}',
            'fish {t}', 'what do {t} eat', 'how big do {t} get',
            'aquarium {t}', 'how to care for {t}',
        ]
        for p in patterns:
            msgs.add(p.format(t=topic))
        for kw in ['goldfish', 'betta', 'cichlid', 'angelfish', 'guppy', 'tetra',
                   'aquarium', 'gills', 'fins', 'spines', 'tank', 'water']:
            for p in ['what is a {kw}', 'what is an {kw}', 'tell me about {kw}',
                      'how do i care for {kw}', 'my {kw} is sick',
                      'how do fish use {kw}', 'explain {kw}']:
                msgs.add(p.format(kw=kw))

    elif expert == 'reptiles':
        topic = low.replace('what is a ', '').replace('what is an ', '').replace('what are ', '').replace('how do ', '').strip()
        patterns = [
            'what is a {t}', 'what is an {t}',
            'what are {t}', 'tell me about {t}',
            'explain {t}', 'my snake {t}', 'my lizard {t}',
            'how do i care for {t}', 'how to care for {t}',
            'reptile {t}', 'snake {t}', 'lizard {t}', 'turtle {t}',
        ]
        for p in patterns:
            msgs.add(p.format(t=topic))
        for kw in ['bearded dragon', 'gecko', 'snake', 'lizard', 'turtle', 'tortoise',
                   'chameleon', 'python', 'viper', 'rattlesnake', 'terrarium', 'scales']:
            for p in ['what is a {kw}', 'what is an {kw}', 'tell me about {kw}',
                      'how do i care for {kw}', 'my {kw} is sick',
                      'explain {kw}', 'how do reptiles {t}']:
                msgs.add(p.format(kw=kw, t=topic))

    elif expert == 'tree':
        topic = low.replace('what is a ', '').replace('what is an ', '').replace('what are ', '').replace('how do ', '').strip()
        patterns = [
            'what is a {t}', 'what is an {t}',
            'what are {t} trees', 'tell me about {t}',
            'explain {t}', 'how do i plant {t}',
            'how to grow {t}', 'my {t} tree', 'my {t} tree is dying',
            'tree {t}', 'oak tree {t}', 'maple {t}', 'pine {t}',
        ]
        for p in patterns:
            msgs.add(p.format(t=topic))
        for kw in ['oak', 'maple', 'pine', 'baobab', 'sequoia', 'birch', 'willow',
                   'redwood', 'sap', 'roots', 'trunk', 'leaves', 'photosynthesis',
                   'acorn', 'conifer']:
            for p in ['what is a {kw} tree', 'what is {kw} sap', 'tell me about {kw}',
                      'how do i plant {kw}', 'why are {kw} leaves', 'explain {kw}']:
                msgs.add(p.format(kw=kw))

    msgs.add(low)
    return list(msgs)


def build_dataset(tokenizer):
    np.random.seed(42)
    by_class = {i: set() for i in range(len(EXPERTS))}

    for i, expert in enumerate(EXPERTS):
        raw = load_expert_messages(expert)
        print(f"  {expert}: {len(raw)} raw messages")
        for msg in raw:
            for a in augment(msg, expert):
                a = a.strip().lower()
                if a and len(a) < 80:
                    by_class[i].add(a)

    # Balance: cap each class at TARGET_PER_CLASS
    for i in range(len(EXPERTS)):
        items = list(by_class[i])
        if len(items) > TARGET_PER_CLASS:
            by_class[i] = set(np.random.choice(items, TARGET_PER_CLASS, replace=False))

    all_examples = []
    for i, msgs in by_class.items():
        for m in msgs:
            all_examples.append((m, i))

    np.random.shuffle(all_examples)

    X, Y = [], []
    for msg, label in all_examples:
        ids = tokenizer.encode(msg)
        if not ids:
            continue
        if len(ids) > SEQ_LEN:
            ids = ids[:SEQ_LEN]
        padded = ids + [0] * (SEQ_LEN - len(ids))
        X.append(padded)
        onehot = np.zeros(len(EXPERTS), dtype=np.float32)
        onehot[label] = 1.0
        Y.append(onehot)

    X = np.array(X, dtype=np.int32)
    Y = np.array(Y, dtype=np.float32)

    print(f"\nTotal: {len(X)} examples")
    for i, name in enumerate(EXPERTS):
        count = int(Y[:, i].sum())
        print(f"  {name:10s}: {count}")

    return X, Y


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def train():
    tokenizer = Tokenizer()
    tokenizer.load(os.path.join(os.path.dirname(__file__), '..', 'data', 'tokenizer.json'))

    X, Y = build_dataset(tokenizer)
    n = X.shape[0]

    split = int(n * 0.85)
    perm = np.random.permutation(n)
    X_train, Y_train = X[perm[:split]], Y[perm[:split]]
    X_val, Y_val = X[perm[split:]], Y[perm[split:]]
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    model = RouterNet(vocab_size=2000, embed_dim=128, n_experts=len(EXPERTS), seq_len=SEQ_LEN)
    print(f"Router params: {model.count_params():,}")

    lr = 1e-3
    wd = 1e-4
    batch_size = 32
    steps = 3000

    print(f"\nTraining {steps} steps, lr={lr}, batch={batch_size}\n")

    for step in range(1, steps + 1):
        idx = np.random.randint(0, len(X_train), size=batch_size)
        x_batch = X_train[idx]
        y_batch = Y_train[idx]

        total_loss = 0
        grads = {k: np.zeros_like(v) for k, v in model.p.items()}

        for i in range(batch_size):
            logits = model.forward(x_batch[i])
            probs = softmax(logits)
            loss = -np.sum(y_batch[i] * np.log(probs + 1e-8))
            total_loss += loss

            d_logits = probs - y_batch[i]

            # Backward
            T = len(x_batch[i])
            x_emb = model.p['wte'][x_batch[i]] + model.p['wpe'][:T]
            # Masked mean + max pooling (must mirror forward)
            C = model.C
            mask = (x_batch[i] != 0).astype(np.float32)
            n = mask.sum()
            if n == 0:
                n = 1
            h_mean = (x_emb * mask[:, None]).sum(axis=0) / n
            x_masked = np.where(mask[:, None] > 0, x_emb, -1e9)
            h_max = x_masked.max(axis=0)
            h = np.concatenate([h_mean, h_max])

            mean1 = h.mean(); var1 = h.var()
            h1n = (h - mean1) / np.sqrt(var1 + 1e-5) * model.p['ln1_g'] + model.p['ln1_b']
            z1 = np.maximum(h1n @ model.p['fc1_w'] + model.p['fc1_b'], 0)
            h_pre2 = h + z1

            mean2 = h_pre2.mean(); var2 = h_pre2.var()
            h2n = (h_pre2 - mean2) / np.sqrt(var2 + 1e-5) * model.p['ln2_g'] + model.p['ln2_b']
            z2 = np.maximum(h2n @ model.p['fc2_w'] + model.p['fc2_b'], 0)
            h_out = h_pre2 + z2

            grads['out_w'] += np.outer(h_out, d_logits) / batch_size
            grads['out_b'] += d_logits / batch_size

            d_h = d_logits @ model.p['out_w'].T
            grads['fc2_w'] += np.outer(h2n, d_h * (z2 > 0)) / batch_size
            grads['fc2_b'] += d_h * (z2 > 0) / batch_size

            # d_h2n is grad of LN2 OUTPUT (h2n)
            d_h2n = d_h * (z2 > 0) @ model.p['fc2_w'].T
            grads['ln2_g'] += d_h2n * h2n / batch_size
            grads['ln2_b'] += d_h2n / batch_size

            # residual 2: h_out = h_pre2 + z2
            # layer-norm backward of LN2 into its input h_pre2 (= h + z1)
            ln2_input = h_pre2
            m2b = ln2_input.mean(); v2b = ln2_input.var()
            xhat2 = (ln2_input - m2b) / np.sqrt(v2b + 1e-5)
            d_xhat2 = d_h2n * model.p['ln2_g']
            dvar2 = (d_xhat2 * (ln2_input - m2b) * -0.5 * (v2b + 1e-5)**-1.5).sum()
            dmean2 = (d_xhat2 * -1 / np.sqrt(v2b + 1e-5)).sum() + dvar2 * (-2 * (ln2_input - m2b)).mean()
            d_ln2_in = d_xhat2 / np.sqrt(v2b + 1e-5) + dvar2 * 2 * (ln2_input - m2b) / len(ln2_input) + dmean2 / len(ln2_input)
            d_h_pre2 = d_h + d_ln2_in

            # block 1: h_pre2 = h + z1
            d_z1 = d_h_pre2
            d_h_accum = d_h_pre2.copy()
            d_pre_fc1 = d_z1 * (z1 > 0)
            d_h1n = d_pre_fc1 @ model.p['fc1_w'].T
            grads['fc1_w'] += np.outer(h1n, d_pre_fc1) / batch_size
            grads['fc1_b'] += d_pre_fc1 / batch_size
            grads['ln1_g'] += d_h1n * h1n / batch_size
            grads['ln1_b'] += d_h1n / batch_size

            # layer-norm backward of LN1 into its input h (pooled)
            ln1_input = h
            m1b = ln1_input.mean(); v1b = ln1_input.var()
            xhat1 = (ln1_input - m1b) / np.sqrt(v1b + 1e-5)
            d_xhat1 = d_h1n * model.p['ln1_g']
            dvar1 = (d_xhat1 * (ln1_input - m1b) * -0.5 * (v1b + 1e-5)**-1.5).sum()
            dmean1 = (d_xhat1 * -1 / np.sqrt(v1b + 1e-5)).sum() + dvar1 * (-2 * (ln1_input - m1b)).mean()
            d_ln1_in = d_xhat1 / np.sqrt(v1b + 1e-5) + dvar1 * 2 * (ln1_input - m1b) / len(ln1_input) + dmean1 / len(ln1_input)
            d_h_pool = d_h_accum + d_ln1_in

            # Split pooled gradient into mean and max halves
            d_h_mean = d_h_pool[:C]
            d_h_max = d_h_pool[C:]

            # Masked mean-pool backward: d_x += d_h_mean / n for real rows only
            d_x = np.zeros_like(x_emb)
            d_x += (mask[:, None] / n) * d_h_mean
            # Masked max-pool backward: d_x at the argmax row among real tokens, per dim
            amax = np.argmax(x_masked, axis=0)
            d_x[amax, np.arange(C)] += d_h_max

            grads['wte'][x_batch[i]] += d_x
            grads['wpe'][:T] += d_x

        model.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for k in model.p:
            grad = grads[k]
            # L2 weight decay on weights (not biases)
            if k.endswith('_w') or k == 'wte':
                grad = grad + wd * model.p[k]
            model.m[k] = beta1 * model.m[k] + (1 - beta1) * grad
            model.v[k] = beta2 * model.v[k] + (1 - beta2) * grad ** 2
            m_hat = model.m[k] / (1 - beta1 ** model.t)
            v_hat = model.v[k] / (1 - beta2 ** model.t)
            model.p[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        if step % 500 == 0:
            avg_loss = total_loss / batch_size
            train_correct = sum(1 for i in range(len(X_train)) if np.argmax(model.forward(X_train[i])) == np.argmax(Y_train[i]))
            val_correct = sum(1 for i in range(len(X_val)) if np.argmax(model.forward(X_val[i])) == np.argmax(Y_val[i]))
            print(f"  step {step:5d}/{steps} | loss {avg_loss:.4f} | train {train_correct/len(X_train):.1%} | val {val_correct/len(X_val):.1%}")
            lr *= 0.95

    os.makedirs(os.path.dirname(ROUTER_PATH), exist_ok=True)
    model.save(ROUTER_PATH)
    print(f"\nSaved to {ROUTER_PATH}")


if __name__ == '__main__':
    train()
