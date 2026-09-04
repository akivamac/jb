"""
Joe Brain local server — Router + Expert selector.
Router auto-selects experts, user can override.
Greedy decoding with probability-averaged blending.

Usage:
  python3 server.py
"""

import json
import os
import sys
import numpy as np
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training'))
from tokenizer import Tokenizer
from model import JoeBrain
from router import RouterNet

DATA = os.path.join(os.path.dirname(__file__), 'data')
PORT = 9090

print("Loading experts...")
try:
    tok = Tokenizer()
    tok.load(os.path.join(DATA, 'tokenizer.json'))

    with open(os.path.join(DATA, 'experts.json')) as f:
        cfg = json.load(f)

    experts = {}
    for entry in cfg.get('experts', []):
        if not entry.get('enabled', True):
            continue
        name = entry['name']
        model_path = os.path.join(DATA, entry['file'])
        if not os.path.exists(model_path):
            print(f"  WARNING: {name} not found at {model_path}, skipping")
            continue
        model = JoeBrain.load(model_path)
        experts[name] = model
        n_params = sum(v.size for v in model.p.values())
        print(f"  Loaded {name}: {n_params:,} params, {model.L} layers, dim {model.C}, seq {model.T}")

    if not experts:
        print("ERROR: No experts found!")
        sys.exit(1)
    print(f"Ready. {len(experts)} expert(s): {', '.join(experts.keys())}")

    # Load router
    router_path = os.path.join(DATA, 'router', 'router.npz')
    router = None
    if os.path.exists(router_path):
        router = RouterNet.load(router_path)
        print(f"Router loaded: {router.count_params():,} params")
    else:
        print("WARNING: No router found, auto-routing disabled")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def get_seq_len():
    return min(m.T for m in experts.values())

def build_prompt(history, msg):
    seq_len = get_seq_len()
    new_turn = f"User: {msg}\nJoe:"
    new_ids = tok.encode(new_turn)
    budget = seq_len - len(new_ids) - 2

    lines = []
    for role, text in history:
        prefix = "User" if role == "user" else "Joe"
        lines.append(f"{prefix}: {text}")

    included = []
    used = 0
    for line in reversed(lines):
        cost = len(tok.encode(line + "\n"))
        if used + cost > budget:
            break
        included.append(line)
        used += cost

    included.reverse()
    history_text = "\n".join(included)
    if history_text:
        return history_text + "\n" + new_turn
    return new_turn

def generate_stream(model, prompt, max_new=150):
    ids = tok.encode(prompt)
    pending = ''
    STOPS = ['\nUser:', '\nJoe:']
    max_safe = max(len(s) for s in STOPS) - 1
    for _ in range(max_new):
        ctx = np.array(ids[-model.T:], dtype=np.int32)
        logits, _ = model.forward(ctx)
        next_id = int(np.argmax(logits[-1]))
        ids.append(next_id)
        token = tok.id_to_token.get(next_id, '')
        if token and all(ord(c) <= 127 for c in token):
            pending += token
            for stop in STOPS:
                if stop in pending:
                    cut = pending.index(stop)
                    if cut > 0:
                        yield pending[:cut]
                    return
            if len(pending) > max_safe:
                safe = pending[:-max_safe]
                pending = pending[-max_safe:]
                yield safe
    if pending:
        yield pending


def softmax(x, temp=1.0):
    x = x - np.max(x)
    e = np.exp(x / temp)
    return e / e.sum()


def generate_blended_stream(models_weights, prompt, max_new=150):
    """Generate using probability-averaged blending across multiple experts."""
    ids = tok.encode(prompt)
    pending = ''
    STOPS = ['\nUser:', '\nJoe:']
    max_safe = max(len(s) for s in STOPS) - 1
    for _ in range(max_new):
        avg_prob = None
        for m, w in models_weights:
            ctx = np.array(ids[-m.T:], dtype=np.int32)
            logits, _ = m.forward(ctx)
            probs = softmax(logits[-1], temp=1.0)
            mn = min(len(probs), 2000)
            if avg_prob is None:
                avg_prob = np.zeros(mn)
            avg_prob[:mn] += probs[:mn] * w
        next_id = int(np.argmax(avg_prob))
        ids.append(next_id)
        token = tok.id_to_token.get(next_id, '')
        if token and all(ord(c) <= 127 for c in token):
            pending += token
            for stop in STOPS:
                if stop in pending:
                    cut = pending.index(stop)
                    if cut > 0:
                        yield pending[:cut]
                    return
            if len(pending) > max_safe:
                safe = pending[:-max_safe]
                pending = pending[-max_safe:]
                yield safe
    if pending:
        yield pending


EXPERT_NAMES = ['greeting', 'emotion', 'knowledge', 'coding', 'cot', 'python', 'horse', 'fish', 'reptiles', 'tree']

def route_message(msg):
    """Use router to predict which experts to use. Returns [(name, weight), ...]"""
    ids = tok.encode(msg.lower())
    if not ids:
        return [('cot', 1.0)]
    if len(ids) > 64:
        ids = ids[:64]
    probs = router.predict(ids)
    top_idx = probs.argsort()[::-1]

    # Pick top-K: if top prob > 0.7, use 1; else blend top-2
    top_prob = probs[top_idx[0]]
    if top_prob > 0.7:
        k = 1
    elif probs[top_idx[1]] > 0.15:
        k = 2
    else:
        k = 1

    selected = []
    for i in top_idx[:k]:
        name = EXPERT_NAMES[i]
        if name in experts:
            selected.append((name, float(probs[i])))
    if not selected:
        selected = [('cot', 1.0)]

    # Normalize weights with power-law: dominant expert leads
    total_w = sum(w ** 2 for _, w in selected)
    selected = [(n, (w ** 2) / total_w) for n, w in selected]
    return selected

class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == '/ping':
            self._json({'ok': True})
            return

        if parsed.path == '/experts':
            info = []
            for name, model in experts.items():
                info.append({
                    'name': name,
                    'params': sum(v.size for v in model.p.values()),
                    'layers': model.L,
                    'dim': model.C,
                })
            self._json({'experts': info})
            return

        if parsed.path == '/router':
            if router is None:
                self._json({'error': 'router not loaded'}, 503)
                return
            qs = parse_qs(parsed.query)
            msg = qs.get('msg', [''])[0].strip()
            if not msg:
                self._json({'error': 'no message'}, 400)
                return
            ids = tok.encode(msg.lower())
            if len(ids) > 64:
                ids = ids[:64]
            probs = router.predict(ids)
            result = {EXPERT_NAMES[i]: float(probs[i]) for i in range(len(EXPERT_NAMES))}
            self._json({'probs': result})
            return

        if parsed.path == '/chat':
            qs = parse_qs(parsed.query)
            msg = qs.get('msg', [''])[0].strip()
            expert_name = qs.get('expert', [''])[0].strip()
            if not msg:
                self._json({'error': 'no message'}, 400)
                return

            if expert_name and expert_name in experts:
                model = experts[expert_name]
                prompt = build_prompt([], msg)
                reply = ''.join(generate_stream(model, prompt))
                self._json({'reply': reply, 'routed_to': [expert_name]})
            elif router is not None:
                selected = route_message(msg)
                prompt = build_prompt([], msg)
                if len(selected) == 1:
                    model = experts[selected[0][0]]
                    reply = ''.join(generate_stream(model, prompt))
                else:
                    models_weights = [(experts[n], w) for n, w in selected]
                    reply = ''.join(generate_blended_stream(models_weights, prompt))
                self._json({'reply': reply, 'routed_to': [n for n, _ in selected]})
            else:
                self._json({'error': f'unknown expert: {expert_name}'}, 400)
            return

        path = parsed.path.lstrip('/')
        if path == '':
            path = 'index.html'

        filepath = os.path.join(os.path.dirname(__file__), path)
        real = os.path.realpath(filepath)
        root = os.path.realpath(os.path.dirname(__file__))
        if not real.startswith(root + os.sep) and real != root:
            self.send_response(403)
            self.end_headers()
            self.wfile.write(b'Forbidden')
            return
        if not os.path.exists(real) or not os.path.isfile(real):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not found')
            return
        filepath = real

        ext = os.path.splitext(filepath)[1]
        types = {'.html': 'text/html', '.js': 'application/javascript', '.css': 'text/css', '.json': 'application/json'}
        ctype = types.get(ext, 'application/octet-stream')

        with open(filepath, 'rb') as f:
            data = f.read()
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', len(data))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == '/chat':
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
            except Exception:
                self._json({'error': 'invalid json'}, 400)
                return
            msg = (data.get('msg') or '').strip()
            expert_name = (data.get('expert') or '').strip()
            if not msg:
                self._json({'error': 'no message'}, 400)
                return

            history = data.get('history', [])
            max_new = int(data.get('max_new', 120))
            prompt = build_prompt(history, msg)

            # Route: if expert specified, use it; otherwise router auto-selects
            if expert_name and expert_name in experts:
                selected = [(expert_name, 1.0)]
                model = experts[expert_name]
                gen_func = lambda p, mn: generate_stream(model, p, mn)
            elif router is not None:
                selected = route_message(msg)
                if len(selected) == 1:
                    model = experts[selected[0][0]]
                    gen_func = lambda p, mn: generate_stream(model, p, mn)
                else:
                    models_weights = [(experts[n], w) for n, w in selected]
                    gen_func = lambda p, mn: generate_blended_stream(models_weights, p, mn)
            else:
                selected = [('cot', 1.0)]
                model = experts['cot']
                gen_func = lambda p, mn: generate_stream(model, p, mn)

            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            routed_to = [n for n, _ in selected]
            reply = ''
            try:
                for ch in gen_func(prompt, max_new):
                    reply += ch
                    msg_data = json.dumps({'char': ch})
                    self.wfile.write(f'data: {msg_data}\n\n'.encode())
                    self.wfile.flush()

                done_data = json.dumps({'done': True, 'reply': reply, 'routed_to': routed_to})
                self.wfile.write(f'data: {done_data}\n\n'.encode())
                self.wfile.flush()
            except Exception:
                pass
            return
        self._json({'error': 'not found'}, 404)

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(data))
        self.end_headers()
        self.wfile.write(data)

if __name__ == '__main__':
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    print(f"Joe Brain running at http://localhost:{PORT}")
    print("Open that URL in your browser. Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
