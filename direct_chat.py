"""
Direct chat server — greedy decoding like training. No sampling tricks.
Port 7357. Single model, selected from dropdown.

Usage:
  python3 direct_chat.py
"""

import json
import os
import sys
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training'))
from tokenizer import Tokenizer
from model import JoeBrain

DATA = os.path.join(os.path.dirname(__file__), 'data')
PORT = 7357

print("Loading experts...")
tok = Tokenizer()
tok.load(os.path.join(DATA, 'tokenizer.json'))

experts = {}
for name in sorted(os.listdir(os.path.join(DATA, 'experts'))):
    path = os.path.join(DATA, 'experts', name, f'{name}.npz')
    if os.path.exists(path):
        model = JoeBrain.load(path)
        n = sum(v.size for v in model.p.values())
        experts[name] = model
        print(f"  {name}: {n:,} params, {model.L}L dim{model.C} seq{model.T}")

print(f"Ready. {len(experts)} experts: {', '.join(experts.keys())}")

T = min(m.T for m in experts.values())

def generate(model, prompt, max_new=200):
    ids = tok.encode(prompt)
    pending = ''
    STOPS = ['\nUser:', '\nJoe:']
    for _ in range(max_new):
        ctx = np.array(ids[-model.T:], dtype=np.int32)
        logits, _ = model.forward(ctx)
        next_id = int(np.argmax(logits[-1]))
        ids.append(next_id)
        t = tok.id_to_token.get(next_id, '')
        pending += t
        for stop in STOPS:
            if stop in pending:
                return pending[:pending.index(stop)]
    return pending

def build_prompt(history, msg):
    new_turn = f"User: {msg}\nJoe:"
    new_ids = tok.encode(new_turn)
    budget = T - len(new_ids) - 2
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

HTML = '''<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Joe Brain - Direct</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#1a1a2e;color:#e0e0e0;font-family:monospace;display:flex;flex-direction:column;height:100vh}
#top{padding:12px 16px;background:#16213e;border-bottom:1px solid #0f3460;display:flex;align-items:center;gap:12px}
#top h1{font-size:1rem;color:#e94560}
#expert-sel{background:#0f3460;color:#e0e0e0;border:1px solid #e94560;border-radius:4px;padding:6px 10px;font-family:monospace;font-size:0.85rem}
#msgs{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:10px}
.msg{max-width:85%;padding:10px 14px;border-radius:8px;line-height:1.5;white-space:pre-wrap;word-break:break-word}
.user{background:#0f3460;align-self:flex-end;border-bottom-right-radius:2px}
.joe{background:#16213e;border:1px solid #333;align-self:flex-start;border-bottom-left-radius:2px}
#input-area{padding:12px 16px;background:#16213e;border-top:1px solid #0f3460;display:flex;gap:8px}
#input{flex:1;background:#0f3460;color:#e0e0e0;border:1px solid #333;border-radius:4px;padding:10px;font-family:monospace;font-size:0.9rem;resize:none;min-height:40px}
#input:focus{outline:none;border-color:#e94560}
#send{background:#e94560;color:white;border:none;border-radius:4px;padding:10px 20px;font-family:monospace;font-size:0.9rem;cursor:pointer}
#send:hover{background:#c73652}
#send:disabled{opacity:0.5;cursor:not-allowed}
.typing{color:#666;font-style:italic}
</style></head><body>
<div id="top"><h1>Joe Brain (Direct/Greedy)</h1>
<select id="expert-sel"></select></div>
<div id="msgs"><div class="msg joe">Greedy decoding — same as training. Pick an expert and talk to it.</div></div>
<div id="input-area">
<textarea id="input" rows="1" placeholder="Type something..." disabled></textarea>
<button id="send" disabled>&gt;</button>
</div>
<script>
const msgs=document.getElementById('msgs'),input=document.getElementById('input'),
send=document.getElementById('send'),sel=document.getElementById('expert-sel');
let history=[],sending=false;

fetch('/experts').then(r=>r.json()).then(d=>{
d.experts.forEach(e=>{const o=document.createElement('option');o.value=e;o.textContent=e;sel.appendChild(o)});
input.disabled=false;send.disabled=false;input.focus();
});

function addMsg(role,text){const d=document.createElement('div');d.className='msg '+role;d.textContent=text;msgs.appendChild(d);msgs.scrollTop=msgs.scrollHeight;return d}

input.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();doSend()}});
send.addEventListener('click',doSend);

async function doSend(){
const text=input.value.trim();if(!text||sending)return;
sending=true;input.value='';send.disabled=true;
addMsg('user',text);history.push(['user',text]);
const typing=addMsg('joe','...');typing.classList.add('typing');

try{
const resp=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({msg:text,expert:sel.value,history:history.slice(0,-1)})});
const data=await resp.json();
typing.classList.remove('typing');typing.textContent=data.reply||'[empty]';
history.push(['joe',data.reply||'']);
}catch(e){typing.textContent='[error: '+e.message+']'}
sending=false;send.disabled=false;input.focus();
}
</script></body></html>'''

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/experts':
            self._json({'experts': list(experts.keys())})
            return
        if parsed.path == '/ping':
            self._json({'ok': True})
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(HTML.encode())

    def do_POST(self):
        if urlparse(self.path).path != '/chat':
            self._json({'error': 'not found'}, 404)
            return
        length = int(self.headers.get('Content-Length', 0))
        data = json.loads(self.rfile.read(length))
        msg = (data.get('msg') or '').strip()
        name = (data.get('expert') or '').strip()
        history = data.get('history', [])

        if not msg:
            self._json({'error': 'no message'}, 400)
            return
        if name not in experts:
            self._json({'error': f'unknown expert: {name}'}, 400)
            return

        prompt = build_prompt(history, msg)
        reply = generate(experts[name], prompt)
        self._json({'reply': reply})

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(data))
        self.end_headers()
        self.wfile.write(data)

if __name__ == '__main__':
    HTTPServer.allow_reuse_address = True
    server = HTTPServer(('0.0.0.0', PORT), Handler)
    print(f"\nDirect chat at http://localhost:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
