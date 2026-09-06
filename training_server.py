"""
JoeBrain Training Dashboard Server
Light-themed web UI for monitoring and controlling expert training.

Usage:
  python3 training_server.py
"""

import json
import os
import re
import sys
import time
import signal
import subprocess
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
EXPERTS_JSON = os.path.join(DATA, 'experts.json')
TRAINING_UI = os.path.join(BASE, 'training_ui')
PORT = 9091

PYTHON = '/usr/bin/python3'
if not os.path.exists(PYTHON):
    PYTHON = '/data/data/com.termux/files/usr/bin/python3'
if not os.path.exists(PYTHON):
    PYTHON = sys.executable

NP = None
try:
    import numpy as _np
    NP = _np
except ImportError:
    for sp in [
        '/usr/lib/python3/dist-packages',
        '/data/data/com.termux/files/usr/lib/python3.14/site-packages',
        '/data/data/com.termux/files/usr/lib/python3.13/site-packages',
        '/data/data/com.termux/files/usr/lib/python3.12/site-packages',
    ]:
        if os.path.isdir(sp) and os.path.exists(os.path.join(sp, 'numpy', '__init__.py')):
            sys.path.insert(0, sp)
            try:
                import numpy as _np
                NP = _np
                break
            except ImportError:
                continue

# --- Tokenizer + model loading for test chat ---
TOK = None
try:
    sys.path.insert(0, os.path.join(BASE, 'training'))
    from tokenizer import Tokenizer
    from model import JoeBrain
    TOK = Tokenizer()
    TOK.load(os.path.join(DATA, 'tokenizer.json'))
except Exception:
    pass
# --- State ---
_state_lock = threading.RLock()
training = {}       # {name: {"pid": int, "target": int, "start_time": float, "params": dict}}
queue = []          # [{"name": str, "params": dict}, ...] ordered waitlist
max_concurrent = 2  # configurable from UI

# --- Helpers ---

def read_experts_config():
    try:
        with open(EXPERTS_JSON) as f:
            return json.load(f)
    except Exception:
        return {'experts': []}

def get_log_path(name):
    return os.path.join(DATA, 'experts', name, 'training.log')

def get_lock_path(name):
    return os.path.join(DATA, 'experts', name, 'training.lock')

def get_model_path(name, file_ref):
    return os.path.join(DATA, file_ref)

def is_process_alive(pid):
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    # Verify PID isn't reused — check cmdline contains train_expert.py
    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as f:
            cmdline = f.read().decode('utf-8', errors='replace')
        return 'train_expert' in cmdline
    except (OSError, FileNotFoundError):
        return False

def _lock_owned_by(name, pid):
    """Check if PID is a train_expert process running this specific expert."""
    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as f:
            cmdline = f.read().decode('utf-8', errors='replace')
        if 'train_expert' not in cmdline:
            return False
        args = [a for a in cmdline.split('\x00') if a]
        if '--name' in args:
            i = args.index('--name')
            return i + 1 < len(args) and args[i + 1] == name
        return False
    except (OSError, FileNotFoundError):
        return False

def count_running():
    with _state_lock:
        return sum(1 for n in training if is_process_alive(training[n]['pid']))

def cleanup_stale():
    """Remove dead processes from training dict."""
    stale = []
    with _state_lock:
        for name, info in training.items():
            if not is_process_alive(info['pid']):
                stale.append(name)
        for name in stale:
            del training[name]
            lock_path = get_lock_path(name)
            if os.path.exists(lock_path):
                try:
                    os.remove(lock_path)
                except OSError:
                    pass
    if stale:
        print(f"Cleaned up stale processes: {', '.join(stale)}")

def _stop_expert(name):
    """Stop a running expert, save checkpoint."""
    pid = None
    with _state_lock:
        if name in training:
            pid = training[name]['pid']
            del training[name]
    lock_path = get_lock_path(name)
    if not pid and os.path.exists(lock_path):
        try:
            with open(lock_path) as f:
                pid = int(f.read().strip())
        except (ValueError, OSError):
            pass
    if pid and is_process_alive(pid):
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        for _ in range(60):
            if not is_process_alive(pid):
                break
            time.sleep(0.5)
    if os.path.exists(lock_path):
        try:
            os.remove(lock_path)
        except OSError:
            pass

def process_queue():
    """Kill excess running experts if over limit, then start queued items."""
    with _state_lock:
        if max_concurrent <= 0:
            return
        running_names = [n for n in training if is_process_alive(training[n]['pid'])]
        while len(running_names) > max_concurrent:
            victim = running_names.pop()
            _stop_expert(victim)
            running_names = [n for n in training if is_process_alive(training[n]['pid'])]
        while queue and count_running() < max_concurrent:
            item = queue.pop(0)
            _start_training(item['name'], item['params'])

def _start_training(name, params):
    """Actually spawn the training subprocess. Returns True on success."""
    try:
        steps = int(params.get('steps', 1000))
    except (TypeError, ValueError):
        return False
    try:
        lr = float(params.get('lr', 3e-4))
    except (TypeError, ValueError):
        return False
    try:
        batch = int(params.get('batch', 8))
    except (TypeError, ValueError):
        return False
    try:
        seq_len = int(params.get('seq_len', 128))
    except (TypeError, ValueError):
        return False
    resume = params.get('resume', True)
    try:
        save_every = int(params.get('save', 0))
    except (TypeError, ValueError):
        save_every = 0
    try:
        push_every = int(params.get('push', 500))
    except (TypeError, ValueError):
        push_every = 500
    try:
        log_every = int(params.get('log', 100))
    except (TypeError, ValueError):
        log_every = 100
    try:
        sample_every = int(params.get('sample', 0))
    except (TypeError, ValueError):
        sample_every = 0

    if steps < 1 or lr <= 0 or batch < 1 or seq_len < 1:
        return False

    # Check already running
    if name in training and is_process_alive(training[name]['pid']):
        return False

    # Clean stale
    if name in training:
        del training[name]

    lock_path = get_lock_path(name)
    if os.path.exists(lock_path):
        try:
            with open(lock_path) as f:
                pid = int(f.read().strip())
            if is_process_alive(pid):
                return False
            else:
                os.remove(lock_path)
        except (ValueError, OSError):
            try:
                os.remove(lock_path)
            except OSError:
                pass

    cmd = [PYTHON, os.path.join(BASE, 'training', 'train_expert.py')]
    cmd += ['--name', name, '--steps', str(steps), '--lr', str(lr),
            '--batch', str(batch), '--seq_len', str(seq_len)]
    if resume:
        cmd.append('--resume')
    if save_every > 0:
        cmd += ['--save', str(save_every)]
    if push_every > 0:
        cmd += ['--push', str(push_every)]
    if log_every != 100:
        cmd += ['--log', str(log_every)]
    if sample_every > 0:
        cmd += ['--sample', str(sample_every)]

    expert_dir = os.path.join(DATA, 'experts', name)
    os.makedirs(expert_dir, exist_ok=True)
    try:
        stderr_log = open(os.path.join(expert_dir, 'train_stderr.log'), 'a')
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=stderr_log,
            cwd=BASE,
            start_new_session=True,
        )
        pid_str = str(proc.pid)
        with open(lock_path, 'w') as f:
            f.write(pid_str)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        training[name] = {
            'pid': proc.pid,
            'target': steps,
            'start_time': time.time(),
            'params': params,
        }
        # Verify process survived initial startup
        time.sleep(2)
        if proc.poll() is not None:
            # Process died immediately
            stderr_log.close()
            with open(os.path.join(DATA, 'experts', name, 'train_stderr.log')) as f:
                err = f.read().strip()
            if name in training:
                del training[name]
            if os.path.exists(lock_path):
                os.remove(lock_path)
            print(f"ERROR: {name} training died immediately (code={proc.returncode}): {err[:500]}")
            return False
        stderr_log.close()
        return True
    except Exception as e:
        print(f"ERROR: failed to start {name}: {e}")
        return False

def parse_last_data_line(log_path):
    if not os.path.exists(log_path):
        return None
    last_data = None
    with open(log_path, 'rb') as f:
        f.seek(0, 2)
        size = f.tell()
        read_size = min(size, 65536)
        f.seek(max(0, size - read_size))
        for line in f.read().decode('utf-8', errors='replace').splitlines():
            line = line.strip()
            if line and not line.startswith('#') and ',' in line:
                parts = line.split(',')
                if len(parts) >= 5:
                    try:
                        int(parts[0])
                        float(parts[1])
                        float(parts[2])
                        float(parts[3])
                        last_data = line
                    except (ValueError, IndexError):
                        pass
    if not last_data:
        return None
    parts = last_data.split(',')
    try:
        ts_str = parts[4]
        if ts_str.startswith('[PHONE]'):
            ts_str = ts_str.replace('[PHONE]', '').lstrip('.')
        timestamp = float(ts_str) if ts_str else time.time()
        return {
            'step': int(parts[0]),
            'loss': float(parts[1]),
            'lr': float(parts[2]),
            'speed': float(parts[3]),
            'timestamp': timestamp,
        }
    except (ValueError, IndexError):
        return None

def parse_target_from_log(log_path):
    if not os.path.exists(log_path):
        return None
    last_match = None
    with open(log_path, 'rb') as f:
        f.seek(0, 2)
        size = f.tell()
        read_size = min(size, 16384)
        f.seek(max(0, size - read_size))
        for line in f.read().decode('utf-8', errors='replace').splitlines():
            line = line.strip()
            if line.startswith('#') and ('START:' in line or 'RESUME:' in line):
                last_match = line
    if not last_match:
        return None
    m = re.search(r'--steps\s+(\d+)', last_match)
    if m:
        return int(m.group(1))
    return None

def get_expert_info(name, cfg_entry):
    log_path = get_log_path(name)
    lock_path = get_lock_path(name)
    model_path = get_model_path(name, cfg_entry.get('file', ''))
    train_path = os.path.join(DATA, 'experts', name, f'{name}_train.txt')

    info = {
        'name': name,
        'running': False,
        'queued': False,
        'step': 0,
        'target': None,
        'loss': None,
        'lr': None,
        'speed': None,
        'params': None,
        'layers': None,
        'dim': None,
        'seq_len': None,
        'file_size': None,
        'train_lines': 0,
    }

    # Model metadata
    if os.path.exists(model_path):
        info['file_size'] = os.path.getsize(model_path)
        try:
            if NP is None:
                raise ImportError("numpy not available")
            data = NP.load(model_path, allow_pickle=False)
            meta_keys = {}
            for k in data.files:
                if k.startswith('__'):
                    v = data[k]
                    meta_keys[k] = v.item() if v.ndim == 0 else v
            param_count = 0
            for k in data.files:
                if k.startswith('p_'):
                    param_count += data[k].size
            info['params'] = param_count
            info['layers'] = int(meta_keys.get('__n_layers__', 0))
            info['dim'] = int(meta_keys.get('__embed_dim__', 0))
            info['seq_len'] = int(meta_keys.get('__seq_len__', 0))
        except Exception:
            pass

    if os.path.exists(train_path):
        try:
            with open(train_path) as f:
                info['train_lines'] = sum(1 for _ in f)
        except Exception:
            pass

    last_data = parse_last_data_line(log_path)
    if last_data:
        info['step'] = last_data['step']
        info['loss'] = last_data['loss']
        info['lr'] = last_data['lr']
        info['speed'] = last_data['speed']

    info['target'] = None
    if name in training and training[name].get('target'):
        info['target'] = int(training[name]['target'])
    if info['target'] is None:
        info['target'] = parse_target_from_log(log_path)
    if info['target'] is None:
        info['target'] = info['step']

    # Running state
    if name in training:
        pid = training[name]['pid']
        if is_process_alive(pid):
            info['running'] = True
        else:
            del training[name]
            if os.path.exists(lock_path):
                try:
                    os.remove(lock_path)
                except OSError:
                    pass

    if not info['running'] and os.path.exists(lock_path):
        try:
            with open(lock_path) as f:
                pid = int(f.read().strip())
            if _lock_owned_by(name, pid):
                info['running'] = True
                training[name] = {
                    'pid': pid,
                    'target': info['target'] or 0,
                    'start_time': time.time(),
                    'params': {},
                }
            else:
                os.remove(lock_path)
        except (ValueError, OSError):
            pass

    # Queue state
    if not info['running']:
        for item in queue:
            if item['name'] == name:
                info['queued'] = True
                break

    return info

def sync_state():
    cfg = read_experts_config()
    with _state_lock:
        for entry in cfg.get('experts', []):
            name = entry['name']
            lock_path = get_lock_path(name)

            if name in training:
                pid = training[name]['pid']
                if not is_process_alive(pid):
                    del training[name]
                    if os.path.exists(lock_path):
                        try:
                            os.remove(lock_path)
                        except OSError:
                            pass

            elif os.path.exists(lock_path):
                try:
                    with open(lock_path) as f:
                        pid = int(f.read().strip())
                    if _lock_owned_by(name, pid):
                        info = get_expert_info(name, entry)
                        training[name] = {
                            'pid': pid,
                            'target': info['target'] or 0,
                            'start_time': time.time(),
                            'params': {},
                        }
                    else:
                        os.remove(lock_path)
                except (ValueError, OSError):
                    pass

        # After syncing, try to start queued items
        process_queue()

# --- Background thread to monitor training and auto-dequeue ---

def monitor_loop():
    while True:
        try:
            time.sleep(5)
            sync_state()
            cleanup_stale()
        except Exception as e:
            ts_log(f"monitor_loop error: {e}")

monitor_thread = threading.Thread(target=monitor_loop, daemon=True)

def ts_log(msg):
    """Write a line to the crash log unconditionally (flushed)."""
    try:
        print(msg, flush=True)
    except Exception:
        pass

# --- Test chat generation ---

def _keyword_match(expected, actual):
    if expected.strip() == actual.strip():
        return True
    def extract(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'about', 'as', 'from', 'up', 'down', 'out', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because', 'until', 'while', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'off'}
        return [w for w in words if w not in stopwords and len(w) > 1]
    exp_keys = set(extract(expected))
    act_keys = set(extract(actual))
    if not exp_keys:
        return False
    overlap = exp_keys & act_keys
    return len(overlap) >= len(exp_keys) * 0.5

def load_expert_model(name):
    if NP is None or TOK is None:
        return None
    cfg = read_experts_config()
    for entry in cfg.get('experts', []):
        if entry['name'] == name:
            model_path = os.path.join(DATA, entry.get('file', ''))
            if os.path.exists(model_path):
                try:
                    return JoeBrain.load(model_path)
                except Exception:
                    return None
    return None

def build_test_prompt(msg):
    new_turn = f"User: {msg}\nJoe:"
    new_ids = TOK.encode(new_turn)
    return new_turn

def generate_stream(model, prompt, max_new=150):
    ids = TOK.encode(prompt)
    pending = ''
    STOPS = ['\nUser:', '\nJoe:']
    max_safe = max(len(s) for s in STOPS) - 1
    for _ in range(max_new):
        ctx = NP.array(ids[-model.T:], dtype=NP.int32)
        logits, _ = model.forward(ctx)
        next_id = int(NP.argmax(logits[-1]))
        ids.append(next_id)
        token = TOK.id_to_token.get(next_id, '')
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

# --- HTTP Handler ---

class TrainingHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == '/api/experts':
            self._handle_experts_list()
            return

        if parsed.path == '/api/logs':
            qs = parse_qs(parsed.query)
            name = qs.get('name', [''])[0].strip()
            self._handle_log_stream(name)
            return

        if parsed.path == '/api/logdata':
            qs = parse_qs(parsed.query)
            name = qs.get('name', [''])[0].strip()
            window = qs.get('window', ['all'])[0].strip()
            self._handle_log_data(name, window)
            return

        if parsed.path == '/api/test':
            qs = parse_qs(parsed.query)
            name = qs.get('name', [''])[0].strip()
            self._handle_test_suite(name)
            return

        if parsed.path == '/api/test-stream':
            qs = parse_qs(parsed.query)
            name = qs.get('name', [''])[0].strip()
            self._handle_test_stream(name)
            return

        # Serve static files
        path = parsed.path.lstrip('/')
        if path == '' or path == '/':
            path = 'index.html'

        try:
            filepath = os.path.realpath(os.path.join(TRAINING_UI, path))
            if not filepath.startswith(os.path.realpath(TRAINING_UI)):
                self.send_response(403)
                self.end_headers()
                self.wfile.write(b'Forbidden')
                return
        except (ValueError, OSError):
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'Bad request')
            return
        if not os.path.exists(filepath) or not os.path.isfile(filepath):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not found')
            return

        ext = os.path.splitext(filepath)[1]
        types = {
            '.html': 'text/html', '.js': 'application/javascript',
            '.css': 'text/css', '.json': 'application/json',
            '.png': 'image/png', '.svg': 'image/svg+xml',
        }
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
        try:
            length = int(self.headers.get('Content-Length', 0))
        except (TypeError, ValueError):
            self._json({'error': 'invalid Content-Length'}, 400)
            return
        if length > 10 * 1024 * 1024:
            self._json({'error': 'request body too large'}, 413)
            return
        body = self.rfile.read(length) if length > 0 else b''
        try:
            data = json.loads(body) if body else {}
        except Exception:
            self._json({'error': 'invalid json'}, 400)
            return
        if not isinstance(data, dict):
            self._json({'error': 'expected JSON object'}, 400)
            return

        if parsed.path == '/api/train':
            self._handle_start_training(data)
            return
        if parsed.path == '/api/stop':
            self._handle_stop_training(data)
            return
        if parsed.path == '/api/queue':
            self._handle_queue_update(data)
            return
        if parsed.path == '/api/dequeue':
            self._handle_dequeue(data)
            return
        if parsed.path == '/api/test':
            self._handle_test_chat(data)
            return

        self._json({'error': 'not found'}, 404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def _handle_experts_list(self):
        cleanup_stale()
        sync_state()
        cfg = read_experts_config()
        with _state_lock:
            experts = []
            for entry in cfg.get('experts', []):
                if not entry.get('enabled', True):
                    continue
                info = get_expert_info(entry['name'], entry)
                experts.append(info)
            payload = {
                'experts': experts,
                'max_concurrent': max_concurrent,
                'running_count': count_running(),
                'queue': [item['name'] for item in queue],
            }
        self._json(payload)

    def _handle_log_stream(self, name):
        cleanup_stale()
        if not name:
            self._json({'error': 'missing name param'}, 400)
            return
        log_path = get_log_path(name)
        if not os.path.exists(log_path):
            self._json({'error': f'no log for {name}'}, 404)
            return

        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        try:
            with open(log_path, 'rb') as f:
                f.seek(0, 2)
                size = f.tell()
                start = max(0, size - 4096)
                f.seek(start)
                for line in f.read().decode('utf-8', errors='replace').splitlines():
                    if line.strip():
                        msg = json.dumps({'line': line})
                        self.wfile.write(f'data: {msg}\n\n'.encode())
                self.wfile.flush()
                initial_end = size
        except Exception:
            initial_end = 0

        last_offset = initial_end
        try:
            with open(log_path, 'rb') as f:
                f.seek(0, 2)
                if last_offset == 0:
                    last_offset = f.tell()
                while True:
                    if last_offset > os.path.getsize(log_path):
                        last_offset = 0
                    f.seek(last_offset)
                    new_data = f.read()
                    if new_data:
                        last_offset = f.tell()
                        for line in new_data.decode('utf-8', errors='replace').splitlines():
                            if line.strip():
                                msg = json.dumps({'line': line})
                                try:
                                    self.wfile.write(f'data: {msg}\n\n'.encode())
                                    self.wfile.flush()
                                except Exception:
                                    return
                    if name not in training and not os.path.exists(get_lock_path(name)):
                        try:
                            msg = json.dumps({'done': True})
                            self.wfile.write(f'data: {msg}\n\n'.encode())
                            self.wfile.flush()
                        except Exception:
                            pass
                        return
                    time.sleep(1)
        except Exception:
            pass

    def _handle_log_data(self, name, window='all'):
        if not name:
            self._json({'error': 'missing name param'}, 400)
            return
        log_path = get_log_path(name)
        if not os.path.exists(log_path):
            self._json({'data': [], 'history': []})
            return

        data_points = []
        history = []
        try:
            with open(log_path, 'r', errors='replace') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('#'):
                        entry = {'raw': line}
                        m = re.search(r'\[(.*?)\]', line)
                        if m:
                            entry['time'] = m.group(1)
                        if 'START:' in line:
                            entry['type'] = 'start'
                        elif 'RESUME:' in line:
                            entry['type'] = 'resume'
                        elif 'FINISH' in line:
                            entry['type'] = 'finish'
                        elif 'STOP' in line:
                            entry['type'] = 'stop'
                        elif 'INTERRUPT' in line:
                            entry['type'] = 'interrupt'
                        elif 'ERROR' in line:
                            entry['type'] = 'error'
                        elif 'SAMPLE' in line:
                            entry['type'] = 'sample'
                            sm = re.search(r'step (\d+)', line)
                            if sm:
                                entry['step'] = int(sm.group(1))
                        else:
                            entry['type'] = 'info'
                        history.append(entry)
                    else:
                        parts = line.split(',')
                        if len(parts) >= 5:
                            try:
                                ts_str = parts[4]
                                if ts_str.startswith('[PHONE]'):
                                    ts_str = ts_str.replace('[PHONE]', '').lstrip('.')
                                timestamp = float(ts_str) if ts_str else 0
                                data_points.append({
                                    'step': int(parts[0]),
                                    'loss': float(parts[1]),
                                    'lr': float(parts[2]),
                                    'speed': float(parts[3]),
                                    'time': timestamp,
                                })
                            except (ValueError, IndexError):
                                pass
        except Exception:
            pass

        now = time.time()
        if window == 'run' and data_points:
            first_ts = data_points[0]['time']
            data_points = [d for d in data_points if d['time'] >= first_ts]
        elif window == '1d':
            cutoff = now - 86400
            data_points = [d for d in data_points if d['time'] >= cutoff]
        elif window == '1w':
            cutoff = now - 604800
            data_points = [d for d in data_points if d['time'] >= cutoff]

        self._json({'data': data_points, 'history': history})

    def _handle_start_training(self, data):
        name = (data.get('name') or '').strip()
        if not name:
            self._json({'error': 'missing name'}, 400)
            return

        cfg = read_experts_config()
        valid_names = [e['name'] for e in cfg.get('experts', []) if e.get('enabled', True)]
        if name not in valid_names:
            self._json({'error': f'unknown expert: {name}'}, 400)
            return

        with _state_lock:
            # Force-clean stale entries (PID reuse or dead process)
            if name in training and not is_process_alive(training[name]['pid']):
                del training[name]
                lock_path = get_lock_path(name)
                if os.path.exists(lock_path):
                    try:
                        os.remove(lock_path)
                    except OSError:
                        pass

            # Already running?
            if name in training and is_process_alive(training[name]['pid']):
                self._json({'error': f'{name} is already training'})
                return

            # Already queued?
            for item in queue:
                if item['name'] == name:
                    self._json({'error': f'{name} is already in the queue'})
                    return

            # Under limit? Start immediately
            if count_running() < max_concurrent:
                ok = _start_training(name, data)
                if ok:
                    self._json({'ok': True, 'started': True})
                else:
                    self._json({'error': f'failed to start {name}'}, 500)
            else:
                # Add to queue
                queue.append({'name': name, 'params': data})
                self._json({'ok': True, 'queued': True, 'position': len(queue)})

    def _handle_stop_training(self, data):
        name = (data.get('name') or '').strip()
        if not name:
            self._json({'error': 'missing name'}, 400)
            return

        result = None
        with _state_lock:
            # Remove from queue if queued
            for i, item in enumerate(queue):
                if item['name'] == name:
                    queue.pop(i)
                    result = {'ok': True, 'removed_from_queue': True}

            if result is not None:
                pass  # already set
            else:
                # Stop running
                stopped = False
                pid = None
                if name in training:
                    pid = training[name]['pid']
                    del training[name]

                lock_path = get_lock_path(name)
                if not pid and os.path.exists(lock_path):
                    try:
                        with open(lock_path) as f:
                            pid = int(f.read().strip())
                    except (ValueError, OSError):
                        pass

                if pid and is_process_alive(pid):
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGTERM)
                        stopped = True
                    except (ProcessLookupError, OSError):
                        pass

                if os.path.exists(lock_path):
                    try:
                        os.remove(lock_path)
                    except OSError:
                        pass

                if stopped:
                    process_queue()
                    result = {'ok': True}
                else:
                    result = {'error': f'{name} is not training'}

        self._json(result)

    def _handle_queue_update(self, data):
        global max_concurrent
        with _state_lock:
            new_max = data.get('max_concurrent')
            if new_max is not None:
                try:
                    new_max = max(1, min(6, int(new_max)))
                except (TypeError, ValueError):
                    new_max = max_concurrent
                max_concurrent = new_max
                process_queue()
            payload = {'ok': True, 'max_concurrent': max_concurrent, 'queue': [i['name'] for i in queue]}
        self._json(payload)

    def _handle_dequeue(self, data):
        name = (data.get('name') or '').strip()
        result = None
        with _state_lock:
            for i, item in enumerate(queue):
                if item['name'] == name:
                    queue.pop(i)
                    result = {'ok': True}
                    break
            if result is None:
                result = {'error': f'{name} not in queue'}
        self._json(result)

    def _handle_test_chat(self, data):
        name = (data.get('expert') or '').strip()
        msg = (data.get('msg') or '').strip()
        if not name:
            self._json({'error': 'missing expert'}, 400)
            return
        if not msg:
            self._json({'error': 'missing msg'}, 400)
            return
        if TOK is None:
            self._json({'error': 'tokenizer not loaded'}, 500)
            return

        model = load_expert_model(name)
        if model is None:
            self._json({'error': f'could not load model for {name}'}, 500)
            return

        prompt = build_test_prompt(msg)
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        try:
            reply = ''
            for chunk in generate_stream(model, prompt):
                reply += chunk
                msg_data = json.dumps({'char': chunk})
                self.wfile.write(f'data: {msg_data}\n\n'.encode())
                self.wfile.flush()
            done_data = json.dumps({'done': True, 'reply': reply})
            self.wfile.write(f'data: {done_data}\n\n'.encode())
            self.wfile.flush()
        except Exception:
            pass

    def _handle_test_suite(self, name):
        if not name:
            self._json({'error': 'missing name param'}, 400)
            return

        cfg = read_experts_config()
        valid_names = [e['name'] for e in cfg.get('experts', []) if e.get('enabled', True)]
        if name not in valid_names:
            self._json({'error': f'unknown expert: {name}'}, 400)
            return

        if TOK is None:
            self._json({'error': 'tokenizer not loaded'}, 500)
            return

        model = load_expert_model(name)
        if model is None:
            self._json({'error': f'could not load model for {name}'}, 500)
            return

        test_path = os.path.join(DATA, 'experts', name, f'{name}_test.txt')
        if not os.path.exists(test_path):
            self._json({'error': f'no test file for {name}'}, 404)
            return

        results = []
        correct = 0

        try:
            with open(test_path) as f:
                content = f.read()
        except Exception:
            self._json({'error': 'failed to read test file'}, 500)
            return

        questions = []
        current_q = None
        current_a = None
        for line in content.strip().split('\n'):
            if line.startswith('User:'):
                if current_q is not None and current_a is not None:
                    questions.append((current_q, current_a))
                current_q = line[5:].strip()
                current_a = None
            elif line.startswith('Joe:'):
                current_a = line[4:].strip()

        if current_q and current_a:
            questions.append((current_q, current_a))

        for question, expected in questions:
            prompt = build_test_prompt(question)
            actual = ''
            try:
                for chunk in generate_stream(model, prompt):
                    actual += chunk
            except Exception:
                pass

            is_correct = _keyword_match(expected, actual)
            if is_correct:
                correct += 1

            results.append({
                'question': question,
                'expected': expected,
                'actual': actual,
                'correct': is_correct,
            })

        total = len(results)
        accuracy = (correct / total * 100) if total > 0 else 0

        self._json({
            'name': name,
            'correct': correct,
            'total': total,
            'accuracy': round(accuracy, 1),
            'results': results,
        })

    def _handle_test_stream(self, name):
        if not name:
            self._json({'error': 'missing name param'}, 400)
            return

        cfg = read_experts_config()
        valid_names = [e['name'] for e in cfg.get('experts', []) if e.get('enabled', True)]
        if name not in valid_names:
            self._json({'error': f'unknown expert: {name}'}, 400)
            return

        if TOK is None:
            self._json({'error': 'tokenizer not loaded'}, 500)
            return

        model = load_expert_model(name)
        if model is None:
            self._json({'error': f'could not load model for {name}'}, 500)
            return

        test_path = os.path.join(DATA, 'experts', name, f'{name}_test.txt')
        if not os.path.exists(test_path):
            self._json({'error': f'no test file for {name}'}, 404)
            return

        try:
            with open(test_path) as f:
                content = f.read()
        except Exception:
            self._json({'error': 'failed to read test file'}, 500)
            return

        questions = []
        current_q = None
        current_a = None
        for line in content.strip().split('\n'):
            if line.startswith('User:'):
                if current_q is not None and current_a is not None:
                    questions.append((current_q, current_a))
                current_q = line[5:].strip()
                current_a = None
            elif line.startswith('Joe:'):
                current_a = line[4:].strip()

        if current_q and current_a:
            questions.append((current_q, current_a))

        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        try:
            correct = 0
            total = len(questions)
            for question, expected in questions:
                prompt = build_test_prompt(question)
                actual = ''
                try:
                    for chunk in generate_stream(model, prompt):
                        actual += chunk
                except Exception:
                    pass
                is_correct = _keyword_match(expected, actual)
                if is_correct:
                    correct += 1
                event = json.dumps({
                    'question': question,
                    'expected': expected,
                    'actual': actual,
                    'correct': is_correct,
                    'correct_so_far': correct,
                    'total': total,
                })
                self.wfile.write(f'data: {event}\n\n'.encode())
                self.wfile.flush()

            done = json.dumps({'done': True, 'correct': correct, 'total': total,
                               'accuracy': round((correct / total * 100) if total else 0, 1)})
            self.wfile.write(f'data: {done}\n\n'.encode())
            self.wfile.flush()
        except Exception:
            pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(data))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        try:
            self.wfile.write(data)
        except BrokenPipeError:
            pass

# --- Main ---

if __name__ == '__main__':
    try:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    except (AttributeError, ValueError):
        pass
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    except (AttributeError, ValueError):
        pass

    def _print_exc(exc_type, exc, tb):
        try:
            import traceback
            print("=== UNCAUGHT EXCEPTION ===", flush=True)
            traceback.print_exception(exc_type, exc, tb, file=sys.stdout)
            sys.stdout.flush()
        except Exception:
            pass
    sys.excepthook = _print_exc

    import faulthandler
    try:
        faulthandler.enable(all_threads=False)
    except Exception:
        pass

    sync_state()
    running = [n for n in training if is_process_alive(training[n]['pid'])]
    if running:
        print(f"Detected running training: {', '.join(running)}")

    monitor_thread.start()

    ThreadingHTTPServer.allow_reuse_address = True
    try:
        server = ThreadingHTTPServer(('127.0.0.1', PORT), TrainingHandler)
    except OSError as e:
        print(f"FATAL: could not bind 127.0.0.1:{PORT}: {e}", flush=True)
        sys.exit(1)
    print(f"Training Dashboard running at http://localhost:{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    except Exception as e:
        print(f"SERVER ERROR: {e}", flush=True)
        try:
            server.server_close()
        except Exception:
            pass
        sys.exit(1)
