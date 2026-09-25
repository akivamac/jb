"""
JoeBrain Training Dashboard Server
Light-themed web UI for monitoring and controlling expert training.

Usage:
  python3 training_server.py
"""

import json
import math
import os
import random
import re
import sys
import time
import signal
import subprocess
import threading
import fcntl
import shutil
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import urllib.request
import json as json_mod
import socket
import io
import zipfile
import tempfile
import hashlib
import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
EXPERTS_JSON = os.path.join(DATA, 'experts.json')
TRAINING_UI = os.path.join(BASE, 'training_ui')
PORT = 9091

NOTIFICATION_URL = os.environ.get('NOTIFICATION_URL', 'http://localhost:9091/notify')

def send_notification(message):
    """Send a push notification to the configured webhook URL."""
    try:
        data = json_mod.dumps({'message': message}).encode('utf-8')
        req = urllib.request.Request(NOTIFICATION_URL, data=data, headers={'Content-Type': 'application/json'})
        try:
            urllib.request.urlopen(req, timeout=2)
        except Exception:
            pass  # Silently fail if notification service unavailable
    except Exception:
        pass

    # Log notification locally as backup
    try:
        with open(os.path.join(BASE, 'data', 'notification.log'), 'a') as f:
            f.write(f'{datetime.datetime.now()}: {message}\n')
    except Exception:
        pass


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
    TOK = None
# --- State ---
_state_lock = threading.RLock()
training = {}       # {name: {"pid": int, "target": int, "start_time": float, "params": dict}}
queue = []          # [{"name": str, "params": dict}, ...] ordered waitlist
max_concurrent = 2  # configurable from UI
just_stopped = {}   # {name: time.time()} — blocks stop-then-re-adopt race for ~20s
child_popens = {}   # {pid: Popen} — known children so we can poll()/reap them
_log_stream_lock = threading.Lock()
_log_stream_active = 0          # current concurrent /api/logs SSE streams
MAX_LOG_STREAMS = 20            # cap to avoid thread/FD exhaustion

_ALLOWED_ORIGINS = {'http://localhost:9091', 'http://127.0.0.1:9091'}

def _cors_headers(origin):
    """CORS headers echoing only the dashboard's own origins; else none."""
    if origin in _ALLOWED_ORIGINS:
        return [('Access-Control-Allow-Origin', origin), ('Vary', 'Origin')]
    return []

def _notify_async(message):
    """Send notification off-thread so callers never block on _state_lock."""
    threading.Thread(target=send_notification, args=(message,), daemon=True).start()

# --- Helpers ---

def read_experts_config():
    try:
        with open(EXPERTS_JSON) as f:
            return json.load(f)
    except Exception:
        return {'experts': []}

EVAL_DIR = os.path.join(DATA, '_eval')
ROUTER_PATH = os.path.join(DATA, 'router', 'router.npz')
# Router class order from training/train_router.py (v4)
ROUTER_NAMES = ['greeting', 'emotion', 'knowledge', 'coding', 'cot',
                'python', 'horse', 'fish', 'reptiles', 'tree']
_eval_lock = threading.Lock()
_router_state = {'obj': None, 'tried': False}

def _format_experts_json(cfg):
    """Re-render data/experts.json in the repo's original compact style
    (one-line expert entries, 2-space outer / 4-space inner objects) so that
    config writes diff only on real changes, not full-file reformatting."""
    parts = ['{', '  "experts": [']
    experts = cfg.get('experts', [])
    for i, e in enumerate(experts):
        suffix = ',' if i < len(experts) - 1 else ''
        parts.append(f'    {json.dumps(e, separators=(", ", ": "))}{suffix}')
    parts.append('  ],')
    ordered = [k for k in ('quality_filter', 'watcher') if k in cfg]
    for k in tuple(cfg):
        if k in ('quality_filter', 'watcher') or k == 'experts':
            continue
        ordered.append(k)
    for i, k in enumerate(ordered):
        if isinstance(cfg[k], dict):
            sub = json.dumps(cfg[k], indent=2).splitlines()
            body = "\n".join([sub[0]] + ['  ' + ln for ln in sub[1:]])
        else:
            body = json.dumps(cfg[k], separators=(", ", ": "))
        suffix = ',' if i < len(ordered) - 1 else ''
        parts.append(f'  "{k}": {body}{suffix}')
    parts.append('}')
    return '\n'.join(parts) + '\n'

def _save_experts_json(cfg):
    """Atomically rewrite data/experts.json (temp file + os.replace)."""
    fd, tmp = tempfile.mkstemp(dir=DATA, suffix='.json.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(_format_experts_json(cfg))
        os.replace(tmp, EXPERTS_JSON)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise

def load_router():
    """Load RouterNet once (cached). Returns None when unavailable/failed."""
    if _router_state['tried']:
        return _router_state['obj']
    _router_state['tried'] = True
    try:
        from router import RouterNet
        if os.path.exists(ROUTER_PATH):
            _router_state['obj'] = RouterNet.load(ROUTER_PATH)
    except Exception:
        _router_state['obj'] = None
    return _router_state['obj']

def suite_config_for(name):
    """Per-expert suite config (n_questions/pass_pct) from data/experts.json."""
    cfg = read_experts_config()
    for e in cfg.get('experts', []):
        if e['name'] == name:
            s = e.get('suite') or {}
            try:
                nq = int(s.get('n_questions', 30))
            except (TypeError, ValueError):
                nq = 30
            try:
                pp = int(s.get('pass_pct', 70))
            except (TypeError, ValueError):
                pp = 70
            return {'n_questions': max(1, min(nq, 300)), 'pass_pct': pp}
    return {'n_questions': 30, 'pass_pct': 70}

def _append_eval(name, rec):
    """Append a test-suite result to data/_eval/{name}.json (flock-protected)."""
    try:
        os.makedirs(EVAL_DIR, exist_ok=True)
        path = os.path.join(EVAL_DIR, f'{name}.json')
        lock_path = os.path.join(EVAL_DIR, '.lock')
        with open(lock_path, 'a'):
            pass
        with _eval_lock:
            with open(lock_path, 'rb+') as lf:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                try:
                    recs = []
                    if os.path.exists(path):
                        try:
                            with open(path) as f:
                                recs = json.load(f)
                            if not isinstance(recs, list):
                                recs = []
                        except Exception:
                            recs = []
                    recs.append(rec)
                    recs = recs[-500:]
                    fd, tmp = tempfile.mkstemp(dir=EVAL_DIR, suffix='.tmp')
                    try:
                        with os.fdopen(fd, 'w') as f:
                            json.dump(recs, f, indent=2)
                        os.replace(tmp, path)
                    except Exception:
                        try:
                            os.remove(tmp)
                        except OSError:
                            pass
                        raise
                finally:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        print(f"ERROR: append_eval {name}: {e}", flush=True)

def build_prompt(history, msg, window=4, seq_len=1024):
    """Port of server.py build_prompt — same User:/Joe: format with token budget.

    history: list of {role, content} dicts or 'User: ...'/'Joe: ...' strings.
    window: max turns (user+Joe pairs) of context to include.
    Falls back to build_test_prompt when no usable history.
    """
    new_turn = f"User: {msg}\nJoe:"
    new_ids = TOK.encode(new_turn)
    budget = seq_len - len(new_ids) - 2
    try:
        window = max(1, min(20, int(window)))
    except (TypeError, ValueError):
        window = 4
    items = history[-window * 2:] if history else []
    lines = []
    for item in items:
        if isinstance(item, str):
            s = item.strip()
            if s.startswith('Joe:'):
                role, text = 'joe', s[4:].strip()
            elif s.startswith('User:'):
                role, text = 'user', s[5:].strip()
            else:
                role, text = 'user', s
        elif isinstance(item, dict):
            role = str(item.get('role', 'user')).lower()
            role = 'joe' if role in ('assistant', 'joe') else 'user'
            text = str(item.get('content', ''))
        else:
            continue
        if not text:
            continue
        lines.append(('User' if role == 'user' else 'Joe', text))
    if not lines:
        return build_test_prompt(msg)
    included = []
    used = 0
    for role, text in reversed(lines):
        cost = len(TOK.encode(f"{role}: {text}\n"))
        if used + cost > budget:
            break
        included.append(f"{role}: {text}")
        used += cost
    included.reverse()
    history_text = "\n".join(included)
    if not history_text:
        return build_test_prompt(msg)
    return history_text + "\n" + new_turn

def _softmax(x):
    x = NP.asarray(x, dtype=float)
    x = x - x.max()
    e = NP.exp(x)
    return e / e.sum()

def generate_variant(model, prompt, max_new=60, temperature=0.5, top_k=0):
    """Sampled/greedy generation for the sweep grid. Returns (text, stopped_at_marker)."""
    ids = TOK.encode(prompt)
    STOPS = ['\nUser:', '\nJoe:']
    buf = ''
    for _ in range(max_new):
        ctx = NP.array(ids[-model.T:], dtype=NP.int32)
        logits, _ = model.forward(ctx)
        logits = logits[-1].astype(NP.float64)
        if top_k and top_k > 0 and top_k < len(logits):
            k = int(top_k)
            keep = NP.argpartition(logits, -k)[-k:]
            masked = NP.full_like(logits, -NP.inf)
            masked[keep] = logits[keep]
            logits = masked
        if temperature and temperature > 0:
            probs = _softmax(logits / temperature)
            if not NP.all(NP.isfinite(probs)):
                probs = NP.ones(len(probs)) / len(probs)
            next_id = int(NP.random.choice(len(probs), p=probs))
        else:
            next_id = int(NP.argmax(logits))
        ids.append(next_id)
        token = TOK.id_to_token.get(next_id, '')
        if not token or not all(ord(c) <= 127 for c in token):
            continue
        buf += token
        for stop in STOPS:
            if stop in buf:
                cut = buf.index(stop)
                return buf[:cut], True
    return buf, False

def _staleness():
    """Feat 48: git staleness + per-expert local steps. Robust, null on failure."""
    result = {
        'behind': False, 'local_sha': None, 'remote_sha': None,
        'remote_date': None, 'note': None, 'experts': [],
    }
    try:
        r = subprocess.run(['git', 'remote'], capture_output=True, text=True, timeout=5, cwd=BASE)
        remotes = [x.strip() for x in r.stdout.splitlines() if x.strip()]
    except Exception:
        remotes = []
    picked = next((x for x in remotes if x in ('jb', 'origin')), remotes[0] if remotes else None)
    try:
        r = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=5, cwd=BASE)
        result['local_sha'] = r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        pass
    if picked:
        try:
            r = subprocess.run(['git', 'ls-remote', picked, 'main'],
                               capture_output=True, text=True, timeout=10, cwd=BASE)
            line = r.stdout.splitlines()
            if line and line[0].split():
                result['remote_sha'] = line[0].split()[0]
        except Exception:
            pass
    tracking = None
    for ref in ('origin/main', 'jb/main'):
        try:
            r = subprocess.run(['git', 'rev-parse', '--verify', '--quiet', ref],
                               capture_output=True, text=True, timeout=5, cwd=BASE)
            if r.returncode == 0:
                tracking = ref
                break
        except Exception:
            break
    if tracking:
        behind_count = None
        try:
            r = subprocess.run(['git', 'rev-list', '--count', f'HEAD..{tracking}'],
                               capture_output=True, text=True, timeout=5, cwd=BASE)
            if r.returncode == 0:
                behind_count = int(r.stdout.strip())
        except Exception:
            pass
        remote_date = None
        try:
            r = subprocess.run(['git', 'log', tracking, '-1', '--format=%ct'],
                               capture_output=True, text=True, timeout=5, cwd=BASE)
            if r.returncode == 0 and r.stdout.strip():
                remote_date = int(r.stdout.strip())
        except Exception:
            pass
        result['remote_date'] = remote_date
        result['behind'] = (behind_count or 0) > 0
        result['note'] = (f'{behind_count} commits behind {tracking}'
                          if result['behind'] else 'up to date')
    else:
        result['note'] = 'no tracking ref; checkout a branch tracking origin/main or jb/main'
    cfg = read_experts_config()
    for e in cfg.get('experts', []):
        if not e.get('enabled', True):
            continue
        info = get_expert_info(e['name'], e)
        result['experts'].append({'name': e['name'], 'step': info['step'], 'loss': info['loss']})
    return result

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
    # On macOS, /proc doesn't exist. Use subprocess to check state + cmdline.
    try:
        result = subprocess.run(['ps', '-p', str(pid), '-o', 'state=,args='],
                                capture_output=True, text=True, timeout=2)
        out = result.stdout.strip()
        if not out:
            return False
        state = out.split(None, 1)[0]
        # Zombies are dead: kill(pid,0) and ps both still "succeed".
        if state.startswith('Z') or 'defunct' in out:
            return False
        return 'train_expert' in out
    except Exception:
        return False

def reap_children():
    """Poll/reap known Popen children so they never linger as zombies."""
    with _state_lock:
        for pid in list(child_popens):
            proc = child_popens[pid]
            try:
                rc = proc.poll()
            except (ChildProcessError, OSError):
                rc = -1
            if rc is not None:
                child_popens.pop(pid, None)

def scan_stale_locks():
    """Return names whose training.lock PID is dead or file is corrupt."""
    stale = []
    expert_dir = os.path.join(DATA, 'experts')
    if not os.path.isdir(expert_dir):
        return stale
    for fname in sorted(os.listdir(expert_dir)):
        fpath = os.path.join(expert_dir, fname, 'training.lock')
        if not os.path.isfile(fpath):
            continue
        try:
            with open(fpath) as f:
                pid = int(f.read().strip())
            if not is_process_alive(pid):
                stale.append(fname)
        except (ValueError, OSError):
            stale.append(fname)
    return stale

def find_expert_pid(name):
    """Find PID of a live train_expert.py process for this expert (cmldine scan)."""
    marker = f'train_expert.py --name {name}'
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    for line in result.stdout.splitlines()[1:]:
        if marker in line:
            parts = line.split()
            if parts:
                return int(parts[1])
    return None

def expert_is_running(name):
    if name in training and is_process_alive(training[name]['pid']):
        return True
    return find_expert_pid(name) is not None

def script_pid(marker):
    """Find PID of a running process whose cmdline contains marker."""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    for line in result.stdout.splitlines()[1:]:
        if marker in line:
            parts = line.split()
            if parts:
                return int(parts[1])
    return None

def script_running(marker):
    return script_pid(marker) is not None

def stop_script(marker):
    """SIGTERM the target process and its direct children (never the whole group)."""
    pid = script_pid(marker)
    if pid is None:
        return False
    children = []
    try:
        result = subprocess.run(['ps', '-o', 'pid=,ppid=,args='], capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 2)
            if len(parts) >= 3 and parts[1] == str(pid):
                try:
                    children.append(int(parts[0]))
                except ValueError:
                    pass
    except Exception:
        pass
    ok = False
    for p in [pid] + children:
        try:
            os.kill(p, signal.SIGTERM)
            ok = True
        except (OSError, ProcessLookupError):
            pass
    return ok

def _lock_owned_by(name, pid):
    """Check if PID is a train_expert process running this specific expert."""
    try:
        result = subprocess.run(['ps', '-p', str(pid), '-o', 'args='],
                               capture_output=True, text=True, timeout=2)
        cmdline = result.stdout
        if 'train_expert' not in cmdline:
            return False
        if '--name' in cmdline:
            args = cmdline.split()
            i = args.index('--name')
            return i + 1 < len(args) and args[i + 1] == name
        return False
    except Exception:
        return False

def count_running():
    with _state_lock:
        return sum(1 for n in training if is_process_alive(training[n]['pid']))

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
    if pid and is_process_alive(pid) and _lock_owned_by(name, pid):
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        for _ in range(60):
            if not is_process_alive(pid):
                break
            time.sleep(0.5)
        if is_process_alive(pid):
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        if os.path.exists(lock_path):
            try:
                os.remove(lock_path)
            except OSError:
                pass
        _notify_async(f"Training stopped for {name}")

def process_queue():
    """Kill excess running experts if over limit, then start queued items."""
    with _state_lock:
        if max_concurrent <= 0:
            return
        running_names = [n for n in training if is_process_alive(training[n]['pid'])]
        while len(running_names) > max_concurrent and running_names:
            victim = running_names.pop()
            _stop_expert(victim)
        failed = []
        while queue and len(running_names) < max_concurrent:
            item = queue.pop(0)
            if _start_training(item['name'], item['params']):
                running_names.append(item['name'])
            else:
                print(f"Queue: failed to start {item['name']}, holding", flush=True)
                failed.append(item)
        # Re-queue failures that aren't actually running (and not already queued)
        # so they don't get silently dropped.
        for item in failed:
            name = item['name']
            if not expert_is_running(name) and not any(q['name'] == name for q in queue):
                print(f"Queue: re-queuing {name}", flush=True)
                queue.append(item)

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
    cmd += ['--backend', 'numpy']  # numpy is always present; don't die if mlx is missing
    cmd += ['--name', name, '--steps', str(steps), '--lr', str(lr),
            '--batch', str(batch), '--seq_len', str(seq_len)]
    if resume:
        cmd.append('--resume')
    if save_every > 0:
        cmd += ['--save', str(save_every)]
    if push_every > 0:
        cmd += ['--push', str(push_every)]
    else:
        cmd += ['--push', '0']
    if log_every != 100:
        cmd += ['--log', str(log_every)]
    if sample_every > 0:
        cmd += ['--sample', str(sample_every)]

    expert_dir = os.path.join(DATA, 'experts', name)
    os.makedirs(expert_dir, exist_ok=True)
    stderr_log = None
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
        child_popens[proc.pid] = proc
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
        _notify_async(f"Training started for {name}")
        return True
    except Exception as e:
        print(f"ERROR: failed to start {name}: {e}")
        return False
    finally:
        if stderr_log is not None:
            stderr_log.close()

def parse_last_data_line(log_path):
    if not os.path.exists(log_path):
        return None
    last_data = None
    with open(log_path, 'rb') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        try:
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
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
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

_bpe_stats_cache = {}  # {name: (train_mtime_ns, tokens, chars, coverage)} -> BPE efficiency (feat 15)

def bpe_stats_for(name):
    """Feature 15: token count, chars->tokens ratio, % of 2000-token vocab used."""
    train_path = os.path.join(DATA, 'experts', name, f'{name}_train.txt')
    try:
        st = os.stat(train_path)
        key = (st.st_mtime_ns, st.st_size)
    except OSError:
        return None
    cached = _bpe_stats_cache.get(name)
    if cached and cached[0] == key:
        return cached[1]
    try:
        tokens = 0
        chars = 0
        vocab = set()
        with open(train_path) as f:
            for line in f:
                if not line.strip():
                    continue
                chars += len(line)
                if TOK is not None:
                    ids = TOK.encode(line.strip())
                    tokens += len(ids)
                    vocab.update(ids)
        stats = {
            'tokens': tokens,
            'chars': chars,
            'coverage': round(len(vocab) / 2000.0, 3) if TOK is not None else None,
        }
        _bpe_stats_cache[name] = (key, stats)
        return stats
    except Exception:
        return None

_data_hash_cache = {}  # {name: (train_mtime_ns, sha1hex)} -> retrain-recommended signal (feat 66)

def data_hash_for(name):
    train_path = os.path.join(DATA, 'experts', name, f'{name}_train.txt')
    try:
        st = os.stat(train_path)
        key = (st.st_mtime_ns, st.st_size)
    except OSError:
        return None
    cached = _data_hash_cache.get(name)
    if cached and cached[0] == key:
        return cached[1]
    try:
        h = hashlib.sha1()
        with open(train_path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 16), b''):
                h.update(chunk)
        digest = h.hexdigest()
        _data_hash_cache[name] = (key, digest)
        return digest
    except Exception:
        return None

def git_dirty_for(path):
    """Feature 83: is a file modified vs HEAD? Cheap git check."""
    try:
        r = subprocess.run(['git', 'status', '--porcelain', '--', path],
                            capture_output=True, text=True, timeout=5, cwd=BASE)
        return bool(r.stdout.strip())
    except Exception:
        return False

def fleet_resources():
    """Feature 71: CPU/RAM/disk meters for the summary bar."""
    try:
        out = subprocess.run(['ps', '-A', '-o', '%cpu=,%mem='],
                            capture_output=True, text=True, timeout=5).stdout.splitlines()
        cpu = ram = 0.0
        from collections import Counter
        for line in out:
            parts = line.split()
            if len(parts) == 2:
                try:
                    cpu += float(parts[0])
                    ram += float(parts[1])
                except ValueError:
                    pass
        disk = shutil.disk_usage(DATA)
        return {'cpu': round(cpu, 1), 'ram': round(ram, 1), 'disk_free_gb': round(disk.free / 1e9, 1)}
    except Exception:
        return None

def smart_eta(cur_step, target, speed, history=None):
    """Features 55/90/98: EMA-trend ETA (hours) to target + cost estimate."""
    if not speed or speed <= 0 or cur_step is None or target is None:
        return None
    steps_left = max(0, int(target) - int(cur_step))
    if history and len(history) >= 5:
        recent = [p.get('speed') for p in history[-12:] if p.get('speed')]
        if recent:
            speed = sum(recent) / len(recent)
    hours = (steps_left / speed) / 3600.0
    return {'steps_left': steps_left, 'hours': round(hours, 2), 'add': 'goal'}

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
        'log_key': None,
        'train_lines': 0,
        'loss_goal': cfg_entry.get('loss_goal', 0.044),
        'priority': int(cfg_entry.get('priority', 0)),
        'skip_thern': bool(cfg_entry.get('skip_thern', False)),
        'enabled': bool(cfg_entry.get('enabled', True)),
        'suite': cfg_entry.get('suite', {}),
        'paused': False,
        'ran_at': None,
        'last_step': None,
        'bpe': None,
        'data_hash': None,
        'git_dirty': False,
        'eta': None,
        'queue_pos': None,
    }
    try:
        st = os.stat(log_path)
        info['log_key'] = f'{st.st_mtime_ns}:{st.st_size}'
    except OSError:
        info['log_key'] = None

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

    # Fallback: live train_expert.py process not covered by training dict or lock
    if not info['running'] and not (time.time() - just_stopped.get(name, 0) < 20):
        live_pid = find_expert_pid(name)
        if live_pid is not None:
            info['running'] = True
            training[name] = {
                'pid': live_pid,
                'target': info['target'] or 0,
                'start_time': time.time(),
                'params': {},
            }

    # Queue state
    if not info['running']:
        for i, item in enumerate(queue):
            if item['name'] == name:
                info['queued'] = True
                info['queue_pos'] = i + 1
                break

    # Last-trained timestamp/step from log (feat 80)
    hist_path = log_path
    if os.path.exists(hist_path):
        try:
            with open(hist_path, 'rb') as f:
                for line in f:
                    pass
            with open(hist_path) as f:
                mark = None
                for line in f:
                    line = line.strip()
                    if line.startswith('#') and ('FINISH' in line or 'START' in line or 'RESUME' in line):
                        mark = line
                if mark:
                    m = re.search(r'\[([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9:]{8})\]', mark)
                    info['ran_at'] = m.group(1) if m else ''
                    sm = re.search(r'step (\d+)', mark)
                    info['last_step'] = int(sm.group(1)) if sm else None
        except Exception:
            pass

    # BPE stats + data hash + git dirty + EMA ETA (feats 15/66/83/55)
    if TOK is not None:
        info['bpe'] = bpe_stats_for(name)
    info['data_hash'] = data_hash_for(name)
    info['git_dirty'] = git_dirty_for(os.path.relpath(train_path, BASE))
    info['eta'] = smart_eta(info['step'], info['target'], info['speed'])
    info['paused'] = bool(training.get(name, {}).get('paused', False)) if info['running'] else False

    return info

def sync_state():
    cfg = read_experts_config()
    with _state_lock:
        for entry in cfg.get('experts', []):
            name = entry['name']
            lock_path = get_lock_path(name)

            if name in training:
                pid = training[name]['pid']
                if not is_process_alive(pid) and find_expert_pid(name) is None:
                    # Training finished normally - notify before cleanup
                    _notify_async(f"Training finished for {name}")
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

def cleanup_stale():
    """Remove dead processes from training dict."""
    stale = []
    with _state_lock:
        for name, info in training.items():
            if not is_process_alive(info['pid']) and find_expert_pid(name) is None:
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

def _watchdog_tick():
    """Features 18/19: divergence watchdog + auto-stop at goal. Runs under _state_lock."""
    cfg = read_experts_config()
    watcher = cfg.get('watcher', {})
    if not watcher.get('enabled', True):
        return
    sigma = float(watcher.get('divergence_sigma', 3.0))
    auto_goal = bool(watcher.get('auto_stop_at_goal', False))
    for name in list(training):
        if name not in training:
            continue
        if not is_process_alive(training[name]['pid']):
            continue
        hist = _recent_losses(name, 40)
        if not hist:
            continue
        recent = hist[-5:]
        recent_avg = sum(recent) / len(recent)
        # --- auto-stop at goal (feat 19) ---
        if auto_goal:
            entry = None
            for e in cfg.get('experts', []):
                if e['name'] == name:
                    entry = e
                    break
            goal = (entry or {}).get('loss_goal', 0.044)
            if recent_avg <= goal:
                _notify_async('auto-stop', f'"{name}" reached loss goal {goal:.4f}')
                _stop_expert(name)
                return
        # --- divergence watchdog (feat 18) ---
        if len(hist) >= 10:
            mean = sum(hist) / len(hist)
            var = sum((x - mean) ** 2 for x in hist) / len(hist)
            std = var ** 0.5
            if std > 0 and recent_avg > mean + sigma * std:
                _notify_async('auto-stop', f'"{name}" diverging (loss {recent_avg:.4f} > {mean:.4f}+{sigma}σ)')
                _stop_expert(name)

def _recent_losses(name, n):
    log_path = get_log_path(name)
    vals = []
    if not os.path.exists(log_path):
        return vals
    try:
        with open(log_path, 'rb') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                for line in f.read().decode('utf-8', errors='replace').splitlines():
                    line = line.strip()
                    if line and not line.startswith('#') and ',' in line:
                        parts = line.split(',')
                        try:
                            if len(parts) >= 2:
                                vals.append(float(parts[1]))
                        except ValueError:
                            pass
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass
    return vals[-n:]

def monitor_loop():
    while True:
        try:
            time.sleep(5)
            reap_children()
            with _state_lock:
                _now = time.time()
                for _n in [n for n, t in just_stopped.items() if _now - t > 20]:
                    just_stopped.pop(_n, None)
            sync_state()
            cleanup_stale()
            with _state_lock:
                _watchdog_tick()
        except Exception as e:
            print(f"monitor_loop error: {e}", flush=True)

monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
_model_cache = {}

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
    if name in _model_cache:
        return _model_cache[name]
    cfg = read_experts_config()
    for entry in cfg.get('experts', []):
        if entry['name'] == name:
            model_path = os.path.join(DATA, entry.get('file', ''))
            if os.path.exists(model_path):
                try:
                    model = JoeBrain.load(model_path)
                    _model_cache[name] = model
                    if len(_model_cache) > 10:
                        _model_cache.clear()
                    return model
                except Exception:
                    return None
    return None

def build_test_prompt(msg):
    new_turn = f"User: {msg}\nJoe:"
    new_ids = TOK.encode(new_turn)
    budget = 1024 - len(new_ids) - 2
    if len(new_ids) > budget:
        msg = msg[:max(1, budget - 4)]
        new_turn = f"User: {msg}\nJoe:"
        new_ids = TOK.encode(new_turn)
    return new_turn

def _load_qa_pairs(name):
    """Parse User:/Joe: Q&A pairs for an expert.

    Prefers the expert's training data ({name}_train.txt) — that's the content
    we actually edit — and falls back to the static {name}_test.txt so old
    hand-maintained lists don't go stale. Returns (pairs, source, available)
    or (None, None, 0) when nothing parseable exists.
    """
    for fname, label in ((f'{name}_train.txt', 'train'), (f'{name}_test.txt', 'test')):
        path = os.path.join(DATA, 'experts', name, fname)
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                content = f.read()
        except Exception:
            continue
        pairs = []
        current_q = None
        current_a = None
        for line in content.strip().split('\n'):
            if line.startswith('User:'):
                if current_q is not None and current_a is not None:
                    pairs.append((current_q, current_a))
                current_q = line[5:].strip()
                current_a = None
            elif line.startswith('Joe:'):
                current_a = line[4:].strip()
        if current_q and current_a:
            pairs.append((current_q, current_a))
        if pairs:
            return pairs, label, len(pairs)
    return None, None, 0

def generate_stream(model, prompt, max_new=150):
    """True per-token streaming generator.

    Yields the safe prefix on every step instead of buffering up to 46 chars,
    so the client sees text appear in real time. Holds back at most
    (max_stop_len - 1) chars so a stop marker (\\nUser: / \\nJoe:) is never
    leaked into the output even when it's split across two model tokens.
    """
    ids = TOK.encode(prompt)
    STOPS = ['\nUser:', '\nJoe:']
    reserve = max(len(s) for s in STOPS) - 1
    buf = ''
    for _ in range(max_new):
        ctx = NP.array(ids[-model.T:], dtype=NP.int32)
        logits, _ = model.forward(ctx)
        next_id = int(NP.argmax(logits[-1]))
        ids.append(next_id)
        token = TOK.id_to_token.get(next_id, '')
        if not token or not all(ord(c) <= 127 for c in token):
            continue
        buf += token
        for stop in STOPS:
            if stop in buf:
                cut = buf.index(stop)
                if cut > 0:
                    yield buf[:cut]
                return
        if len(buf) > reserve:
            yield buf[:-reserve]
            buf = buf[-reserve:]
    if buf:
        yield buf

# --- HTTP Handler ---

class TrainingHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == '/api/experts':
            self._handle_experts_list()
            return

        if parsed.path == '/ping':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'ok')
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
            seed = qs.get('seed', [''])[0].strip()
            self._handle_test_suite(name, seed=seed)
            return

        if parsed.path == '/api/test-stream':
            qs = parse_qs(parsed.query)
            name = qs.get('name', [''])[0].strip()
            seed = qs.get('seed', [''])[0].strip()
            self._handle_test_stream(name, seed=seed)
            return

        if parsed.path == '/api/events':
            self._handle_events_stream()
            return

        if parsed.path == '/api/evals':
            qs = parse_qs(parsed.query)
            name = qs.get('name', [''])[0].strip()
            self._handle_evals(name)
            return

        if parsed.path == '/api/router-info':
            self._handle_router_info()
            return

        if parsed.path == '/api/rotation':
            self._handle_rotation_get()
            return

        if parsed.path == '/api/config':
            self._handle_config_get()
            return

        if parsed.path == '/api/bundle':
            qs = parse_qs(parsed.query)
            name = qs.get('name', [''])[0].strip()
            self._handle_bundle(name)
            return

        if parsed.path == '/api/staleness':
            self._handle_staleness()
            return

        if parsed.path == '/api/paused-external':
            self._handle_paused_external()
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
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
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
        if parsed.path == '/api/thern':
            self._handle_start_thern()
            return
        if parsed.path == '/api/start_36':
            self._handle_start_36h()
            return
        if parsed.path == '/api/test':
            self._handle_test_chat(data)
            return
        if parsed.path == '/api/ask-all':
            self._handle_ask_all(data)
            return
        if parsed.path == '/api/retrain-router':
            self._handle_retrain_router()
            return
        if parsed.path == '/api/rotation':
            self._handle_rotation_post(data)
            return
        if parsed.path == '/api/queue-move':
            self._handle_queue_move(data)
            return
        if parsed.path == '/api/queue-all':
            self._handle_queue_all()
            return
        if parsed.path == '/api/pause-fleet':
            self._handle_pause_fleet(data)
            return
        if parsed.path == '/api/pause':
            self._handle_pause(data)
            return
        if parsed.path == '/api/config':
            self._handle_config_post(data)
            return
        if parsed.path == '/api/tags':
            self._handle_tags(data)
            return
        if parsed.path == '/api/sweep':
            self._handle_sweep(data)
            return
        if parsed.path == '/api/clean-shall':
            self._handle_clean_shall(data)
            return
        if parsed.path == '/api/tokenize':
            self._handle_tokenize(data)
            return
        if parsed.path == '/api/samples':
            self._handle_samples(data)
            return
        if parsed.path == '/api/merge-review':
            self._handle_merge_review(data)
            return
        if parsed.path == '/api/wizard':
            self._handle_wizard(data)
            return

        self._json({'error': 'not found'}, 404)

    def do_OPTIONS(self):
        self.send_response(200)
        for k, v in _cors_headers(self.headers.get('Origin')):
            self.send_header(k, v)
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
                'thern_running': script_running('thern.py'),
                'start36_running': script_running('start_36h.sh'),
                'watcher': cfg.get('watcher', {}),
                'resources': fleet_resources(),
                'avg_loss': (round(sum(e['loss'] for e in experts if e.get('loss') is not None) /
                                   max(1, sum(1 for e in experts if e.get('loss') is not None)), 4)
                             if any(e.get('loss') is not None for e in experts) else None),
                'total_step': sum(e.get('step') or 0 for e in experts),
            }
        # Squad health (feat 100): % of experts within 1.5x goal (feat 100)
        healthy = 0
        scored = 0
        for e in experts:
            goal = e.get('loss_goal') or 0.044
            loss = e.get('loss')
            if loss is not None:
                scored += 1
                if loss <= goal * 1.5:
                    healthy += 1
        payload['squad_health'] = round(100.0 * healthy / max(1, scored), 0)
        self._json(payload)

    def _handle_events_stream(self):
        """Feature 77: SSE live card updates replacing 5s polling."""
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        for k, v in _cors_headers(self.headers.get('Origin')):
            self.send_header(k, v)
        self.end_headers()
        try:
            self.connection.settimeout(20)
        except Exception:
            pass
        try:
            interval = 2.0
            max_duration = 600
            start = time.time()
            sent_key = None
            while time.time() - start < max_duration:
                cfg = read_experts_config()
                with _state_lock:
                    payload = self._snapshot(cfg)
                stable = str([(e['name'], e['running'], e['queued']) for e in payload['experts']])
                stats = str(payload.get('total_step')) + '|' + str(payload.get('running_count'))
                digest = stable + stats + '|' + json.dumps(payload.get('resources', {}))
                if digest != sent_key:
                    sent_key = digest
                    try:
                        self.wfile.write(b'data: ' + json.dumps(payload).encode() + b'\n\n')
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        break
                time.sleep(interval)
            try:
                self.wfile.write(b'event: close\ndata: {"reason":"max_duration"}\n\n')
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
        except Exception:
            try:
                self.wfile.write(b'event: close\ndata: {"reason":"error"}\n\n')
                self.wfile.flush()
            except Exception:
                pass

    def _snapshot(self, cfg):
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
            'thern_running': script_running('thern.py'),
            'start36_running': script_running('start_36h.sh'),
            'watcher': cfg.get('watcher', {}),
            'resources': fleet_resources(),
            'avg_loss': (round(sum(e['loss'] for e in experts if e.get('loss') is not None) /
                               max(1, sum(1 for e in experts if e.get('loss') is not None)), 4)
                         if any(e.get('loss') is not None for e in experts) else None),
            'total_step': sum(e.get('step') or 0 for e in experts),
        }
        healthy = 0
        scored = 0
        for e in experts:
            goal = e.get('loss_goal') or 0.044
            loss = e.get('loss')
            if loss is not None:
                scored += 1
                if loss <= goal * 1.5:
                    healthy += 1
        payload['squad_health'] = round(100.0 * healthy / max(1, scored), 0)
        return payload

    def _handle_log_stream(self, name):
        global _log_stream_active
        cleanup_stale()
        if not name:
            self._json({'error': 'missing name param'}, 400)
            return
        cfg = read_experts_config()
        valid_names = {e['name'] for e in cfg.get('experts', [])}
        if name not in valid_names:
            self._json({'error': 'invalid expert name'}, 400)
            return
        log_path = get_log_path(name)
        if not os.path.exists(log_path):
            self._json({'error': f'no log for {name}'}, 404)
            return

        with _log_stream_lock:
            if _log_stream_active >= MAX_LOG_STREAMS:
                self._json({'error': 'too many active log streams'}, 429)
                return
            _log_stream_active += 1
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            for k, v in _cors_headers(self.headers.get('Origin')):
                self.send_header(k, v)
            self.end_headers()
            try:
                self.connection.settimeout(120)
            except Exception:
                pass

            stream_start = time.time()
            max_duration = 120

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
                        if time.time() - stream_start > max_duration:
                            msg = json.dumps({'done': True, 'reason': 'max_duration'})
                            self.wfile.write(f'data: {msg}\n\n'.encode())
                            self.wfile.flush()
                            return
                        time.sleep(1)
            except Exception:
                # Never leave the client hanging: send a final done event.
                try:
                    msg = json.dumps({'done': True, 'reason': 'error'})
                    self.wfile.write(f'data: {msg}\n\n'.encode())
                    self.wfile.flush()
                except Exception:
                    pass
        finally:
            with _log_stream_lock:
                _log_stream_active -= 1

    def _handle_log_data(self, name, window='all'):
        if not name:
            self._json({'error': 'missing name param'}, 400)
            return
        cfg = read_experts_config()
        valid_names = {e['name'] for e in cfg.get('experts', [])}
        if name not in valid_names:
            self._json({'error': 'invalid expert name'}, 400)
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
                                loss = float(parts[1])
                                lr = float(parts[2])
                                speed = float(parts[3])
                                if not (math.isfinite(loss) and math.isfinite(lr) and math.isfinite(speed) and math.isfinite(timestamp)):
                                    continue
                                data_points.append({
                                    'step': int(parts[0]),
                                    'loss': loss,
                                    'lr': lr,
                                    'speed': speed,
                                    'time': timestamp,
                                })
                            except (ValueError, IndexError):
                                pass
        except Exception:
            pass

        now = time.time()
        if window == 'run' and data_points:
            run_start = None
            for e in history:
                if e.get('type') in ('start', 'resume') and e.get('time'):
                    try:
                        run_start = time.mktime(time.strptime(e['time'], '%Y-%m-%d %H:%M:%S'))
                    except (ValueError, OSError, OverflowError):
                        continue
            if run_start is not None:
                data_points = [d for d in data_points if d['time'] >= run_start]
            else:
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

        cfg = read_experts_config()
        valid_names = {e['name'] for e in cfg.get('experts', [])}
        if name not in valid_names:
            self._json({'error': f'invalid expert name: {name}'}, 400)
            return

        result = None
        with _state_lock:
            # Remove from queue if queued
            for i, item in enumerate(queue):
                if item['name'] == name:
                    queue.pop(i)
                    result = {'ok': True, 'removed_from_queue': True}

            process_queue()

            if result is not None:
                pass  # already set
            else:
                # Stop running
                stopped = False
                pid = None
                pid_from_training = False
                if name in training:
                    pid = training[name]['pid']
                    del training[name]
                    pid_from_training = True

                lock_path = get_lock_path(name)
                if not pid and os.path.exists(lock_path):
                    try:
                        with open(lock_path) as f:
                            pid = int(f.read().strip())
                    except (ValueError, OSError):
                        pass

                # Only kill a pid we own: from the training dict, or a lock
                # whose cmdline actually matches this expert's train_expert.
                if pid and is_process_alive(pid) and (pid_from_training or _lock_owned_by(name, pid)):
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGTERM)
                        stopped = True
                    except (ProcessLookupError, OSError):
                        pass

                if stopped:
                    just_stopped[name] = time.time()
                    # Wait (up to ~10s) for the pid to actually exit BEFORE
                    # removing the lock, so it can't be re-adopted mid-death.
                    try:
                        child_popens.pop(pid, None)
                    except Exception:
                        pass
                    for _ in range(20):
                        if not is_process_alive(pid):
                            break
                        time.sleep(0.5)
                    if os.path.exists(lock_path):
                        try:
                            os.remove(lock_path)
                        except OSError:
                            pass
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
            process_queue()
            if result is None:
                result = {'error': f'{name} not in queue'}
        self._json(result)

    def _handle_test_chat(self, data):
        self.close_connection = True
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

        history = data.get('history') or []
        window = 4
        try:
            window = max(1, min(20, int(data.get('window', 4))))
        except (TypeError, ValueError):
            window = 4
        if history:
            prompt = build_prompt(history, msg, window=window, seq_len=model.T)
        else:
            prompt = build_test_prompt(msg)
        # Sampling params (feats 11/60): temp=0 -> greedy streaming; else generate_variant
        temperature = 1.0
        top_k = 40
        max_new = 150
        try:
            temperature = float(data.get('temp', 1.0))
        except (TypeError, ValueError):
            pass
        try:
            top_k = int(data.get('top_k', 40))
        except (TypeError, ValueError):
            pass
        try:
            max_new = max(1, min(500, int(data.get('max_new', 150))))
        except (TypeError, ValueError):
            pass
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'close')
        for k, v in _cors_headers(self.headers.get('Origin')):
            self.send_header(k, v)
        self.end_headers()

        try:
            reply = ""
            token_count = 0
            start_time = time.time()

            def _emit(chunk):
                nonlocal reply, token_count
                reply += chunk
                if chunk:
                    token_count = len(TOK.encode(reply))
                elapsed = time.time() - start_time
                tok_s = token_count / elapsed if elapsed > 0 else 0
                msg_data = json.dumps({'char': chunk, 'tok_s': round(tok_s, 1), 'tokens': token_count})
                self.wfile.write(f'data: {msg_data}\n\n'.encode())
                self.wfile.flush()

            if temperature and temperature > 0:
                full, _stopped = generate_variant(model, prompt, max_new=max_new, temperature=temperature, top_k=top_k)
                for chunk in [full[i:i + 4] for i in range(0, len(full), 4)]:
                    _emit(chunk)
            else:
                for chunk in generate_stream(model, prompt, max_new=max_new):
                    _emit(chunk)
            elapsed = time.time() - start_time
            avg_tok_s = token_count / elapsed if elapsed > 0 else 0
            done_data = json.dumps({'done': True, 'reply': reply, 'avg_tok_s': round(avg_tok_s, 1), 'total_tokens': token_count})
            self.wfile.write(f'data: {done_data}\n\n'.encode())
            self.wfile.flush()
        except Exception as e:
            try:
                err_data = json.dumps({'done': True, 'error': str(e)[:500]})
                self.wfile.write(f'data: {err_data}\n\n'.encode())
                self.wfile.flush()
            except Exception:
                pass

    def _handle_start_thern(self):
        """Toggle thern.py: start if stopped, stop if running."""
        if script_running('thern.py'):
            if stop_script('thern.py'):
                self._json({'ok': True, 'stopped': True, 'running': False})
            else:
                self._json({'error': 'failed to stop thern.py'}, 500)
            return
        try:
            proc = subprocess.Popen(
                [PYTHON, os.path.join(BASE, 'thern.py')],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                cwd=BASE,
            )
            child_popens[proc.pid] = proc
            self._json({'ok': True, 'started': True, 'running': True, 'pid': proc.pid})
        except Exception as e:
            self._json({'error': f'failed to start thern.py: {e}'}, 500)

    def _handle_start_36h(self):
        """Toggle start_36h.sh: start if stopped, stop if running."""
        if script_running('start_36h.sh'):
            if stop_script('start_36h.sh'):
                self._json({'ok': True, 'stopped': True, 'running': False})
            else:
                self._json({'error': 'failed to stop start_36h.sh'}, 500)
            return
        try:
            proc = subprocess.Popen(
                ['bash', os.path.join(BASE, 'start_36h.sh')],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                cwd=BASE,
            )
            child_popens[proc.pid] = proc
            self._json({'ok': True, 'started': True, 'running': True, 'pid': proc.pid})
        except Exception as e:
            self._json({'error': f'failed to start start_36h.sh: {e}'}, 500)


    def _handle_test_suite(self, name, seed=None):
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

        sconf = suite_config_for(name)
        n_questions = sconf['n_questions']

        seed_val = None
        if seed:
            try:
                seed_val = int(seed)
            except (TypeError, ValueError):
                seed_val = None
        if seed_val is not None:
            random.seed(seed_val)

        qa, source, available = _load_qa_pairs(name)
        if qa is None:
            self._json({'error': f'no training/test file for {name}'}, 404)
            return

        query_set = random.sample(qa, min(n_questions, len(qa)))

        results = []
        correct = 0
        for question, expected in query_set:
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

        try:
            _append_eval(name, {
                'ts': time.time(),
                'seed': seed_val,
                'correct': correct,
                'total': total,
                'accuracy': accuracy,
                'source': source,
            })
        except Exception as e:
            print(f"ERROR: persist eval for {name}: {e}", flush=True)

        self._json({
            'name': name,
            'source': source,
            'available': available,
            'sampled': len(query_set),
            'correct': correct,
            'total': total,
            'accuracy': round(accuracy, 1),
            'pass_pct': sconf['pass_pct'],
            'seed': seed_val,
            'results': results,
        })

    def _handle_test_stream(self, name, seed=None):
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

        sconf = suite_config_for(name)
        n_questions = sconf['n_questions']
        pass_pct = sconf['pass_pct']

        seed_val = None
        if seed:
            try:
                seed_val = int(seed)
            except (TypeError, ValueError):
                seed_val = None
        if seed_val is not None:
            random.seed(seed_val)

        qa, source, available = _load_qa_pairs(name)
        if qa is None:
            self._json({'error': f'no training/test file for {name}'}, 404)
            return

        query_set = random.sample(qa, min(n_questions, len(qa)))

        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'close')
        for k, v in _cors_headers(self.headers.get('Origin')):
            self.send_header(k, v)
        self.end_headers()

        accuracy = 0.0
        try:
            correct = 0
            total = len(query_set)
            count = 0
            for question, expected in query_set:
                count += 1
                prompt = build_test_prompt(question)
                actual = ''
                tok_count = 0
                q_start = time.time()
                try:
                    for chunk in generate_stream(model, prompt):
                        actual += chunk
                        if chunk:
                            tok_count = len(TOK.encode(actual))
                except Exception:
                    pass
                is_correct = _keyword_match(expected, actual)
                if is_correct:
                    correct += 1
                elapsed = time.time() - q_start
                avg_tok_s = tok_count / elapsed if elapsed > 0 else 0
                event = json.dumps({
                    'count': count,
                    'source': source,
                    'available': available,
                    'question': question,
                    'expected': expected,
                    'actual': actual,
                    'correct': is_correct,
                    'correct_so_far': correct,
                    'total': total,
                    'sampled': len(query_set),
                    'n_questions': n_questions,
                    'pass_pct': pass_pct,
                    'seed': seed_val,
                    'tok_s': round(avg_tok_s, 1),
                    'tokens': tok_count,
                })
                self.wfile.write(f'data: {event}\n\n'.encode())
                self.wfile.flush()

            accuracy = round((correct / total * 100) if total else 0, 1)
            done = json.dumps({'done': True, 'correct': correct, 'total': total,
                               'accuracy': accuracy, 'pass_pct': pass_pct, 'seed': seed_val})
            self.wfile.write(f'data: {done}\n\n'.encode())
            self.wfile.flush()
        except Exception as e:
            try:
                err_data = json.dumps({'done': True, 'error': str(e)[:500]})
                self.wfile.write(f'data: {err_data}\n\n'.encode())
                self.wfile.flush()
            except Exception:
                pass

        # Persist result to data/_eval/{name}.json (feature 1: suite persistence/history)
        try:
            _append_eval(name, {
                'ts': time.time(),
                'seed': seed_val,
                'correct': correct,
                'total': total,
                'accuracy': accuracy,
                'source': source,
            })
        except Exception as e:
            print(f"ERROR: persist eval for {name}: {e}", flush=True)


    def _handle_evals(self, name=''):
        """GET /api/evals[?name=] — persisted suite history."""
        if name:
            cfg = read_experts_config()
            valid_names = {e['name'] for e in cfg.get('experts', [])}
            if name not in valid_names:
                self._json({'error': f'invalid expert name: {name}'}, 400)
                return
            recs = []
            path = os.path.join(EVAL_DIR, f'{name}.json')
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        recs = json.load(f)
                    if not isinstance(recs, list):
                        recs = []
                except Exception:
                    recs = []
            self._json(recs)
            return
        cfg = read_experts_config()
        out = {}
        for e in cfg.get('experts', []):
            if not e.get('enabled', True):
                continue
            recs = []
            path = os.path.join(EVAL_DIR, f"{e['name']}.json")
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        recs = json.load(f)
                    if not isinstance(recs, list):
                        recs = []
                except Exception:
                    recs = []
            out[e['name']] = recs
        self._json(out)

    def _handle_router_info(self):
        self._json({'exists': os.path.exists(ROUTER_PATH)})

    def _handle_rotation_get(self):
        cfg = read_experts_config()
        experts = []
        for e in cfg.get('experts', []):
            experts.append({
                'name': e['name'],
                'priority': int(e.get('priority', 0)),
                'skip_thern': bool(e.get('skip_thern', False)),
                'enabled': bool(e.get('enabled', True)),
            })
        order = [e['name'] for e in sorted(experts, key=lambda x: (x['priority'], x['name']))]
        self._json({'experts': experts, 'order': order})

    def _handle_rotation_post(self, data):
        name = (data.get('name') or '').strip()
        cfg = read_experts_config()
        valid = {e['name'] for e in cfg.get('experts', [])}
        if name not in valid:
            self._json({'error': f'invalid expert name: {name}'}, 400)
            return
        changed = False
        for e in cfg['experts']:
            if e['name'] != name:
                continue
            if data.get('priority') is not None:
                try:
                    e['priority'] = max(0, min(1000, int(data['priority'])))
                    changed = True
                except (TypeError, ValueError):
                    self._json({'error': 'priority must be an int'}, 400)
                    return
            if data.get('skip_thern') is not None:
                e['skip_thern'] = bool(data['skip_thern'])
                changed = True
            break
        if not changed:
            self._json({'error': 'nothing to update (provide priority or skip_thern)'}, 400)
            return
        try:
            _save_experts_json(cfg)
        except Exception as ex:
            self._json({'error': f'failed to write config: {ex}'}, 500)
            return
        self._json({'ok': True})

    def _handle_config_get(self):
        cfg = read_experts_config()
        self._json({'quality_filter': cfg.get('quality_filter', {}),
                    'watcher': cfg.get('watcher', {})})

    def _handle_config_post(self, data):
        cfg = read_experts_config()
        qf = data.get('quality_filter')
        if isinstance(qf, dict):
            cfg['quality_filter'] = {**cfg.get('quality_filter', {}), **qf}
        w = data.get('watcher')
        if isinstance(w, dict):
            cfg['watcher'] = {**cfg.get('watcher', {}), **w}
        if not isinstance(qf, dict) and not isinstance(w, dict):
            self._json({'error': 'provide quality_filter and/or watcher'}, 400)
            return
        try:
            _save_experts_json(cfg)
        except Exception as ex:
            self._json({'error': f'failed to write config: {ex}'}, 500)
            return
        self._json({'ok': True, 'quality_filter': cfg.get('quality_filter', {}),
                    'watcher': cfg.get('watcher', {})})

    def _handle_bundle(self, name):
        if not name:
            self._json({'error': 'missing name param'}, 400)
            return
        cfg = read_experts_config()
        valid_names = {e['name'] for e in cfg.get('experts', [])}
        if name not in valid_names:
            self._json({'error': f'invalid expert name: {name}'}, 400)
            return
        src_dir = os.path.join(DATA, 'experts', name)
        if not os.path.isdir(src_dir):
            self._json({'error': f'no directory for {name}'}, 404)
            return
        wanted = {f'{name}.npz', f'{name}_train.txt', 'training.log', 'train_stderr.log'}
        found = []
        for root, _dirs, files in os.walk(src_dir):
            for fn in files:
                if fn in wanted:
                    found.append(os.path.join(root, fn))
        if not found:
            self._json({'error': f'no bundle files for {name}'}, 404)
            return
        try:
            bio = io.BytesIO()
            with zipfile.ZipFile(bio, 'w', zipfile.ZIP_DEFLATED) as zf:
                for fp in sorted(found):
                    zf.write(fp, arcname=os.path.basename(fp))
        except Exception as ex:
            self._json({'error': f'zip failed: {ex}'}, 500)
            return
        data_b = bio.getvalue()
        self.send_response(200)
        self.send_header('Content-Type', 'application/zip')
        self.send_header('Content-Length', len(data_b))
        self.send_header('Content-Disposition', f'attachment; filename="{name}.zip"')
        for k, v in _cors_headers(self.headers.get('Origin')):
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(data_b)
        except BrokenPipeError:
            pass

    def _handle_staleness(self):
        self._json(_staleness())

    def _handle_paused_external(self):
        # console_watch.py (grep'd) has NO marker/state mechanism — only the
        # mom_on_console() check. No invented mechanism here.
        self._json({'paused': False, 'mechanism': None})

    def _handle_ask_all(self, data):
        msg = (data.get('msg') or '').strip()
        if not msg:
            self._json({'error': 'missing msg'}, 400)
            return
        if TOK is None:
            self._json({'error': 'tokenizer not loaded'}, 500)
            return
        cfg = read_experts_config()
        names = [e['name'] for e in cfg.get('experts', []) if e.get('enabled', True)]
        router = load_router()
        router_probs = None
        router_picked = None
        if router is not None:
            router_probs = {}
            try:
                ids = TOK.encode(msg.lower())
                if not ids:
                    ids = [1]
                probs = router.predict(ids)
                for i, n in enumerate(ROUTER_NAMES):
                    if i < len(probs):
                        router_probs[n] = float(probs[i])
                top_idx = int(NP.argmax(probs))
                if top_idx < len(ROUTER_NAMES):
                    router_picked = ROUTER_NAMES[top_idx]
            except Exception as ex:
                print(f"ask-all router warn: {ex}")
                router_probs = None
        history = data.get('history') or []
        experts = []
        for name in names:
            entry = {'name': name, 'reply': None, 'router_prob': None, 'error': None}
            if router_probs:
                entry['router_prob'] = router_probs.get(name)
            try:
                model = load_expert_model(name)
                if model is None:
                    entry['error'] = 'model load failed'
                else:
                    if history:
                        prompt = build_prompt(history, msg, seq_len=model.T)
                    else:
                        prompt = build_test_prompt(msg)
                    entry['reply'] = ''.join(generate_stream(model, prompt, 150))
            except Exception as ex:
                entry['error'] = str(ex)[:300]
            experts.append(entry)
        self._json({'experts': experts, 'router': router_probs,
                    'router_picked': router_picked})

    def _handle_retrain_router(self):
        try:
            os.makedirs(os.path.join(DATA, 'router'), exist_ok=True)
            log_path = os.path.join(DATA, 'router', 'train.log')
            logf = open(log_path, 'a')
            try:
                proc = subprocess.Popen(
                    [PYTHON, os.path.join(BASE, 'training', 'train_router.py')],
                    stdout=logf, stderr=subprocess.STDOUT,
                    cwd=BASE, start_new_session=True,
                )
            except Exception:
                logf.close()
                raise
            logf.close()
            child_popens[proc.pid] = proc
            _router_state['tried'] = False
            _router_state['obj'] = None
            self._json({'ok': True, 'pid': proc.pid, 'log': log_path})
        except Exception as ex:
            self._json({'error': f'failed to start router training: {ex}'}, 500)

    def _handle_queue_move(self, data):
        name = (data.get('name') or '').strip()
        delta = data.get('delta')
        direction = (data.get('direction') or '').strip().lower()
        if delta is None and direction:
            delta = -1 if direction == 'up' else (1 if direction == 'down' else None)
        try:
            delta = int(delta)
        except (TypeError, ValueError):
            self._json({'error': 'delta or direction required'}, 400)
            return
        if delta == 0:
            self._json({'error': 'delta must be nonzero'}, 400)
            return
        with _state_lock:
            idx = None
            for i, item in enumerate(queue):
                if item['name'] == name:
                    idx = i
                    break
            if idx is None:
                self._json({'error': f'{name} not in queue'}, 400)
                return
            ni = min(max(0, idx + delta), len(queue) - 1)
            item = queue.pop(idx)
            queue.insert(ni, item)
            qnames = [q['name'] for q in queue]
        self._json({'ok': True, 'queue': qnames})

    def _handle_queue_all(self):
        with _state_lock:
            cfg = read_experts_config()
            queued = []
            for e in cfg.get('experts', []):
                if not e.get('enabled', True):
                    continue
                name = e['name']
                if expert_is_running(name) or any(q['name'] == name for q in queue):
                    continue
                info = get_expert_info(name, e)
                goal = float(e.get('loss_goal', 0.044))
                loss = info.get('loss')
                if loss is None or loss > goal:
                    queue.append({'name': name, 'params': {}})
                    queued.append(name)
            qnames = [q['name'] for q in queue]
            process_queue()
        self._json({'ok': True, 'queued': queued, 'queue': qnames})

    def _handle_pause_fleet(self, data):
        paused = bool(data.get('paused', False))
        sig = signal.SIGSTOP if paused else signal.SIGCONT
        n = 0
        with _state_lock:
            for nm in list(training):
                t = training[nm]
                if is_process_alive(t['pid']):
                    try:
                        os.kill(t['pid'], sig)
                        t['paused'] = paused
                        n += 1
                    except (OSError, ProcessLookupError):
                        pass
        self._json({'ok': True, 'paused': paused, 'n_signaled': n})

    def _handle_pause(self, data):
        name = (data.get('name') or '').strip()
        paused = bool(data.get('paused', False))
        cfg = read_experts_config()
        valid = {e['name'] for e in cfg.get('experts', [])}
        if name not in valid:
            self._json({'error': f'invalid expert name: {name}'}, 400)
            return
        with _state_lock:
            pid = None
            if name in training:
                pid = training[name]['pid']
                if not is_process_alive(pid):
                    pid = None
            if pid is None:
                lock_path = get_lock_path(name)
                if os.path.exists(lock_path):
                    try:
                        with open(lock_path) as f:
                            pid = int(f.read().strip())
                        if not (is_process_alive(pid) and _lock_owned_by(name, pid)):
                            pid = None
                    except (ValueError, OSError):
                        pid = None
            if pid is None:
                self._json({'error': f'{name} is not training'}, 400)
                return
            sig = signal.SIGSTOP if paused else signal.SIGCONT
            try:
                os.kill(pid, sig)
            except (OSError, ProcessLookupError):
                self._json({'error': f'failed to signal pid {pid}'}, 500)
                return
            if name in training:
                training[name]['paused'] = paused
        self._json({'ok': True, 'paused': paused, 'pid': pid})

    def _handle_clean_shall(self, data):
        """Remove stale training.lock files (dead PID or corrupt), report count."""
        removed = []
        for nm in scan_stale_locks():
            if find_expert_pid(nm) is not None:
                continue
            lock_path = get_lock_path(nm)
            try:
                os.remove(lock_path)
                removed.append(nm)
            except OSError:
                pass
        if removed:
            print(f"Cleaned {len(removed)} stale lock(s): {', '.join(removed)}")
        self._json({'ok': True, 'cleaned': removed})

    def _handle_tags(self, data):
        name = (data.get('name') or '').strip()
        note = (data.get('note') or '').strip()
        if not name or not note:
            self._json({'error': 'missing name or note'}, 400)
            return
        cfg = read_experts_config()
        valid = {e['name'] for e in cfg.get('experts', [])}
        if name not in valid:
            self._json({'error': f'invalid expert name: {name}'}, 400)
            return
        step = data.get('step')
        step_str = f" step={int(step)}" if step is not None else ''
        try:
            ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_path = get_log_path(name)
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'a') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(f"# TAG [{ts}]{step_str} {note}\n")
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as ex:
            self._json({'error': f'failed to write tag: {ex}'}, 500)
            return
        self._json({'ok': True})

    def _handle_sweep(self, data):
        msg = (data.get('msg') or '').strip()
        expert = (data.get('expert') or '').strip()
        temps = data.get('temps') or []
        topks = data.get('topks') or []
        if not msg:
            self._json({'error': 'missing msg'}, 400)
            return
        if TOK is None:
            self._json({'error': 'tokenizer not loaded'}, 500)
            return
        cfg = read_experts_config()
        valid = {e['name'] for e in cfg.get('experts', []) if e.get('enabled', True)}
        if expert not in valid:
            self._json({'error': f'unknown expert: {expert}'}, 400)
            return
        model = load_expert_model(expert)
        if model is None:
            self._json({'error': f'could not load model for {expert}'}, 500)
            return
        clean_temps = []
        for t in temps:
            try:
                clean_temps.append(float(t))
            except (TypeError, ValueError):
                pass
        if not clean_temps:
            clean_temps = [0.5]
        clean_topks = []
        for k in topks:
            try:
                clean_topks.append(max(0, int(k)))
            except (TypeError, ValueError):
                pass
        if not clean_topks:
            clean_topks = [0]
        while len(clean_temps) * len(clean_topks) > 16:
            if len(clean_temps) >= len(clean_topks):
                clean_temps.pop()
            else:
                clean_topks.pop()
        try:
            max_new = max(1, min(120, int(data.get('max_new', 60))))
        except (TypeError, ValueError):
            max_new = 60
        prompt = build_test_prompt(msg)
        grid = []
        for t in clean_temps:
            for k in clean_topks:
                text = ''
                escaped = False
                try:
                    text, escaped = generate_variant(model, prompt, max_new, t, k)
                except Exception as ex:
                    print(f"sweep variant warn: {ex}")
                toks = TOK.encode(text)
                if len(toks) >= 2:
                    bg = list(zip(toks, toks[1:]))
                    repeat = round(1.0 - (len(set(bg)) / len(bg)), 3)
                else:
                    repeat = 0.0
                grid.append({
                    'temp': t, 'top_k': k, 'text': text,
                    'tokens': len(toks),
                    'repeat_score': repeat,
                    'stop_escape': escaped,
                })
        self._json({'ok': True, 'expert': expert, 'msg': msg, 'grid': grid})

    def _handle_tokenize(self, data):
        text = (data.get('text') or '').strip()
        if not text:
            self._json({'error': 'missing text'}, 400); return
        ids = TOK.encode(text)
        tokens = [{'token': TOK.id_to_token.get(i, ''), 'id': i} for i in ids]
        self._json({'ok': True, 'tokens': tokens, 'count': len(tokens)})

    def _handle_samples(self, data):
        name = (data.get('name') or '').strip()
        cfg = read_experts_config()
        valid = {e['name'] for e in cfg.get('experts', []) if e.get('enabled', True)}
        if name not in valid:
            self._json({'error': f'unknown expert: {name}'}, 400); return
        limit = data.get('limit')
        args = [sys.executable, os.path.join(BASE, 'training', 'panel_sample_gallery.py'), '--name', name]
        if limit: args += ['--limit', str(int(limit))]
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                self._json({'error': r.stderr[:500]}, 500); return
            samples = json.loads(r.stdout.strip() or '[]')
            self._json({'ok': True, 'samples': samples})
        except subprocess.TimeoutExpired:
            self._json({'error': 'timeout'}, 504)
        except Exception as e:
            self._json({'error': str(e)[:500]}, 500)

    def _handle_merge_review(self, data):
        action = (data.get('action') or '').strip()
        args = [sys.executable, os.path.join(BASE, 'training', 'panel_merge_review.py')]
        if action == 'list':
            args += ['--list']
        elif action == 'show':
            chunk = (data.get('chunk') or '').strip()
            if not chunk: self._json({'error': 'missing chunk'}, 400); return
            args += ['--show', chunk]
        elif action == 'verdict':
            chunk = (data.get('chunk') or '').strip()
            idx = data.get('index')
            verdict = (data.get('verdict') or '').strip()
            if not chunk or idx is None or verdict not in ('ACCEPT','REJECT'):
                self._json({'error': 'missing chunk/index/verdict'}, 400); return
            reviewer = (data.get('reviewer') or 'v1').strip()
            args += ['--verdict', chunk, str(int(idx)), verdict]
            if reviewer: args += ['--reviewer', reviewer]
        elif action == 'merge':
            expert = (data.get('expert') or '').strip()
            if not expert: self._json({'error': 'missing expert'}, 400); return
            args += ['--merge', expert]
        else:
            self._json({'error': 'unknown action'}, 400); return
        args += ['--json']
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                self._json({'error': r.stderr[:500]}, 500); return
            result = json.loads(r.stdout.strip() or '[]')
            self._json({'ok': True, 'result': result})
        except subprocess.TimeoutExpired:
            self._json({'error': 'timeout'}, 504)
        except Exception as e:
            self._json({'error': str(e)[:500]}, 500)

    def _handle_wizard(self, data):
        new_name = (data.get('new_name') or '').strip()
        if not re.match(r'^[a-z_][a-z0-9_]*$', new_name):
            self._json({'error': 'invalid name (lowercase _/digits, start with letter)'}, 400); return
        steps = data.get('steps')
        force = bool(data.get('force', False))
        args = [sys.executable, os.path.join(BASE, 'training', 'panel_wizard.py'), '--new-name', new_name]
        if steps: args += ['--steps', str(int(steps))]
        if force: args += ['--force']
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=300)
            lines = []
            for ln in r.stdout.strip().splitlines():
                ln = ln.strip()
                if ln.startswith('{'):
                    try: lines.append(json.loads(ln))
                    except: lines.append({'cmd': ln[:60], 'ok': None})
            self._json({'ok': True, 'steps': lines, 'rc': r.returncode, 'stderr': r.stderr[-500:]})
        except subprocess.TimeoutExpired:
            self._json({'error': 'wizard timed out after 300s'}, 504)
        except Exception as e:
            self._json({'error': str(e)[:500]}, 500)

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(data))
        for k, v in _cors_headers(self.headers.get('Origin')):
            self.send_header(k, v)
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

    class DualStackServer(ThreadingHTTPServer):
        address_family = socket.AF_INET6
        def server_bind(self):
            try:
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            except (AttributeError, OSError):
                pass
            super().server_bind()

    import threading as _threading
    _servers = []
    def _run_server(srv):
        srv.serve_forever()

    try:
        _srv6 = DualStackServer(('::1', PORT), TrainingHandler)
        _t6 = _threading.Thread(target=_run_server, args=(_srv6,), daemon=True)
        _t6.start()
        _servers.append(_srv6)
        print(f"IPv6 server on [::1]:{PORT}", flush=True)
    except OSError as e:
        print(f"IPv6 server failed: {e}", flush=True)

    try:
        _srv4 = ThreadingHTTPServer(('127.0.0.1', PORT), TrainingHandler)
        _t4 = _threading.Thread(target=_run_server, args=(_srv4,), daemon=True)
        _t4.start()
        _servers.append(_srv4)
        print(f"IPv4 server on [IP_ADDRESS]:{PORT}", flush=True)
    except OSError as e:
        print(f"IPv4 server failed: {e}", flush=True)

    if not _servers:
        print(f"FATAL: could not bind port {PORT}", flush=True)
        sys.exit(1)
    print(f"Training Dashboard running at http://localhost:{PORT}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    except Exception as e:
        print(f"SERVER ERROR: {e}", flush=True)
        try:
            for _s in _servers: _s.server_close()
        except Exception:
            pass
