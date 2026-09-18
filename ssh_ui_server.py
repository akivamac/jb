"""
JoeBrain SSH-Aware Training Dashboard Gateway

Runs on port 6666 on the Mac. Tablet accesses via localhost:6666
(through SSH tunnel). Checks SSH connectivity and starts the
training server if needed. Serves the gateway page with state.
"""

import json
import os
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

BASE = os.path.dirname(os.path.abspath(__file__))
PORT = 6666
TRAINING_PORT = 9091
SSH_USER = 'dev'
SSH_HOST = 'localhost'
INTERNET_CHECK_HOST = '8.8.8.8'
GATEWAY_HTML = os.path.join(BASE, 'ssh_ui.html')

_training_server_proc = None
_training_server_lock = threading.Lock()


def check_ssh(timeout=3):
    try:
        result = subprocess.run(
            ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=' + str(timeout),
             '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null',
             f'{SSH_USER}@{SSH_HOST}', 'echo', 'ok'],
            capture_output=True, text=True, timeout=timeout + 2
        )
        return result.returncode == 0 and 'ok' in result.stdout
    except Exception:
        return False


def check_internet(timeout=3):
    try:
        result = subprocess.run(
            ['curl', '-s', '--max-time', str(timeout), '--connect-timeout', str(timeout),
             'https://www.google.com'],
            capture_output=True, text=True, timeout=timeout + 2
        )
        if result.returncode == 0:
            return True
        result2 = subprocess.run(
            ['ping', '-c', '1', '-W', str(timeout), '8.8.8.8'],
            capture_output=True, text=True, timeout=timeout + 2
        )
        return result2.returncode == 0
    except Exception:
        return False


def is_training_server_running():
    try:
        result = subprocess.run(
            ['curl', '-s', '--max-time', '1', f'http://localhost:{TRAINING_PORT}/ping'],
            capture_output=True, text=True, timeout=2
        )
        return result.returncode == 0 and 'ok' in result.stdout
    except Exception:
        return False


def ensure_training_server():
    global _training_server_proc
    with _training_server_lock:
        if is_training_server_running():
            return True
        server_path = os.path.join(BASE, 'training', 'training_server.py')
        if not os.path.exists(server_path):
            return False
        try:
            _training_server_proc = subprocess.Popen(
                ['python3', '-u', server_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            time.sleep(2)
            return is_training_server_running()
        except Exception:
            return False


def get_state():
    if check_ssh():
        ensure_training_server()
        return 'ssh_ok'
    if check_internet():
        return 'no_ssh'
    return 'no_internet'


class GatewayHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/ping':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'ok')
            return
        if self.path == '/api/state':
            state = get_state()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'state': state}).encode())
            return
        if self.path == '/api/training':
            running = is_training_server_running()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'running': running}).encode())
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        try:
            with open(GATEWAY_HTML) as f:
                html = f.read()
            html = html.replace('__STATE__', json.dumps(get_state()))
            self.wfile.write(html.encode())
        except Exception:
            self.wfile.write(b'<html><body><h1>Gateway error</h1></body></html>')

    def log_message(self, format, *args):
        pass


def main():
    server = ThreadingHTTPServer(('', PORT), GatewayHandler)
    print(f"[gateway] SSH training gateway running on port {PORT}", flush=True)
    print(f"[gateway] SSH check: {SSH_USER}@{SSH_HOST}", flush=True)
    print(f"[gateway] Training server port: {TRAINING_PORT}", flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
