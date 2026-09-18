"""
JoeBrain SSH-Aware Training Dashboard Gateway

Runs on port 6666 on the Mac. Tablet accesses via localhost:6666
(through SSH tunnel). When SSH is connected, proxies to the
training server on port 9091 so the tablet sees the training
dashboard directly. When SSH is not available, shows a status
page with connection instructions.
"""

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

BASE = os.path.dirname(os.path.abspath(__file__))
PORT = 8888
TRAINING_PORT = 9091
SSH_USER = 'dev'
SSH_HOST = 'localhost'
INTERNET_CHECK_HOST = '[IP_ADDRESS]'
GATEWAY_HTML = os.path.join(BASE, 'ssh_ui.html')

_training_server_proc = None
_training_server_lock = threading.Lock()


def check_ssh(timeout=3):
    try:
        result = subprocess.run(
            ['ps', 'aux'], capture_output=True, text=True, timeout=2
        )
        return 'sshd' in result.stdout
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
            ['ping', '-c', '1', '-W', str(timeout), '[IP_ADDRESS]'],
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
        server_path = os.path.join(BASE, 'training_server.py')
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


def proxy_to_training(path, method='GET', body=None):
    """Proxy a request to the training server on localhost:9091."""
    try:
        url = f'http://localhost:{TRAINING_PORT}{path}'
        data = body.encode() if body else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header('Content-Type', 'application/json')
        with urllib.request.urlopen(req, timeout=5) as resp:
            resp_body = resp.read()
            headers = dict(resp.headers)
            return resp.status, headers, resp_body
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()
    except Exception:
        return None, None, None


class GatewayHandler(BaseHTTPRequestHandler):
    def _proxy(self, method):
        # Gateway's own API endpoints (never proxied)
        if self.path == '/ping':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'ok')
            return
        if self.path == '/api/state':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'state': get_state()}).encode())
            return
        if self.path == '/api/training':
            running = is_training_server_running()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'running': running}).encode())
            return

        # If SSH is connected, proxy to training server
        if check_ssh():
            content_length = int(self.headers.get('Content-Length', 0))
            post_body = self.rfile.read(content_length).decode() if content_length > 0 else None
            status, headers, resp_body = proxy_to_training(self.path, method, post_body)
            if status is not None:
                self.send_response(status)
                for key, val in headers.items():
                    if key.lower() not in ('content-length', 'transfer-encoding', 'connection'):
                        self.send_header(key, val)
                self.send_header('Content-Length', str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
                return

        # SSH not connected — serve gateway HTML (GET only)
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

    def do_GET(self):
        self._proxy('GET')

    def do_POST(self):
        self._proxy('POST')

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

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
