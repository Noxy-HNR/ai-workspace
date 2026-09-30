"""Disposable loopback service for launcher lifecycle verification.

This fixture has no bot database, Telegram client, model client, or credentials.
It is copied into a temporary folder as app.py by test-management-lifecycle.ps1.
"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


FIXTURE_CSRF = "public-disposable-fixture-token"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_json(self, body, status=200):
        encoded = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)
        self.wfile.flush()

    def trusted(self, mutate=False):
        local = {f"127.0.0.1:{self.server.server_port}",
                 f"localhost:{self.server.server_port}"}
        if self.headers.get("Host") not in local:
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {"http://" + host for host in local}:
            return False
        return not mutate or self.headers.get("X-Relay-CSRF") == FIXTURE_CSRF

    def do_GET(self):
        if not self.trusted():
            return self.send_json({"error": "Untrusted request"}, 403)
        if self.path == "/api/health":
            self.send_json({"service": "relay-station", "version": "fixture-lifecycle"})
        elif self.path == "/api/session":
            self.send_json({"csrf": FIXTURE_CSRF})
        elif self.path == "/api/poller":
            self.send_json({"poller": {"running": False}})
        else:
            self.send_json({"error": "Unknown fixture route"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        if 0 <= length <= 8192:
            self.rfile.read(length)
        else:
            return self.send_json({"error": "Invalid fixture body length"}, 400)
        if not self.trusted(mutate=True):
            return self.send_json({"error": "Untrusted request"}, 403)
        if self.path != "/api/shutdown":
            return self.send_json({"error": "Unknown fixture route"}, 404)
        self.send_json({"ok": True, "shutting_down": True})
        threading.Thread(target=self.server.shutdown, daemon=True).start()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
