import json
import os
import time
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TARGET_HOST = "127.0.0.1"
TARGET_PORT = 10000
LISTEN_PORT = 10001
LOG_DIR = r"C:\AI\logs\llama-requests"

os.makedirs(LOG_DIR, exist_ok=True)

class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)

        stamp = int(time.time() * 1000)
        filename = os.path.join(LOG_DIR, f"{stamp}.json")
        with open(filename, "wb") as f:
            f.write(body)

        print(f"\n>>> POST {self.path}  body={len(body)} bytes  file={filename}")

        try:
            data = json.loads(body)
            msgs = data.get("messages", [])

            print(f"    messages={len(msgs)}")

            if msgs:
                print(f"    FIRST: role={msgs[0].get('role')!r}")
                print(f"    LAST : role={msgs[-1].get('role')!r}")

                user_count = sum(1 for m in msgs if m.get("role") == "user")
                tool_count = sum(1 for m in msgs if m.get("role") == "tool")
                assistant_count = sum(1 for m in msgs if m.get("role") == "assistant")

                print(
                    f"    users={user_count} "
                    f"assistants={assistant_count} "
                    f"tools={tool_count}"
                )
        except Exception as e:
            print(f"    JSON parse error: {e}")

        conn = http.client.HTTPConnection(
            TARGET_HOST,
            TARGET_PORT,
            timeout=600
        )

        try:
            headers = {}

            for k, v in self.headers.items():
                if k.lower() not in (
                    "host",
                    "content-length",
                    "connection",
                    "accept-encoding",
                ):
                    headers[k] = v

            headers["Host"] = f"{TARGET_HOST}:{TARGET_PORT}"
            headers["Content-Length"] = str(len(body))

            conn.request(
                "POST",
                self.path,
                body=body,
                headers=headers
            )

            resp = conn.getresponse()

            print(f"<<< RESPONSE {resp.status} {resp.reason} for POST {self.path}")

            self.send_response(resp.status, resp.reason)

            response_headers = {
                k.lower(): (k, v)
                for k, v in resp.getheaders()
            }

            for lk, (k, v) in response_headers.items():
                if lk in ("connection", "transfer-encoding"):
                    continue
                self.send_header(k, v)

            if "transfer-encoding" in response_headers:
                self.send_header(
                    "Transfer-Encoding",
                    response_headers["transfer-encoding"][1]
                )
            elif "content-length" in response_headers:
                self.send_header(
                    "Content-Length",
                    response_headers["content-length"][1]
                )

            self.end_headers()

            transfer = response_headers.get(
                "transfer-encoding", (None, "")
            )[1].lower()

            if "chunked" in transfer:
                while True:
                    line = resp.fp.readline()
                    if not line:
                        break

                    self.wfile.write(line)

                    size_text = line.split(b";", 1)[0].strip()

                    try:
                        size = int(size_text, 16)
                    except ValueError:
                        break

                    if size == 0:
                        while True:
                            trailer = resp.fp.readline()
                            if not trailer or trailer == b"\r\n":
                                self.wfile.flush()
                                return
                            self.wfile.write(trailer)

                    remaining = size + 2

                    while remaining > 0:
                        chunk = resp.fp.read(min(65536, remaining))
                        if not chunk:
                            return

                        self.wfile.write(chunk)
                        remaining -= len(chunk)

                    self.wfile.flush()

            else:
                while True:
                    chunk = resp.read(65536)

                    if not chunk:
                        break

                    self.wfile.write(chunk)
                    self.wfile.flush()

        except Exception as e:
            print(f"!!! PROXY ERROR: {type(e).__name__}: {e}")

        finally:
            conn.close()

    def do_GET(self):
        print(f"\n>>> GET {self.path}")

        conn = http.client.HTTPConnection(
            TARGET_HOST,
            TARGET_PORT,
            timeout=60
        )

        try:
            conn.request(
                "GET",
                self.path,
                headers={"Host": f"{TARGET_HOST}:{TARGET_PORT}"}
            )

            resp = conn.getresponse()

            print(f"<<< RESPONSE {resp.status} {resp.reason} for GET {self.path}")

            self.send_response(resp.status, resp.reason)

            for k, v in resp.getheaders():
                if k.lower() in ("connection", "transfer-encoding"):
                    continue
                self.send_header(k, v)

            self.end_headers()

            while True:
                chunk = resp.read(65536)

                if not chunk:
                    break

                self.wfile.write(chunk)
                self.wfile.flush()

        finally:
            conn.close()

    def log_message(self, fmt, *args):
        pass

print("Proxy listening on http://127.0.0.1:10001")
print("Forwarding to http://127.0.0.1:10000")

ThreadingHTTPServer(("127.0.0.1", LISTEN_PORT), Proxy).serve_forever()
