"""Loopback-only NPU service. Run with the Lecture Notes venv."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import runtime as runtime_module
from runtime import Runtime

runtime = Runtime()
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def reply(self,data,status=200):
        payload=json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(payload)))
        self.end_headers(); self.wfile.write(payload)
    def allowed(self):
        return (self.headers.get('Host') in ('127.0.0.1:8788','localhost:8788')
                and not self.headers.get('Origin'))
    def do_GET(self):
        if not self.allowed():return self.reply({'error':'Local clients only'},403)
        self.reply({'devices':{'embedding':runtime_module.EMBEDDING_DEVICE,'person':runtime_module.PERSON_DEVICE},
                    'service':'project-npu','loaded':list(runtime.models)})
    def do_POST(self):
        if not self.allowed() or self.headers.get('X-NPU-Client')!='1':
            return self.reply({'error':'Local clients only'},403)
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=16*1024*1024:raise ValueError('Invalid request size')
            data=json.loads(self.rfile.read(size))
            self.reply(runtime.run(self.path.strip('/'),data))
        except Exception as exc:
            self.reply({'error':str(exc),'fallback':False},503)

if __name__=='__main__':
    ThreadingHTTPServer(('127.0.0.1',8788),Handler).serve_forever()
