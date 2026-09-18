"""Standard-library client. Background advisers never block capture/rendering."""
import json
import threading
import time
import urllib.request
import subprocess
from pathlib import Path

_startup_lock = threading.Lock()
_ready = False

def ensure_service():
    global _ready
    with _startup_lock:
        if _ready: return
        def healthy():
            try:
                with urllib.request.urlopen('http://127.0.0.1:8788',timeout=2) as response:
                    return json.load(response).get('service')=='project-npu'
            except Exception:return False
        if not healthy():
            subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',
                            str(Path(__file__).with_name('start.ps1'))],
                           creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=20,
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)
            for _ in range(20):
                if healthy():break
                time.sleep(.25)
        if not healthy(): raise RuntimeError('NPU service did not start; see server-error.log')
        _ready=True

def request(action, payload, timeout=10):
    req=urllib.request.Request('http://127.0.0.1:8788/'+action,
        json.dumps(payload).encode(),headers={'Content-Type':'application/json','X-NPU-Client':'1'})
    try:
        ensure_service()
        with urllib.request.urlopen(req,timeout=timeout) as response:return json.load(response)
    except Exception as exc:
        global _ready
        _ready = False
        raise RuntimeError('NPU service unavailable. Start C:\\AI\\Tools\\npu-services\\start.ps1. No CPU/GPU fallback.') from exc

class Advisor:
    def __init__(self,action,interval=5):
        self.action=action;self.interval=interval;self.last=0;self.busy=False
        self.result={'label':'NPU connecting','boxes':[]};self.updated=0
    def ready(self):
        return not self.busy and time.monotonic()-self.last>=self.interval
    def submit(self,payload):
        now=time.monotonic()
        if self.busy or now-self.last<self.interval:return
        self.last=now;self.busy=True
        def work():
            try:self.result=request(self.action,payload)
            except RuntimeError:self.result={'label':'NPU unavailable','boxes':[]}
            finally:self.updated=time.monotonic();self.busy=False
        threading.Thread(target=work,daemon=True).start()
