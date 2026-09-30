"""Bot queue: runs delegated build jobs through OpenCode models, unattended, while nobody is watching.

Each job is a folder under state/botqueue/jobs/<name>/ holding job.json and TASK.md. The runner:
- starts a job once every job it depends on is done,
- runs at most MAX_CLOUD cloud jobs at once and ONE local-model job at a time,
- for a local job, first loads the requested model through the dashboard (verified, RAM-checked), and never while
  state/LLAMA_BUSY exists (something is measuring llama-server),
- seeds each attempt's workdir with a fresh snapshot of the project (so bots build on the latest interfaces),
- runs `opencode run --auto` in that workdir, then runs the job's check command (default: pytest) and records the
  outcome, timing and token use in result.json,
- on failure, retries with the next model in the job's `models` list.

It never touches C:/AI outside the tuner's state folder except through the dashboard's load API.

Start detached:   python devtools/botqueue.py run
Status:           python devtools/botqueue.py status
Stop gracefully:  create state/botqueue/STOP
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
QROOT = PROJECT / "state" / "botqueue"
JOBS = QROOT / "jobs"
STOP = QROOT / "STOP"
STATUS = QROOT / "status.json"
LOG = QROOT / "queue.log"
LLAMA_LOCK = PROJECT / "state" / "LLAMA_BUSY"
DASH = "http://127.0.0.1:9090"
LLAMA = "http://127.0.0.1:10000"
OPENCODE = shutil.which("opencode") or str(Path.home() / ".opencode" / "bin" / "opencode.exe")
MAX_CLOUD = 3
POLL_S = 20
MAX_RUNTIME_H = float(os.environ.get("BOTQUEUE_MAX_HOURS", "14"))
LOCAL_PREFIX = "llama-local/"
SEED_ITEMS = ["modeltuner", "tests", "DESIGN.md", "pyproject.toml", "devtools/SNAPSHOT_NOTES.md"]
IGNORE = shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc", ".venv")

_lock = threading.Lock()
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)
# OpenCode's standalone servers fail to start when several launch in the same moment
# ("Standalone server exited before reporting readiness"), so launches are spaced out.
_launch_lock = threading.Lock()
_last_launch = [0.0]
LAUNCH_GAP_S = 15


def log(msg: str) -> None:
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    with _lock:
        QROOT.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    print(line, flush=True)


# ---------- RAM + dashboard helpers (local jobs) ----------

class _MEM(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]


def avail_gb() -> float:
    m = _MEM()
    m.dwLength = ctypes.sizeof(_MEM)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 1024 ** 3


def _get(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _post(path: str, body: dict, timeout: float = 120) -> dict:
    req = urllib.request.Request(DASH + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def loaded_model() -> str | None:
    try:
        return Path(_get(LLAMA + "/props", 10)["model_path"]).name
    except Exception:
        return None


class _PWR(ctypes.Structure):
    _fields_ = [("ac", ctypes.c_ubyte), ("flag", ctypes.c_ubyte), ("pct", ctypes.c_ubyte), ("saver", ctypes.c_ubyte),
                ("life", ctypes.c_ulong), ("full", ctypes.c_ulong)]


def on_battery() -> bool:
    s = _PWR()
    ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s))
    return s.ac == 0


def tuner_running() -> bool:
    cur = PROJECT / "state" / "current.json"
    try:
        return json.loads(cur.read_text(encoding="utf-8")).get("status") in ("running", "starting")
    except Exception:
        return False


def local_blocked() -> str | None:
    """Why a local-model job must wait right now, or None."""
    if LLAMA_LOCK.exists() or tuner_running():
        return "a Model Tuner run is using llama-server"
    if on_battery():
        return "on battery (local jobs would drain it)"
    return None


def loaded_ctx() -> int | None:
    try:
        return int(_get(LLAMA + "/props", 10)["default_generation_settings"]["n_ctx"])
    except Exception:
        return None


def ensure_local_model(filename: str, settings: dict | None = None, timeout: float = 420) -> None:
    """Serve `filename` (with optional non-persisted setting overrides) unless it already is. Verified by /props."""
    want_ctx = (settings or {}).get("contextSize")
    if loaded_model() == filename and (want_ctx is None or loaded_ctx() == want_ctx):
        return
    a = avail_gb()
    if a < 1.5:
        raise RuntimeError(f"only {a:.2f} GB RAM available; not loading {filename}")
    body = {"filename": filename}
    if settings:
        body.update(persist=False, settings=settings)
    resp = _post("/api/models/load", body)
    if resp.get("success") is False:
        raise RuntimeError(f"dashboard refused to load {filename}: {resp.get('message')}")
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(3)
        if loaded_model() == filename and (want_ctx is None or loaded_ctx() == want_ctx):
            try:
                if _get(LLAMA + "/v1/models", 5).get("data"):
                    log(f"loaded {filename} {settings or ''} in {time.time() - t0:.0f}s (avail RAM {avail_gb():.2f} GB)")
                    return
            except Exception:
                pass
    raise RuntimeError(f"{filename} did not come up within {timeout:.0f}s")


# ---------- jobs ----------

@dataclass
class Job:
    dir: Path
    spec: dict

    @property
    def name(self) -> str:
        return self.dir.name

    @property
    def result_path(self) -> Path:
        return self.dir / "result.json"

    def result(self) -> dict:
        try:
            return json.loads(self.result_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @property
    def state(self) -> str:
        return self.result().get("state", "pending")

    def save_result(self, **kw) -> None:
        r = self.result()
        r.update(kw)
        self.result_path.write_text(json.dumps(r, indent=2), encoding="utf-8")


def load_jobs() -> list[Job]:
    jobs = []
    for d in sorted(JOBS.glob("*")):
        f = d / "job.json"
        if d.is_dir() and f.exists():
            try:
                jobs.append(Job(d, json.loads(f.read_text(encoding="utf-8"))))
            except Exception as e:
                log(f"bad job.json in {d.name}: {e}")
    return jobs


def seed_workdir(job: Job, attempt: int) -> Path:
    wd = job.dir / f"attempt-{attempt}"
    if wd.exists():
        shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True)
    for item in SEED_ITEMS + list(job.spec.get("extra_seed", [])):
        src = PROJECT / item
        if src.is_dir():
            shutil.copytree(src, wd / item, ignore=IGNORE)
        elif src.exists():
            (wd / item).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, wd / item)
    for extra in job.dir.glob("*"):
        if extra.is_file() and extra.name not in ("job.json", "result.json"):
            shutil.copy2(extra, wd / extra.name)
    subprocess.run(["git", "init", "-q"], cwd=wd, creationflags=NOWIN)
    subprocess.run(["git", "add", "-A"], cwd=wd, creationflags=NOWIN)
    subprocess.run(["git", "-c", "user.name=seed", "-c", "user.email=seed@local", "commit", "-qm", "seed"], cwd=wd, creationflags=NOWIN)
    return wd


def _token_totals(jsonl: Path) -> dict:
    tot = {"input": 0, "output": 0, "reasoning": 0, "tool_calls": 0, "steps": 0}
    try:
        for line in jsonl.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                ev = json.loads(line)
            except Exception:
                continue
            s = json.dumps(ev)
            if '"tool_use"' in s or '"type": "tool"' in s:
                tot["tool_calls"] += 1
            part = ev.get("part") or {}
            toks = part.get("tokens") or ev.get("tokens")
            if isinstance(toks, dict):
                tot["input"] += int(toks.get("input") or 0)
                tot["output"] += int(toks.get("output") or 0)
                tot["reasoning"] += int(toks.get("reasoning") or 0)
                tot["steps"] += 1
    except FileNotFoundError:
        pass
    return tot


def run_check(job: Job, wd: Path) -> tuple[bool, str]:
    cmd = job.spec.get("check", "uv run --with pytest pytest -q")
    try:
        p = subprocess.run(cmd, cwd=wd, shell=True, capture_output=True, text=True, timeout=900,
                           encoding="utf-8", errors="replace", creationflags=NOWIN)
        out = (p.stdout + p.stderr)[-3000:]
        return p.returncode == 0, out
    except subprocess.TimeoutExpired:
        return False, "check timed out after 900s"


def run_attempt(job: Job, model: str, attempt: int) -> bool:
    wd = seed_workdir(job, attempt)
    prompt = job.spec.get("prompt") or (
        "Read TASK.md and DESIGN.md in the current directory, then do the task in TASK.md. "
        "Work only inside the current directory. When you finish, make sure the tests pass.")
    if model.startswith(LOCAL_PREFIX):
        prompt += (" IMPORTANT: each of your replies is cut off after about 500 tokens, so never write a whole file in "
                   "one tool call. Create each file with a short first part, then add the rest with several small "
                   "edit calls (each under about 350 tokens of code).")
    timeout = int(job.spec.get("timeout_s", 3600))
    log(f"[{job.name}] attempt {attempt} with {model} in {wd}")
    job.save_result(state="running", model=model, attempt=attempt, workdir=str(wd), started=time.time())
    t0 = time.time()
    cmd = [OPENCODE, "run", "--standalone", "--auto", "--format", "json", "-m", model,
           "--title", f"modeltuner {job.name} ({model})", prompt]
    for launch_try in range(3):
        with _launch_lock:
            wait = LAUNCH_GAP_S - (time.time() - _last_launch[0])
            if wait > 0:
                time.sleep(wait)
            _last_launch[0] = time.time()
        t_launch = time.time()
        with (wd / "opencode.jsonl").open("w", encoding="utf-8") as out, (wd / "opencode.err").open("w", encoding="utf-8") as err:
            try:
                p = subprocess.run(cmd, cwd=wd, stdout=out, stderr=err, timeout=timeout, creationflags=NOWIN)
                rc = p.returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        errtext = (wd / "opencode.err").read_text(encoding="utf-8", errors="replace")
        if rc != 0 and time.time() - t_launch < 60 and "before reporting readiness" in errtext:
            log(f"[{job.name}] OpenCode failed to start (launch collision); retrying the same model")
            continue
        break
    elapsed = time.time() - t0
    ok, check_out = run_check(job, wd)
    changed = subprocess.run(["git", "status", "--porcelain"], cwd=wd, capture_output=True, text=True, creationflags=NOWIN).stdout
    tokens = _token_totals(wd / "opencode.jsonl")
    hist = job.result().get("attempts", [])
    hist.append({"model": model, "attempt": attempt, "rc": rc, "seconds": round(elapsed), "check_ok": ok,
                 "files_changed": len(changed.splitlines()), "tokens": tokens, "check_tail": check_out[-1500:]})
    job.save_result(attempts=hist)
    log(f"[{job.name}] {model} finished rc={rc} in {elapsed / 60:.1f} min; check {'PASSED' if ok else 'FAILED'}; "
        f"{len(changed.splitlines())} files changed")
    return ok and bool(changed.strip())


def worker(job: Job) -> None:
    models = job.spec["models"]
    start_attempt = len(job.result().get("attempts", [])) + 1
    for i, model in enumerate(models[start_attempt - 1:], start=start_attempt):
        if STOP.exists():
            job.save_result(state="pending")
            return
        try:
            if model.startswith(LOCAL_PREFIX):
                while (why := local_blocked()) and not STOP.exists():
                    job.save_result(state="running", waiting=why)
                    time.sleep(POLL_S)
                if job.spec.get("local_model"):
                    ensure_local_model(job.spec["local_model"], job.spec.get("local_settings"))
            if run_attempt(job, model, i):
                job.save_result(state="done", finished=time.time(), winner=model)
                return
        except Exception as e:  # keep the queue alive whatever one job does
            log(f"[{job.name}] attempt {i} with {model} errored: {e!r}")
            hist = job.result().get("attempts", [])
            hist.append({"model": model, "attempt": i, "error": repr(e)})
            job.save_result(attempts=hist)
    job.save_result(state="failed", finished=time.time())


def write_status(running: dict[str, threading.Thread]) -> None:
    rows = []
    for j in load_jobs():
        r = j.result()
        rows.append({"job": j.name, "state": r.get("state", "pending"), "model": r.get("model"),
                     "attempts": len(r.get("attempts", [])), "winner": r.get("winner")})
    STATUS.write_text(json.dumps({"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "pid": os.getpid(),
                                  "running": sorted(running), "jobs": rows}, indent=2), encoding="utf-8")


def run() -> None:
    QROOT.mkdir(parents=True, exist_ok=True)
    JOBS.mkdir(parents=True, exist_ok=True)
    log(f"queue runner started (pid {os.getpid()})")
    running: dict[str, threading.Thread] = {}
    # Anything left "running" by a previous runner goes back to pending (its attempt restarts).
    for j in load_jobs():
        if j.state == "running":
            j.save_result(state="pending")
    t_end = time.time() + MAX_RUNTIME_H * 3600
    while time.time() < t_end:
        for name in [n for n, t in running.items() if not t.is_alive()]:
            del running[name]
        if STOP.exists():
            if not running:
                log("STOP file found and nothing running; exiting")
                break
        else:
            jobs = load_jobs()
            states = {j.name: j.state for j in jobs}
            for j in jobs:
                if j.name in running or states[j.name] != "pending":
                    continue
                if any(states.get(dep) != "done" for dep in j.spec.get("after", [])):
                    if any(states.get(dep) == "failed" for dep in j.spec.get("after", [])) and not j.spec.get("run_if_dep_failed"):
                        j.save_result(state="blocked")
                    continue
                is_local = any(m.startswith(LOCAL_PREFIX) for m in j.spec["models"])
                by_name = {x.name: x for x in jobs}
                n_local = sum(1 for n in running
                              if n in by_name and any(m.startswith(LOCAL_PREFIX) for m in by_name[n].spec["models"]))
                n_cloud = len(running) - n_local
                if is_local and (n_local >= 1 or local_blocked()):
                    continue
                if not is_local and n_cloud >= MAX_CLOUD:
                    continue
                t = threading.Thread(target=worker, args=(j,), name=j.name, daemon=True)
                running[j.name] = t
                t.start()
                states[j.name] = "running"
        write_status(running)
        time.sleep(POLL_S)
    write_status(running)
    log("queue runner exiting")


def status() -> None:
    if STATUS.exists():
        print(STATUS.read_text(encoding="utf-8"))
    for j in load_jobs():
        r = j.result()
        print(f"{j.name:32} {r.get('state', 'pending'):8} winner={r.get('winner')}")
        for a in r.get("attempts", []):
            print(f"    - {a.get('model')}: rc={a.get('rc')} {a.get('seconds')}s check_ok={a.get('check_ok')} "
                  f"files={a.get('files_changed')} err={a.get('error', '')}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"run": run, "status": status}[cmd]()
