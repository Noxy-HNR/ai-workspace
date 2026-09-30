"""Dashboard client. The tuner never starts llama-server itself; every load goes through the dashboard, which stops the
old server and waits for it to exit before starting the next (the rule that prevents the 2026-09-26 crash)."""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from modeltuner import paths
from modeltuner.llama import LlamaClient


class DashboardError(RuntimeError):
    pass


def _nowin() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


class Dashboard:
    def __init__(self, base_url: str = paths.DASHBOARD_URL, llama: LlamaClient | None = None):
        self.base_url = base_url.rstrip("/")
        self.llama = llama or LlamaClient()

    def _call(self, method: str, path: str, body: dict | None = None, timeout: float = 120) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8-sig", "replace"))
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError) as e:
            raise DashboardError(f"dashboard {method} {path} failed: {e}") from e

    def alive(self) -> bool:
        try:
            self._call("GET", "/api/status", timeout=30)
            return True
        except DashboardError:
            return False

    def trial_load(self, model_file: str, settings: dict[str, Any]) -> dict:
        """Load with setting overrides, without saving anything to models.json."""
        resp = self._call("POST", "/api/models/load",
                          {"filename": model_file, "persist": False, "settings": settings}, timeout=180)
        if resp.get("success") is False:
            raise DashboardError(resp.get("message") or "dashboard refused the load")
        if not resp.get("trial"):
            raise DashboardError("this dashboard does not support trial loads (no 'trial' flag in its reply); "
                                 "restart it so it picks up the persist=false support")
        return resp

    def load_saved(self, model_file: str) -> dict:
        """Normal load with the model's saved settings (used to restore the user's model)."""
        resp = self._call("POST", "/api/models/load", {"filename": model_file}, timeout=180)
        if resp.get("success") is False:
            raise DashboardError(resp.get("message") or "dashboard refused the load")
        return resp

    def stop_llama(self) -> None:
        self._call("POST", "/api/stop", {"service": "llama-server"}, timeout=120)

    def models_config(self) -> dict:
        return json.loads(paths.MODELS_JSON.read_text(encoding="utf-8-sig"))

    def save_settings(self, model_file: str, settings: dict[str, Any]) -> dict:
        """Write a model's settings to models.json through the dashboard (only used by `apply`)."""
        return self._call("POST", "/api/models/config", {"filename": model_file, "settings": settings})


def llama_pid(port: int = paths.LLAMA_PORT) -> int | None:
    """PID listening on the llama-server port (netstat; no third-party packages)."""
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, timeout=30,
                             creationflags=_nowin()).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[1].endswith(f":{port}") and parts[3] == "LISTENING":
            try:
                return int(parts[4])
            except ValueError:
                return None
    return None


def process_alive(pid: int) -> bool:
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True,
                             timeout=30, creationflags=_nowin()).stdout
    except (OSError, subprocess.TimeoutExpired):
        return True
    return str(pid) in out


def llama_log_tail(lines: int = 60) -> str:
    log = paths.AI_ROOT / "Logs" / "llama.stderr.log"
    try:
        return "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


OOM_MARKERS = ("out of memory", "cudaMalloc failed", "failed to allocate", "unable to allocate",
               "not enough memory", "CUDA error: out of memory", "failed to create context")


def classify_failure(log_tail: str) -> str:
    low = log_tail.lower()
    if any(m.lower() in low for m in OOM_MARKERS):
        return "oom"
    return "crash"


def wait_for_model(dash: Dashboard, model_file: str, want_ctx: int, old_pid: int | None,
                   timeout: float = 360) -> tuple[int | None, float]:
    """Wait until a NEW server process serves `model_file` with `want_ctx`. Returns (pid, seconds).

    Raises DashboardError("oom"/"crash"/"timeout" prefix) when the server dies or never comes up. Checking for a new PID
    matters: the old server can keep answering for a few seconds after a load call.
    """
    t0 = time.time()
    seen_pid = None
    while time.time() - t0 < timeout:
        time.sleep(1.5)
        pid = llama_pid()
        if pid is not None and pid != old_pid:
            seen_pid = pid
            p = None
            try:
                p = dash.llama.props(timeout=5)
            except Exception:
                p = None
            if p and Path(p.get("model_path", "")).name == model_file:
                n_ctx = int((p.get("default_generation_settings") or {}).get("n_ctx") or 0)
                if n_ctx == want_ctx and dash.llama.ready():
                    return pid, time.time() - t0
                if n_ctx and n_ctx != want_ctx:
                    raise DashboardError(f"verify: server reports n_ctx {n_ctx}, expected {want_ctx}")
        elif seen_pid is not None and pid is None and not process_alive(seen_pid):
            kind = classify_failure(llama_log_tail())
            raise DashboardError(f"{kind}: llama-server exited while loading")
        elif seen_pid is None and time.time() - t0 > 20 and pid is None:
            # never came up: look at the log for why
            tail = llama_log_tail()
            if any(m.lower() in tail.lower() for m in OOM_MARKERS) or "error" in tail.lower().split("\n")[-1]:
                raise DashboardError(f"{classify_failure(tail)}: llama-server failed to start")
    raise DashboardError(f"timeout: {model_file} (ctx {want_ctx}) not serving after {timeout:.0f}s")
