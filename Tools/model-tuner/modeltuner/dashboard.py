"""Dashboard client. The tuner never starts llama-server itself; every load goes through the dashboard, which stops the
old server and waits for it to exit before starting the next (the rule that prevents the 2026-09-26 crash)."""

from __future__ import annotations

import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from modeltuner import paths
from modeltuner.llama import LlamaClient


class DashboardError(RuntimeError):
    """A dashboard or load failure with a machine-readable `kind`: "timeout" (took too long; says nothing about the
    config), "refused" (the dashboard said no), "oom" / "crash" (the server died), "verify" (it came up with the
    wrong settings) or "unreadable" (a file the dashboard owns could not be read). When no kind is given it is
    inferred from the message, which keeps older call sites and tests working."""

    KINDS = ("timeout", "refused", "oom", "crash", "verify", "unreadable")

    def __init__(self, message: str, kind: str | None = None):
        super().__init__(message)
        self.kind = kind or _infer_kind(message)


def _infer_kind(message: str) -> str:
    low = message.lower()
    prefix = low.split(":", 1)[0].strip()
    if prefix in DashboardError.KINDS:
        return prefix
    if "timed out" in low or "timeout" in low:
        return "timeout"
    return "refused"


def _nowin() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


class Dashboard:
    def __init__(self, base_url: str = paths.DASHBOARD_URL, llama: LlamaClient | None = None):
        self.base_url = base_url.rstrip("/")
        self.llama = llama or LlamaClient()

    def _call(self, method: str, path: str, body: dict | None = None, timeout: float = 120) -> dict:
        """One request with a hard total deadline. urllib's own timeout applies per socket read, so a peer that
        trickles its reply could hold the tuner indefinitely; the request runs in a worker thread instead and is
        abandoned when the deadline passes."""
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data else {})
        box: dict[str, Any] = {}

        def worker() -> None:
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    box["body"] = r.read()
            except BaseException as e:  # noqa: BLE001 -- handed to the caller below
                box["error"] = e

        t = threading.Thread(target=worker, name="dashboard-call", daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            raise DashboardError(f"dashboard {method} {path} failed: no complete reply within {timeout:.0f}s",
                                 kind="timeout")
        err = box.get("error")
        if err is not None:
            kind = "timeout" if isinstance(err, TimeoutError) or "timed out" in str(err).lower() else "refused"
            raise DashboardError(f"dashboard {method} {path} failed: {err}", kind=kind) from err
        try:
            return json.loads(box["body"].decode("utf-8-sig", "replace"))
        except ValueError as e:
            raise DashboardError(f"dashboard {method} {path} returned invalid JSON", kind="refused") from e

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

    def stop_llama(self, timeout: float = 120) -> None:
        self._call("POST", "/api/stop", {"service": "llama-server"}, timeout=timeout)

    def models_config(self) -> dict:
        """models.json as the dashboard last wrote it. It can be mid-rewrite, so read errors are DashboardErrors."""
        try:
            return json.loads(paths.MODELS_JSON.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as e:
            raise DashboardError(f"could not read models.json: {e}", kind="unreadable") from e

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


def wait_for_model(dash: Dashboard, model_file: str, want_ctx: int | None, old_pid: int | None,
                   timeout: float = 360, verify_grace_s: float = 10.0) -> tuple[int | None, float]:
    """Wait until a NEW server process serves `model_file` (with `want_ctx`, unless it is None). Returns (pid, secs).

    Raises DashboardError with a kind. The rules, each learned the hard way:
    - A new PID must be seen: the old server can answer for a few seconds after a load call.
    - The server log is only evidence once the new server has started (a new PID was seen). Before that the file on
      disk is the previous server's, and a slow load must not be failed with its errors.
    - A server that held the port and then let go of it for two polls has exited, even if `tasklist` cannot say so.
    - A context mismatch only counts once it has persisted for `verify_grace_s`: the first /props can still describe
      the previous settings.
    """
    t0 = time.time()
    seen_pid = None
    gone_polls = 0
    mismatch_since: float | None = None
    while time.time() - t0 < timeout:
        time.sleep(1.5)
        pid = llama_pid()
        if pid is not None and pid != old_pid:
            seen_pid, gone_polls = pid, 0
            try:
                p = dash.llama.props(timeout=5)
            except Exception:  # noqa: BLE001 -- not up yet
                p = None
            if p and Path(p.get("model_path", "")).name == model_file:
                n_ctx = int((p.get("default_generation_settings") or {}).get("n_ctx") or 0)
                if (want_ctx is None or n_ctx == want_ctx) and dash.llama.ready():
                    return pid, time.time() - t0
                if want_ctx is not None and n_ctx and n_ctx != want_ctx:
                    mismatch_since = mismatch_since if mismatch_since is not None else time.time()
                    if time.time() - mismatch_since >= verify_grace_s:
                        raise DashboardError(f"verify: server reports n_ctx {n_ctx}, expected {want_ctx}",
                                             kind="verify")
                else:
                    mismatch_since = None
        elif seen_pid is not None and pid is None:
            gone_polls += 1
            if gone_polls >= 2 or not process_alive(seen_pid):
                kind = classify_failure(llama_log_tail())
                raise DashboardError(f"{kind}: llama-server exited while loading", kind=kind)
    if seen_pid is not None:
        tail = llama_log_tail()
        if any(m.lower() in tail.lower() for m in OOM_MARKERS):
            raise DashboardError(f"oom: {model_file} started but never served; its log shows a memory failure",
                                 kind="oom")
    raise DashboardError(f"timeout: {model_file} (ctx {want_ctx}) not serving after {timeout:.0f}s", kind="timeout")
