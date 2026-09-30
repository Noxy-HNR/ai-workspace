"""llama-fit-params wrapper: memory projections for a configuration without loading the model.

llama-fit-params does a no-allocation dry run of model and context creation and prints a memory breakdown per device.
It takes about 2 seconds and reads only the file header, so the tuner can afford dozens of projections to find the
largest context or the most GPU layers that fit.

Two things it cannot be trusted for on this machine:
- Its "free" column. Under Windows WDDM, CUDA reports free memory as if other processes' VRAM could be paged out, so
  it can say 10 GB is free while nvidia-smi shows 500 MiB. The tuner computes its own budget from nvidia-smi.
- Speculative-decoding draft contexts (MTP). It rejects --spec-type, so that overhead is learned by calibration.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from modeltuner import paths
from modeltuner.knobs import Config, fit_params_args

_ROW = re.compile(
    r"breakdown_print:\s*\|\s*-\s*(?P<dev>.+?)\s*\|\s*"
    r"(?:(?P<total>\d+)\s*=\s*(?P<free>-?\d+)\s*\+\s*\(\s*)?"
    r"(?P<self>\d+)\s*=\s*(?P<model>\d+)\s*\+\s*(?P<context>\d+)\s*\+\s*(?P<compute>\d+)"
    r"(?:\s*\)\s*\+\s*(?P<unacc>-?\d+))?"
)
_FITTED = re.compile(r"^-c\s+(\d+)(?:\s+-ngl\s+(\d+))?", re.M)


class FitError(RuntimeError):
    pass


@dataclass(frozen=True)
class Projection:
    device_model_mib: int
    device_context_mib: int
    device_compute_mib: int
    host_model_mib: int
    host_context_mib: int
    host_compute_mib: int
    device_total_mib: int
    fitted_ngl: int | None
    fitted_ctx: int | None
    seconds: float

    @property
    def device_self_mib(self) -> int:
        return self.device_model_mib + self.device_context_mib + self.device_compute_mib

    @property
    def host_self_mib(self) -> int:
        return self.host_model_mib + self.host_context_mib + self.host_compute_mib

    def as_dict(self) -> dict:
        return {
            "device": {"model": self.device_model_mib, "context": self.device_context_mib,
                       "compute": self.device_compute_mib, "self": self.device_self_mib},
            "host": {"model": self.host_model_mib, "context": self.host_context_mib,
                     "compute": self.host_compute_mib, "self": self.host_self_mib},
            "device_total_mib": self.device_total_mib, "fitted_ngl": self.fitted_ngl,
            "fitted_ctx": self.fitted_ctx, "seconds": round(self.seconds, 2),
        }


def parse_output(text: str, seconds: float = 0.0) -> Projection:
    """Parse llama-fit-params -lv 4 output. The FIRST breakdown block describes the requested parameters; later blocks
    are the tool's own fitting attempts and are ignored."""
    dev = host = None
    for line in text.splitlines():
        m = _ROW.search(line)
        if not m:
            continue
        is_host = m.group("dev").strip().lower().startswith("host")
        if is_host and host is None:
            host = m
        elif not is_host and dev is None:
            dev = m
        if dev is not None and host is not None:
            break
    if dev is None:
        tail = "\n".join(text.strip().splitlines()[-5:])
        raise FitError(f"llama-fit-params printed no device memory breakdown. Last lines:\n{tail}")
    fitted = _FITTED.search(text)
    return Projection(
        device_model_mib=int(dev.group("model")),
        device_context_mib=int(dev.group("context")),
        device_compute_mib=int(dev.group("compute")),
        host_model_mib=int(host.group("model")) if host else 0,
        host_context_mib=int(host.group("context")) if host else 0,
        host_compute_mib=int(host.group("compute")) if host else 0,
        device_total_mib=int(dev.group("total") or 0),
        fitted_ctx=int(fitted.group(1)) if fitted else None,
        fitted_ngl=int(fitted.group(2)) if fitted and fitted.group(2) else None,
        seconds=seconds,
    )


def binary_for(server_binary: str | None) -> Path:
    """llama-fit-params next to the llama-server build the model runs on."""
    if server_binary:
        server = Path(server_binary)
        if not server.is_absolute():
            server = paths.AI_ROOT / server
        candidate = server.parent / "llama-fit-params.exe"
    else:
        candidate = paths.LLAMA_NATIVE_BIN / "llama-fit-params.exe"
    if not candidate.exists():
        raise FitError(f"llama-fit-params.exe not found at {candidate}")
    return candidate


class FitParams:
    """Projects memory for Configs of one model file, with a per-process cache (projections are deterministic)."""

    def __init__(self, model_path: Path, server_binary: str | None = None, timeout_s: float = 90.0):
        self.model_path = Path(model_path)
        self.exe = binary_for(server_binary)
        self.timeout_s = timeout_s
        self._cache: dict[tuple, Projection] = {}
        self.calls = 0

    def project(self, cfg: Config) -> Projection:
        args = tuple(fit_params_args(cfg))
        if args in self._cache:
            return self._cache[args]
        cmd = [str(self.exe), "-m", str(self.model_path), "-lv", "4", *args]
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_s,
                                  cwd=str(self.exe.parent), encoding="utf-8", errors="replace",
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired as e:
            raise FitError(f"llama-fit-params timed out after {self.timeout_s:.0f}s") from e
        self.calls += 1
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if "invalid argument" in out:
            raise FitError(out.strip().splitlines()[-1])
        proj = parse_output(out, time.perf_counter() - t0)
        self._cache[args] = proj
        return proj
