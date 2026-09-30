"""RAM safety for unattended runs (see C:\\AI\\AGENTS.md). This PC has 15.4 GB of RAM; careless loading crashed it.

- Preflight: available RAM must cover the load's projected host footprint plus 1 GB.
- Watchdog: if available RAM stays under 0.8 GB for 3 seconds, stop llama-server, write the trip flag and exit
  immediately. The tuner refuses to start while the flag exists; a person must look and delete it.
- Stop file: the dashboard (or a person) can ask a run to stop; the strategy checks between steps.
- LLAMA_BUSY lock: held while measuring, so nothing else uses llama-server meanwhile.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Callable

from modeltuner import hardware, paths

WATCHDOG_MIN_GB = 0.8
WATCHDOG_SECONDS = 3
PREFLIGHT_MARGIN_GB = 1.0


class SafetyError(RuntimeError):
    pass


def refuse_if_tripped() -> None:
    if paths.TRIP_FLAG.exists():
        raise SafetyError(
            f"refusing to run: {paths.TRIP_FLAG} exists, so a RAM watchdog stopped an earlier run. Check what used the "
            f"memory, then delete that file.\n{paths.TRIP_FLAG.read_text(encoding='utf-8', errors='replace')}")


def preflight(host_need_mib: int, *, margin_gb: float = PREFLIGHT_MARGIN_GB) -> float:
    """Raise SafetyError unless available RAM covers host_need + margin. Returns available GB."""
    avail = hardware.memory().avail_gb
    need = host_need_mib / 1024 + margin_gb
    if avail < need:
        raise SafetyError(f"not enough free RAM for this load: {avail:.2f} GB available, {need:.2f} GB needed "
                          f"({host_need_mib} MiB on the host + {margin_gb:.1f} GB margin). Close something, or wait.")
    return avail


class Watchdog:
    def __init__(self, on_trip: Callable[[float], None], min_gb: float = WATCHDOG_MIN_GB,
                 seconds: int = WATCHDOG_SECONDS, interval: float = 1.0, exit_process: bool = True):
        self.on_trip = on_trip
        self.min_gb = min_gb
        self.seconds = seconds
        self.interval = interval
        self.exit_process = exit_process
        self.tripped = False
        self.lowest_gb = float("inf")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="ram-watchdog", daemon=True)

    def start(self) -> "Watchdog":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        low_since = None
        while not self._stop.is_set():
            try:
                avail = hardware.memory().avail_gb
            except Exception:
                avail = float("inf")
            self.lowest_gb = min(self.lowest_gb, avail)
            if avail < self.min_gb:
                low_since = low_since or time.monotonic()
                if time.monotonic() - low_since >= self.seconds:
                    self._trip(avail)
                    return
            else:
                low_since = None
            self._stop.wait(self.interval)

    def _trip(self, avail: float) -> None:
        self.tripped = True
        paths.STATE_DIR.mkdir(parents=True, exist_ok=True)
        paths.TRIP_FLAG.write_text(
            time.strftime("%Y-%m-%d %H:%M:%S") + f" available RAM {avail:.2f} GB < {self.min_gb} GB for "
            f"{self.seconds}s; llama-server stopped\n", encoding="utf-8")
        try:
            self.on_trip(avail)
        finally:
            if self.exit_process:
                os._exit(3)


def stop_requested() -> bool:
    return paths.STOP_FILE.exists()


def clear_stop() -> None:
    try:
        paths.STOP_FILE.unlink()
    except FileNotFoundError:
        pass


class LlamaLock:
    """Marks llama-server as busy with measurements (a file; advisory, for other local tooling)."""

    def __init__(self, path: Path = paths.LLAMA_LOCK):
        self.path = path

    def __enter__(self) -> "LlamaLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")
        return self

    def __exit__(self, *exc) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
