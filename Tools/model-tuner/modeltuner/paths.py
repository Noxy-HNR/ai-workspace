"""Every path, port and URL Model Tuner uses, in one place."""

from __future__ import annotations

import os
from pathlib import Path

AI_ROOT = Path(os.environ.get("MODELTUNER_AI_ROOT", r"C:\AI"))
MODELS_DIR = AI_ROOT / "models"
MODELS_JSON = AI_ROOT / "Config" / "models.json"
MODEL_STORAGE_JSON = AI_ROOT / "Config" / "model-storage.json"
LLAMA_NATIVE_BIN = AI_ROOT / "Tools" / "llama-native" / "bin"
SWITCHBOARD_ROOT = AI_ROOT / "Switchboard"
SWITCHBOARD_MODELS_TOML = SWITCHBOARD_ROOT / "config" / "models.toml"
TUNING_LOG = AI_ROOT / "Logs" / "overnight-tuning-log.md"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = Path(os.environ.get("MODELTUNER_STATE_DIR", PROJECT_ROOT / "state"))
RUNS_DIR = STATE_DIR / "runs"
DB_PATH = STATE_DIR / "tuner.db"
CURRENT_RUN = STATE_DIR / "current.json"
STOP_FILE = STATE_DIR / "STOP"
TRIP_FLAG = STATE_DIR / "WATCHDOG_TRIPPED"
# Held while anything is using llama-server for measurement, so the bot queue does not run local-model jobs then.
LLAMA_LOCK = STATE_DIR / "LLAMA_BUSY"

DASHBOARD_URL = os.environ.get("MODELTUNER_DASHBOARD_URL", "http://127.0.0.1:9090")
LLAMA_URL = os.environ.get("MODELTUNER_LLAMA_URL", "http://127.0.0.1:10000")
LLAMA_PORT = 10000


def _storage() -> dict:
    import json
    try:
        data = json.loads(MODEL_STORAGE_JSON.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


# A file on the share that changed in the last minute may still be arriving (same rule as the dashboard).
SETTLE_SECONDS = 60


def _pinned(cfg: dict, model_file: str) -> bool:
    # Windows file names are case-insensitive, and so is the dashboard's pin check.
    return model_file.casefold() in {str(p).casefold() for p in (cfg.get("pinnedLocal") or [])}


def _network_dir(cfg: dict) -> Path | None:
    net = str(cfg.get("networkDir") or "").rstrip("\\")
    if not net:
        return None
    try:
        return Path(net) if Path(net).is_dir() else None
    except OSError:
        return None


def _settled(path: Path) -> bool:
    import time
    try:
        return time.time() - path.stat().st_mtime >= SETTLE_SECONDS
    except OSError:
        return False


def model_dirs(model_file: str = "") -> list[Path]:
    """Directories that may hold a model, in load order. Pinned models load from this PC, the rest from the
    model server (Config/model-storage.json), the same rule the dashboard uses. A share that can't be
    opened is skipped, so this PC's models still work."""
    cfg = _storage()
    net = _network_dir(cfg)
    if net is None:
        return [MODELS_DIR]
    return [MODELS_DIR, net] if _pinned(cfg, model_file) else [net, MODELS_DIR]


def resolve_model(model_file: str) -> Path:
    cfg = _storage()
    local = MODELS_DIR / model_file
    if _pinned(cfg, model_file) and local.is_file():
        return local                      # a pinned local model never touches the share
    for d in model_dirs(model_file):
        p = d / model_file
        if p.is_file() and (d == MODELS_DIR or _settled(p)):
            return p
    return local


def model_files() -> dict[str, Path]:
    """Every top-level .gguf on this PC and the model server, one per name (ignoring case), each the copy
    resolve_model would load."""
    names: dict[str, str] = {}
    net = _network_dir(_storage())
    for d in ([net] if net else []) + [MODELS_DIR]:
        for p in d.glob("*.gguf"):
            if d == MODELS_DIR or _settled(p):
                names.setdefault(p.name.casefold(), p.name)
    out = {}
    for name in names.values():
        p = resolve_model(name)
        if p.is_file():
            out[p.name] = p
    return out


def ensure_state_dirs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
