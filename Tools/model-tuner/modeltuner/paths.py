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


def model_dirs(model_file: str = "") -> list[Path]:
    """Directories that may hold a model, in load order. Pinned models load from this PC, the rest from the
    model server (Config/model-storage.json), the same rule the dashboard uses. A share that can't be
    opened is skipped, so this PC's models still work."""
    cfg = _storage()
    net = str(cfg.get("networkDir") or "").rstrip("\\")
    if not net:
        return [MODELS_DIR]
    try:
        reachable = Path(net).is_dir()
    except OSError:
        reachable = False
    if not reachable:
        return [MODELS_DIR]
    pinned = cfg.get("pinnedLocal") or []
    return [MODELS_DIR, Path(net)] if model_file in pinned else [Path(net), MODELS_DIR]


def resolve_model(model_file: str) -> Path:
    for d in model_dirs(model_file):
        if (d / model_file).is_file():
            return d / model_file
    return MODELS_DIR / model_file


def ensure_state_dirs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
