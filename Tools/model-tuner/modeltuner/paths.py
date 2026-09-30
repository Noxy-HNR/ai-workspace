"""Every path, port and URL Model Tuner uses, in one place."""

from __future__ import annotations

import os
from pathlib import Path

AI_ROOT = Path(os.environ.get("MODELTUNER_AI_ROOT", r"C:\AI"))
MODELS_DIR = AI_ROOT / "models"
MODELS_JSON = AI_ROOT / "Config" / "models.json"
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


def ensure_state_dirs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
