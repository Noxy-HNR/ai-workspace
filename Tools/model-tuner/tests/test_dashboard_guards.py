"""Live checks of the dashboard's Model Tuner guards (from the bot-queue dashboard review, job 06, 2026-09-30).

Every request here is one the dashboard must refuse before doing anything, so no model is loaded or stopped.
Skipped when the dashboard is not running.
"""

import json
import urllib.error
import urllib.request

import pytest

from modeltuner import paths


def _post(path: str, body: dict) -> dict:
    req = urllib.request.Request(paths.DASHBOARD_URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8-sig"))


def _alive() -> bool:
    try:
        with urllib.request.urlopen(paths.DASHBOARD_URL + "/api/optimizer/status", timeout=10):
            return True
    except (urllib.error.URLError, OSError):
        return False


pytestmark = pytest.mark.skipif(not _alive(), reason="the dashboard is not running")


@pytest.mark.parametrize("name", [r"..\..\Windows\win.ini", "../models/x.gguf", "C:x.gguf", "sub\\x.gguf", ""])
def test_model_loads_refuse_paths(name):
    for persist in (False, True):
        r = _post("/api/models/load", {"filename": name, "persist": persist, "settings": {"contextSize": 4096}})
        assert r.get("success") is False, r


@pytest.mark.parametrize("extra", ["--model ..\\x.gguf", "-m x.gguf", "--lora a.gguf", "--port 1234",
                                   "--spec-type ngram-mod --host 0.0.0.0", "--mmproj=p.gguf"])
def test_trial_extra_args_cannot_choose_model_lora_or_port(extra):
    r = _post("/api/models/load", {"filename": "Qwen3.5-9B-UD-Q6_K_XL-mtp.gguf", "persist": False,
                                   "settings": {"extraArgs": extra}})
    assert r.get("success") is False and "may not" in r.get("message", ""), r


@pytest.mark.parametrize("action", [{"action": "start", "model": r"..\secret.gguf"},
                                    {"action": "plan", "model": "a/b.gguf"},
                                    {"action": "apply", "run_id": r"..\..\x"},
                                    {"action": "nonsense"}])
def test_optimizer_actions_refuse_bad_input(action):
    r = _post("/api/optimizer", action)
    assert r.get("success") is False, r
