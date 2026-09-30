import tomllib

import pytest

from modeltuner.switchboard_sync import fields_from_report, update_measured_text

SAMPLE = '''# header comment

[[model]]
file = "A.gguf"
id = "a"
enabled = true

[model.capability]
general = 2.0

[model.measured]
gen_tps = 85.0
gen_tps_source = "hand, session 9"
load_s = 19.0
load_s_source = "hand"

[[model]]
file = "B.gguf"
# no measured table yet
enabled = true

[model.capability]
general = 1.0
'''


def test_updates_existing_keys_and_adds_missing_ones_in_place():
    out = update_measured_text(SAMPLE, "A.gguf", {"gen_tps": 91.2, "gen_tps_source": 'tuner "x"',
                                                  "prompt_tps": 1150.0})
    data = tomllib.loads(out)
    a = data["model"][0]["measured"]
    assert a["gen_tps"] == 91.2 and a["gen_tps_source"] == 'tuner "x"' and a["prompt_tps"] == 1150.0
    assert a["load_s"] == 19.0                      # untouched keys stay
    assert "# header comment" in out                # comments survive
    assert data["model"][1]["file"] == "B.gguf" and "measured" not in data["model"][1]


def test_creates_the_measured_table_when_missing():
    out = update_measured_text(SAMPLE, "B.gguf", {"gen_tps": 40.0})
    data = tomllib.loads(out)
    assert data["model"][1]["measured"]["gen_tps"] == 40.0
    assert data["model"][1]["capability"]["general"] == 1.0
    assert "# no measured table yet" in out


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        update_measured_text(SAMPLE, "C.gguf", {"gen_tps": 1.0})


def test_fields_from_report():
    rep = {"run_id": "r1", "model": "A.gguf", "mode": "quick", "finished": "2026-09-30 01:00:00",
           "power_key": "ac|best_performance|saver-off", "best_metrics": {"prose": 37.44, "prefill": 1155.9},
           "trials": [{"phase": "confirm", "load_s": 10.2}, {"phase": "fit", "load_s": 30.0}]}
    f = fields_from_report(rep)
    assert f["gen_tps"] == 37.4 and f["prompt_tps"] == 1155.9 and f["load_s"] == 10.2
    assert "r1" in f["gen_tps_source"] and "ac|best_performance" in f["gen_tps_source"]
