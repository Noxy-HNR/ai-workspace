"""Tests for modeltuner.report: the JSON and Markdown run reports.

The TuneResult is a real one: the strategy runs against the simulated backend (no GPU, no
llama-server), seeded so it replays exactly. The reports are then checked against the run's
actual trials, decisions and metrics.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from modeltuner import report
from modeltuner.knobs import Config, to_settings
from modeltuner.profile import ModelProfile
from modeltuner.sim import SimBackend, SimGpu, SimModel
from modeltuner.strategy import TuneOptions, Tuner, TuneResult

RUN_ID = "20260930-000000-sim-9b"
BASE = Config(ctx=32768, spec="none", ubatch=512, cache_k="f16", cache_v="f16")
# The model's saved settings: everything the tuner owns matches the baseline, plus two keys the
# tuner does not own, which the proposed settings must carry over untouched.
CURRENT = to_settings(BASE) | {
    "serverBinary": "C:/AI/Tools/llama-native/bin/llama-server.exe",
    "loraFile": "C:/AI/models/sim-lora.gguf",
}
META = {"mode": "quick", "power_key": "ac|best_performance|saver-off",
        "started": "2026-09-30 00:00:00", "finished": "2026-09-30 01:00:00"}


def _profile(m: SimModel, **kw) -> ModelProfile:
    d = dict(file=m.name, path="x", size_bytes=m.weights_mib << 20, arch="qwen35", name="sim", n_layer=m.n_layer,
              n_mtp_layers=1 if m.has_mtp else 0, train_ctx=m.train_ctx, n_embd=4096, n_head=16, n_head_kv=4,
              head_dim_k=256, head_dim_v=256, full_attention_interval=4, sliding_window=None, n_expert=0,
              weights_bytes=m.weights_mib << 20, layer_bytes={}, template_supports_tools=True,
              template_has_thinking=True, template_honours_effort=False, server_binary=None, current=CURRENT)
    d.update(kw)
    return ModelProfile(**d)


@pytest.fixture(scope="module")
def tuned() -> TuneResult:
    """One real strategy run against the simulated backend (seed 3 replays exactly)."""
    b = SimBackend(SimModel(), SimGpu(), seed=3)
    return Tuner(b, _profile(SimModel()), BASE, TuneOptions(), emit=lambda k, m, **d: None).run(topo=(8, 20))


class TestToJson:
    def test_keys_and_scalar_fields(self, tuned: TuneResult):
        data = report.to_json(RUN_ID, _profile(SimModel()), tuned, META)
        assert set(data) == {
            "run_id", "model", "status", "mode", "power_key", "started", "finished", "seconds", "loads",
            "improvement", "baseline", "best", "baseline_metrics", "best_metrics", "proposed_settings",
            "changes", "decisions", "notes", "quality", "trials",
        }
        assert data["run_id"] == RUN_ID
        assert data["model"] == "sim-9b.gguf"
        assert data["status"] == "ok"
        assert data["mode"] == "quick" and data["power_key"] == META["power_key"]
        assert data["started"] == META["started"] and data["finished"] == META["finished"]
        assert data["seconds"] == round(tuned.seconds)
        assert data["loads"] == tuned.loads
        assert data["improvement"] == tuned.improvement
        assert data["baseline"] == asdict(tuned.baseline)
        assert data["best"] == asdict(tuned.best)
        assert data["baseline_metrics"] == tuned.baseline_metrics
        assert data["best_metrics"] == tuned.best_metrics
        assert data["decisions"] == tuned.decisions
        assert data["notes"] == tuned.notes

    def test_proposed_settings_keep_keys_the_tuner_does_not_own(self, tuned: TuneResult):
        proposed = report.to_json(RUN_ID, _profile(SimModel()), tuned, META)["proposed_settings"]
        assert proposed["serverBinary"] == CURRENT["serverBinary"]
        assert proposed["loraFile"] == CURRENT["loraFile"]
        # the tuner's own keys reflect the winning config
        assert proposed["contextSize"] == tuned.best.ctx
        assert proposed["cacheTypeK"] == tuned.best.cache_k
        assert proposed["extraArgs"]  # rebuilt, with the enforced flags in it

    def test_changes_list_exactly_the_differing_keys(self, tuned: TuneResult):
        data = report.to_json(RUN_ID, _profile(SimModel()), tuned, META)
        proposed, changes = data["proposed_settings"], data["changes"]
        assert changes, "the run improved on the baseline, so changes must not be empty"
        for k, v in proposed.items():
            if CURRENT.get(k) != v:
                assert k in changes, f"{k} differs from the saved settings but is not listed in changes"
                assert changes[k] == [CURRENT.get(k), v]
            else:
                assert k not in changes, f"{k} matches the saved settings but is listed in changes"
        assert all(isinstance(v, list) and len(v) == 2 for v in changes.values())
        assert "serverBinary" not in changes and "loraFile" not in changes
        assert changes["contextSize"] == [32768, tuned.best.ctx]

    def test_the_whole_document_is_json_serialisable(self, tuned: TuneResult):
        data = report.to_json(RUN_ID, _profile(SimModel()), tuned, META)
        back = json.loads(json.dumps(data))
        # JSON has no tuples: the passthrough flags come back as lists.
        back["baseline"]["passthrough"] = tuple(back["baseline"]["passthrough"])
        back["best"]["passthrough"] = tuple(back["best"]["passthrough"])
        assert back == data

    def test_trials_carry_phase_config_outcome_and_medians(self, tuned: TuneResult):
        data = report.to_json(RUN_ID, _profile(SimModel()), tuned, META)
        assert len(data["trials"]) == len(tuned.trials) > 0
        for got, t in zip(data["trials"], tuned.trials):
            assert got["phase"] == t.phase
            assert got["config"] == t.cfg.describe()
            assert got["outcome"] == t.outcome
            assert got["medians"] == t.medians()
            assert got["reason"] == t.reason[:300]


class TestToMarkdown:
    def test_result_table(self, tuned: TuneResult):
        md = report.to_markdown(RUN_ID, _profile(SimModel()), tuned, META)
        assert md.startswith("# Model Tuner report: sim-9b.gguf")
        assert f"Run `{RUN_ID}`, mode **quick**, status **ok**, {tuned.loads} loads" in md
        assert "| Workload | Current | Tuned | Change |" in md
        assert "| Decode, prose |" in md and "| Decode, code edit |" in md and "| Prompt processing |" in md
        assert f"{tuned.improvement * 100:+.1f}%" in md
        assert f"**Current:** {tuned.baseline.describe()}" in md
        assert f"**Tuned:** {tuned.best.describe()}" in md

    def test_proposed_changes_table(self, tuned: TuneResult):
        profile = _profile(SimModel())
        md = report.to_markdown(RUN_ID, profile, tuned, META)
        assert "## Proposed changes to models.json" in md
        assert "| Setting | Current | Proposed |" in md
        for k, (old, new) in report.settings_changes(profile, tuned.best).items():
            assert f"| {k} | `{old}` | `{new}` |" in md
        assert "Apply with: `python -m modeltuner apply" in md

    def test_every_decision_is_listed(self, tuned: TuneResult):
        md = report.to_markdown(RUN_ID, _profile(SimModel()), tuned, META)
        assert "## Decisions, in order" in md
        assert tuned.decisions
        for d in tuned.decisions:
            assert d in md

    def test_trials_table(self, tuned: TuneResult):
        md = report.to_markdown(RUN_ID, _profile(SimModel()), tuned, META)
        assert "## Every configuration tried" in md
        assert "| Phase | Configuration | Outcome | Prose | Code | Prefill | VRAM (proj. / actual MiB) |" in md
        assert tuned.trials
        for t in tuned.trials:
            assert f"| {t.phase} | {t.cfg.describe()} | {t.outcome} |" in md

    def test_what_the_tuner_knew_is_included(self, tuned: TuneResult):
        profile = _profile(SimModel())
        md = report.to_markdown(RUN_ID, profile, tuned, META)
        for line in profile.summary_lines():
            assert line in md

    def test_when_the_current_settings_are_already_the_best(self):
        cfg = Config(ctx=32768)
        res = TuneResult(status="ok", baseline=cfg, best=cfg, baseline_metrics={"prose": 50.0, "code": 25.0},
                         best_metrics={"prose": 50.0, "code": 25.0}, improvement=0.0, decisions=[], notes=[],
                         trials=[], loads=0, seconds=1.0)
        profile = _profile(SimModel(), current=to_settings(cfg))
        md = report.to_markdown("run-x", profile, res, {"mode": "quick", "power_key": "p"})
        assert "already the best found. Nothing to change." in md
        assert "Proposed changes" not in md


class TestWrite:
    def test_writes_both_files_and_returns_the_json(self, tuned: TuneResult, tmp_path):
        profile = _profile(SimModel())
        data = report.write(tmp_path, RUN_ID, profile, tuned, META)
        assert data == report.to_json(RUN_ID, profile, tuned, META)
        back = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
        back["baseline"]["passthrough"] = tuple(back["baseline"]["passthrough"])
        back["best"]["passthrough"] = tuple(back["best"]["passthrough"])
        assert back == data
        assert (tmp_path / "report.md").read_text(encoding="utf-8") == report.to_markdown(RUN_ID, profile, tuned, META)
