"""Shared helpers for the proposed tests (copied from tests/test_strategy_sim.py so these files stand alone)."""

from modeltuner.knobs import Config
from modeltuner.profile import ModelProfile
from modeltuner.sim import SimBackend, SimGpu, SimModel
from modeltuner.strategy import TuneOptions, Tuner


def profile(m: SimModel, **kw) -> ModelProfile:
    d = dict(file=m.name, path="x", size_bytes=m.weights_mib << 20, arch="qwen35", name="sim", n_layer=m.n_layer,
             n_mtp_layers=1 if m.has_mtp else 0, train_ctx=m.train_ctx, n_embd=4096, n_head=16, n_head_kv=4,
             head_dim_k=256, head_dim_v=256, full_attention_interval=4, sliding_window=None, n_expert=0,
             weights_bytes=m.weights_mib << 20, layer_bytes={}, template_supports_tools=True,
             template_has_thinking=True, template_honours_effort=False, server_binary=None, current={})
    d.update(kw)
    return ModelProfile(**d)


def tuner(backend, m: SimModel, base: Config, prof: ModelProfile | None = None, **opts):
    """A Tuner that records every event it emits."""
    events: list[tuple[str, str]] = []
    t = Tuner(backend, prof or profile(m), base, TuneOptions(**opts),
              emit=lambda kind, msg, **d: events.append((kind, msg)))
    return t, events


def decisions(events) -> list[str]:
    return [msg for kind, msg in events if kind == "decision"]


def compares(events) -> list[str]:
    return [msg for kind, msg in events if kind == "compare"]


def quality_compares(events) -> list[str]:
    return [msg for msg in compares(events) if msg.startswith("quality ")]


def loads_of(events) -> list[str]:
    return [msg for kind, msg in events if kind == "load"]


__all__ = ["Config", "SimBackend", "SimGpu", "SimModel", "TuneOptions", "Tuner",
           "profile", "tuner", "decisions", "compares", "quality_compares", "loads_of"]
