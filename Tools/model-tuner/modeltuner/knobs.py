"""Typed llama-server settings (`Config`) and the round trip to and from the dashboard's per-model settings.

The dashboard stores a model's settings as a flat dict (contextSize, gpuLayers, ... extraArgs). Several knobs the
tuner cares about (speculative decoding, the thinking cap) live inside the free-text `extraArgs`, so this module
parses extraArgs into typed fields and rebuilds it, keeping every flag it does not understand verbatim.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

# Always present in extraArgs. The defaults (memory-mapped loading and an 8 GB prompt cache) exhausted RAM before.
ENFORCED_ARGS: tuple[str, ...] = ("--load-mode", "none", "--cache-ram", "0")

SPEC_NONE = "none"
KNOWN_SPEC_TYPES = ("none", "ngram-mod", "draft-mtp", "draft-mtp,ngram-mod", "ngram-simple", "ngram-map-k", "ngram-cache")
CACHE_TYPES = ("f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl")
EFFORTS = ("default", "low", "medium", "high", "xhigh")

# extraArgs flags that map to typed Config fields: flag -> (field, converter). Aliases share a field.
_VALUE_FLAGS: dict[str, tuple[str, type]] = {
    "--spec-type": ("spec", str),
    "--spec-draft-n-max": ("draft_n_max", int),
    "--draft-max": ("draft_n_max", int),
    "--spec-ngram-mod-n-match": ("ngram_n_match", int),
    "--reasoning-budget": ("reasoning_budget", int),
    "-tb": ("threads_batch", int),
    "--threads-batch": ("threads_batch", int),
    "-ncmoe": ("n_cpu_moe", int),
    "--n-cpu-moe": ("n_cpu_moe", int),
}
# The one -ot rule the tuner writes itself (`n_cpu_ffn`): the FFN weights of blocks 0..N-1 stay in system RAM while
# attention, the KV cache and every other block run on the GPU. Any other -ot rule is kept verbatim in passthrough.
_FFN_OT = re.compile(r"^blk\\\.\(([0-9|]+)\)\\\.ffn_\(up\|gate\|down\)\\\.weight=CPU$")
_OT_FLAGS = ("-ot", "--override-tensor")
# Flags the tuner owns through first-class settings; they must not also appear in extraArgs.
_OWNED_ELSEWHERE = {
    "-c", "--ctx-size", "-ngl", "--n-gpu-layers", "--gpu-layers", "-t", "--threads", "-b", "--batch-size",
    "-ub", "--ubatch-size", "-ctk", "--cache-type-k", "-ctv", "--cache-type-v", "-fa", "--flash-attn",
    "--reasoning-effort",
}


@dataclass(frozen=True)
class Config:
    """One complete llama-server configuration for one model."""

    ctx: int
    ngl: int = 999
    threads: int = 8
    batch: int = 1024
    ubatch: int = 256
    cache_k: str = "q8_0"
    cache_v: str = "q8_0"
    flash_attn: bool = True
    spec: str = SPEC_NONE
    draft_n_max: int | None = None
    ngram_n_match: int | None = None
    reasoning_budget: int | None = None
    reasoning_effort: str = "default"
    threads_batch: int | None = None
    n_cpu_moe: int | None = None
    n_cpu_ffn: int | None = None        # dense models: FFN weights of the first N blocks on the CPU (see ffn_override)
    parallel: int = 1
    jinja: bool = True
    passthrough: tuple[str, ...] = field(default_factory=tuple)

    def with_(self, **changes: Any) -> "Config":
        return replace(self, **changes)

    @property
    def uses_mtp(self) -> bool:
        return "draft-mtp" in self.spec

    @property
    def uses_ngram(self) -> bool:
        return "ngram" in self.spec

    def key(self) -> str:
        """Stable identity used to deduplicate trials and look results up in history."""
        d = asdict(self)
        d["passthrough"] = " ".join(self.passthrough)
        if d["n_cpu_ffn"] is None:
            del d["n_cpu_ffn"]      # added later; leaving it out keeps keys of older history rows matching
        return "|".join(f"{k}={d[k]}" for k in sorted(d))

    def describe(self) -> str:
        """Short human summary, e.g. 'ctx 131072, all layers on GPU, q8_0 cache, MTP+ngram (n_max 2)'."""
        parts = [f"ctx {self.ctx}"]
        parts.append("all layers on GPU" if self.ngl >= 999 else f"{self.ngl} layers on GPU")
        if self.n_cpu_ffn:
            parts.append(f"FFN of {self.n_cpu_ffn} layers on CPU")
        parts.append(f"{self.cache_k} cache" if self.cache_k == self.cache_v else f"K {self.cache_k}/V {self.cache_v}")
        if self.spec != SPEC_NONE:
            s = self.spec.replace("draft-mtp", "MTP").replace("ngram-mod", "ngram").replace(",", "+")
            extra = []
            if self.uses_mtp and self.draft_n_max:
                extra.append(f"n_max {self.draft_n_max}")
            if self.uses_ngram and self.ngram_n_match:
                extra.append(f"match {self.ngram_n_match}")
            parts.append(s + (f" ({', '.join(extra)})" if extra else ""))
        else:
            parts.append("no speculation")
        parts.append(f"batch {self.batch}/{self.ubatch}")
        parts.append(f"{self.threads} threads" + (f" (+{self.threads_batch} batch)" if self.threads_batch else ""))
        if self.reasoning_budget is not None:
            parts.append(f"thinking cap {self.reasoning_budget}")
        if self.reasoning_effort != "default":
            parts.append(f"effort {self.reasoning_effort}")
        if self.n_cpu_moe:
            parts.append(f"{self.n_cpu_moe} MoE layers on CPU")
        return ", ".join(parts)


def ffn_override(n: int) -> str:
    """The -ot value that keeps the FFN weights of blocks 0..n-1 on the CPU."""
    return "blk\\.(" + "|".join(str(i) for i in range(n)) + ")\\.ffn_(up|gate|down)\\.weight=CPU"


def parse_ffn_override(value: str) -> int | None:
    """n for a value `ffn_override(n)` wrote, else None (a hand-written rule the tuner must not reinterpret)."""
    m = _FFN_OT.match(value)
    if not m:
        return None
    blocks = m.group(1).split("|")
    n = len(blocks)
    return n if blocks == [str(i) for i in range(n)] else None


def split_args(text: str) -> list[str]:
    """Tokenise an extraArgs string the same way the dashboard does (quotes group, whitespace splits)."""
    if not text or not text.strip():
        return []
    return [m.strip('"').strip("'") for m in re.findall(r'"[^"]*"|\'[^\']*\'|\S+', text)]


def parse_extra_args(text: str) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Split extraArgs into typed values (for Config fields) and everything else (kept verbatim, in order).

    The enforced RAM-safety flags are dropped from the passthrough here and re-added by `build_extra_args`, so they are
    always present exactly once.
    """
    toks = split_args(text)
    typed: dict[str, Any] = {}
    rest: list[str] = []
    i = 0
    while i < len(toks):
        tok = toks[i]
        name, eq, inline = tok.partition("=")
        if name in _VALUE_FLAGS:
            fld, conv = _VALUE_FLAGS[name]
            if eq:
                raw, i = inline, i + 1
            elif i + 1 < len(toks):
                raw, i = toks[i + 1], i + 2
            else:
                raise ValueError(f"{name} in extraArgs has no value")
            try:
                val = conv(raw)
            except ValueError as e:
                raise ValueError(f"{name} in extraArgs has a bad value {raw!r}") from e
            if fld == "reasoning_budget" and val < 0:
                val = None  # -1 means unlimited
            typed[fld] = val
            continue
        if name in _OT_FLAGS and "n_cpu_ffn" not in typed:
            raw = inline if eq else (toks[i + 1] if i + 1 < len(toks) else "")
            n = parse_ffn_override(raw)
            if n:
                typed["n_cpu_ffn"] = n
                i += 1 if eq else 2
                continue
        if name in ("--load-mode", "--cache-ram") and not eq and i + 1 < len(toks):
            i += 2
            continue
        if name in ("--load-mode", "--cache-ram") and eq:
            i += 1
            continue
        if name in _OWNED_ELSEWHERE:
            raise ValueError(f"{name} belongs in the model's own settings, not extraArgs")
        rest.append(tok)
        i += 1
    return typed, tuple(rest)


def build_extra_args(cfg: Config) -> str:
    """Rebuild extraArgs from a Config: typed flags first, then the enforced flags, then the passthrough."""
    out: list[str] = []
    if cfg.spec and cfg.spec != SPEC_NONE:
        out += ["--spec-type", cfg.spec]
        if cfg.draft_n_max is not None and cfg.uses_mtp:
            out += ["--spec-draft-n-max", str(cfg.draft_n_max)]
        if cfg.ngram_n_match is not None and cfg.uses_ngram:
            out += ["--spec-ngram-mod-n-match", str(cfg.ngram_n_match)]
    if cfg.threads_batch is not None:
        out += ["-tb", str(cfg.threads_batch)]
    if cfg.n_cpu_moe:
        out += ["--n-cpu-moe", str(cfg.n_cpu_moe)]
    if cfg.n_cpu_ffn:
        out += ["-ot", ffn_override(cfg.n_cpu_ffn)]
    out += list(ENFORCED_ARGS)
    if cfg.reasoning_budget is not None:
        out += ["--reasoning-budget", str(cfg.reasoning_budget)]
    out += list(cfg.passthrough)
    return " ".join(shlex.quote(t) if " " in t else t for t in out)


def from_settings(settings: Mapping[str, Any]) -> Config:
    """Dashboard settings dict -> Config. Unknown extraArgs flags are preserved in `passthrough`."""
    typed, rest = parse_extra_args(str(settings.get("extraArgs") or ""))
    effort = str(settings.get("reasoningEffort") or "default")
    return Config(
        ctx=int(settings.get("contextSize") or 16384),
        ngl=int(settings.get("gpuLayers") if settings.get("gpuLayers") is not None else 999),
        threads=int(settings.get("threads") or 8),
        batch=int(settings.get("batchSize") or 2048),
        ubatch=int(settings.get("ubatchSize") or 512),
        cache_k=str(settings.get("cacheTypeK") or "f16"),
        cache_v=str(settings.get("cacheTypeV") or "f16"),
        flash_attn=bool(settings.get("flashAttn", True)),
        parallel=int(settings.get("parallel") or 1),
        jinja=bool(settings.get("jinja", True)),
        reasoning_effort=effort if effort in EFFORTS else "default",
        spec=typed.get("spec", SPEC_NONE),
        draft_n_max=typed.get("draft_n_max"),
        ngram_n_match=typed.get("ngram_n_match"),
        reasoning_budget=typed.get("reasoning_budget"),
        threads_batch=typed.get("threads_batch"),
        n_cpu_moe=typed.get("n_cpu_moe"),
        n_cpu_ffn=typed.get("n_cpu_ffn"),
        passthrough=rest,
    )


def to_settings(cfg: Config, base: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Config -> dashboard settings dict. Keys the tuner does not own (serverBinary, loraFile, rope settings...) are
    copied from `base` unchanged."""
    out: dict[str, Any] = dict(base or {})
    out.update(
        contextSize=cfg.ctx,
        gpuLayers=cfg.ngl,
        flashAttn=cfg.flash_attn,
        parallel=cfg.parallel,
        jinja=cfg.jinja,
        batchSize=cfg.batch,
        ubatchSize=cfg.ubatch,
        threads=cfg.threads,
        cacheTypeK=cfg.cache_k,
        cacheTypeV=cfg.cache_v,
        reasoningEffort=cfg.reasoning_effort,
        extraArgs=build_extra_args(cfg),
    )
    out.setdefault("ropeFreqBase", 0)
    out.setdefault("ropeFreqScale", 0)
    return out


def fit_params_args(cfg: Config) -> list[str]:
    """The subset of llama-server arguments that llama-fit-params needs to project memory for this Config."""
    args = [
        "-c", str(cfg.ctx),
        "-b", str(cfg.batch),
        "-ub", str(cfg.ubatch),
        "-ctk", cfg.cache_k,
        "-ctv", cfg.cache_v,
        "-fa", "on" if cfg.flash_attn else "off",
        "--parallel", str(cfg.parallel),
        "--load-mode", "none",
    ]
    if cfg.ngl < 999:
        args += ["-ngl", str(cfg.ngl)]
    if cfg.n_cpu_moe:
        args += ["--n-cpu-moe", str(cfg.n_cpu_moe)]
    if cfg.n_cpu_ffn:
        args += ["-ot", ffn_override(cfg.n_cpu_ffn)]
    # llama-fit-params rejects --spec-type, so a draft context (MTP) is not in its projection. The calibration step
    # measures that overhead per architecture instead.
    return args


def diff(a: Config, b: Config) -> dict[str, tuple[Any, Any]]:
    """Fields that differ between two configs: {field: (a_value, b_value)}."""
    da, db = asdict(a), asdict(b)
    return {k: (da[k], db[k]) for k in da if da[k] != db[k]}
