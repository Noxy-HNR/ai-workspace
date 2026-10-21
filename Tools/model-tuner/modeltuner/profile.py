"""What the tuner knows about a model before loading it: `ModelProfile`, built from the GGUF header and models.json.

Everything here drives decisions later. Examples:
- A hybrid model (qwen35: attention on every 4th layer, recurrent state elsewhere) needs very little context memory,
  so its context can be pushed far higher than its file size suggests.
- A model with an MTP (NextN) head can use `--spec-type draft-mtp`; one without it would crash with that flag.
- A chat template that reads `reasoning_effort` makes the effort setting meaningful; one that ignores it makes the
  thinking cap (`--reasoning-budget`) the only lever on thinking length.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from modeltuner import paths
from modeltuner.gguf import ArraySummary, GGUFFile, read_gguf
from modeltuner.knobs import Config, from_settings


@dataclass(frozen=True)
class ModelProfile:
    file: str
    path: str
    size_bytes: int
    arch: str
    name: str
    n_layer: int                      # repeating blocks, including an MTP block if present
    n_mtp_layers: int                 # NextN / MTP prediction blocks at the end
    train_ctx: int
    n_embd: int
    n_head: int
    n_head_kv: int
    head_dim_k: int
    head_dim_v: int
    full_attention_interval: int | None   # hybrid attention/recurrent models (qwen35)
    sliding_window: int | None            # gemma3-style sliding-window layers
    n_expert: int                         # 0 = dense
    weights_bytes: int
    layer_bytes: dict[int, int]
    template_supports_tools: bool
    template_has_thinking: bool
    template_honours_effort: bool
    server_binary: str | None
    current: dict[str, Any] = field(default_factory=dict)   # the model's saved dashboard settings

    # ---- derived facts ----
    @property
    def is_hybrid(self) -> bool:
        return bool(self.full_attention_interval and self.full_attention_interval > 1)

    @property
    def has_mtp(self) -> bool:
        return self.n_mtp_layers > 0

    @property
    def is_moe(self) -> bool:
        return self.n_expert > 0

    @property
    def weights_mib(self) -> int:
        return self.weights_bytes // (1024 * 1024)

    @property
    def attention_layers(self) -> int:
        """Layers that keep a KV cache (the MTP block always has attention)."""
        base = self.n_layer - self.n_mtp_layers
        if self.is_hybrid:
            n = base // self.full_attention_interval
        else:
            n = base
        return n + self.n_mtp_layers

    def kv_bytes_per_token(self, cache_k: str = "f16", cache_v: str = "f16") -> float:
        """Analytic context-memory estimate per token (the fallback when llama-fit-params is unavailable)."""
        per = {"f32": 4, "f16": 2, "bf16": 2, "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32,
               "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32}
        k = self.n_head_kv * self.head_dim_k * per.get(cache_k, 2)
        v = self.n_head_kv * self.head_dim_v * per.get(cache_v, 2)
        layers = self.attention_layers
        if self.sliding_window:
            # gemma3: 5 of every 6 layers only keep `sliding_window` tokens; charge the full layers only.
            layers = max(1, round(layers / 6))
        return layers * (k + v)

    @property
    def family(self) -> str:
        """Group used for priors and calibration: models in one family behave alike."""
        tags = [self.arch]
        if self.is_hybrid:
            tags.append("hybrid")
        if self.is_moe:
            tags.append("moe")
        return "-".join(tags)

    def current_config(self) -> Config:
        return from_settings(self.current) if self.current else Config(ctx=min(self.train_ctx, 16384))

    def summary_lines(self) -> list[str]:
        att = f"{self.attention_layers} of {self.n_layer} layers keep a KV cache" if self.is_hybrid else \
            f"all {self.n_layer} layers keep a KV cache"
        lines = [
            f"{self.file}: {self.arch}, {self.size_bytes / 2**30:.2f} GiB, {self.n_layer} layers, trained for "
            f"{self.train_ctx:,} tokens of context.",
            f"Attention: {att}; {self.n_head_kv} KV heads x {self.head_dim_k}; about "
            f"{self.kv_bytes_per_token('q8_0', 'q8_0') / 1024:.1f} KiB per token of context at q8_0.",
        ]
        if self.sliding_window:
            lines.append(f"Sliding-window attention ({self.sliding_window} tokens) on most layers.")
        lines.append("MTP (NextN) head present: draft-mtp speculation is possible." if self.has_mtp
                     else "No MTP head: only n-gram speculation applies.")
        if self.is_moe:
            lines.append(f"Mixture of experts ({self.n_expert} experts): --n-cpu-moe is a lever.")
        t = []
        t.append("tool calls" if self.template_supports_tools else "NO tool calls")
        if self.template_has_thinking:
            t.append("thinking" + (" (honours reasoning effort)" if self.template_honours_effort
                                   else " (ignores reasoning effort; the thinking cap is the lever)"))
        lines.append("Chat template: " + ", ".join(t) + ".")
        if self.server_binary:
            lines.append(f"Runs on a custom llama-server build: {self.server_binary}.")
        return lines

    def as_dict(self) -> dict:
        d = asdict(self)
        d["layer_bytes"] = {str(k): v for k, v in self.layer_bytes.items()}
        d.update(is_hybrid=self.is_hybrid, has_mtp=self.has_mtp, is_moe=self.is_moe,
                 attention_layers=self.attention_layers, family=self.family)
        return d


def _int(v: Any, default: int = 0) -> int:
    if isinstance(v, list):
        v = max(v) if v else default     # per-layer arrays (e.g. head_count_kv): the max is the safe estimate
    if isinstance(v, ArraySummary) or v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def saved_settings(model_file: str, models_json: Path = paths.MODELS_JSON) -> dict[str, Any]:
    try:
        data = json.loads(models_json.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return dict((data.get("models") or {}).get(model_file) or {})


def analyse_template(template: str) -> tuple[bool, bool, bool]:
    """(supports tools, has a thinking mode, honours reasoning_effort) from a Jinja chat template."""
    t = template or ""
    tools = bool(re.search(r"\btools\b", t)) and ("tool_call" in t or "tool_calls" in t)
    thinking = any(s in t for s in ("<think>", "enable_thinking", "reasoning_content", "thinking"))
    effort = "reasoning_effort" in t
    return tools, thinking, effort


def build_profile(model_file: str, *, models_dir: Path | None = None,
                  models_json: Path = paths.MODELS_JSON, gguf: GGUFFile | None = None) -> ModelProfile:
    path = models_dir / model_file if models_dir else paths.resolve_model(model_file)
    g = gguf or read_gguf(path)
    arch = g.architecture or "unknown"
    n_layer = _int(g.arch_get("block_count"))
    n_head = _int(g.arch_get("attention.head_count"))
    n_embd = _int(g.arch_get("embedding_length"))
    head_k = _int(g.arch_get("attention.key_length")) or (n_embd // n_head if n_head else 0)
    head_v = _int(g.arch_get("attention.value_length")) or head_k
    tools, thinking, effort = analyse_template(str(g.get("tokenizer.chat_template") or ""))
    settings = saved_settings(model_file, models_json)
    return ModelProfile(
        file=model_file,
        path=str(path),
        size_bytes=g.file_size,
        arch=arch,
        name=str(g.get("general.name") or model_file),
        n_layer=n_layer,
        n_mtp_layers=_int(g.arch_get("nextn_predict_layers")),
        train_ctx=_int(g.arch_get("context_length"), 4096),
        n_embd=n_embd,
        n_head=n_head,
        n_head_kv=_int(g.arch_get("attention.head_count_kv")) or n_head,
        head_dim_k=head_k,
        head_dim_v=head_v,
        full_attention_interval=_int(g.arch_get("full_attention_interval")) or None,
        sliding_window=_int(g.arch_get("attention.sliding_window")) or None,
        n_expert=_int(g.arch_get("expert_count")),
        weights_bytes=g.total_tensor_bytes(),
        layer_bytes=g.layer_bytes(),
        template_supports_tools=tools,
        template_has_thinking=thinking,
        template_honours_effort=effort,
        server_binary=settings.get("serverBinary") or None,
        current=settings,
    )


def list_models(models_dir: Path | None = None) -> list[str]:
    """Loadable model files on this PC and the model server (LoRA adapters excluded by their general.type)."""
    out = []
    files = {p.name: p for p in models_dir.glob("*.gguf")} if models_dir else paths.model_files()
    for p in (files[n] for n in sorted(files)):
        try:
            g = read_gguf(p, max_array_len=0)
        except Exception:
            continue
        if str(g.get("general.type") or "model") != "adapter":
            out.append(p.name)
    return out
