"""Minimal llama-server HTTP client with hard timeouts on every call. Standard library only."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modeltuner import paths


class LlamaError(RuntimeError):
    pass


@dataclass
class ChatResult:
    content: str
    reasoning: str
    tool_calls: list[dict]
    finish_reason: str | None
    timings: dict
    usage: dict
    wall_s: float
    raw: dict


class LlamaClient:
    def __init__(self, base_url: str = paths.LLAMA_URL, default_timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.default_timeout = default_timeout

    def _request(self, method: str, path: str, body: dict | None = None, timeout: float | None = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.default_timeout) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise LlamaError(f"{method} {path} -> HTTP {e.code}: {detail}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise LlamaError(f"{method} {path} failed: {e}") from e

    def props(self, timeout: float = 10) -> dict:
        return self._request("GET", "/props", timeout=timeout)

    def ready(self, timeout: float = 5) -> bool:
        try:
            return bool(self._request("GET", "/v1/models", timeout=timeout).get("data"))
        except LlamaError:
            return False

    def loaded_model(self) -> str | None:
        try:
            return Path(self.props()["model_path"]).name
        except (LlamaError, KeyError, TypeError):
            return None

    def n_ctx(self) -> int | None:
        try:
            return int(self.props()["default_generation_settings"]["n_ctx"])
        except (LlamaError, KeyError, TypeError, ValueError):
            return None

    def tokenize(self, text: str, timeout: float = 120) -> int:
        return len(self._request("POST", "/tokenize", {"content": text}, timeout=timeout)["tokens"])

    def chat(self, messages: list[dict], *, max_tokens: int = 1024, temperature: float = 0.0, seed: int | None = 1234,
             tools: list[dict] | None = None, thinking: bool | None = None, timeout: float | None = None,
             **extra: Any) -> ChatResult:
        """One /v1/chat/completions call. `thinking=False` disables the model's thinking block when the template
        supports it (enable_thinking); None leaves the template default."""
        body: dict[str, Any] = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                                "stream": False}
        if seed is not None:
            body["seed"] = seed
        if tools:
            body["tools"] = tools
        if thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
        body.update(extra)
        t0 = time.perf_counter()
        raw = self._request("POST", "/v1/chat/completions", body, timeout=timeout)
        wall = time.perf_counter() - t0
        choice = (raw.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        return ChatResult(
            content=msg.get("content") or "",
            reasoning=msg.get("reasoning_content") or "",
            tool_calls=msg.get("tool_calls") or [],
            finish_reason=choice.get("finish_reason"),
            timings=raw.get("timings") or {},
            usage=raw.get("usage") or {},
            wall_s=wall,
            raw=raw,
        )
