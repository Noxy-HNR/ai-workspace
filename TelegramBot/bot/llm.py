"""LLM interaction: context building, chat calls, reply chunking.

Context is always built from a single user's stored messages. The caller
passes the user id and the already-loaded history; there is no global
message pool, so cross-user mixing is impossible by construction.
"""
from __future__ import annotations

import json
import html
import re
import urllib.request
import urllib.error
import urllib.parse
from .http_json import request_json


def estimate_tokens(text: str) -> int:
    return (len(text or "") + 3) // 4


def build_context(history: list[dict], system_prompt: str,
                  history_limit: int, max_tokens: int,
                  max_user_chars: int) -> dict:
    """Split one user's stored history into effective context vs. archived.

    Returns {system, messages, included_ids, excluded_ids, est_tokens,
    dropped_for_count, dropped_for_tokens}.
    """
    eligible = [m for m in history
                if m.get("in_context") and m.get("status") == "done"
                and m.get("kind") in ("text", "command")
                and m.get("direction") in ("in", "out")
                and (m.get("text") or "").strip()]
    total = len(eligible)
    window = eligible[-max(1, history_limit):]
    newest_user = next((m for m in reversed(eligible) if m["direction"] == "in"), None)
    if newest_user is not None and all(m["id"] != newest_user["id"] for m in window):
        window = [newest_user] + window[1:]
    dropped_for_count = total - len(window)

    # Reserve room for the conversation before dropping history. Truncating
    # this later would discard useful turns just because the initial system
    # prompt alone was too large for the owner's chosen budget.
    system_truncated = estimate_tokens(system_prompt) >= max_tokens
    if system_truncated:
        system_prompt = system_prompt[:max(0, (max_tokens // 2) * 4)]

    msgs: list[dict] = []
    char_trimmed_ids, token_trimmed_ids = set(), set()
    for m in window:
        text = m["text"]
        if m["direction"] == "in" and len(text) > max_user_chars:
            char_trimmed_ids.add(m["id"])
            text = (text[:max_user_chars]
                    + f"\n\n[…message trimmed for context: {len(m['text'])}"
                    f" chars stored, first {max_user_chars} sent…]")
        msgs.append({"role": "user" if m["direction"] == "in" else "assistant",
                     "content": text, "_id": m["id"]})

    # Token cap: drop oldest first, always keep the newest user message.
    dropped_for_tokens = 0
    newest_user = next((m["_id"] for m in reversed(msgs) if m["role"] == "user"), None)
    while True:
        est = estimate_tokens(system_prompt) + sum(
            estimate_tokens(m["content"]) for m in msgs)
        if est <= max_tokens or len(msgs) <= 1:
            break
        # Never drop the final user message; drop oldest assistant instead.
        if msgs[-1]["role"] == "user" and len(msgs) == 1:
            break
        drop_idx = next((i for i,m in enumerate(msgs) if m["_id"] != newest_user), 0)
        msgs.pop(drop_idx)
        dropped_for_tokens += 1

    # Even one oversized message plus the system prompt must fit the cap.
    remaining = max(1, max_tokens - estimate_tokens(system_prompt))
    if len(msgs) == 1 and estimate_tokens(msgs[0]["content"]) > remaining:
        token_trimmed_ids.add(msgs[0]["_id"])
        msgs[0]["content"] = msgs[0]["content"][:remaining * 4]

    est = estimate_tokens(system_prompt) + sum(
        estimate_tokens(m["content"]) for m in msgs)
    included = {m.pop("_id") for m in msgs}
    excluded = [m["id"] for m in eligible if m["id"] not in included]
    return {"system": system_prompt, "messages": msgs,
            "included_ids": sorted(included), "excluded_ids": excluded,
            "est_tokens": est, "eligible": total,
            "dropped_for_count": dropped_for_count,
            "dropped_for_tokens": dropped_for_tokens,
            "system_truncated": system_truncated,
            "content_trimmed_ids": sorted((char_trimmed_ids | token_trimmed_ids) & included),
            "trimmed_for_chars": len(char_trimmed_ids & included),
            "trimmed_for_tokens": len(token_trimmed_ids & included),
            "token_budget": max_tokens}


def chat_completion(base_url: str, model: str, api_key: str,
                    system: str, messages: list[dict], temperature: float,
                    max_tokens: int, top_p: float,
                    timeout: int, cancel_event=None) -> str:
    """Single OpenAI-compatible chat call. Raises RuntimeError on failure."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload_msgs = [{"role": "system", "content": system}] if system else []
    payload_msgs += [{"role": m["role"], "content": m["content"]}
                     for m in messages]
    payload: dict = {"model": model or "default", "messages": payload_msgs,
                     "temperature": temperature, "top_p": top_p,
                     "stream": False}
    if max_tokens and max_tokens > 0:
        payload["max_tokens"] = max_tokens
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    parsed = urllib.parse.urlsplit(base_url)
    guard = parsed.hostname in ("localhost", "127.0.0.1", "::1") and parsed.port == 10000
    body = request_json(url, payload, dict(req.header_items()), timeout,
                        guard_model=guard, cancel_event=cancel_event)
    try:
        content = body["choices"][0]["message"]["content"]
        if content is not None and not isinstance(content, str):
            raise TypeError("Expected text")
        return content or ""
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError("LLM returned an invalid chat response") from None


def list_models(base_url: str, api_key: str, timeout=10) -> list[str]:
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url)
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    body = request_json(url, headers=dict(req.header_items()), timeout=timeout)
    try:
        data = body["data"]
        if not isinstance(data, list):
            raise TypeError("Expected model list")
        return [m["id"] for m in data if isinstance(m, dict) and
                isinstance(m.get("id"), str) and m["id"]]
    except (KeyError, TypeError):
        raise RuntimeError("LLM returned an invalid model list") from None


def chunk_reply(text: str, limit=4000) -> list[str]:
    """Split text without losing whitespace, counting UTF-16 units safely.

    Paragraph and line endings are preferred; an emoji's two UTF-16 units
    always remain together. Joining the chunks reproduces the original text.
    """
    if limit < 2:
        raise ValueError("Reply chunk limit must be at least 2")
    text = text or ""
    parts, start = [], 0
    while start < len(text):
        end, units = start, 0
        while end < len(text):
            width = 2 if ord(text[end]) > 0xFFFF else 1
            if units + width > limit:
                break
            end += 1
            units += width
        if end < len(text):
            segment = text[start:end]
            for separator in ("\n\n", "\n", " "):
                at = segment.rfind(separator)
                if at >= len(segment) // 2:
                    end = start + at + len(separator)
                    break
        parts.append(text[start:end])
        start = end
    return parts or [""]


_MARKDOWN = re.compile(
    r"```(?:[a-zA-Z0-9_+-]*\n)?([\s\S]*?)```|"
    r"`([^`\n]+)`|\*\*([^*]+)\*\*|__([^_]+)__|"
    r"(?<!\w)\*([^*\n]+)\*(?!\w)|~~([^~]+)~~|"
    r"\[([^\]\n]+)\]\((https?://[^\s()]+)\)")


def telegram_html(text: str) -> str:
    """Render a small safe Markdown subset using Telegram's HTML tags.

    Raw model HTML is escaped. Links accept only HTTP(S); fenced and inline
    code never gain active markup. Unsupported syntax remains readable text.
    """
    pieces, previous = [], 0
    for match in _MARKDOWN.finditer(text):
        pieces.append(html.escape(text[previous:match.start()], quote=False))
        code, inline, bold, under_bold, italic, strike, label, url = match.groups()
        if code is not None:
            rendered = "<pre>" + html.escape(code, quote=False) + "</pre>"
        elif inline is not None:
            rendered = "<code>" + html.escape(inline, quote=False) + "</code>"
        elif bold is not None or under_bold is not None:
            rendered = "<b>" + html.escape(bold if bold is not None else under_bold, quote=False) + "</b>"
        elif italic is not None:
            rendered = "<i>" + html.escape(italic, quote=False) + "</i>"
        elif strike is not None:
            rendered = "<s>" + html.escape(strike, quote=False) + "</s>"
        else:
            rendered = '<a href="' + html.escape(url, quote=True) + '">' + html.escape(label, quote=False) + "</a>"
        pieces.append(rendered)
        previous = match.end()
    pieces.append(html.escape(text[previous:], quote=False))
    return "".join(pieces)
