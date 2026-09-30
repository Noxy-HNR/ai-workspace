"""Settings definitions, defaults, validation and secret masking."""
from __future__ import annotations
import math
import urllib.parse

# key -> (default, kind, label, section, help)
# kind: text | secret | int | float | bool | choice | textarea
DEFS = {
    # Telegram
    "tg_bot_token": ("", "secret", "Bot token", "telegram",
                     "From @BotFather. Stored server-side only; never sent to the browser."),
    "tg_enabled": ("true", "bool", "Enable Telegram polling", "telegram",
                   "Turn off to pause the bot without clearing configuration."),
    "tg_welcome": ("Hello! I'm a chat assistant powered by a local model. Just talk to me.",
                   "textarea", "Welcome message (/start reply)", "telegram", ""),
    "tg_help": ("Send me any message and I'll reply.\n\nCommands:\n/start - hello\n/reset - start a fresh conversation\n/help - this help",
                "textarea", "Help text (/help reply)", "telegram", ""),
    "tg_error": ("Sorry — the model is unavailable right now. Please try again in a bit.",
                 "textarea", "Model-error reply", "telegram",
                 "Sent to the user when the LLM request fails."),
    "tg_unsupported": ("I can only read text messages for now.",
                       "text", "Non-text reply", "telegram",
                       "Reply to photos, stickers, voice, etc. Empty = stay silent."),
    "tg_group_mode": ("mention", "choice:mention,all", "Group chat mode", "telegram",
                      "'mention' replies only when addressed (@bot or reply to bot). 'all' answers every group message."),
    "tg_typing": ("true", "bool", "Show typing indicator", "telegram",
                  "Sends 'typing...' while the model generates."),
    # LLM
    "llm_base_url": ("http://127.0.0.1:10000/v1", "text", "LLM base URL", "llm",
                     "OpenAI-compatible endpoint, e.g. llama-server /v1."),
    "llm_model": ("", "text", "Model name", "llm",
                  "Empty = first model reported by /v1/models (llama-server default)."),
    "llm_api_key": ("", "secret", "API key (optional)", "llm",
                    "Usually empty for local llama-server."),
    "llm_timeout": ("120", "int", "Request timeout (seconds)", "llm", ""),
    # Generation
    "gen_temperature": ("0.7", "float", "Temperature", "generation", ""),
    "gen_max_tokens": ("1024", "int", "Max reply tokens", "generation",
                       "Finite upper bound per reply."),
    "gen_top_p": ("0.9", "float", "Top-p", "generation", ""),
    "gen_system_prompt": ("You are a helpful chat assistant. Keep replies conversational and concise unless asked for detail.",
                          "textarea", "System prompt (global default)", "generation",
                          "Sent at the top of every model context. Users can have overrides."),
    # Context
    "ctx_history_limit": ("30", "int", "Recent messages in context", "context",
                          "How many of the user's most recent messages are sent to the model. Older ones stay stored and viewable."),
    "ctx_max_tokens": ("6000", "int", "Context token estimate cap", "context",
                       "Estimated (chars/4). Oldest messages are dropped first until under this cap."),
    "ctx_max_user_chars": ("12000", "int", "Max chars per user message in context", "context",
                           "Very long messages are trimmed (head kept, note appended) for the model only; the full text stays stored."),
    # Bot behavior / concurrency
    "bot_max_parallel_llm": ("1", "int", "Max parallel LLM requests", "behavior",
                             "Global cap so concurrent users don't overload the model server."),
    "bot_queue_notice": ("false", "bool", "Tell user they are queued", "behavior",
                         "If on, a queued user gets a short 'one moment' notice."),
    "bot_queue_notice_text": ("One moment — finishing your previous reply first.",
                              "text", "Queued notice text", "behavior", ""),
}

DEFS.update({
    "tg_reply_format": ("html", "choice:html,plain", "Reply formatting", "telegram", "Render model Markdown as Telegram formatting, or send plain text. Raw model HTML is escaped."),
    "tg_reply_to_message": ("true", "bool", "Reply to the original message", "telegram", "Keep replies attached to the message that triggered them, including forum topics."),
    "bot_queue_limit": ("8", "int", "Queued messages per user", "behavior", "New messages are refused when this user's queue is full."),
    "bot_access_mode": ("all", "choice:all,admins,allowlist", "Who can chat", "behavior", "Restrict model access to administrators or the allowed Telegram IDs."),
    "bot_allowed_users": ("", "text", "Allowed Telegram user IDs", "behavior", "Comma-separated numeric IDs. Administrators always have access."),
    "bot_admin_user_id": ("", "text", "Administrator Telegram ID", "behavior", "Optional explicit owner. Otherwise the first private-chat user claims admin."),
    "llm_allow_remote": ("false", "bool", "Allow remote model endpoint", "llm", "Enable explicitly before sending conversations outside this computer."),
})

SECRET_KEYS = {"tg_bot_token", "llm_api_key"}

INT_BOUNDS = {
    "llm_timeout": (5, 600),
    "gen_max_tokens": (1, 32000),
    "ctx_history_limit": (1, 500),
    "ctx_max_tokens": (256, 200000),
    "ctx_max_user_chars": (500, 100000),
    "bot_max_parallel_llm": (1, 8),
    "bot_queue_limit": (1, 50),
}


def coerce(key: str, raw: str):
    kind = DEFS[key][1]
    if kind == "bool":
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    if kind == "int":
        v = int(str(raw).strip())
        if key in INT_BOUNDS:
            lo, hi = INT_BOUNDS[key]
            v = max(lo, min(hi, v))
        return v
    if kind == "float":
        value = float(str(raw).strip())
        if not math.isfinite(value):
            raise ValueError("must be a finite number")
        lo, hi = (0, 2) if key == "gen_temperature" else (0.01, 1)
        if not lo <= value <= hi:
            raise ValueError(f"must be between {lo} and {hi}")
        return value
    if kind.startswith("choice"):
        choices = kind.split(":", 1)[1].split(",")
        v = str(raw).strip()
        if v not in choices:
            raise ValueError(f"must be one of {choices}")
        return v
    return str(raw)


def typed_settings(raw: dict) -> dict:
    out = {}
    for key, (default, *_rest) in DEFS.items():
        val = raw.get(key, default)
        try:
            out[key] = coerce(key, val)
        except (ValueError, TypeError):
            out[key] = coerce(key, default)
    return out


def validate_settings(incoming: dict, current: dict) -> dict:
    """Validate the whole edit before writing any part of it."""
    if not isinstance(incoming, dict):
        raise ValueError("settings must be an object")
    values = {}
    for key, value in incoming.items():
        if key not in DEFS:
            raise ValueError(f"Unknown setting: {key}")
        if key in SECRET_KEYS and value in (None, ""):
            continue
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"{key}: expected a scalar value")
        text = str(value).strip() if DEFS[key][1] != "textarea" else str(value)
        try:
            typed = coerce(key, text)
            if key in INT_BOUNDS and int(text) != typed:
                raise ValueError(f"must be between {INT_BOUNDS[key][0]} and {INT_BOUNDS[key][1]}")
            if DEFS[key][1] == "bool" and text.lower() not in ("true", "false", "1", "0", "yes", "no", "on", "off"):
                raise ValueError("must be true or false")
            if len(text) > 16000:
                raise ValueError("too long (maximum 16000 characters)")
            if key == "bot_admin_user_id" and text and (not text.isdigit() or int(text) <= 0):
                raise ValueError("must be a positive Telegram user ID")
            if key == "bot_allowed_users" and any(not x.strip().isdigit() or int(x.strip()) <= 0 for x in text.split(",") if x.strip()):
                raise ValueError("use comma-separated positive Telegram user IDs")
        except (TypeError, ValueError) as e:
            raise ValueError(f"{key}: {e}") from None
        values[key] = text
    combined = typed_settings({**current, **values})
    validate_endpoint(combined["llm_base_url"], combined["llm_allow_remote"])
    return values


def validate_endpoint(url: str, allow_remote=False):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("LLM URL must be an HTTP(S) endpoint without credentials, query or fragment")
    if not allow_remote and parsed.hostname.lower() not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Remote LLM endpoints require explicit Allow remote model endpoint")
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("LLM endpoint port is invalid") from None
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("LLM endpoint port must be between 1 and 65535")


def mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "••••••••"
    return "••••••••" + value[-4:]


def public_settings(raw: dict) -> dict:
    """Settings safe to send to the browser: secrets masked, never raw."""
    pub = {}
    for key, (default, kind, label, section, help_) in DEFS.items():
        val = raw.get(key, default)
        entry = {"key": key, "label": label, "section": section,
                 "help": help_, "kind": kind.split(":")[0],
                 "choices": kind.split(":")[1].split(",") if kind.startswith("choice") else []}
        if key in SECRET_KEYS:
            entry["set"] = bool(val)
            entry["masked"] = mask_secret(val)
            entry["value"] = ""
        else:
            entry["value"] = val
        pub[key] = entry
    return pub
