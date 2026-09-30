"""Minimal Telegram Bot API client over bounded HTTP (no dependencies).

Secrets never appear in exceptions or logs: every error path redacts the
token before the message leaves this module.
"""
from __future__ import annotations

import time
from .http_json import request_json


class TelegramError(RuntimeError):
    def __init__(self, message, code=None, retry_after=None):
        super().__init__(message)
        self.code = code
        self.retry_after = retry_after


class TelegramClient:
    supports_reply_options = True

    def __init__(self, token: str, timeout=30):
        self._token = token or ""
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self._token and ":" in self._token)

    def _redact(self, text: str) -> str:
        if self._token and self._token in text:
            text = text.replace(self._token, "<bot-token>")
        return text

    def call(self, method: str, params: dict | None = None,
             timeout: int | None = None, cancel_event=None):
        if not self.configured:
            raise TelegramError("Bot token is not configured.")
        url = f"https://api.telegram.org/bot{self._token}/{method}"
        try:
            status, body = request_json(url, params or {}, timeout=timeout or self.timeout,
                                        cancel_event=cancel_event, include_status=True)
        except Exception as e:
            raise TelegramError(
                self._redact(f"Telegram {method} failed: {e}")) from None
        if not isinstance(body, dict):
            raise TelegramError(f"Telegram {method} returned an invalid response.", code=status)
        if not 200 <= status < 300 or body.get("ok") is not True:
            desc = str(body.get("description", "unknown error"))[:300]
            retry = None
            params_out = body.get("parameters") or {}
            if isinstance(params_out, dict):
                try:
                    retry = max(0, int(params_out.get("retry_after", 0))) or None
                except (ValueError, TypeError, OverflowError):
                    pass
            raise TelegramError(self._redact(f"Telegram {method}: {desc}"),
                                code=body.get("error_code", status),
                                retry_after=retry)
        return body.get("result")

    def get_me(self, *, cancel_event=None):
        return self.call("getMe", timeout=15,cancel_event=cancel_event)

    def get_updates(self, offset=0, timeout_secs=25, *, cancel_event=None):
        # Long-poll: Bot API holds the connection up to `timeout` seconds.
        return self.call("getUpdates", {"offset": offset, "timeout": timeout_secs,
                                        "allowed_updates": ["message", "edited_message"]},
                         timeout=timeout_secs + 5,cancel_event=cancel_event) or []

    def send_chat_action(self, chat_id, action="typing", message_thread_id=0):
        try:
            params = {"chat_id": chat_id, "action": action}
            if message_thread_id:
                params["message_thread_id"] = message_thread_id
            self.call("sendChatAction", params,
                      timeout=10)
        except TelegramError:
            pass

    def send_text(self, chat_id, text: str, max_retries=2, *, message_thread_id=0,
                  reply_to_message_id=None, parse_mode=None, cancel_event=None, thread_id=None,
                  delivery_deadline=None):
        """Send once; retry only explicit rate-limit rejections.

        A connection error can occur after Telegram accepted a message. Retrying
        that uncertain delivery automatically would create duplicate replies.
        """
        attempt = 0
        if thread_id is not None:
            message_thread_id = thread_id
        deadline = time.monotonic() + 65
        if delivery_deadline is not None:
            deadline = min(deadline, delivery_deadline)
        params = {"chat_id": chat_id, "text": text,
                  "link_preview_options": {"is_disabled": True}}
        if message_thread_id:
            params["message_thread_id"] = message_thread_id
        if reply_to_message_id:
            params["reply_parameters"] = {"message_id": reply_to_message_id,
                                          "allow_sending_without_reply": True}
        if parse_mode:
            params["parse_mode"] = parse_mode
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise TelegramError("Telegram delivery cancelled.")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TelegramError("Telegram delivery exceeded its total time limit.")
            try:
                res = self.call("sendMessage", params, timeout=min(20, remaining),
                                cancel_event=cancel_event)
                return (res or {}).get("message_id", 0)
            except TelegramError as e:
                if e.code == 429 and e.retry_after and attempt < max_retries:
                    if e.retry_after > 30 or e.retry_after >= deadline - time.monotonic():
                        raise  # Mark failure for explicit retry, never retry too early.
                    if cancel_event is not None:
                        if cancel_event.wait(e.retry_after):
                            raise TelegramError("Telegram delivery cancelled.") from None
                    else:
                        time.sleep(e.retry_after)
                    attempt += 1
                    continue
                raise
