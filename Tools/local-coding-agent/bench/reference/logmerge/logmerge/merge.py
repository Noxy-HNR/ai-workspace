"""Merge JSONL event files."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    return value.isoformat(timespec="auto").replace("+00:00", "Z")


def merge_files(paths: Iterable[str | Path]) -> list[dict]:
    selected: dict[str, tuple[datetime, str, dict]] = {}
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                event_id, source = event.get("id"), event.get("source")
                instant = _parse_timestamp(event.get("timestamp"))
                if not isinstance(event_id, str) or not event_id or not isinstance(source, str) or instant is None:
                    continue
                candidate = (instant, source, event.copy())
                current = selected.get(event_id)
                if current is None or (instant, source) < (current[0], current[1]):
                    selected[event_id] = candidate
    rows = []
    for instant, _source, event in selected.values():
        event["timestamp"] = _format_timestamp(instant)
        rows.append(event)
    rows.sort(key=lambda event: (event["timestamp"], event["id"]))
    return rows
