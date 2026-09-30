"""Deliberately wrong implementation used to validate the grader."""

import json
from pathlib import Path


def merge_files(paths):
    # Bug: keeps every valid-looking line, compares timestamp strings without
    # timezone normalization, and returns input order rather than sorted order.
    rows = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and isinstance(event.get("id"), str):
                rows.append(event)
    return rows
