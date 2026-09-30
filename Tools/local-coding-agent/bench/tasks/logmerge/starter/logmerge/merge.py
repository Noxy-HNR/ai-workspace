"""Merge JSONL event files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def merge_files(paths: Iterable[str | Path]) -> list[dict]:
    """Read, deduplicate, normalize, and order event records."""
    raise NotImplementedError
