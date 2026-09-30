"""Tests from the bot-queue review of the real backend (job 02-review-real-backend, Space Bunny, 2026-09-30).
Each one pinned a defect in real.py / dashboard.py / runner.py / bench.py that has since been fixed; see
REVIEW-real-backend.md for the findings. `fakes.py` stands in for the machine: nothing here opens llama-server,
the dashboard, a GPU or C:\AI."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
