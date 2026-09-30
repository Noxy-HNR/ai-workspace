"""Runs INSIDE Switchboard's virtual environment (it has typesafe_sdk and loads the key through Switchboard's own
config loader, so the tuner never reads or handles the key). Reads one request as JSON on stdin:

    {"state": {...}, "questions": {"name": {"instructions": "...", "criteria": {"label": "description", ...}}}}

and prints {"answers": {"name": {"choice": label, "probabilities": {label: p}}}} or {"error": "..."} on stdout.
"""

from __future__ import annotations

import asyncio
import json
import sys


async def _main() -> dict:
    req = json.load(sys.stdin)
    try:
        from switchboard.config import load_config
        from typesafe_sdk import AsyncTypeSafeClient, Choice
    except ImportError as e:
        return {"error": f"Switchboard environment is missing a package: {e.name}"}
    cfg = load_config()
    if not cfg.typesafe_api_key:
        return {"error": "no TypeSafe key configured in Switchboard"}
    questions = {name: Choice(instructions=q["instructions"], criteria=q["criteria"])
                 for name, q in req["questions"].items()}
    client = AsyncTypeSafeClient(api_key=cfg.typesafe_api_key, model=cfg.decisions.jev.model,
                                 timeout=float(req.get("timeout_s", 10)))
    try:
        resp = await client.system_one(req["state"], questions)
    except Exception as e:  # noqa: BLE001 -- report the kind only; provider messages may echo input
        return {"error": f"Jev request failed ({type(e).__name__})"}
    finally:
        close = getattr(client, "aclose", None) or getattr(client, "close", None)
        if close is not None:
            res = close()
            if asyncio.iscoroutine(res):
                await res
    return {"answers": {name: {"choice": a.choice, "probabilities": dict(a.probabilities)}
                        for name, a in resp.choices.items()}}


if __name__ == "__main__":
    print(json.dumps(asyncio.run(_main())))
