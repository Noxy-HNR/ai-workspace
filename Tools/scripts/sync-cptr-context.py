#!/usr/bin/env python3
"""Keep cptr's context-compaction threshold in step with llama-server's real context size.

cptr never asks llama-server for its n_ctx. It compacts (summarizes) a chat once the chat passes
chat.compact_token_threshold (default 80000), so when the loaded model -- and with it the context
size -- changes (24K ... 180K here), a fixed threshold is wrong in both directions: too high and the
context overflows before cptr compacts, too low and cptr compacts long before it has to.

cptr resolves the threshold on every request from the config key "chat.models" (re-read from its
database each time, so no restart is needed). A "*" entry there applies to every model, including
the "switchboard-auto" router, and beats the global default. This script writes exactly that entry
(threshold = ratio * n_ctx, rounded down to a multiple of 512) and touches nothing else.

The write goes through cptr's own Config.upsert, which also mirrors the value into
~/.cptr/config.toml. That matters: cptr re-seeds its database from config.toml on every start
(toml wins), so a database-only write would be reverted by a stale mirror after the next restart.
If cptr can't be imported, it falls back to a direct SQLite write (database only).

Usage: sync-cptr-context.py --n-ctx 24576 [--ratio 0.70] [--data-dir DIR] [--log FILE]
Exit codes: 0 = written or already correct, 2 = nothing to do / refused (message says why), 1 = error.
"""
import argparse
import asyncio
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

KEY = "chat.models"
STEP = 512
MIN_THRESHOLD = 2048


class Refused(Exception):
    """The existing value isn't something we should modify."""


def threshold_for(n_ctx: int, ratio: float) -> int:
    return max(MIN_THRESHOLD, int(n_ctx * ratio) // STEP * STEP)


def with_threshold(current, threshold: int):
    """Return (new_value, old_threshold). Keeps every other entry and parameter as it was."""
    if current is None:
        current = {}
    if isinstance(current, (str, bytes)):
        try:
            current = json.loads(current)
        except ValueError:
            raise Refused(f"existing {KEY} is not valid JSON, leaving it alone")
    if not isinstance(current, dict):
        raise Refused(f"existing {KEY} is not an object, leaving it alone")
    wildcard = current.setdefault("*", {})
    if not isinstance(wildcard, dict) or not isinstance(wildcard.setdefault("params", {}), dict):
        raise Refused(f"existing {KEY}['*'] has an unexpected shape, leaving it alone")
    params = wildcard["params"]
    old = params.get("compact_token_threshold")
    params["compact_token_threshold"] = threshold
    return current, old


def write_via_cptr(threshold: int) -> str:
    """Write through cptr's own config layer (database + config.toml mirror)."""
    from cptr.models.config import Config
    from cptr.utils.config import load_app_config_from_toml

    async def run() -> str:
        db_value = await Config.get(KEY)
        new_value, old = with_threshold(json.loads(json.dumps(db_value)) if db_value is not None else None, threshold)
        toml_value = load_app_config_from_toml().get(KEY)
        if old == threshold and toml_value == new_value:
            return f"unchanged: compact_token_threshold={threshold} (database and config.toml agree)"
        await Config.upsert({KEY: new_value})  # also mirrors to config.toml
        note = "" if old != threshold else " (database was already right; config.toml mirror refreshed)"
        return f"updated: compact_token_threshold {old} -> {threshold}{note}"

    return asyncio.run(run())


def write_via_sqlite(db_path: Path, threshold: int) -> str:
    con = sqlite3.connect(str(db_path), timeout=15, isolation_level=None)
    try:
        con.execute("PRAGMA busy_timeout=15000")
        if not con.execute("select 1 from sqlite_master where type='table' and name='config'").fetchone():
            raise Refused("cptr database has no config table yet")
        con.execute("BEGIN IMMEDIATE")
        try:
            row = con.execute("select value from config where key=?", (KEY,)).fetchone()
            new_value, old = with_threshold(row[0] if row else None, threshold)
            if old == threshold:
                con.execute("ROLLBACK")
                return f"unchanged: compact_token_threshold={threshold} (database only)"
            value = json.dumps(new_value, ensure_ascii=False)
            if row is None:
                con.execute("insert into config(key, value, updated_at) values(?, ?, NULL)", (KEY, value))
            else:
                con.execute("update config set value=? where key=?", (value, KEY))
            con.execute("COMMIT")
            return f"updated: compact_token_threshold {old} -> {threshold} (database only; cptr not importable)"
        except BaseException:
            try:
                con.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
    finally:
        con.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-ctx", type=int, required=True, help="llama-server per-slot context size")
    ap.add_argument("--ratio", type=float, default=0.70, help="fraction of n_ctx at which cptr compacts")
    ap.add_argument("--data-dir", default=None, help="cptr data dir (default: $CPTR_DATA_DIR or ~/.cptr)")
    ap.add_argument("--log", default=None, help="append the result line to this file")
    args = ap.parse_args()

    def finish(code: int, msg: str) -> int:
        print(msg)
        if args.log:
            try:
                with open(args.log, "a", encoding="utf-8") as fh:
                    fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} exit={code} {msg} [n_ctx {args.n_ctx}]\n")
            except OSError:
                pass
        return code

    if args.n_ctx <= 0 or not (0.1 <= args.ratio <= 0.95):
        return finish(2, f"refused: n_ctx={args.n_ctx} ratio={args.ratio} out of range")
    threshold = threshold_for(args.n_ctx, args.ratio)

    # cptr reads its data dir from the environment when it is imported, so set it first.
    if args.data_dir:
        os.environ["CPTR_DATA_DIR"] = args.data_dir
    data_dir = Path(os.environ.get("CPTR_DATA_DIR") or (Path.home() / ".cptr"))
    db_path = data_dir / "app.db"
    if not db_path.exists():
        return finish(2, f"skipped: no cptr database at {db_path} (cptr has never run)")

    try:
        try:
            msg = write_via_cptr(threshold)
        except Refused:
            raise
        except Exception as e:  # cptr missing, renamed internals, DB layer problem: fall back
            print(f"note: cptr's own config layer unavailable ({type(e).__name__}: {e}); using direct SQLite", file=sys.stderr)
            msg = write_via_sqlite(db_path, threshold)
        return finish(0, f"{msg} (n_ctx {args.n_ctx} x {args.ratio:g})")
    except Refused as e:
        return finish(2, f"refused: {e}")
    except sqlite3.Error as e:
        return finish(1, f"error: {e}")


if __name__ == "__main__":
    sys.exit(main())
