"""Command-line interface for merging JSONL event files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .merge import merge_files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args(argv)
    try:
        rows = merge_files(args.inputs)
    except OSError as error:
        print(f"cannot read input: {error}", file=sys.stderr)
        return 2
    output = Path(args.output)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    except OSError as error:
        print(f"cannot write output: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
