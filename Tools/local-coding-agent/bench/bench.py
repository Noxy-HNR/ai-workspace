"""Prepare and score two synthetic coding-agent repository tasks.

No provider, server, network, or third-party package is used.  The scorer runs
candidate code, so run it only against a checkout you trust to execute locally.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
TASKS = ("queue", "logmerge")


def _overlay(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        if path.is_file():
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _run(task: str, repo: Path, temp: Path) -> tuple[bool, str]:
    if not repo.is_dir():
        raise ValueError(f"candidate repository does not exist: {repo}")
    temp.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for key in ("TMP", "TEMP", "TMPDIR"):
        env[key] = str(temp)
    try:
        result = subprocess.run(
            [sys.executable, "-B", "-m", "unittest", "discover", "-s", str(HERE / "grader" / task), "-p", "test_*.py", "-v"],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT: authoritative tests exceeded 30 seconds"
    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0, output


def prepare(task: str, output: Path) -> None:
    if output.exists():
        raise ValueError(f"output already exists; choose a fresh path: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(HERE / "tasks" / task / "starter", output)
    print(f"Prepared {task} at {output}")
    print(f"Task prompt: {output / 'TASK.md'}")


def grade(task: str, repo: Path) -> bool:
    with tempfile.TemporaryDirectory(prefix="grade-", dir=HERE) as scratch:
        passed, details = _run(task, repo.resolve(), Path(scratch))
    print(f"{task}: {'PASS' if passed else 'FAIL'}")
    print(details)
    return passed


def selfcheck() -> bool:
    all_good = True
    with tempfile.TemporaryDirectory(prefix="selfcheck-", dir=HERE) as scratch:
        base = Path(scratch)
        for task in TASKS:
            for variant, should_pass in (("reference", True), ("negative", False)):
                repo = base / f"{task}-{variant}"
                shutil.copytree(HERE / "tasks" / task / "starter", repo)
                _overlay(HERE / "reference" / task, repo)
                if variant == "negative":
                    _overlay(HERE / "negative" / task, repo)
                passed, details = _run(task, repo, base / f"tmp-{task}-{variant}")
                good = passed == should_pass
                all_good &= good
                print(f"{task}/{variant}: {'OK' if good else 'BROKEN'} "
                      f"(grader {'passed' if passed else 'failed'} as expected={should_pass})")
                if not good:
                    print(details)
    return all_good


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="copy a fresh starter repository")
    prep.add_argument("task", choices=TASKS)
    prep.add_argument("output", type=Path)
    score = sub.add_parser("grade", help="run authoritative tests on a candidate repository")
    score.add_argument("task", choices=TASKS)
    score.add_argument("repo", type=Path)
    sub.add_parser("selfcheck", help="verify references pass and wrong versions fail")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.task, args.output)
        return 0
    if args.command == "grade":
        return 0 if grade(args.task, args.repo) else 1
    return 0 if selfcheck() else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from None
