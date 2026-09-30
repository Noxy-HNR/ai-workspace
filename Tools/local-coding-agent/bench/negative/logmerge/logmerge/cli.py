"""Deliberately wrong command-line implementation."""

import sys


def main(argv=None):
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
