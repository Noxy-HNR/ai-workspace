# Task: merge JSONL event logs

Implement the functions in `logmerge/merge.py` and command-line entry point in
`logmerge/cli.py`.

Read events from any number of UTF-8 JSONL files. Ignore blank lines and
malformed JSON lines. Keep only JSON objects with a nonempty string `id`, an
ISO-8601 `timestamp` that includes a timezone, and a string `source`. When the
same ID appears more than once, keep the event with the earliest timestamp;
for equal instants, keep the one with lexicographically smallest `source`.
Normalize output timestamps to UTC with a trailing `Z`. Sort output by UTC
timestamp, then ID. Preserve every other field from the selected event.

`merge_files(paths)` returns the event list. The CLI must support
`python -m logmerge.cli --output OUTPUT INPUT...`, write compact UTF-8 JSONL,
create the output parent directory if needed, and return a nonzero exit status
when no input paths are provided or an input file cannot be read. Do not read
the output file as an input, and do not mutate parsed input dictionaries. Add
tests and run the available unittest tests.
