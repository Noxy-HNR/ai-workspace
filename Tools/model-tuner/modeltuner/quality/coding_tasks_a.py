"""Coding-eval tasks, part A: easy/medium (E1-E8) and hard algorithmic (H1-H4).

Each task: id, category, prompt (what the model sees), tests (hidden asserts run after the model's
code in the same module namespace), reference (a solution that must pass), wrong (solutions that
must FAIL, to prove the tests can fail). Helper names in tests start with _t_ to avoid clashes.

Carried over from `_legacy/codeeval/tasks_a.py` unchanged: same prompts (including the shared
`COMMON` block the model sees), same hidden asserts, same reference and wrong solutions. Part B
appends to the same `TASKS` list, so `coding.py` imports it for that side effect. Nothing here
loads a model, runs anything or scores a run.
"""

from __future__ import annotations

__all__ = ["COMMON", "TASKS", "task"]

COMMON = (
    "\n\nRules: use only the Python standard library. Reply with the complete solution in one "
    "```python code block and nothing after it. Do not include example usage, tests or a __main__ block."
)

TASKS = []


def task(id, category, prompt, tests, reference, wrong=None):
    TASKS.append(dict(id=id, category=category, prompt=prompt.strip() + COMMON, tests=tests, reference=reference, wrong=wrong or []))


# ---------------------------------------------------------------------------------------- E1
task(
    "E1_parse_duration", "easy",
    r"""
Write `parse_duration(s: str) -> int` that converts a duration string to a whole number of seconds.

A duration is one or more components, each a non-negative integer followed by a unit: `d` (86400 s),
`h` (3600 s), `m` (60 s) or `s` (1 s). Units are case-insensitive and may appear in any order, but each
unit may appear at most once. Whitespace is allowed before, after and between components and between a
number and its unit. Anything else raises `ValueError`: an empty or whitespace-only string, a number
with no unit (such as "90"), a unit with no number, an unknown unit, a repeated unit, a negative number,
a decimal number.
""",
    r'''
assert parse_duration("1h30m") == 5400
assert parse_duration("2d 4h") == 2 * 86400 + 4 * 3600
assert parse_duration("45S") == 45
assert parse_duration("  10m 5s ") == 605
assert parse_duration("1d2h3m4s") == 86400 + 7200 + 180 + 4
assert parse_duration("3m 2h") == 7380
assert parse_duration("0s") == 0
assert parse_duration("1 h 30 m") == 5400
assert isinstance(parse_duration("5s"), int)
for _t_bad in ["", "   ", "90", "h", "5x", "1h1h", "-5s", "1.5h", "1h 30", "1h,30m", "m5", "1 2h", "1hh"]:
    try:
        parse_duration(_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for %r" % (_t_bad,))
''',
    r'''
import re
def parse_duration(s):
    if not re.fullmatch(r"\s*(\d+\s*[dhmsDHMS]\s*)+", s):
        raise ValueError("bad duration")
    mult = {"d": 86400, "h": 3600, "m": 60, "s": 1}
    seen = set()
    total = 0
    for num, unit in re.findall(r"(\d+)\s*([dhmsDHMS])", s):
        u = unit.lower()
        if u in seen:
            raise ValueError("repeated unit")
        seen.add(u)
        total += int(num) * mult[u]
    return total
''',
)

# ---------------------------------------------------------------------------------------- E2
task(
    "E2_merge_intervals", "easy",
    r"""
Write `merge_intervals(intervals)` that merges integer intervals.

Each interval is a pair `(start, end)` (a tuple or a list) of integers with `start <= end`, and both
ends are inclusive. Two intervals are merged when they overlap OR are adjacent integers (for example
`[1, 3]` and `[4, 6]` merge into `[1, 6]`). Return a new list of `[start, end]` lists sorted by start.
The input may be unsorted and must not be modified. An empty input returns `[]`.
""",
    r'''
assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]
assert merge_intervals([[1, 2], [3, 4]]) == [[1, 4]]
assert merge_intervals([(5, 7), (1, 2)]) == [[1, 2], [5, 7]]
assert merge_intervals([]) == []
assert merge_intervals([[1, 10], [2, 3]]) == [[1, 10]]
assert merge_intervals([[1, 2], [4, 5]]) == [[1, 2], [4, 5]]
assert merge_intervals([[3, 3]]) == [[3, 3]]
_t_data = [[9, 12], [1, 2], [2, 4], [6, 7]]
_t_copy = [list(x) for x in _t_data]
merge_intervals(_t_data)
assert _t_data == _t_copy
_t_out = merge_intervals([(1, 2)])
assert isinstance(_t_out[0], list)
''',
    r'''
def merge_intervals(intervals):
    items = sorted((tuple(x) for x in intervals))
    out = []
    for s, e in items:
        if out and s <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out
''',
    wrong=[
        r'''
def merge_intervals(intervals):
    items = sorted(tuple(x) for x in intervals)
    out = []
    for s, e in items:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out
'''
    ],
)

# ---------------------------------------------------------------------------------------- E3
task(
    "E3_roman_to_int", "easy",
    r"""
Write `roman_to_int(s: str) -> int` that converts a Roman numeral to an integer, accepting ONLY
canonical numerals in the range 1 to 3999 written in upper case (M, D, C, L, X, V, I with the usual
subtractive pairs CM, CD, XC, XL, IX, IV). Anything else raises `ValueError`, including the empty
string, lower-case letters, and non-canonical forms such as "IIII", "VX", "IC", "XM" or "MMMM".
""",
    r'''
for _t_s, _t_v in [("I", 1), ("III", 3), ("IV", 4), ("IX", 9), ("LVIII", 58), ("XL", 40), ("XC", 90),
                   ("CD", 400), ("CM", 900), ("MCMXCIV", 1994), ("MMMCMXCIX", 3999), ("MMXXIV", 2024), ("DCCC", 800)]:
    assert roman_to_int(_t_s) == _t_v, _t_s
for _t_bad in ["", "IIII", "VX", "IC", "XM", "MMMM", "iv", "IXI", "VV", "LL", "DD", "XXXX", "CMC", "A", "IL", "MMMMCMXCIX", " I"]:
    try:
        roman_to_int(_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for %r" % (_t_bad,))
''',
    r'''
import re
_PAT = re.compile(r"M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})")
def roman_to_int(s):
    if not s or not _PAT.fullmatch(s):
        raise ValueError("bad numeral")
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, ch in enumerate(s):
        v = vals[ch]
        if i + 1 < len(s) and vals[s[i + 1]] > v:
            total -= v
        else:
            total += v
    return total
''',
)

# ---------------------------------------------------------------------------------------- E4
task(
    "E4_word_wrap", "easy",
    r"""
Write `word_wrap(text: str, width: int) -> list[str]` that wraps text greedily into lines of at most
`width` characters.

Split `text` on any whitespace into words and join the words on a line with single spaces. Put as many
words on a line as fit. A word longer than `width` is first placed on a fresh line (finish the current
line first if it is not empty), then split into pieces of exactly `width` characters; the final,
possibly shorter, piece stays as the current line, which later words may continue. If `width < 1`
raise `ValueError`. Text with no words returns `[]`.
""",
    r'''
assert word_wrap("the quick brown fox", 10) == ["the quick", "brown fox"]
assert word_wrap("a b c", 1) == ["a", "b", "c"]
assert word_wrap("abcdefghij kl", 4) == ["abcd", "efgh", "ij", "kl"]
assert word_wrap("ab abcdefghij", 4) == ["ab", "abcd", "efgh", "ij"]
assert word_wrap("abcdefghi jk", 5) == ["abcde", "fghi", "jk"]
assert word_wrap("hello   world\n\nagain", 11) == ["hello world", "again"]
assert word_wrap("", 5) == []
assert word_wrap("   \n ", 5) == []
assert word_wrap("exact fit", 5) == ["exact", "fit"]
assert word_wrap("one two", 7) == ["one two"]
assert word_wrap("abcdef", 3) == ["abc", "def"]
assert word_wrap("abcdefg", 3) == ["abc", "def", "g"]
for _t_w in [0, -3]:
    try:
        word_wrap("x", _t_w)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
''',
    r'''
def word_wrap(text, width):
    if width < 1:
        raise ValueError("width")
    lines = []
    cur = ""
    for w in text.split():
        if len(w) > width:
            if cur:
                lines.append(cur)
                cur = ""
            while len(w) > width:
                lines.append(w[:width])
                w = w[width:]
            cur = w
            continue
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines
''',
)

# ---------------------------------------------------------------------------------------- E5
task(
    "E5_flatten", "easy",
    r"""
Write `flatten(obj) -> list` that flattens arbitrarily nested lists and tuples into one flat list,
preserving left-to-right order. Only `list` and `tuple` are containers to be flattened; every other
object (strings, bytes, dicts, sets, numbers, None, ...) is an atom kept as it is. A non-container
argument returns a one-element list containing it. It must work for extremely deep nesting (for
example 5000 levels) without raising `RecursionError`.
""",
    r'''
assert flatten([1, [2, [3, (4, 5)], "ab"], {"a": 1}, [], ()]) == [1, 2, 3, 4, 5, "ab", {"a": 1}]
assert flatten(5) == [5]
assert flatten("abc") == ["abc"]
assert flatten([]) == []
assert flatten([[], [[]], ((),)]) == []
assert flatten([None, [None]]) == [None, None]
assert flatten(({1, 2}, [b"xy"])) == [{1, 2}, b"xy"]
_t_x = 0
for _t_i in range(5000):
    _t_x = [_t_x]
assert flatten(_t_x) == [0]
_t_y = [1, 2]
for _t_i in range(5000):
    _t_y = [_t_y, _t_i]
_t_r = flatten(_t_y)
assert _t_r[:2] == [1, 2] and len(_t_r) == 5002 and _t_r[-1] == 4999
''',
    r'''
def flatten(obj):
    if not isinstance(obj, (list, tuple)):
        return [obj]
    out = []
    stack = [iter(obj)]
    while stack:
        try:
            item = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        if isinstance(item, (list, tuple)):
            stack.append(iter(item))
        else:
            out.append(item)
    return out
''',
    wrong=[
        r'''
def flatten(obj):
    if not isinstance(obj, (list, tuple)):
        return [obj]
    out = []
    for item in obj:
        out.extend(flatten(item))
    return out
'''
    ],
)

# ---------------------------------------------------------------------------------------- E6
task(
    "E6_top_k_frequent", "easy",
    r"""
Write `top_k_frequent(words, k)` that returns the `k` most frequent items of the list `words`, most
frequent first. Ties are broken by ascending alphabetical order (normal string comparison). If `k` is
larger than the number of distinct words return all of them; if `k <= 0` return `[]`.
""",
    r'''
assert top_k_frequent(["a", "b", "a", "c", "b", "a"], 2) == ["a", "b"]
assert top_k_frequent(["b", "a", "c"], 2) == ["a", "b"]
assert top_k_frequent(["x", "y", "y", "x", "z"], 3) == ["x", "y", "z"]
assert top_k_frequent(["a"], 5) == ["a"]
assert top_k_frequent([], 3) == []
assert top_k_frequent(["a", "b"], 0) == []
assert top_k_frequent(["a", "b"], -1) == []
assert top_k_frequent(["Banana", "apple", "apple", "Banana", "cherry"], 2) == ["Banana", "apple"]
assert top_k_frequent(["i", "love", "leetcode", "i", "love", "coding"], 2) == ["i", "love"]
''',
    r'''
from collections import Counter
def top_k_frequent(words, k):
    if k <= 0:
        return []
    c = Counter(words)
    return [w for w, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]
''',
)

# ---------------------------------------------------------------------------------------- E7
task(
    "E7_is_balanced", "easy",
    r"""
Write `is_balanced(s: str) -> bool` that reports whether the round, square and curly brackets in `s`
are correctly nested and matched: `()`, `[]`, `{}`. All other characters are ignored. A backslash
escapes the single character that follows it: an escaped character (including an escaped bracket or an
escaped backslash) is ignored entirely. A backslash at the very end of the string is ignored.
""",
    r'''
assert is_balanced("([]{})") is True
assert is_balanced("([)]") is False
assert is_balanced("(") is False
assert is_balanced(")") is False
assert is_balanced("") is True
assert is_balanced("a(b)c[d]{e}") is True
assert is_balanced("\\(") is True
assert is_balanced("(\\)") is False
assert is_balanced("\\\\(") is False
assert is_balanced("\\\\()") is True
assert is_balanced("{[()()]}") is True
assert is_balanced("(()") is False
assert is_balanced("())") is False
assert is_balanced("abc\\") is True
assert is_balanced("[\\]]") is True
assert is_balanced("[\\]") is False
''',
    r'''
def is_balanced(s):
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "\\":
            i += 2
            continue
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
        i += 1
    return not stack
''',
)

# ---------------------------------------------------------------------------------------- E8
task(
    "E8_chunk_by_sum", "easy",
    r"""
Write `chunk_by_sum(nums, limit)` that splits the list `nums` of non-negative numbers into consecutive
chunks (lists), greedily: keep adding the next number to the current chunk while the chunk's sum stays
`<= limit`, otherwise start a new chunk. The order of the numbers is preserved. If any single number is
greater than `limit` raise `ValueError`. An empty list returns `[]`.
""",
    r'''
assert chunk_by_sum([1, 2, 3, 4, 5], 6) == [[1, 2, 3], [4], [5]]
assert chunk_by_sum([5, 5, 5], 5) == [[5], [5], [5]]
assert chunk_by_sum([1, 1, 1], 10) == [[1, 1, 1]]
assert chunk_by_sum([], 3) == []
assert chunk_by_sum([0, 0, 4, 1, 1], 5) == [[0, 0, 4, 1], [1]]
assert chunk_by_sum([2, 2, 2, 2], 4) == [[2, 2], [2, 2]]
assert chunk_by_sum([3, 1, 3], 4) == [[3, 1], [3]]
assert chunk_by_sum([1.5, 1.5, 2], 3) == [[1.5, 1.5], [2]]
for _t_args in [([7], 5), ([1, 2, 6], 5)]:
    try:
        chunk_by_sum(*_t_args)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
''',
    r'''
def chunk_by_sum(nums, limit):
    for x in nums:
        if x > limit:
            raise ValueError("element exceeds limit")
    out = []
    cur = []
    total = 0
    for x in nums:
        if cur and total + x > limit:
            out.append(cur)
            cur = []
            total = 0
        cur.append(x)
        total += x
    if cur:
        out.append(cur)
    return out
''',
)

# ---------------------------------------------------------------------------------------- H1
task(
    "H1_eval_expr", "hard",
    r"""
Write `eval_expr(s: str) -> float` that evaluates an arithmetic expression and returns a `float`.
Do not use `eval`, `exec`, `compile` or the `ast` module: write a parser.

Supported: decimal numbers (`12`, `3.5`, `.5`, `3.`), binary `+ - * /`, exponentiation `**`, unary `+`
and `-`, parentheses, and arbitrary whitespace between tokens. Precedence from loosest to tightest:
`+ -` (left-associative), `* /` (left-associative), unary `+ -`, `**`. `**` is right-associative and its
right operand may itself carry a unary sign, so `2**3**2 == 512`, `2**-1 == 0.5` and, because unary minus
binds looser than `**`, `-2**2 == -4`. Division by zero raises `ZeroDivisionError`. A malformed
expression (empty, dangling operator, unbalanced parentheses, two numbers in a row, unknown
character) raises `ValueError`.
""",
    r'''
def _t_close(a, b):
    return abs(a - b) <= 1e-9 * max(1.0, abs(b))
_t_cases = {
    "1+2": 3.0, "1 + 2 * 3": 7.0, "(1+2)*3": 9.0, "10/4": 2.5, "2**3**2": 512.0, "-2**2": -4.0,
    "2**-1": 0.5, "--3": 3.0, "+3": 3.0, "-(2+3)*2": -10.0, "2*-3": -6.0, "1-2-3": -4.0, "8/4/2": 1.0,
    ".5+.5": 1.0, "3.": 3.0, " 7 ": 7.0, "((4))": 4.0, "2 ** 0.5 ** 2": 2 ** (0.5 ** 2), "-3**-2": -(3 ** -2),
    "1e": None,
}
for _t_e, _t_v in _t_cases.items():
    if _t_v is None:
        continue
    _t_r = eval_expr(_t_e)
    assert isinstance(_t_r, float), _t_e
    assert _t_close(_t_r, _t_v), (_t_e, _t_r, _t_v)
for _t_bad in ["", "   ", "1+", "*2", "(1", "1)", "1 2", "abc", "1+*2", "()", "1..2", "2**", "1 +", "3 $ 4", "1e"]:
    try:
        eval_expr(_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for %r" % (_t_bad,))
for _t_bad in ["1/0", "5/(2-2)", "0**-1"]:
    try:
        eval_expr(_t_bad)
    except ZeroDivisionError:
        pass
    else:
        raise AssertionError("expected ZeroDivisionError for %r" % (_t_bad,))
''',
    r'''
import re
_TOK = re.compile(r"\s*(?:(\d+\.?\d*|\.\d+)|(\*\*|[-+*/()]))")
def eval_expr(s):
    pos = 0
    toks = []
    s = s.rstrip()
    while pos < len(s):
        m = _TOK.match(s, pos)
        if not m:
            raise ValueError("bad char")
        toks.append(("n", float(m.group(1))) if m.group(1) is not None else ("o", m.group(2)))
        pos = m.end()
    if not toks:
        raise ValueError("empty")
    i = [0]
    def peek():
        return toks[i[0]] if i[0] < len(toks) else None
    def take():
        t = toks[i[0]]
        i[0] += 1
        return t
    def expr():
        v = term()
        while peek() in (("o", "+"), ("o", "-")):
            op = take()[1]
            r = term()
            v = v + r if op == "+" else v - r
        return v
    def term():
        v = unary()
        while peek() in (("o", "*"), ("o", "/")):
            op = take()[1]
            r = unary()
            v = v * r if op == "*" else v / r
        return v
    def unary():
        if peek() in (("o", "+"), ("o", "-")):
            op = take()[1]
            v = unary()
            return -v if op == "-" else v
        return power()
    def power():
        b = atom()
        if peek() == ("o", "**"):
            take()
            e = unary()
            return b ** e
        return b
    def atom():
        t = peek()
        if t is None:
            raise ValueError("unexpected end")
        if t[0] == "n":
            take()
            return t[1]
        if t == ("o", "("):
            take()
            v = expr()
            if peek() != ("o", ")"):
                raise ValueError("missing )")
            take()
            return v
        raise ValueError("unexpected token")
    v = expr()
    if i[0] != len(toks):
        raise ValueError("trailing tokens")
    return float(v)
''',
)

# ---------------------------------------------------------------------------------------- H2
task(
    "H2_smallest_toposort", "hard",
    r"""
Write `smallest_toposort(n, edges)` for a directed graph with nodes `0 .. n-1`. `edges` is a list of
pairs `(u, v)` meaning `u` must come before `v` (duplicate edges may occur). Return the topological
order that is lexicographically smallest as a list of node numbers (compare orders element by
element), or `None` if the graph contains a cycle (a self-loop counts as a cycle). `n = 0` returns
`[]`. It must be fast enough for `n` = 200000 with a similar number of edges.
""",
    r'''
import itertools, random
def _t_brute(n, edges):
    for perm in itertools.permutations(range(n)):
        pos = {v: i for i, v in enumerate(perm)}
        if all(pos[u] < pos[v] for u, v in edges):
            return list(perm)
    return None
assert smallest_toposort(3, [(2, 0), (2, 1)]) == [2, 0, 1]
assert smallest_toposort(4, []) == [0, 1, 2, 3]
assert smallest_toposort(2, [(0, 1), (1, 0)]) is None
assert smallest_toposort(1, [(0, 0)]) is None
assert smallest_toposort(0, []) == []
assert smallest_toposort(4, [(3, 0), (3, 1), (1, 2)]) == [3, 0, 1, 2]
assert smallest_toposort(3, [(0, 1), (0, 1), (1, 2)]) == [0, 1, 2]
assert smallest_toposort(5, [(4, 3), (3, 2), (2, 1), (1, 0)]) == [4, 3, 2, 1, 0]
_t_rng = random.Random(7)
for _t_it in range(300):
    _t_n = _t_rng.randint(1, 7)
    _t_e = [(_t_rng.randrange(_t_n), _t_rng.randrange(_t_n)) for _ in range(_t_rng.randint(0, 9))]
    assert smallest_toposort(_t_n, _t_e) == _t_brute(_t_n, _t_e), (_t_n, _t_e)
_t_N = 200000
_t_chain = [(i + 1, i) for i in range(_t_N - 1)]
assert smallest_toposort(_t_N, _t_chain) == list(range(_t_N - 1, -1, -1))
_t_fan = [(0, i) for i in range(1, _t_N)]
assert smallest_toposort(_t_N, _t_fan) == list(range(_t_N))
''',
    r'''
import heapq
def smallest_toposort(n, edges):
    adj = [[] for _ in range(n)]
    indeg = [0] * n
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1
    heap = [i for i in range(n) if indeg[i] == 0]
    heapq.heapify(heap)
    out = []
    while heap:
        u = heapq.heappop(heap)
        out.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, v)
    return out if len(out) == n else None
''',
    wrong=[
        r'''
from collections import deque
def smallest_toposort(n, edges):
    adj = [[] for _ in range(n)]
    indeg = [0] * n
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1
    q = deque(i for i in range(n) if indeg[i] == 0)
    out = []
    while q:
        u = q.popleft()
        out.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return out if len(out) == n else None
'''
    ],
)

# ---------------------------------------------------------------------------------------- H3
task(
    "H3_peak_load", "hard",
    r"""
Write `peak_load(intervals)` for half-open integer intervals `[start, end)`. Return a tuple
`(max_concurrent, earliest_time)`: the largest number of intervals that are active at the same time,
and the smallest time `t` at which that maximum is reached. An interval is active at `t` when
`start <= t < end`, so intervals that merely touch (`[1,3)` and `[3,5)`) do not overlap. Intervals with
`start >= end` are empty and are ignored. If there is nothing to count return `(0, None)`.
""",
    r'''
import random
def _t_brute(iv):
    iv = [(s, e) for s, e in iv if s < e]
    if not iv:
        return (0, None)
    best, bt = 0, None
    for t in range(min(s for s, _ in iv), max(e for _, e in iv) + 1):
        c = sum(1 for s, e in iv if s <= t < e)
        if c > best:
            best, bt = c, t
    return (best, bt)
assert peak_load([(1, 4), (2, 5), (3, 6)]) == (3, 3)
assert peak_load([(1, 3), (3, 5)]) == (1, 1)
assert peak_load([(5, 8), (1, 2), (1, 3)]) == (2, 1)
assert peak_load([]) == (0, None)
assert peak_load([(2, 2)]) == (0, None)
assert peak_load([(4, 3), (6, 6)]) == (0, None)
assert peak_load([(1, 10), (2, 3), (2, 3), (5, 6)]) == (3, 2)
assert peak_load([(0, 5), (5, 10), (10, 15)]) == (1, 0)
_t_rng = random.Random(11)
for _t_it in range(300):
    _t_iv = []
    for _ in range(_t_rng.randint(0, 8)):
        _t_s = _t_rng.randint(-5, 15)
        _t_iv.append((_t_s, _t_s + _t_rng.randint(-1, 8)))
    assert peak_load(_t_iv) == _t_brute(_t_iv), _t_iv
_t_big = [(i, i + 1000) for i in range(100000)]
assert peak_load(_t_big) == (1000, 999)
''',
    r'''
def peak_load(intervals):
    events = []
    for s, e in intervals:
        if s < e:
            events.append((s, 1))
            events.append((e, -1))
    if not events:
        return (0, None)
    events.sort()
    cur = 0
    best = 0
    bt = None
    for t, d in events:
        cur += d
        if cur > best:
            best, bt = cur, t
    return (best, bt)
''',
    wrong=[
        r'''
def peak_load(intervals):
    events = []
    for s, e in intervals:
        if s < e:
            events.append((s, 1))
            events.append((e, -1))
    if not events:
        return (0, None)
    events.sort(key=lambda x: (x[0], -x[1]))
    cur = 0
    best = 0
    bt = None
    for t, d in events:
        cur += d
        if cur > best:
            best, bt = cur, t
    return (best, bt)
'''
    ],
)

# ---------------------------------------------------------------------------------------- H4
task(
    "H4_ttl_cache", "hard",
    r"""
Write a class `TTLCache(capacity, ttl, clock)`. `clock` is a zero-argument function returning the
current time as a number (injected so behaviour is deterministic). Raise `ValueError` if
`capacity < 1` or `ttl <= 0`.

- `put(key, value)`: store the value with timestamp `clock()`; this also makes `key` the most recently
  used entry and restarts its lifetime. If the cache would now hold more than `capacity` LIVE entries,
  first discard every expired entry; if it is still over capacity, evict least recently used live
  entries until it fits.
- `get(key, default=None)`: return the value if the key exists and is live, else `default`. A hit makes
  the key the most recently used entry but does NOT restart its lifetime. An expired key is removed.
- An entry is expired when `clock() - stored_at >= ttl` (it expires exactly at `ttl`).
- `len(cache)` is the number of live (unexpired) entries. `key in cache` is True only for a live key
  and, like `len`, must not count or keep expired entries.
- `delete(key)` removes a key and returns True if it was present and live, otherwise False.
""",
    r'''
class _T_Clock:
    def __init__(self):
        self.t = 0.0
    def __call__(self):
        return self.t
_t_c = _T_Clock()
_t_cache = TTLCache(2, 10, _t_c)
assert len(_t_cache) == 0
_t_cache.put("a", 1)
_t_c.t = 3
_t_cache.put("b", 2)
assert len(_t_cache) == 2 and "a" in _t_cache and "b" in _t_cache
assert _t_cache.get("a") == 1
_t_c.t = 9.99
assert _t_cache.get("a") == 1
_t_c.t = 10
assert _t_cache.get("a") is None and "a" not in _t_cache
assert _t_cache.get("a", "dflt") == "dflt"
assert len(_t_cache) == 1
assert _t_cache.get("b") == 2
_t_c.t = 12.9
assert _t_cache.get("b") == 2
_t_c.t = 13
assert len(_t_cache) == 0
# LRU eviction with recency refreshed by get
_t_c = _T_Clock()
_t_cache = TTLCache(2, 100, _t_c)
_t_cache.put("x", 1)
_t_cache.put("y", 2)
assert _t_cache.get("x") == 1          # x is now most recent
_t_cache.put("z", 3)                   # evicts y
assert "y" not in _t_cache and "x" in _t_cache and "z" in _t_cache
# put on an existing key refreshes recency and lifetime, and never evicts
_t_c = _T_Clock()
_t_cache = TTLCache(2, 10, _t_c)
_t_cache.put("p", 1)
_t_c.t = 6
_t_cache.put("q", 2)
_t_c.t = 8
_t_cache.put("p", 11)                  # restarts p's lifetime, p most recent
_t_c.t = 12
assert _t_cache.get("p") == 11         # would have expired at 10 without the restart
assert _t_cache.get("q") is None       # q stored at 6, expired at 16 -> still live at 12
''' + "\n" if False else r'''
class _T_Clock:
    def __init__(self):
        self.t = 0.0
    def __call__(self):
        return self.t
_t_c = _T_Clock()
_t_cache = TTLCache(2, 10, _t_c)
assert len(_t_cache) == 0
_t_cache.put("a", 1)
_t_c.t = 3
_t_cache.put("b", 2)
assert len(_t_cache) == 2 and "a" in _t_cache and "b" in _t_cache
assert _t_cache.get("a") == 1
_t_c.t = 9.99
assert _t_cache.get("a") == 1
_t_c.t = 10
assert _t_cache.get("a") is None and "a" not in _t_cache
assert _t_cache.get("a", "dflt") == "dflt"
assert len(_t_cache) == 1
assert _t_cache.get("b") == 2
_t_c.t = 12.9
assert _t_cache.get("b") == 2
_t_c.t = 13
assert len(_t_cache) == 0
_t_c = _T_Clock()
_t_cache = TTLCache(2, 100, _t_c)
_t_cache.put("x", 1)
_t_cache.put("y", 2)
assert _t_cache.get("x") == 1
_t_cache.put("z", 3)
assert "y" not in _t_cache and "x" in _t_cache and "z" in _t_cache
_t_c = _T_Clock()
_t_cache = TTLCache(2, 10, _t_c)
_t_cache.put("p", 1)
_t_c.t = 6
_t_cache.put("q", 2)
_t_c.t = 8
_t_cache.put("p", 11)
_t_c.t = 12
assert _t_cache.get("p") == 11
assert _t_cache.get("q") == 2
assert len(_t_cache) == 2
# expired entries are purged BEFORE a live one is evicted
_t_c = _T_Clock()
_t_cache = TTLCache(2, 5, _t_c)
_t_cache.put("old", 1)
_t_c.t = 1
_t_cache.put("mid", 2)
_t_c.t = 5
_t_cache.put("new", 3)                 # "old" is expired at t=5, so "mid" must survive
assert "mid" in _t_cache and "new" in _t_cache and "old" not in _t_cache
assert len(_t_cache) == 2
# get does not restart the lifetime
_t_c = _T_Clock()
_t_cache = TTLCache(3, 10, _t_c)
_t_cache.put("k", 1)
_t_c.t = 9
assert _t_cache.get("k") == 1
_t_c.t = 10
assert _t_cache.get("k") is None
# delete
_t_c = _T_Clock()
_t_cache = TTLCache(3, 10, _t_c)
_t_cache.put("k", 1)
assert _t_cache.delete("k") is True
assert _t_cache.delete("k") is False
_t_cache.put("e", 1)
_t_c.t = 11
assert _t_cache.delete("e") is False
for _t_args in [(0, 5, _t_c), (3, 0, _t_c), (3, -1, _t_c)]:
    try:
        TTLCache(*_t_args)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
''',
    r'''
from collections import OrderedDict
class TTLCache:
    def __init__(self, capacity, ttl, clock):
        if capacity < 1 or ttl <= 0:
            raise ValueError("bad args")
        self.capacity = capacity
        self.ttl = ttl
        self.clock = clock
        self.d = OrderedDict()  # key -> (value, stored_at), LRU first
    def _expired(self, stored_at, now):
        return now - stored_at >= self.ttl
    def _purge(self):
        now = self.clock()
        for k in [k for k, (_, t) in self.d.items() if self._expired(t, now)]:
            del self.d[k]
    def put(self, key, value):
        self.d.pop(key, None)
        self.d[key] = (value, self.clock())
        if len(self.d) > self.capacity:
            self._purge()
            while len(self.d) > self.capacity:
                self.d.popitem(last=False)
    def get(self, key, default=None):
        item = self.d.get(key)
        if item is None:
            return default
        if self._expired(item[1], self.clock()):
            del self.d[key]
            return default
        self.d.move_to_end(key)
        return item[0]
    def __len__(self):
        self._purge()
        return len(self.d)
    def __contains__(self, key):
        item = self.d.get(key)
        if item is None:
            return False
        if self._expired(item[1], self.clock()):
            del self.d[key]
            return False
        return True
    def delete(self, key):
        item = self.d.get(key)
        if item is None:
            return False
        live = not self._expired(item[1], self.clock())
        del self.d[key]
        return live
''',
    wrong=[
        r'''
from collections import OrderedDict
class TTLCache:
    def __init__(self, capacity, ttl, clock):
        if capacity < 1 or ttl <= 0:
            raise ValueError("bad args")
        self.capacity = capacity
        self.ttl = ttl
        self.clock = clock
        self.d = OrderedDict()
    def put(self, key, value):
        self.d.pop(key, None)
        self.d[key] = (value, self.clock())
        while len(self.d) > self.capacity:
            self.d.popitem(last=False)
    def get(self, key, default=None):
        item = self.d.get(key)
        if item is None:
            return default
        if self.clock() - item[1] >= self.ttl:
            del self.d[key]
            return default
        self.d.move_to_end(key)
        return item[0]
    def __len__(self):
        return len(self.d)
    def __contains__(self, key):
        return key in self.d
    def delete(self, key):
        return self.d.pop(key, None) is not None
'''
    ],
)
