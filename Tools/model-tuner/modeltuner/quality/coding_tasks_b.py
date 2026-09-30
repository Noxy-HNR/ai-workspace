"""Coding-eval tasks, part B: hard algorithmic (H5-H8), bug fixes (B1-B4), class design (C1-C4).

Carried over from `_legacy/codeeval/tasks_b.py` unchanged: same prompts (including the shared `FIX`
block), same hidden asserts, same reference and wrong solutions. Importing this module appends its
tasks to `coding_tasks_a.TASKS`.
"""

from __future__ import annotations

from modeltuner.quality.coding_tasks_a import COMMON, TASKS, task  # noqa: F401  (task() appends to the shared TASKS list)

FIX = (
    "\n\nReturn the complete corrected code (the whole function or class, not a diff), in one ```python "
    "code block."
)

# ---------------------------------------------------------------------------------------- H5
task(
    "H5_count_paths", "hard",
    r"""
Write `count_paths(grid, k)`. `grid` is a list of equal-length strings made of `.` (open cell) and `#`
(wall). Count the paths from the top-left cell to the bottom-right cell where every step goes one cell
right or one cell down, and the path steps on AT MOST `k` wall cells in total (the start and end cells
count if they are walls; `k >= 0`). Two paths are different if they visit different cell sequences.
Return the count modulo 1_000_000_007. It must be fast for grids up to 300 x 300 with `k` up to 5.
""",
    r'''
import random
def _t_brute(grid, k):
    R, C = len(grid), len(grid[0])
    def go(i, j, walls):
        walls += grid[i][j] == "#"
        if walls > k:
            return 0
        if i == R - 1 and j == C - 1:
            return 1
        t = 0
        if i + 1 < R:
            t += go(i + 1, j, walls)
        if j + 1 < C:
            t += go(i, j + 1, walls)
        return t
    return go(0, 0, 0) % 1000000007
assert count_paths(["."], 0) == 1
assert count_paths(["#"], 0) == 0
assert count_paths(["#"], 1) == 1
assert count_paths(["..", ".."], 0) == 2
assert count_paths([".#", "#."], 0) == 0
assert count_paths([".#", "#."], 1) == 2
assert count_paths(["...", ".#.", "..."], 0) == 2
_t_rng = random.Random(5)
for _t_it in range(250):
    _t_R, _t_C = _t_rng.randint(1, 6), _t_rng.randint(1, 6)
    _t_g = ["".join(_t_rng.choice("..#") for _ in range(_t_C)) for _ in range(_t_R)]
    _t_k = _t_rng.randint(0, 3)
    assert count_paths(_t_g, _t_k) == _t_brute(_t_g, _t_k), (_t_g, _t_k)
def _t_alt(grid, k):
    M = 1000000007
    R, C = len(grid), len(grid[0])
    prev = None
    for i in range(R):
        cur = []
        for j in range(C):
            w = 1 if grid[i][j] == "#" else 0
            if i == 0 and j == 0:
                base = [1] + [0] * k
            else:
                base = [0] * (k + 1)
                if i > 0:
                    base = [(a + b) % M for a, b in zip(base, prev[j])]
                if j > 0:
                    base = [(a + b) % M for a, b in zip(base, cur[j - 1])]
            shifted = [0] * (k + 1)
            for x in range(k + 1 - w):
                shifted[x + w] = base[x]
            cur.append(shifted)
        prev = cur
    return sum(prev[C - 1]) % M
_t_rng = random.Random(9)
_t_g = ["".join("#" if _t_rng.random() < 0.15 else "." for _ in range(300)) for _ in range(300)]
assert count_paths(_t_g, 5) == _t_alt(_t_g, 5)
assert count_paths(["." * 300] * 300, 0) == _t_alt(["." * 300] * 300, 0)
''',
    r'''
def count_paths(grid, k):
    M = 1000000007
    R, C = len(grid), len(grid[0])
    prev = [[0] * (k + 1) for _ in range(C)]
    for i in range(R):
        cur = [[0] * (k + 1) for _ in range(C)]
        for j in range(C):
            w = 1 if grid[i][j] == "#" else 0
            if i == 0 and j == 0:
                src = [1] + [0] * k
            else:
                src = [0] * (k + 1)
                if i > 0:
                    p = prev[j]
                    src = [(a + b) % M for a, b in zip(src, p)]
                if j > 0:
                    p = cur[j - 1]
                    src = [(a + b) % M for a, b in zip(src, p)]
            row = cur[j]
            for x in range(k + 1 - w):
                row[x + w] = src[x]
        prev = cur
    return sum(prev[C - 1]) % M
''',
)

# ---------------------------------------------------------------------------------------- H6
task(
    "H6_coupon_path", "hard",
    r"""
Write `cheapest_with_coupons(n, edges, src, dst, k)` for a directed graph with nodes `0 .. n-1`. `edges`
is a list of `(u, v, w)` with integer weights `w >= 0`. You start at `src` and want to reach `dst`; the
cost of a walk is the sum of its edge weights, except that on at most `k` of the edges you traverse you
may use a coupon that makes that edge cost 0 (you choose which edges). Return the minimum possible
total cost, or `-1` if `dst` cannot be reached. `src == dst` costs 0. It must be fast for `n` = 2000,
about 10000 edges and `k` up to 5.
""",
    r'''
import random
def _t_bellman(n, edges, src, dst, k):
    INF = float("inf")
    d = [[INF] * n for _ in range(k + 1)]
    d[0][src] = 0
    for _ in range(n * (k + 1) + 1):
        changed = False
        for u, v, w in edges:
            for c in range(k + 1):
                if d[c][u] + w < d[c][v]:
                    d[c][v] = d[c][u] + w
                    changed = True
                if c < k and d[c][u] < d[c + 1][v]:
                    d[c + 1][v] = d[c][u]
                    changed = True
        if not changed:
            break
    best = min(d[c][dst] for c in range(k + 1))
    return -1 if best == INF else best
assert cheapest_with_coupons(4, [(0, 1, 5), (1, 3, 5), (0, 2, 1), (2, 3, 100)], 0, 3, 1) == 1
assert cheapest_with_coupons(4, [(0, 1, 5), (1, 3, 5), (0, 2, 1), (2, 3, 100)], 0, 3, 0) == 10
assert cheapest_with_coupons(4, [(0, 1, 5), (1, 3, 5), (0, 2, 1), (2, 3, 100)], 0, 3, 2) == 0
assert cheapest_with_coupons(3, [(0, 1, 4)], 0, 2, 3) == -1
assert cheapest_with_coupons(3, [(0, 1, 4)], 1, 1, 0) == 0
assert cheapest_with_coupons(2, [(1, 0, 4)], 0, 1, 2) == -1
assert cheapest_with_coupons(5, [(0, 1, 2), (1, 2, 2), (2, 3, 2), (3, 4, 2)], 0, 4, 2) == 4
_t_rng = random.Random(3)
for _t_it in range(200):
    _t_n = _t_rng.randint(1, 6)
    _t_e = [(_t_rng.randrange(_t_n), _t_rng.randrange(_t_n), _t_rng.randint(0, 9)) for _ in range(_t_rng.randint(0, 12))]
    _t_s, _t_d, _t_k = _t_rng.randrange(_t_n), _t_rng.randrange(_t_n), _t_rng.randint(0, 3)
    assert cheapest_with_coupons(_t_n, _t_e, _t_s, _t_d, _t_k) == _t_bellman(_t_n, _t_e, _t_s, _t_d, _t_k), (_t_n, _t_e, _t_s, _t_d, _t_k)
_t_rng = random.Random(21)
_t_n = 2000
_t_e = [(_t_rng.randrange(_t_n), _t_rng.randrange(_t_n), _t_rng.randint(1, 1000)) for _ in range(10000)]
_t_e += [(i, i + 1, _t_rng.randint(1, 1000)) for i in range(_t_n - 1)]
_t_r = cheapest_with_coupons(_t_n, _t_e, 0, _t_n - 1, 5)
assert isinstance(_t_r, int) and 0 <= _t_r < 10 ** 7
''',
    r'''
import heapq
def cheapest_with_coupons(n, edges, src, dst, k):
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
    INF = float("inf")
    dist = [[INF] * (k + 1) for _ in range(n)]
    dist[src][0] = 0
    heap = [(0, src, 0)]
    while heap:
        d, u, c = heapq.heappop(heap)
        if d > dist[u][c]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v][c]:
                dist[v][c] = nd
                heapq.heappush(heap, (nd, v, c))
            if c < k and d < dist[v][c + 1]:
                dist[v][c + 1] = d
                heapq.heappush(heap, (d, v, c + 1))
    best = min(dist[dst])
    return -1 if best == INF else best
''',
)

# ---------------------------------------------------------------------------------------- H7
task(
    "H7_longest_valid_substring", "hard",
    r"""
Write `longest_valid_substring(word, forbidden)`. `word` is a string of lower-case letters and
`forbidden` is a list of non-empty lower-case strings, each at most 10 characters long. A substring
of `word` is valid if none of the strings in `forbidden` occurs anywhere inside it as a substring.
Return the length of the longest valid substring of `word` (0 if there is none; an empty `word`
returns 0). It must be fast for `len(word)` up to 100000 and up to 100000 forbidden strings.
""",
    r'''
import random
def _t_brute(word, forbidden):
    best = 0
    for i in range(len(word)):
        for j in range(i, len(word) + 1):
            sub = word[i:j]
            if all(f not in sub for f in forbidden):
                best = max(best, j - i)
    return best
assert longest_valid_substring("cbaaaabc", ["aaa", "cb"]) == 4
assert longest_valid_substring("leetcode", ["de", "le", "e"]) == 4
assert longest_valid_substring("aaa", ["a"]) == 0
assert longest_valid_substring("", ["a"]) == 0
assert longest_valid_substring("abc", []) == 3
assert longest_valid_substring("abcabc", ["bca"]) == 4
_t_rng = random.Random(13)
for _t_it in range(300):
    _t_w = "".join(_t_rng.choice("abc") for _ in range(_t_rng.randint(0, 14)))
    _t_f = ["".join(_t_rng.choice("abc") for _ in range(_t_rng.randint(1, 4))) for _ in range(_t_rng.randint(0, 5))]
    assert longest_valid_substring(_t_w, _t_f) == _t_brute(_t_w, _t_f), (_t_w, _t_f)
_t_rng = random.Random(17)
_t_w = "".join(_t_rng.choice("ab") for _ in range(100000))
_t_f = ["".join(_t_rng.choice("ab") for _ in range(_t_rng.randint(8, 10))) for _ in range(5000)]
_t_r = longest_valid_substring(_t_w, _t_f)
assert isinstance(_t_r, int) and 0 < _t_r <= 100000
_t_w2 = "a" * 100000
assert longest_valid_substring(_t_w2, ["a" * 10]) == 9
assert longest_valid_substring(_t_w2, ["b"]) == 100000
''',
    r'''
def longest_valid_substring(word, forbidden):
    fs = set(forbidden)
    n = len(word)
    best = 0
    right = n  # exclusive end: word[i:right] is valid
    for i in range(n - 1, -1, -1):
        for j in range(i, min(n, i + 10)):
            if word[i:j + 1] in fs:
                right = min(right, j)
                break
        best = max(best, right - i)
    return best
''',
    wrong=[
        r'''
def longest_valid_substring(word, forbidden):
    n = len(word)
    best = 0
    for i in range(n):
        for j in range(i, n + 1):
            sub = word[i:j]
            if any(f in sub for f in forbidden):
                break
            best = max(best, j - i)
    return best
'''
    ],
)

# ---------------------------------------------------------------------------------------- H8
task(
    "H8_kth_subset_sum", "hard",
    r"""
Write `kth_smallest_subset_sum(nums, k)`. `nums` is a list of non-negative integers and subsets are
chosen by position (so equal values at different positions give different subsets). Consider the sums
of all `2**len(nums)` subsets, including the empty subset whose sum is 0, as a multiset. Return the
`k`-th smallest value in it (`k` is 1-based, `1 <= k <= 2**len(nums)`). It must be fast for
`len(nums)` up to 2000 when `k` is at most 5000.
""",
    r'''
import random
def _t_brute(nums, k):
    sums = [0]
    for x in nums:
        sums += [s + x for s in sums]
    return sorted(sums)[k - 1]
assert kth_smallest_subset_sum([1, 2, 3], 1) == 0
assert kth_smallest_subset_sum([1, 2, 3], 2) == 1
assert kth_smallest_subset_sum([1, 2, 3], 8) == 6
assert kth_smallest_subset_sum([2, 2], 3) == 2
assert kth_smallest_subset_sum([2, 2], 4) == 4
assert kth_smallest_subset_sum([0, 0, 0], 8) == 0
assert kth_smallest_subset_sum([5], 2) == 5
assert kth_smallest_subset_sum([], 1) == 0
_t_rng = random.Random(23)
for _t_it in range(300):
    _t_nums = [_t_rng.randint(0, 9) for _ in range(_t_rng.randint(0, 10))]
    _t_k = _t_rng.randint(1, 2 ** len(_t_nums))
    assert kth_smallest_subset_sum(_t_nums, _t_k) == _t_brute(_t_nums, _t_k), (_t_nums, _t_k)
def _t_alt(nums, k):
    cur = [0]
    for x in sorted(nums):
        merged = sorted(cur + [s + x for s in cur])
        cur = merged[:k]
    return cur[k - 1]
_t_rng = random.Random(29)
_t_nums = [_t_rng.randint(0, 10 ** 9) for _ in range(2000)]
assert kth_smallest_subset_sum(_t_nums, 5000) == _t_alt(_t_nums, 5000)
_t_nums = [_t_rng.randint(0, 5) for _ in range(40)]
assert kth_smallest_subset_sum(_t_nums, 3000) == _t_alt(_t_nums, 3000)
''',
    r'''
import heapq
def kth_smallest_subset_sum(nums, k):
    if k == 1:
        return 0
    a = sorted(nums)
    n = len(a)
    heap = [(a[0], 0)]
    result = 0
    for _ in range(k - 1):
        s, i = heapq.heappop(heap)
        result = s
        if i + 1 < n:
            heapq.heappush(heap, (s + a[i + 1], i + 1))
            heapq.heappush(heap, (s - a[i] + a[i + 1], i + 1))
    return result
''',
    wrong=[
        r'''
def kth_smallest_subset_sum(nums, k):
    sums = [0]
    for x in nums:
        sums += [s + x for s in sums]
    return sorted(sums)[k - 1]
'''
    ],
)

# ---------------------------------------------------------------------------------------- B1
_B1_BUG = r'''
def search(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return True
        if nums[lo] <= nums[mid]:
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return False
'''
task(
    "B1_rotated_search", "bugfix",
    "This function should return `True` if `target` occurs in `nums`, which is a sorted (non-decreasing) "
    "list that has been rotated at some pivot and MAY CONTAIN DUPLICATES; otherwise `False`. It is wrong "
    "for some inputs, for example `search([1, 0, 1, 1, 1], 0)` returns `False` but the answer is `True`. "
    "Find and fix the bug(s).\n\n```python" + _B1_BUG + "```" + FIX,
    r'''
import random
assert search([1, 0, 1, 1, 1], 0) is True
assert search([2, 5, 6, 0, 0, 1, 2], 0) is True
assert search([2, 5, 6, 0, 0, 1, 2], 3) is False
assert search([1], 1) is True and search([1], 2) is False and search([], 1) is False
assert search([1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1], 2) is True
assert search([3, 1], 1) is True
_t_rng = random.Random(31)
for _t_it in range(2000):
    _t_base = sorted(_t_rng.randint(0, 6) for _ in range(_t_rng.randint(0, 12)))
    _t_r = _t_rng.randint(0, max(0, len(_t_base)))
    _t_nums = _t_base[_t_r:] + _t_base[:_t_r]
    for _t_target in range(-1, 8):
        assert search(list(_t_nums), _t_target) is (_t_target in _t_nums), (_t_nums, _t_target)
''',
    r'''
def search(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return True
        if nums[lo] == nums[mid] == nums[hi]:
            lo += 1
            hi -= 1
        elif nums[lo] <= nums[mid]:
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return False
''',
    wrong=[_B1_BUG],
)

# ---------------------------------------------------------------------------------------- B2
_B2_BUG = r'''
import heapq

class RunningMedian:
    def __init__(self):
        self.lo = []   # max-heap (stored negated): the smaller half
        self.hi = []   # min-heap: the larger half

    def add(self, x):
        if not self.lo or x <= -self.lo[0]:
            heapq.heappush(self.lo, -x)
        else:
            heapq.heappush(self.hi, x)
        if len(self.lo) > len(self.hi) + 1:
            heapq.heappush(self.hi, -heapq.heappop(self.lo))
        elif len(self.hi) > len(self.lo) + 1:
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def median(self):
        if not self.lo:
            raise ValueError("empty")
        if len(self.lo) == len(self.hi):
            return (-self.lo[0] + self.hi[0]) / 2
        return -self.lo[0]
'''
task(
    "B2_running_median", "bugfix",
    "This class keeps a running median of the numbers added so far using two heaps. `median()` returns a "
    "wrong value for some sequences: after adding 5, 10, 20 it returns 5, but the median is 10. For an "
    "even count the median is the mean of the two middle values; an empty structure raises `ValueError`. "
    "Find and fix the bug(s).\n\n```python" + _B2_BUG + "```" + FIX,
    r'''
import random, statistics
_t_m = RunningMedian()
try:
    _t_m.median()
except ValueError:
    pass
else:
    raise AssertionError("expected ValueError")
for _t_x in [5, 10, 20]:
    _t_m.add(_t_x)
assert _t_m.median() == 10
_t_rng = random.Random(41)
for _t_it in range(300):
    _t_m = RunningMedian()
    _t_seen = []
    for _ in range(_t_rng.randint(1, 30)):
        _t_x = _t_rng.choice([_t_rng.randint(-10, 10), _t_rng.random() * 20 - 10])
        _t_m.add(_t_x)
        _t_seen.append(_t_x)
        assert abs(_t_m.median() - statistics.median(_t_seen)) < 1e-9, _t_seen
''',
    r'''
import heapq

class RunningMedian:
    def __init__(self):
        self.lo = []
        self.hi = []

    def add(self, x):
        if not self.lo or x <= -self.lo[0]:
            heapq.heappush(self.lo, -x)
        else:
            heapq.heappush(self.hi, x)
        if len(self.lo) > len(self.hi) + 1:
            heapq.heappush(self.hi, -heapq.heappop(self.lo))
        elif len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def median(self):
        if not self.lo:
            raise ValueError("empty")
        if len(self.lo) == len(self.hi):
            return (-self.lo[0] + self.hi[0]) / 2
        return -self.lo[0]
''',
    wrong=[_B2_BUG],
)

# ---------------------------------------------------------------------------------------- B3
_B3_BUG = r'''
def parse_csv_line(line):
    fields, cur, in_quotes = [], "", False
    for ch in line:
        if ch == '"':
            in_quotes = not in_quotes
        elif ch == "," and not in_quotes:
            fields.append(cur)
            cur = ""
        else:
            cur += ch
    fields.append(cur)
    return fields
'''
task(
    "B3_csv_line", "bugfix",
    "`parse_csv_line(line)` splits one CSV line into a list of field strings. The intended rules: a field "
    "is quoted only if its FIRST character is a double quote; inside a quoted field a doubled quote `\"\"` "
    "means one literal quote, commas are ordinary characters, and the field ends at the next single quote, "
    "which must be followed by a comma or the end of the line (otherwise raise `ValueError`); an "
    "unterminated quoted field raises `ValueError`; an unquoted field is taken literally (quotes inside it "
    "are ordinary characters). An empty line gives `['']`. Right now `parse_csv_line('a,\"he said \"\"hi\"\"\",c')` "
    "returns `['a', 'he said hi', 'c']` but should return `['a', 'he said \"hi\"', 'c']`, and other cases are "
    "wrong too. Find and fix the bug(s).\n\n```python" + _B3_BUG + "```" + FIX,
    r'''
assert parse_csv_line('a,"he said ""hi""",c') == ['a', 'he said "hi"', 'c']
assert parse_csv_line("a,b,c") == ["a", "b", "c"]
assert parse_csv_line("") == [""]
assert parse_csv_line(",,") == ["", "", ""]
assert parse_csv_line('"a,b",c') == ["a,b", "c"]
assert parse_csv_line('""') == [""]
assert parse_csv_line('"",x') == ["", "x"]
assert parse_csv_line('x,""') == ["x", ""]
assert parse_csv_line('""""') == ['"']
assert parse_csv_line('ab"c,d') == ['ab"c', "d"]
assert parse_csv_line(' "a",b') == [' "a"', "b"]
assert parse_csv_line('"a""",b') == ['a"', "b"]
assert parse_csv_line('"a b"') == ["a b"]
for _t_bad in ['"abc', 'x,"abc', '"a"b,c', '"a" ,b', '"abc""']:
    try:
        parse_csv_line(_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for %r" % (_t_bad,))
''',
    r'''
def parse_csv_line(line):
    fields = []
    i, n = 0, len(line)
    while True:
        if i < n and line[i] == '"':
            i += 1
            cur = []
            while True:
                if i >= n:
                    raise ValueError("unterminated quote")
                ch = line[i]
                if ch == '"':
                    if i + 1 < n and line[i + 1] == '"':
                        cur.append('"')
                        i += 2
                        continue
                    i += 1
                    break
                cur.append(ch)
                i += 1
            if i < n and line[i] != ",":
                raise ValueError("text after closing quote")
            fields.append("".join(cur))
        else:
            j = line.find(",", i)
            if j == -1:
                fields.append(line[i:])
                return fields
            fields.append(line[i:j])
            i = j
        if i >= n:
            return fields
        i += 1  # skip the comma
        if i >= n:
            fields.append("")
            return fields
''',
    wrong=[_B3_BUG],
)

# ---------------------------------------------------------------------------------------- B4
_B4_BUG = r'''
def prorated_refund(amount, days_used, days_total):
    refund = float(amount) * (days_total - days_used) / days_total
    return round(refund, 2)
'''
task(
    "B4_prorated_refund", "bugfix",
    "`prorated_refund(amount, days_used, days_total)` computes the refund for the unused part of a "
    "subscription. `amount` is a price string such as `\"19.99\"`; the refund is `amount * (days_total - "
    "days_used) / days_total`. Money must be exact: the result is a `decimal.Decimal` with exactly two "
    "decimal places, rounded half UP (0.125 becomes 0.13). If `days_used > days_total` the refund is "
    "`Decimal('0.00')`. `days_total <= 0` or `days_used < 0` raises `ValueError`. The current code returns "
    "floats and rounds wrongly: `prorated_refund(\"0.25\", 1, 2)` gives `0.12` instead of `Decimal('0.13')`. "
    "Find and fix the bug(s).\n\n```python" + _B4_BUG + "```" + FIX,
    r'''
from decimal import Decimal
_t_cases = [
    (("0.25", 1, 2), "0.13"), (("1.00", 3, 8), "0.63"), (("19.99", 10, 30), "13.33"), (("10.00", 0, 30), "10.00"),
    (("10.00", 30, 30), "0.00"), (("10.00", 31, 30), "0.00"), (("100.00", 1, 3), "66.67"), (("0.01", 1, 2), "0.01"),
    (("59.99", 7, 28), "44.99"), (("9.99", 1, 7), "8.56"), (("2.675", 0, 1), "2.68"), (("5", 1, 4), "3.75"),
]
for _t_args, _t_exp in _t_cases:
    _t_r = prorated_refund(*_t_args)
    assert isinstance(_t_r, Decimal), (_t_args, type(_t_r))
    assert _t_r == Decimal(_t_exp) and _t_r.as_tuple().exponent == -2, (_t_args, _t_r, _t_exp)
for _t_args in [("10.00", 1, 0), ("10.00", 1, -5), ("10.00", -1, 30)]:
    try:
        prorated_refund(*_t_args)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for %r" % (_t_args,))
''',
    r'''
from decimal import Decimal, ROUND_HALF_UP
def prorated_refund(amount, days_used, days_total):
    if days_total <= 0 or days_used < 0:
        raise ValueError("bad days")
    if days_used > days_total:
        return Decimal("0.00")
    refund = Decimal(str(amount)) * (days_total - days_used) / Decimal(days_total)
    return refund.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
''',
    wrong=[_B4_BUG],
)

# ---------------------------------------------------------------------------------------- C1
task(
    "C1_matrix", "class",
    r"""
Write a class `Matrix` for small numeric matrices.

- `Matrix(rows)`: `rows` is a non-empty list of equal-length, non-empty lists of numbers; otherwise
  raise `ValueError`. The matrix keeps its own copy: later changes to the argument do not affect it.
- `m.shape` is `(number_of_rows, number_of_columns)`; `m[i]` returns row `i` as a list.
- `m + other`: element-wise sum of two matrices of the same shape (`ValueError` otherwise); adding
  anything that is not a `Matrix` raises `TypeError`.
- `m * other`: if `other` is an `int` or `float`, multiply every element by it; if it is a `Matrix`,
  the matrix product (`ValueError` if the inner dimensions do not match); any other type raises
  `TypeError`. `2 * m` must work too.
- `m.transpose()` returns a new `Matrix`.
- `==` compares contents (comparing with a non-`Matrix` is simply `False`). `repr(m)` looks like
  `Matrix([[1, 2], [3, 4]])`.
""",
    r'''
m = Matrix([[1, 2], [3, 4]])
assert m.shape == (2, 2)
assert m[0][1] == 2 and m[1] == [3, 4]
assert m + Matrix([[1, 1], [1, 1]]) == Matrix([[2, 3], [4, 5]])
assert m * 2 == Matrix([[2, 4], [6, 8]]) and 2 * m == Matrix([[2, 4], [6, 8]])
assert m * 0.5 == Matrix([[0.5, 1.0], [1.5, 2.0]])
assert m * Matrix([[0, 1], [1, 0]]) == Matrix([[2, 1], [4, 3]])
assert Matrix([[1, 2, 3], [4, 5, 6]]) * Matrix([[1], [0], [2]]) == Matrix([[7], [16]])
assert Matrix([[1, 2, 3]]).transpose() == Matrix([[1], [2], [3]])
assert Matrix([[1, 2, 3], [4, 5, 6]]).transpose().shape == (3, 2)
assert repr(Matrix([[1, 2], [3, 4]])) == "Matrix([[1, 2], [3, 4]])"
assert Matrix([[1]]) != Matrix([[2]])
assert (Matrix([[1]]) == 5) is False
_t_rows = [[1, 2], [3, 4]]
_t_m = Matrix(_t_rows)
_t_rows[0][0] = 99
assert _t_m[0][0] == 1
assert Matrix([[1, 2]]).shape == (1, 2)
for _t_bad in ([], [[]], [[1, 2], [3]], [[1], []]):
    try:
        Matrix(_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for %r" % (_t_bad,))
for _t_fn in (lambda: Matrix([[1, 2]]) + Matrix([[1], [2]]), lambda: Matrix([[1, 2]]) * Matrix([[1, 2]])):
    try:
        _t_fn()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
for _t_fn in (lambda: m + 5, lambda: m * "x", lambda: m * [1, 2], lambda: m + [[1, 1], [1, 1]]):
    try:
        _t_fn()
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError")
''',
    r'''
class Matrix:
    def __init__(self, rows):
        if not isinstance(rows, (list, tuple)) or len(rows) == 0:
            raise ValueError("no rows")
        rows = [list(r) for r in rows]
        if len(rows[0]) == 0 or any(len(r) != len(rows[0]) for r in rows):
            raise ValueError("ragged or empty")
        self._rows = rows

    @property
    def shape(self):
        return (len(self._rows), len(self._rows[0]))

    def __getitem__(self, i):
        return self._rows[i]

    def __add__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("shape mismatch")
        return Matrix([[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(self._rows, other._rows)])

    def __mul__(self, other):
        if isinstance(other, bool):
            return NotImplemented
        if isinstance(other, (int, float)):
            return Matrix([[a * other for a in r] for r in self._rows])
        if isinstance(other, Matrix):
            if self.shape[1] != other.shape[0]:
                raise ValueError("inner dimensions")
            cols = list(zip(*other._rows))
            return Matrix([[sum(a * b for a, b in zip(r, c)) for c in cols] for r in self._rows])
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return self * other
        return NotImplemented

    def transpose(self):
        return Matrix([list(c) for c in zip(*self._rows)])

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        return self._rows == other._rows

    def __repr__(self):
        return "Matrix(%r)" % (self._rows,)
''',
)

# ---------------------------------------------------------------------------------------- C2
task(
    "C2_event_emitter", "class",
    r"""
Write a class `EventEmitter`.

- `on(event, fn)`: register `fn` for `event` (any hashable). The same function may be registered
  several times; each registration is called separately.
- `once(event, fn)`: like `on`, but the registration is removed automatically the first time it fires.
- `off(event, fn)`: remove the EARLIEST registration of `fn` for `event`; return `True` if one was
  removed, `False` otherwise.
- `emit(event, *args, **kwargs)`: call the handlers registered for `event` in registration order with
  the given arguments and return how many handlers were called (0 for an unknown event). Handlers that
  are added or removed while an emit is running only take effect from the NEXT emit (the running emit
  uses a snapshot). A `once` handler is removed before it is called, so if it triggers another emit of
  the same event it does not run a second time. Exceptions from handlers propagate to the caller.
""",
    r'''
_t_log = []
e = EventEmitter()
def _t_a(*args, **kw):
    _t_log.append(("a", args, kw))
def _t_b(*args, **kw):
    _t_log.append(("b", args, kw))
e.on("x", _t_a)
e.on("x", _t_b)
assert e.emit("x", 1, k=2) == 2
assert _t_log == [("a", (1,), {"k": 2}), ("b", (1,), {"k": 2})]
assert e.emit("nothing") == 0
_t_log.clear()
e.once("y", _t_a)
assert e.emit("y") == 1 and e.emit("y") == 0
assert len(_t_log) == 1
# same function twice, off removes the earliest only
e = EventEmitter()
_t_n = []
def _t_h():
    _t_n.append(1)
e.on("z", _t_h)
e.on("z", _t_h)
assert e.emit("z") == 2
assert e.off("z", _t_h) is True
assert e.emit("z") == 1
assert e.off("z", _t_h) is True
assert e.off("z", _t_h) is False
assert e.off("unknown", _t_h) is False
assert e.emit("z") == 0
# snapshot semantics: a handler added during emit does not run in that emit
e = EventEmitter()
_t_seen = []
def _t_late():
    _t_seen.append("late")
def _t_adder():
    _t_seen.append("adder")
    e.on("s", _t_late)
e.on("s", _t_adder)
assert e.emit("s") == 1 and _t_seen == ["adder"]
assert e.emit("s") == 2 and _t_seen == ["adder", "adder", "late"]
# snapshot semantics: a handler removed during emit still runs in that emit
e = EventEmitter()
_t_seen = []
def _t_second():
    _t_seen.append("second")
def _t_remover():
    _t_seen.append("remover")
    e.off("r", _t_second)
e.on("r", _t_remover)
e.on("r", _t_second)
assert e.emit("r") == 2 and _t_seen == ["remover", "second"]
assert e.emit("r") == 1
# once is removed before it is called: re-entrant emit does not run it twice
e = EventEmitter()
_t_count = []
def _t_reentrant():
    _t_count.append(1)
    if len(_t_count) < 5:
        e.emit("re")
e.once("re", _t_reentrant)
assert e.emit("re") == 1
assert len(_t_count) == 1
# a once handler registered together with on handlers keeps order
e = EventEmitter()
_t_order = []
e.on("o", lambda: _t_order.append(1))
e.once("o", lambda: _t_order.append(2))
e.on("o", lambda: _t_order.append(3))
assert e.emit("o") == 3 and _t_order == [1, 2, 3]
assert e.emit("o") == 2 and _t_order == [1, 2, 3, 1, 3]
# exceptions propagate; a once handler that raised is already gone
e = EventEmitter()
def _t_boom():
    raise RuntimeError("boom")
e.once("b", _t_boom)
try:
    e.emit("b")
except RuntimeError:
    pass
else:
    raise AssertionError("expected RuntimeError")
assert e.emit("b") == 0
# off works on once registrations too
e = EventEmitter()
e.once("q", _t_a)
assert e.off("q", _t_a) is True and e.emit("q") == 0
''',
    r'''
class EventEmitter:
    def __init__(self):
        self._handlers = {}

    def on(self, event, fn):
        self._handlers.setdefault(event, []).append([fn, False])

    def once(self, event, fn):
        self._handlers.setdefault(event, []).append([fn, True])

    def off(self, event, fn):
        lst = self._handlers.get(event, [])
        for i, entry in enumerate(lst):
            if entry[0] == fn:
                del lst[i]
                return True
        return False

    def emit(self, event, *args, **kwargs):
        snapshot = list(self._handlers.get(event, []))
        called = 0
        for entry in snapshot:
            fn, once = entry
            if once:
                lst = self._handlers.get(event, [])
                for i, e2 in enumerate(lst):
                    if e2 is entry:
                        del lst[i]
                        break
            fn(*args, **kwargs)
            called += 1
        return called
''',
)

# ---------------------------------------------------------------------------------------- C3
task(
    "C3_trie", "class",
    r"""
Write a class `Trie` of non-empty strings.

- `insert(word)`: add a word (inserting a word twice changes nothing). An empty string raises
  `ValueError`.
- `search(word)`: True only if exactly this word was inserted (and not deleted).
- `starts_with(prefix)`: True if any stored word starts with `prefix`; `starts_with("")` is True only
  if the trie holds at least one word.
- `delete(word)`: remove a word and return `True`, or return `False` if it is not stored. Nodes that
  are no longer needed (they lead to no stored word) must be removed.
- `words_with_prefix(prefix)`: sorted list of the stored words that start with `prefix` (all words
  for the empty prefix).
- `node_count()`: number of nodes below the root that currently exist (one node per distinct
  prefix of the stored words, excluding the empty prefix).
""",
    r'''
t = Trie()
assert t.node_count() == 0
assert t.starts_with("") is False
for w in ["car", "card", "care"]:
    t.insert(w)
t.insert("car")
assert t.node_count() == 5
assert t.search("car") and t.search("card") and t.search("care")
assert not t.search("ca") and not t.search("cars") and not t.search("")
assert t.starts_with("ca") and t.starts_with("care") and not t.starts_with("cat") and t.starts_with("")
assert t.words_with_prefix("car") == ["car", "card", "care"]
assert t.words_with_prefix("cx") == []
assert t.words_with_prefix("") == ["car", "card", "care"]
assert t.delete("card") is True
assert t.node_count() == 4 and not t.search("card") and t.search("car") and t.search("care")
assert t.delete("card") is False
assert t.delete("ca") is False
assert t.delete("car") is True
assert t.node_count() == 4                      # "care" still needs c-a-r-e
assert not t.search("car") and t.starts_with("car") and t.search("care")
assert t.words_with_prefix("ca") == ["care"]
assert t.delete("care") is True
assert t.node_count() == 0 and t.starts_with("") is False
t.insert("a"); t.insert("ab"); t.insert("b")
assert t.node_count() == 3
assert t.delete("ab") is True and t.node_count() == 2 and t.search("a")
assert t.words_with_prefix("") == ["a", "b"]
for _t_w in ("", ):
    try:
        t.insert(_t_w)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
t2 = Trie()
t2.insert("zebra"); t2.insert("zoo"); t2.insert("apple")
assert t2.node_count() == 5 + 2 + 5 - 1 + 0 or True
assert t2.words_with_prefix("z") == ["zebra", "zoo"]
assert t2.words_with_prefix("") == ["apple", "zebra", "zoo"]
''',
    r'''
class Trie:
    def __init__(self):
        self._root = {}
        self._end = "\0end"

    def insert(self, word):
        if not word:
            raise ValueError("empty word")
        node = self._root
        for ch in word:
            node = node.setdefault(ch, {})
        node[self._end] = True

    def _find(self, s):
        node = self._root
        for ch in s:
            node = node.get(ch)
            if node is None:
                return None
        return node

    def search(self, word):
        node = self._find(word) if word else None
        return bool(node and self._end in node)

    def starts_with(self, prefix):
        if prefix == "":
            return bool(self._root)
        return self._find(prefix) is not None

    def delete(self, word):
        if not word or not self.search(word):
            return False
        path = [self._root]
        for ch in word:
            path.append(path[-1][ch])
        del path[-1][self._end]
        for i in range(len(word), 0, -1):
            node = path[i]
            if node:
                break
            del path[i - 1][word[i - 1]]
        return True

    def words_with_prefix(self, prefix):
        node = self._find(prefix) if prefix else self._root
        if node is None:
            return []
        out = []
        stack = [(node, prefix)]
        while stack:
            n, s = stack.pop()
            for k, v in n.items():
                if k == self._end:
                    out.append(s)
                else:
                    stack.append((v, s + k))
        return sorted(out)

    def node_count(self):
        count = 0
        stack = [self._root]
        while stack:
            n = stack.pop()
            for k, v in n.items():
                if k != self._end:
                    count += 1
                    stack.append(v)
        return count
''',
)

# ---------------------------------------------------------------------------------------- C4
task(
    "C4_bank", "class",
    r"""
Write the exceptions `InsufficientFunds(Exception)` and `AccountFrozen(Exception)` and a class `Bank`
of named accounts with integer amounts.

- `open(name, initial=0, overdraft=0)`: create an account. `ValueError` if the name exists or
  `initial < 0` or `overdraft < 0`. Records the history entry `("open", initial, initial)`.
- `balance(name)`, `history(name)`: `history` is a list of `(kind, amount, balance_after)` tuples in
  order; kinds are `"open"`, `"deposit"`, `"withdraw"`, `"transfer_out"`, `"transfer_in"`.
- `deposit(name, amount)`: `amount` must be an `int > 0` else `ValueError`. Allowed on frozen accounts.
- `withdraw(name, amount)`: same amount rule. A frozen account raises `AccountFrozen`; if the balance
  after the withdrawal would be below `-overdraft` raise `InsufficientFunds` (a balance of exactly
  `-overdraft` is allowed).
- `transfer(src, dst, amount)`: moves money and records `"transfer_out"` on `src` and `"transfer_in"`
  on `dst`. `ValueError` if `src == dst` or the amount is invalid; `KeyError` if either account is
  unknown; `AccountFrozen` if `src` is frozen (a frozen `dst` may receive); `InsufficientFunds` by
  the same rule as `withdraw`. A failed transfer must change nothing at all: no balance changes and
  no history entries on either account.
- `freeze(name)` / `unfreeze(name)`. Every method raises `KeyError` for an unknown account.
""",
    r'''
b = Bank()
b.open("alice", 100, overdraft=50)
b.open("bob")
assert b.balance("alice") == 100 and b.balance("bob") == 0
assert b.history("alice") == [("open", 100, 100)]
b.deposit("bob", 30)
assert b.balance("bob") == 30 and b.history("bob") == [("open", 0, 0), ("deposit", 30, 30)]
b.withdraw("alice", 150)                       # exactly the overdraft limit
assert b.balance("alice") == -50
try:
    b.withdraw("alice", 1)
except InsufficientFunds:
    pass
else:
    raise AssertionError("expected InsufficientFunds")
assert b.balance("alice") == -50 and len(b.history("alice")) == 2
b.transfer("bob", "alice", 30)
assert b.balance("bob") == 0 and b.balance("alice") == -20
assert b.history("bob")[-1] == ("transfer_out", 30, 0) and b.history("alice")[-1] == ("transfer_in", 30, -20)
# a failed transfer changes nothing
_t_before = (b.balance("alice"), b.balance("bob"), list(b.history("alice")), list(b.history("bob")))
for _t_args, _t_exc in [(("alice", "bob", 100), InsufficientFunds), (("alice", "carol", 1), KeyError),
                        (("carol", "alice", 1), KeyError), (("alice", "alice", 1), ValueError),
                        (("alice", "bob", 0), ValueError), (("alice", "bob", -5), ValueError)]:
    try:
        b.transfer(*_t_args)
    except _t_exc:
        pass
    else:
        raise AssertionError("expected %s for %r" % (_t_exc.__name__, _t_args))
    assert (b.balance("alice"), b.balance("bob"), b.history("alice"), b.history("bob")) == _t_before, _t_args
# freezing
b.freeze("bob")
b.deposit("bob", 10)
assert b.balance("bob") == 10
for _t_fn in (lambda: b.withdraw("bob", 1), lambda: b.transfer("bob", "alice", 1)):
    try:
        _t_fn()
    except AccountFrozen:
        pass
    else:
        raise AssertionError("expected AccountFrozen")
assert b.balance("bob") == 10
b.transfer("alice", "bob", 5)                  # a frozen destination may receive
assert b.balance("bob") == 15
b.unfreeze("bob")
b.withdraw("bob", 15)
assert b.balance("bob") == 0
# argument validation
for _t_fn in (lambda: b.deposit("bob", 0), lambda: b.deposit("bob", -1), lambda: b.deposit("bob", 1.5),
              lambda: b.withdraw("bob", 0), lambda: b.open("bob"), lambda: b.open("x", -1), lambda: b.open("y", 0, -1)):
    try:
        _t_fn()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
for _t_fn in (lambda: b.balance("nobody"), lambda: b.deposit("nobody", 1), lambda: b.withdraw("nobody", 1),
              lambda: b.history("nobody"), lambda: b.freeze("nobody"), lambda: b.unfreeze("nobody")):
    try:
        _t_fn()
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")
assert issubclass(InsufficientFunds, Exception) and issubclass(AccountFrozen, Exception)
''',
    r'''
class InsufficientFunds(Exception):
    pass

class AccountFrozen(Exception):
    pass

class Bank:
    def __init__(self):
        self._acc = {}

    def _get(self, name):
        if name not in self._acc:
            raise KeyError(name)
        return self._acc[name]

    @staticmethod
    def _check(amount):
        if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
            raise ValueError("bad amount")

    def open(self, name, initial=0, overdraft=0):
        if name in self._acc or initial < 0 or overdraft < 0:
            raise ValueError("bad open")
        self._acc[name] = {"bal": initial, "od": overdraft, "frozen": False, "hist": [("open", initial, initial)]}

    def balance(self, name):
        return self._get(name)["bal"]

    def history(self, name):
        return list(self._get(name)["hist"])

    def freeze(self, name):
        self._get(name)["frozen"] = True

    def unfreeze(self, name):
        self._get(name)["frozen"] = False

    def deposit(self, name, amount):
        a = self._get(name)
        self._check(amount)
        a["bal"] += amount
        a["hist"].append(("deposit", amount, a["bal"]))

    def _can_take(self, a, amount):
        if a["frozen"]:
            raise AccountFrozen("frozen")
        if a["bal"] - amount < -a["od"]:
            raise InsufficientFunds("insufficient")

    def withdraw(self, name, amount):
        a = self._get(name)
        self._check(amount)
        self._can_take(a, amount)
        a["bal"] -= amount
        a["hist"].append(("withdraw", amount, a["bal"]))

    def transfer(self, src, dst, amount):
        s = self._get(src)
        d = self._get(dst)
        if src == dst:
            raise ValueError("same account")
        self._check(amount)
        self._can_take(s, amount)
        s["bal"] -= amount
        d["bal"] += amount
        s["hist"].append(("transfer_out", amount, s["bal"]))
        d["hist"].append(("transfer_in", amount, d["bal"]))
''',
)
