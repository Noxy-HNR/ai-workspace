"""Agent tasks for the tool suite: a small Python repo, task prompts, hidden-test graders and reference solutions.

Carried over from the hand-run harness unchanged: the same base repository, the same task prompts, the same
graders (hidden unittest suites, source-text checks, a todo-trace check) and the same reference solutions.
`tools.py` drives a model through them in a sandbox; nothing here loads a model, measures anything or scores
a run, and no grader depends on anything outside this module.
"""

from __future__ import annotations

import random
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

__all__ = ["BASE", "BASE_TESTS", "GRADE", "LONG_PRELUDE_USER", "TASKS", "TASK", "make_sandbox", "py_files",
           "run_hidden_tests", "todo_valid"]

# ---------------------------------------------------------------- base repo

BASE = {
    "README.md": "# shop\n\nA tiny order-pricing library. Run the tests with `python -m unittest`.\n",
    "shop/__init__.py": "",
    "shop/models.py": '''from dataclasses import dataclass, field


@dataclass
class Item:
    name: str
    price_cents: int
    qty: int = 1


@dataclass
class Order:
    customer: str
    items: list = field(default_factory=list)
    loyalty_years: int = 0
''',
    "shop/pricing.py": '''from shop.models import Order


def calc_total(order: Order) -> int:
    """Sum of item prices times quantities, in cents."""
    return sum(item.price_cents * item.qty for item in order.items)


def apply_loyalty_discount(total_cents: int, years: int) -> int:
    """5% off after 2 years, 10% off after 5 years."""
    if years >= 5:
        return total_cents - total_cents // 10
    if years >= 2:
        return total_cents - total_cents // 20
    return total_cents


def parse_price(text: str) -> int:
    """Parse a price like "12.34" into cents."""
    value = float(text.strip())
    return int(round(value * 100))
''',
    "shop/tax.py": '''def compute_tax(amount_cents: int, rate: float) -> int:
    """Tax in cents, rounded to the nearest cent."""
    return round(amount_cents * rate)
''',
    "shop/utils.py": '''def format_cents(cents: int) -> str:
    """1234 -> "$12.34"."""
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}${cents // 100}.{cents % 100:02d}"


def clamp(value, low, high):
    return max(low, min(high, value))
''',
    "shop/config.py": '''# Service configuration defaults.
DEFAULT_TIMEOUT_S = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_REGION = "us-east"
''',
    "shop/cli.py": '''import sys

from shop.models import Item, Order
from shop.pricing import calc_total
from shop.utils import format_cents


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    order = Order(customer="cli", items=[Item("thing", int(p)) for p in argv])
    print(format_cents(calc_total(order)))


if __name__ == "__main__":
    main()
''',
    "shop/report.py": '''from shop.pricing import apply_loyalty_discount, calc_total
from shop.utils import format_cents


def summarize(order) -> str:
    total = calc_total(order)
    discounted = apply_loyalty_discount(total, order.loyalty_years)
    return f"{order.customer}: {format_cents(discounted)}"
''',
    "handlers/__init__.py": "",
    "handlers/orders.py": "def handle_order(payload):\n    return {'status': 'ok', 'kind': 'order', 'id': payload.get('id')}\n",
    "handlers/refunds.py": "def handle_refund(payload):\n    return {'status': 'ok', 'kind': 'refund', 'id': payload.get('id')}\n",
    "handlers/shipping.py": "def handle_shipping(payload):\n    return {'status': 'ok', 'kind': 'shipping', 'id': payload.get('id')}\n",
    "config/settings.yaml": "service:\n  name: shop-api\n  host: 0.0.0.0\n  port: 8431\n  workers: 4\n",
    "tests/__init__.py": "",
    "tests/test_pricing.py": '''import unittest

from shop.models import Item, Order
from shop.pricing import apply_loyalty_discount, calc_total, parse_price


class PricingTests(unittest.TestCase):
    def test_total(self):
        order = Order("a", [Item("x", 250, 2), Item("y", 100)])
        self.assertEqual(calc_total(order), 600)

    def test_loyalty(self):
        self.assertEqual(apply_loyalty_discount(1000, 1), 1000)
        self.assertEqual(apply_loyalty_discount(1000, 2), 950)
        self.assertEqual(apply_loyalty_discount(1000, 5), 900)

    def test_parse_price(self):
        self.assertEqual(parse_price("12.34"), 1234)
        self.assertEqual(parse_price(" 0.5 "), 50)


if __name__ == "__main__":
    unittest.main()
''',
    "tests/test_tax.py": '''import unittest

from shop.tax import compute_tax


class TaxTests(unittest.TestCase):
    def test_rounds_to_nearest_cent(self):
        self.assertEqual(compute_tax(1999, 0.0825), 165)
        self.assertEqual(compute_tax(1000, 0.07), 70)
        self.assertEqual(compute_tax(4321, 0.0625), 270)


if __name__ == "__main__":
    unittest.main()
''',
    "tests/test_utils.py": '''import unittest

from shop.utils import clamp, format_cents


class UtilsTests(unittest.TestCase):
    def test_format_cents(self):
        self.assertEqual(format_cents(1234), "$12.34")
        self.assertEqual(format_cents(-5), "-$0.05")

    def test_clamp(self):
        self.assertEqual(clamp(5, 0, 3), 3)


if __name__ == "__main__":
    unittest.main()
''',
    "tests/test_orders.py": "import unittest\n\nfrom handlers.orders import handle_order\n\n\nclass OrdersTests(unittest.TestCase):\n    def test_ok(self):\n        self.assertEqual(handle_order({'id': 1})['status'], 'ok')\n",
    "tests/test_shipping.py": "import unittest\n\nfrom handlers.shipping import handle_shipping\n\n\nclass ShippingTests(unittest.TestCase):\n    def test_ok(self):\n        self.assertEqual(handle_shipping({'id': 1})['kind'], 'shipping')\n",
}


def _app_log() -> str:
    rng = random.Random(11)
    services = ["api", "worker", "scheduler", "billing", "auth", "mailer"]
    msgs = [
        "request completed in {n}ms", "cache hit ratio {p}%", "retry {a}/5 for job {j}", "queue depth {n}",
        "user {u} logged in", "payment batch {j} settled", "slow query took {n}ms", "health check ok",
        "connection pool size {a}", "rotated log file", "job {j} finished with status ok",
    ]
    lines = []
    for i in range(620):
        m = rng.choice(msgs).format(n=rng.randint(3, 900), p=rng.randint(40, 99), a=rng.randint(1, 5),
                                    j=rng.randint(1000, 9999), u=rng.randint(100, 999))
        h, mi, s = 1 + i // 240, (i // 4) % 60, (i * 7) % 60
        lines.append(f"2026-09-14 {h:02d}:{mi:02d}:{s:02d} INFO [{rng.choice(services)}] {m}")
    lines.insert(413, "2026-09-14 02:43:31 INFO [scheduler] config reload: max_retries changed from 3 to 7 (ticket OPS-2291)")
    return "\n".join(lines) + "\n"


BASE["logs/app.log"] = _app_log()

# ---------------------------------------------------------------- long module (long-context tasks)

GRACE_BLOCK = "    grace = 10\n"


def _legacy_billing(n_funcs: int = 620) -> str:
    rng = random.Random(23)
    regions = ["AB", "BC", "MB", "NB", "NL", "NS", "ON", "PE", "QC", "SK", "YT", "NT", "NU"]
    out = ['"""Legacy billing rules. Machine-migrated from the old system; do not reformat."""\n\n',
           "LATE_FEE_GRACE_DAYS = 10\nLATE_FEE_DAILY_CENTS = 150\nROUNDING_MODE = 'half-up'\n\n\n"]
    sign_off_at, late_fee_at = int(n_funcs * 0.41), int(n_funcs * 0.68)
    for i in range(n_funcs):
        if i == sign_off_at:
            out.append("# finance sign-off: approval code FIN-4471-QX (do not remove)\n\n\n")
        if i == late_fee_at:
            out.append('def late_fee(days_late):\n'
                       '    """Late fee in cents for a payment made `days_late` days after the due date."""\n'
                       + GRACE_BLOCK +
                       '    if days_late <= grace:\n'
                       '        return 0\n'
                       '    return (days_late - grace) * LATE_FEE_DAILY_CENTS\n\n\n')
        r, a, b, c = rng.choice(regions), rng.randint(2, 40), rng.randint(5, 60), rng.randint(1, 9)
        kind = i % 3
        if kind == 0:
            body = (f"    base = amount_cents * {a} // 1000\n    if base < {b}:\n        base = {b}\n"
                    f"    return base + {c}\n")
        elif kind == 1:
            body = (f"    if amount_cents > {b * 100}:\n        return amount_cents * {a} // 10000\n"
                    f"    return {c * 10}\n")
        else:
            body = (f"    tiers = [{a}, {b}, {c * 7}]\n    idx = min(len(tiers) - 1, amount_cents // {b * 250 + 1})\n"
                    f"    return tiers[idx]\n")
        out.append(f'def fee_rule_{i:04d}(amount_cents):\n    """Legacy fee rule {i} for region {r}."""\n{body}\n\n')
    return "".join(out)


LEGACY = _legacy_billing()
LEGACY_TEST = '''import unittest

from shop.legacy_billing import late_fee


class LateFeeTests(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(late_fee(10), 0)
        self.assertEqual(late_fee(12), 300)


if __name__ == "__main__":
    unittest.main()
'''

# ---------------------------------------------------------------- sandbox + hidden tests


def make_sandbox(task: dict, root: Path) -> Path:
    """Write the base repo plus this task's extra files into `root` (which must not exist yet or is replaced)."""
    if root.exists():
        shutil.rmtree(root)
    for rel, text in BASE.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    for rel, text in task.get("extra_files", {}).items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return root


def run_hidden_tests(sandbox: Path, hidden: dict[str, str], drop: tuple[str, ...] = ()) -> tuple[bool, str]:
    """Copy the sandbox's code, replace its tests with the hidden ones, run unittest. Returns (ok, output)."""
    with tempfile.TemporaryDirectory() as tmp:
        g = Path(tmp) / "g"
        shutil.copytree(sandbox, g, ignore=shutil.ignore_patterns("__pycache__"))
        if (g / "tests").exists():
            shutil.rmtree(g / "tests")
        (g / "tests").mkdir()
        (g / "tests/__init__.py").write_text("")
        for rel, text in hidden.items():
            if rel in drop:
                continue
            (g / rel).write_text(text, encoding="utf-8")
        p = subprocess.run([sys.executable, "-m", "unittest", "-q"], cwd=g, capture_output=True, text=True, timeout=120)
        return p.returncode == 0, (p.stdout + p.stderr)[-1500:]


BASE_TESTS = {k: v for k, v in BASE.items() if k.startswith("tests/test_")}


def py_files(sandbox: Path) -> list[Path]:
    return [p for p in sandbox.rglob("*.py") if "__pycache__" not in p.parts]


# ---------------------------------------------------------------- tasks

GRADE = Callable[[Path, str, list], tuple[bool, str]]

TASKS: list[dict] = []


def task(**kw) -> None:
    kw.setdefault("long", False)
    kw.setdefault("extra_files", {})
    TASKS.append(kw)


# T1 -------------------------------------------------------------
task(id="T1_locate",
     prompt="Which function in this repository applies the loyalty discount, and which file is it in? "
            "Answer with the function name and the file path.",
     grade=lambda sb, final, trace: ("apply_loyalty_discount" in final and "pricing.py" in final,
                                      "needs apply_loyalty_discount + pricing.py"),
     reference=lambda sb: "It is apply_loyalty_discount in shop/pricing.py.")

# T2 -------------------------------------------------------------
_BUGGY_TAX = BASE["shop/tax.py"].replace("return round(amount_cents * rate)", "return int(amount_cents * rate)")


def _grade_t2(sb, final, trace):
    ok, out = run_hidden_tests(sb, BASE_TESTS)
    return ok, "hidden suite" + ("" if ok else ": " + out[-300:])


def _ref_t2(sb):
    (sb / "shop/tax.py").write_text(BASE["shop/tax.py"])
    return "Fixed the truncation in compute_tax; tests pass."


task(id="T2_bugfix", extra_files={"shop/tax.py": _BUGGY_TAX},
     prompt="The tax tests are failing. Find the bug in the source code and fix it (do not change the tests). "
            "Run the tests to confirm they pass.",
     grade=_grade_t2, reference=_ref_t2)

# T3 -------------------------------------------------------------
_T3_HIDDEN = dict(BASE_TESTS)
_T3_HIDDEN["tests/test_pricing.py"] = BASE_TESTS["tests/test_pricing.py"].replace("calc_total", "compute_order_total")
_T3_HIDDEN["tests/test_rename.py"] = '''import unittest

import shop.cli
from shop.models import Item, Order
from shop.report import summarize


class RenameTests(unittest.TestCase):
    def test_report_uses_new_name(self):
        self.assertEqual(summarize(Order("z", [Item("a", 1000)], loyalty_years=5)), "z: $9.00")

    def test_cli_imports(self):
        self.assertTrue(hasattr(shop.cli, "main"))
'''


def _grade_t3(sb, final, trace):
    left = [str(p.relative_to(sb)) for p in py_files(sb) if "calc_total" in p.read_text(encoding="utf-8", errors="replace")]
    if left:
        return False, f"calc_total still in {left}"
    ok, out = run_hidden_tests(sb, _T3_HIDDEN)
    return ok, "hidden suite" + ("" if ok else ": " + out[-300:])


def _ref_t3(sb):
    for p in py_files(sb):
        t = p.read_text(encoding="utf-8")
        if "calc_total" in t:
            p.write_text(t.replace("calc_total", "compute_order_total"), encoding="utf-8")
    return "Renamed everywhere."


task(id="T3_rename",
     prompt="Rename the function `calc_total` to `compute_order_total` everywhere in the project, including every "
            "call site, import and test. Then run the tests.",
     grade=_grade_t3, reference=_ref_t3)

# T4 -------------------------------------------------------------
_T4_HIDDEN = dict(BASE_TESTS)
_T4_HIDDEN["tests/test_slugify_hidden.py"] = '''import unittest

from shop.utils import slugify


class SlugifyHidden(unittest.TestCase):
    def test_cases(self):
        self.assertEqual(slugify("Hello, World!"), "hello-world")
        self.assertEqual(slugify("  --Already--Slugged--  "), "already-slugged")
        self.assertEqual(slugify("a  b\\tc"), "a-b-c")
        self.assertEqual(slugify("C++ & Python 3"), "c-python-3")
        self.assertEqual(slugify(""), "")
'''


def _grade_t4(sb, final, trace):
    tu = (sb / "tests/test_utils.py").read_text(encoding="utf-8", errors="replace")
    if "slugify" not in tu:
        return False, "no slugify test added to tests/test_utils.py"
    ok, out = run_hidden_tests(sb, _T4_HIDDEN)
    if not ok:
        return False, "hidden suite: " + out[-300:]
    ok2, out2 = run_hidden_tests(sb, {"tests/test_utils.py": tu})
    return ok2, "model's own test_utils.py" + ("" if ok2 else " fails: " + out2[-200:])


def _ref_t4(sb):
    u = sb / "shop/utils.py"
    u.write_text("import re\n\n" + u.read_text() + '''

def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
''')
    tu = sb / "tests/test_utils.py"
    tu.write_text(tu.read_text().replace("from shop.utils import clamp, format_cents",
                                          "from shop.utils import clamp, format_cents, slugify").replace(
        "    def test_clamp(self):", "    def test_slugify(self):\n        self.assertEqual(slugify('A b'), 'a-b')\n\n    def test_clamp(self):"))
    return "Added slugify and a test."


task(id="T4_feature",
     prompt="Add a function `slugify(text: str) -> str` to shop/utils.py. It should lowercase the text, replace every "
            "run of characters that are not a-z or 0-9 with a single hyphen, and strip leading and trailing hyphens. "
            "Add unit tests for it in tests/test_utils.py, then run the tests.",
     grade=_grade_t4, reference=_ref_t4)

# T5 -------------------------------------------------------------
_T5_HIDDEN = dict(BASE_TESTS)
_T5_HIDDEN["tests/test_parse_hidden.py"] = '''import unittest

from shop.pricing import parse_price


class ParseHidden(unittest.TestCase):
    def test_negative_rejected(self):
        with self.assertRaises(ValueError):
            parse_price("-5.00")

    def test_positive_ok(self):
        self.assertEqual(parse_price("12.34"), 1234)
'''

VALID_STATUS, VALID_PRIORITY = {"pending", "in_progress", "completed"}, {"high", "medium", "low"}


def todo_valid(todos) -> bool:
    if not isinstance(todos, list) or not todos:
        return False
    for t in todos:
        if not isinstance(t, dict) or not isinstance(t.get("content"), str) or not t["content"].strip():
            return False
        if t.get("status") not in VALID_STATUS or t.get("priority") not in VALID_PRIORITY:
            return False
    return True


def _grade_t5(sb, final, trace):
    plans = [c["args"].get("todos") for c in trace if c["name"] == "todowrite" and c["ok"]]
    if not any(todo_valid(p) and len(p) >= 3 for p in plans):
        return False, "no valid 3-item todowrite call"
    ok, out = run_hidden_tests(sb, _T5_HIDDEN)
    return ok, "hidden suite" + ("" if ok else ": " + out[-300:])


def _ref_t5(sb):
    p = sb / "shop/pricing.py"
    p.write_text(p.read_text().replace('    value = float(text.strip())\n',
                                       '    value = float(text.strip())\n    if value < 0:\n        raise ValueError("negative price")\n'))
    return "Done."


task(id="T5_todo_then_edit",
     prompt="Before changing anything, record a plan with the todo tool as exactly three items: (1) make parse_price "
            "reject negative prices, (2) add a test for that, (3) run the tests. Then do item 1 only: make "
            "`parse_price` in shop/pricing.py raise ValueError when the price is negative.",
     grade=_grade_t5, reference=_ref_t5,
     ref_trace=[{"name": "todowrite", "ok": True, "args": {"todos": [
         {"content": "a", "status": "in_progress", "priority": "high"},
         {"content": "b", "status": "pending", "priority": "medium"},
         {"content": "c", "status": "pending", "priority": "low"}]}}])

# T6 -------------------------------------------------------------


def _grade_t6(sb, final, trace):
    t = (sb / "shop/config.py").read_text(encoding="utf-8", errors="replace")
    ok = re.search(r"^DEFAULT_MAX_RETRIES\s*=\s*7\b", t, re.M) is not None
    return ok, "DEFAULT_MAX_RETRIES must be 7"


def _ref_t6(sb):
    p = sb / "shop/config.py"
    p.write_text(p.read_text().replace("DEFAULT_MAX_RETRIES = 3", "DEFAULT_MAX_RETRIES = 7"))
    return "Updated to 7."


task(id="T6_log_to_config",
     prompt="Someone changed our service's max_retries setting recently. Find the new value in logs/app.log and "
            "update DEFAULT_MAX_RETRIES in shop/config.py to match it.",
     grade=_grade_t6, reference=_ref_t6)

# T7 -------------------------------------------------------------
task(id="T7_no_tool",
     prompt="Quick question, no need to look at the code or use any tools: in one or two sentences, what is the "
            "difference between a list and a tuple in Python?",
     grade=lambda sb, final, trace: (not trace and bool(final.strip()) and "mutable" in final.lower(),
                                      "must answer without tools and mention mutability"),
     reference=lambda sb: "Lists are mutable; tuples are immutable.")

# T8 -------------------------------------------------------------
task(id="T8_recover_wrong_path",
     prompt="Read src/settings.yaml and tell me which port the service is configured to listen on.",
     grade=lambda sb, final, trace: ("8431" in final, "needs 8431"),
     reference=lambda sb: "The file is at config/settings.yaml; the port is 8431.")

# T9 -------------------------------------------------------------
task(id="T9_missing_test",
     prompt="Which module in the handlers/ folder has no corresponding test file in tests/? Answer with the module name.",
     grade=lambda sb, final, trace: ("refund" in final.lower(), "needs refunds"),
     reference=lambda sb: "handlers/refunds.py has no test file.")

# Long-context tasks (only for models with >= 96K context) --------
LONG_EXTRA = {"shop/legacy_billing.py": LEGACY, "tests/test_legacy_billing.py": LEGACY_TEST}

task(id="L1_recall_in_long_session", long=True, extra_files=LONG_EXTRA,
     prompt="What is the finance sign-off approval code mentioned in a comment in shop/legacy_billing.py?",
     grade=lambda sb, final, trace: ("FIN-4471-QX" in final, "needs FIN-4471-QX"),
     reference=lambda sb: "FIN-4471-QX")

_L2_HIDDEN = dict(BASE_TESTS)
_L2_HIDDEN["tests/test_legacy_billing.py"] = LEGACY_TEST
_L2_HIDDEN["tests/test_late_fee_hidden.py"] = '''import unittest

import shop.legacy_billing as lb


class LateFeeHidden(unittest.TestCase):
    def test_uses_constant(self):
        old = lb.LATE_FEE_GRACE_DAYS
        try:
            lb.LATE_FEE_GRACE_DAYS = 3
            self.assertEqual(lb.late_fee(3), 0)
            self.assertEqual(lb.late_fee(5), 300)
        finally:
            lb.LATE_FEE_GRACE_DAYS = old
        self.assertEqual(lb.late_fee(12), 300)
'''


def _grade_l2(sb, final, trace):
    ok, out = run_hidden_tests(sb, _L2_HIDDEN)
    return ok, "hidden suite" + ("" if ok else ": " + out[-300:])


def _ref_l2(sb):
    p = sb / "shop/legacy_billing.py"
    p.write_text(p.read_text().replace(GRACE_BLOCK, "    grace = LATE_FEE_GRACE_DAYS\n"))
    return "Done."


task(id="L2_edit_in_long_session", long=True, extra_files=LONG_EXTRA,
     prompt="In that file, `late_fee` uses a hard-coded grace period. Change it to use the module-level constant "
            "LATE_FEE_GRACE_DAYS (read at call time), then run the tests.",
     grade=_grade_l2, reference=_ref_l2)

LONG_PRELUDE_USER = "I'm going to ask you about the legacy billing module. Read shop/legacy_billing.py in full first."
