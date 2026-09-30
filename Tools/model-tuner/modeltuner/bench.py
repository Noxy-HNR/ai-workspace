"""Speed workloads, measured from llama-server's own `timings` (not wall clock).

Three workloads, because speculative decoding helps them very differently:
- prose: free writing. MTP drafting helps; n-gram lookup barely does.
- code: return a file with a small edit. Output copies input, so n-gram lookup can draft long runs (2-3x).
- prefill: a long prompt with a tiny answer, for prompt-processing speed (batch/ubatch).

Every request uses temperature 0, a fixed seed, thinking disabled (so all tokens are answer tokens and lengths are
stable) and cache_prompt=false (a cached prefix would fake prompt speed). A discarded warm-up runs first: Windows
pages an idle model's VRAM out, and the first request after idle can take 10-20 s.
"""

from __future__ import annotations

import time
from typing import Callable

from modeltuner.backend import Sample
from modeltuner.llama import LlamaClient, LlamaError

_CODE_FILE = '''\
"""Inventory helpers for a small shop."""

from dataclasses import dataclass, field


@dataclass
class Item:
    sku: str
    name: str
    price_cents: int
    quantity: int = 0
    tags: list[str] = field(default_factory=list)

    def total_value(self) -> int:
        return self.price_cents * self.quantity


class Inventory:
    def __init__(self) -> None:
        self._items: dict[str, Item] = {}

    def add(self, item: Item) -> None:
        if item.sku in self._items:
            raise ValueError(f"duplicate sku {item.sku}")
        self._items[item.sku] = item

    def remove(self, sku: str) -> Item:
        try:
            return self._items.pop(sku)
        except KeyError:
            raise KeyError(f"unknown sku {sku}") from None

    def restock(self, sku: str, amount: int) -> int:
        if amount <= 0:
            raise ValueError("amount must be positive")
        item = self._items[sku]
        item.quantity += amount
        return item.quantity

    def sell(self, sku: str, amount: int) -> int:
        item = self._items[sku]
        if amount > item.quantity:
            raise ValueError(f"only {item.quantity} of {sku} left")
        item.quantity -= amount
        return item.price_cents * amount

    def by_tag(self, tag: str) -> list[Item]:
        return [i for i in self._items.values() if tag in i.tags]

    def low_stock(self, threshold: int = 3) -> list[Item]:
        return sorted((i for i in self._items.values() if i.quantity <= threshold), key=lambda i: i.quantity)

    def total_value(self) -> int:
        return sum(i.total_value() for i in self._items.values())

    def report(self) -> str:
        lines = [f"{i.sku:<10} {i.name:<24} {i.quantity:>5} {i.price_cents / 100:>9.2f}"
                 for i in sorted(self._items.values(), key=lambda i: i.sku)]
        lines.append(f"{'TOTAL':<40} {self.total_value() / 100:>9.2f}")
        return "\\n".join(lines)
'''

_PREFILL_UNIT = (
    "The lighthouse stood on a basalt shelf at the edge of the bay, and every evening the keeper climbed its 114 "
    "steps to trim the wick, wind the clockwork and polish the lens until the brass shone. Ships counted on the beam "
    "the way farmers count on the seasons. "
)

_TOPICS = [
    "how lighthouses changed coastal navigation",
    "the history of the printing press",
    "how honeybees communicate",
    "why Roman aqueducts worked so well",
    "the invention of refrigeration and what it changed",
    "how tides are caused",
    "the building of the transcontinental railroad",
    "how vaccines train the immune system",
]
# Identifier sets for the code workload: same structure, different tokens, so the n-gram speculator cannot replay a
# previous repetition's answer from its cross-request cache (that inflated early measurements 5x).
_CODE_NAMES = [
    ("Item", "Inventory", "sku", "low_stock", "items_below"),
    ("Product", "Warehouse", "code", "restock_needed", "under_threshold"),
    ("Article", "Catalog", "ref", "scarce", "running_low"),
    ("Part", "Depot", "part_no", "short_supply", "below_min"),
    ("Record", "Ledger", "key", "thin_stock", "near_empty"),
    ("Widget", "Store", "tag_id", "few_left", "almost_out"),
    ("Good", "Pantry", "label", "dwindling", "needs_refill"),
    ("Unit", "Stockroom", "serial", "sparse", "under_par"),
]
_PREFILL_UNITS = [
    _PREFILL_UNIT,
    "The orchard ran down to the river in terraced rows, and each autumn the pickers carried ladders from tree to "
    "tree while the cider press groaned in the barn. Buyers came from three towns to taste the first pressing. ",
    "The observatory dome turned on iron wheels, and on clear nights the astronomer logged every variable star by "
    "lamplight, pencil in hand, until the eastern sky grew pale. Students copied the ledgers each morning. ",
    "The ferry crossed the strait twice a day in any weather, its engine thumping under the deck while gulls "
    "followed the wake. The captain knew every rock by name and every current by season. ",
]


def _variant(workload: str, i: int) -> tuple[list[dict], int]:
    if workload == "prose":
        topic = _TOPICS[i % len(_TOPICS)]
        return [{"role": "user", "content": f"Write a detailed, 500-word essay about {topic}. "
                                            f"Use plain prose, no headings or lists."}], 400
    if workload == "code":
        item, inv, sku, old, new = _CODE_NAMES[i % len(_CODE_NAMES)]
        src = (_CODE_FILE.replace("Inventory", inv).replace("Item", item).replace("sku", sku)
               .replace("low_stock", old))
        return [{"role": "user", "content": f"Return the complete file below with one change: rename the method "
                                            f"`{old}` to `{new}` (keep everything else identical). Output only the "
                                            f"code.\n\n```python\n{src}```"}], 700
    if workload == "prefill":
        unit = _PREFILL_UNITS[i % len(_PREFILL_UNITS)]
        return [{"role": "user", "content": (unit * 90) + "\n\nIn one word: what is this passage about?"}], 8
    raise KeyError(workload)


WORKLOADS = ("prose", "code", "prefill")
# Unrelated to every workload, so warming up does not seed the n-gram cache with an answer we then measure.
WARMUP = {"messages": [{"role": "user", "content": "Name three primary colours, comma-separated."}], "max_tokens": 16}


def _sample(workload: str, r, power_key: str) -> Sample:
    t = r.timings or {}
    return Sample(
        workload=workload,
        gen_tps=float(t["predicted_per_second"]) if t.get("predicted_per_second") else None,
        prompt_tps=float(t["prompt_per_second"]) if t.get("prompt_per_second") else None,
        prompt_n=int(t.get("prompt_n") or 0),
        predicted_n=int(t.get("predicted_n") or 0),
        draft_n=int(t["draft_n"]) if t.get("draft_n") is not None else None,
        draft_accepted=int(t["draft_n_accepted"]) if t.get("draft_n_accepted") is not None else None,
        wall_s=r.wall_s,
        power_key=power_key,
    )


def warm_up(client: LlamaClient, timeout: float = 180) -> float:
    t0 = time.perf_counter()
    client.chat(WARMUP["messages"], max_tokens=WARMUP["max_tokens"], thinking=False, timeout=timeout,
                cache_prompt=False)
    return time.perf_counter() - t0


def measure(client: LlamaClient, workload: str, reps: int, power_key: Callable[[], str], *, start: int = 0,
            timeout: float = 300) -> list[Sample]:
    """`reps` measurements of one workload, repetition i using prompt variant `start + i`. Every config sees the same
    variant sequence, so rep i of config A is compared with rep i of config B on identical text."""
    out: list[Sample] = []
    for i in range(start, start + reps):
        messages, max_tokens = _variant(workload, i)
        r = client.chat(messages, max_tokens=max_tokens, temperature=0.0, seed=42, thinking=False,
                        timeout=timeout, cache_prompt=False)
        out.append(_sample(workload, r, power_key()))
    return out
