"""Proposed tests for modeltuner.bench: which text each repetition gets.

Run with: uv run --with pytest pytest -q proposed_tests
"""

from __future__ import annotations

from modeltuner import bench
from modeltuner.bench import _variant


def _texts(workload: str, first: int, reps: int) -> list[str]:
    return [_variant(workload, i)[0][0]["content"] for i in range(first, first + reps)]


def test_a_run_never_measures_the_same_prompt_twice_for_one_config():
    # strategy.challenge() asks for o.reps + 2 samples when the first comparison is inconclusive, so one
    # config is measured 5 times. llama-server's n-gram speculator keeps a cache across requests, so the
    # second time a prompt comes round it can replay the previous answer -- the 5x inflation recorded on
    # 2026-09-30. prefill has only 4 passages, so repetition 5 is repetition 1.
    for workload in bench.WORKLOADS:
        seen = _texts(workload, 0, 5)
        assert len(set(seen)) == len(seen), f"{workload}: one config was asked the same prompt twice"


def test_a_second_round_of_measurements_does_not_repeat_a_prompt():
    # RealBackend keys its variant counter on the config, so a config measured in two rounds continues the
    # sequence. Between rounds the sequence must not land back on a prompt that config already answered.
    for workload in bench.WORKLOADS:
        first = _texts(workload, 0, 3)
        second = _texts(workload, 3, 3)
        assert not set(first) & set(second), f"{workload}: the second round repeated a prompt from the first"


def test_every_workload_offers_a_distinct_prompt_per_repetition():
    # How many distinct texts a workload can offer at all. A run can ask for o.reps + 2 = 5 measurements of
    # one config, and prefill cycles 4 passages, so it has to hand out the same text twice.
    pools = {w: len({_variant(w, i)[0][0]["content"] for i in range(16)}) for w in bench.WORKLOADS}
    assert min(pools.values()) >= 5, f"a run can ask for 5 repetitions, but the pools are {pools}"


def test_the_warm_up_prompt_is_not_one_of_the_workload_prompts():
    warm = bench.WARMUP["messages"][0]["content"]
    for workload in bench.WORKLOADS:
        for i in range(8):
            assert _variant(workload, i)[0][0]["content"] != warm
