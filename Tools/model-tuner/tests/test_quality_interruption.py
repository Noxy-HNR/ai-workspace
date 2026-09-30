from modeltuner.backend import QualityResult
from modeltuner.strategy import interruption


def _r(rates):
    return QualityResult("reasoning", 8, 8, 0, 0.0, [{"tokens": 1000, "seconds": 1000 / x} for x in rates])


def test_steady_suite_is_not_flagged():
    assert interruption(_r([62, 64, 61, 63, 60]), "ac", "ac") is None


def test_one_stalled_task_is_flagged():
    # the real case: 7 tasks near 62 tok/s, one frozen by Modern Standby at 1.9 tok/s
    why = interruption(_r([62, 63, 1.9, 60, 64, 63, 62, 64]), "ac", "ac")
    assert why and "stalled" in why


def test_power_change_is_flagged():
    assert "power state changed" in interruption(_r([62, 63, 61]), "ac|best_performance", "battery|best_efficiency")


def test_too_few_tasks_to_judge():
    assert interruption(_r([60, 2]), "ac", "ac") is None
