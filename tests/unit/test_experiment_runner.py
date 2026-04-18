"""Tests for experiment_runner helpers — pure functions only.

The full ExperimentRunner needs a vLLM server and GPU so we don't
exercise it here. summarize_results is a pure function that handles all
the budget-accounting math and benefits from focused unit tests.
"""
from __future__ import annotations

from sloguard.experiment_runner import summarize_results
from sloguard.types import EvalResult


def _ok(goodput: float = 100.0, eval_s: float = 1.0) -> EvalResult:
    return EvalResult(
        objective_value=goodput, feasible=True, crashed=False, eval_time_s=eval_s,
    )


def _infeasible(eval_s: float = 1.0) -> EvalResult:
    return EvalResult(feasible=False, crashed=False, eval_time_s=eval_s)


def _crashed(eval_s: float = 1.0) -> EvalResult:
    return EvalResult(feasible=False, crashed=True, eval_time_s=eval_s)


def test_summarize_all_feasible():
    s = summarize_results([_ok(), _ok(), _ok()], budget=3)
    assert s["feasible"] == 3
    assert s["crashed"] == 0
    assert s["infeasible"] == 0
    assert s["wasted_s"] == 0
    assert s["crashed_pct"] == 0.0


def test_summarize_mixed():
    results = [_ok(eval_s=2.0), _crashed(eval_s=5.0), _infeasible(eval_s=3.0)]
    s = summarize_results(results, budget=3)
    assert s["feasible"] == 1
    assert s["crashed"] == 1
    assert s["infeasible"] == 1
    assert s["wasted_s"] == 8.0  # 5 (crashed) + 3 (infeasible)
    assert s["crashed_pct"] == round(100 / 3, 2) or s["crashed_pct"] > 33


def test_summarize_handles_partial_budget():
    """Budget allocates 5 trials but only 2 ran — the missing 3 count
    as infeasible (waste from the optimizer's POV)."""
    results = [_ok(), _crashed()]
    s = summarize_results(results, budget=5)
    assert s["feasible"] == 1
    assert s["crashed"] == 1
    assert s["infeasible"] == 3


def test_summarize_zero_budget_no_div_by_zero():
    s = summarize_results([], budget=0)
    assert s["crashed_pct"] == 0.0
    assert s["infeasible_pct"] == 0.0
    assert s["wasted_s"] == 0


def test_crashed_trials_count_as_wasted_even_if_feasible_flag_unset():
    # A crashed result should never have feasible=True, but if it did
    # we still want to count it as crashed (not feasible).
    weird = EvalResult(feasible=True, crashed=True, eval_time_s=4.0)
    s = summarize_results([weird], budget=1)
    assert s["crashed"] == 1
    assert s["feasible"] == 0  # crashed beats feasible
    assert s["wasted_s"] == 4.0
