"""Tests for experiment_runner helpers — pure functions only.

The full ExperimentRunner needs a vLLM server and GPU so we don't
exercise it here. summarize_results and compute_utility are pure
functions that benefit from focused unit tests.
"""
from __future__ import annotations

from sloguard.experiment_runner import (
    DEFAULT_CRASH_PENALTY,
    DEFAULT_TIME_PENALTY,
    compute_utility,
    summarize_results,
)
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


# -----------------------------------------------------------------------------
# compute_utility
# -----------------------------------------------------------------------------


def test_utility_feasible_deducts_time_cost():
    r = EvalResult(
        feasible=True, crashed=False,
        goodput_tokens_per_sec=200.0,
        server_startup_time_s=10.0, eval_time_s=20.0,
    )
    # U = 200 - 1.0 * 30 = 170
    assert compute_utility(r) == 170.0


def test_utility_crashed_is_finite_and_negative():
    r = EvalResult(
        crashed=True, eval_time_s=5.0, server_startup_time_s=0.0,
    )
    # U = -1000 - 1.0 * 5 = -1005
    u = compute_utility(r)
    assert u == -1005.0


def test_utility_fast_crash_ranks_above_slow_crash():
    """Key incentive: the optimizer should prefer failing fast over
    failing slow, so fast-crash utility > slow-crash utility."""
    fast = EvalResult(crashed=True, server_startup_time_s=1.0, eval_time_s=2.0)
    slow = EvalResult(crashed=True, server_startup_time_s=60.0, eval_time_s=120.0)
    assert compute_utility(fast) > compute_utility(slow)


def test_utility_feasible_beats_any_crash():
    """At default penalties, even a low-goodput feasible result should
    outrank a crash."""
    feasible_low = EvalResult(
        feasible=True, goodput_tokens_per_sec=10.0,
        server_startup_time_s=5.0, eval_time_s=5.0,
    )
    crashed = EvalResult(
        crashed=True, server_startup_time_s=1.0, eval_time_s=1.0,
    )
    assert compute_utility(feasible_low) > compute_utility(crashed)


def test_utility_custom_penalties():
    r = EvalResult(
        feasible=True, goodput_tokens_per_sec=100.0,
        server_startup_time_s=10.0, eval_time_s=10.0,
    )
    # time_penalty=2 -> cost = 2*20 = 40; U = 100 - 40 = 60
    assert compute_utility(r, time_penalty=2.0) == 60.0

    crashed = EvalResult(
        crashed=True, server_startup_time_s=5.0, eval_time_s=5.0,
    )
    # crash_penalty=500, time_penalty=0 -> U = -500
    assert compute_utility(crashed, crash_penalty=500.0, time_penalty=0.0) == -500.0


def test_utility_defaults_match_exported_constants():
    """Sanity: the defaults published in the module are what the function uses."""
    r = EvalResult(feasible=True, goodput_tokens_per_sec=50.0, eval_time_s=10.0)
    explicit = compute_utility(
        r, crash_penalty=DEFAULT_CRASH_PENALTY, time_penalty=DEFAULT_TIME_PENALTY,
    )
    implicit = compute_utility(r)
    assert explicit == implicit


def test_utility_missing_goodput_treated_as_zero():
    """A trial that completed but never measured goodput (e.g. early
    infeasibility) should get U = 0 - time_cost, not crash."""
    r = EvalResult(
        feasible=False, crashed=False,
        goodput_tokens_per_sec=None,
        eval_time_s=15.0,
    )
    assert compute_utility(r) == -15.0
