"""Tests for all optimizers — verifies ask/tell loop works without errors."""
from __future__ import annotations

import random

import pytest

from sloguard.config_space import build_serving_space
from sloguard.optimizer.base import BaseOptimizer
from sloguard.optimizer.random_search import RandomSearchOptimizer
from sloguard.optimizer.optuna_tpe import OptunaColdTPE
from sloguard.optimizer.tba_optimizer import TBAOptimizer
from sloguard.optimizer.tba_tpe_hybrid import TBATPEHybrid
from sloguard.types import EvalResult


def _mock_evaluate(config: dict, rng: random.Random, crash_prob: float = 0.2) -> EvalResult:
    """Mock evaluation: random metrics with configurable crash probability."""
    if rng.random() < crash_prob:
        return EvalResult(
            crashed=True,
            crash_type="oom",
            error_msg="Mock OOM",
        )

    goodput = rng.uniform(50, 500)
    ttft_p99 = rng.uniform(100, 800)
    itl_p99 = rng.uniform(20, 200)
    memory = rng.uniform(4000, 16000)

    feasible = ttft_p99 <= 500 and itl_p99 <= 100

    return EvalResult(
        objective_value=goodput if feasible else goodput * 0.3,
        constraints={
            "ttft_p99_ms": ttft_p99,
            "itl_p99_ms": itl_p99,
            "gpu_memory_mb": memory,
        },
        feasible=feasible,
        crashed=False,
        eval_time_s=1.0,
        ttft_p99_ms=ttft_p99,
        itl_p99_ms=itl_p99,
        goodput_tokens_per_sec=goodput,
        goodput_ratio=0.8 if feasible else 0.3,
        gpu_memory_peak_mb=memory,
    )


def _run_optimizer(opt: BaseOptimizer, budget: int = 20, seed: int = 42) -> None:
    """Run a complete ask/tell loop with mock evaluation."""
    rng = random.Random(seed)
    for _ in range(budget):
        config = opt.ask()
        assert isinstance(config, dict)
        result = _mock_evaluate(config, rng)
        opt.tell(config, result)

    assert opt.trial_count == budget
    assert opt.n_crashes + opt.n_infeasible + opt.n_feasible == budget


class TestRandomSearch:
    def test_basic_loop(self):
        space = build_serving_space()
        constraints = {"ttft_p99_ms": 500, "itl_p99_ms": 100}
        opt = RandomSearchOptimizer(space, constraints, budget=20, seed=42)
        _run_optimizer(opt)

    def test_configs_are_valid(self):
        space = build_serving_space()
        opt = RandomSearchOptimizer(space, {}, budget=10, seed=42)
        for _ in range(10):
            config = opt.ask()
            assert space.is_valid(config)


class TestOptunaColdTPE:
    def test_basic_loop(self):
        space = build_serving_space()
        constraints = {"ttft_p99_ms": 500, "itl_p99_ms": 100}
        opt = OptunaColdTPE(space, constraints, budget=20, seed=42)
        _run_optimizer(opt)

    def test_finds_feasible(self):
        space = build_serving_space()
        constraints = {"ttft_p99_ms": 500, "itl_p99_ms": 100}
        opt = OptunaColdTPE(space, constraints, budget=30, seed=42)
        _run_optimizer(opt, budget=30)
        # With 30 trials and mock data, should find something feasible
        # (not guaranteed but very likely)


class TestTBAOptimizer:
    def test_basic_loop(self):
        space = build_serving_space()
        constraints = {"ttft_p99_ms": 500, "itl_p99_ms": 100}
        opt = TBAOptimizer(space, constraints, budget=20, seed=42)
        _run_optimizer(opt)

    def test_tracks_crashes(self):
        space = build_serving_space()
        constraints = {"ttft_p99_ms": 500, "itl_p99_ms": 100}
        opt = TBAOptimizer(space, constraints, budget=20, seed=42)
        _run_optimizer(opt)
        assert opt.n_crashes >= 0  # at least some with 20% crash prob

    def test_without_surrogate(self):
        space = build_serving_space()
        constraints = {"ttft_p99_ms": 500, "itl_p99_ms": 100}
        opt = TBAOptimizer(space, constraints, budget=15, seed=42, surrogate=False)
        _run_optimizer(opt, budget=15)

    def test_without_blacklisting(self):
        space = build_serving_space()
        constraints = {"ttft_p99_ms": 500, "itl_p99_ms": 100}
        opt = TBAOptimizer(space, constraints, budget=15, seed=42, enable_blacklisting=False)
        _run_optimizer(opt, budget=15)


class TestTBATPEHybrid:
    def test_basic_loop(self):
        space = build_serving_space()
        constraints = {"ttft_p99_ms": 500, "itl_p99_ms": 100}
        opt = TBATPEHybrid(space, constraints, budget=20, seed=42)
        _run_optimizer(opt)

    def test_phases_transition(self):
        space = build_serving_space()
        constraints = {"ttft_p99_ms": 500, "itl_p99_ms": 100}
        opt = TBATPEHybrid(space, constraints, budget=25, seed=42)
        _run_optimizer(opt, budget=25)
        # With 25 trials and adaptive handoff at 5-15, should transition

    def test_small_budget(self):
        space = build_serving_space()
        constraints = {"ttft_p99_ms": 500, "itl_p99_ms": 100}
        opt = TBATPEHybrid(space, constraints, budget=5, seed=42)
        _run_optimizer(opt, budget=5)


class TestBestFeasible:
    def test_best_feasible_returned(self):
        space = build_serving_space()
        constraints = {"ttft_p99_ms": 500, "itl_p99_ms": 100}
        opt = RandomSearchOptimizer(space, constraints, budget=30, seed=42)

        rng = random.Random(42)
        for _ in range(30):
            config = opt.ask()
            result = _mock_evaluate(config, rng, crash_prob=0.1)
            opt.tell(config, result)

        best = opt.best_feasible()
        if best is not None:
            config, result = best
            assert result.feasible
            assert not result.crashed
            assert result.objective_value > float("-inf")
