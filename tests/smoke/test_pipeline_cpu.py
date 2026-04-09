"""End-to-end smoke test — runs the full pipeline with mock evaluation.

No GPU required. Tests that the entire optimization loop, logging,
and report generation work together correctly.
"""
from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import pytest

from sloguard.config_space import build_serving_space
from sloguard.optimizer.random_search import RandomSearchOptimizer
from sloguard.optimizer.tba_tpe_hybrid import TBATPEHybrid
from sloguard.report_generator import ReportGenerator
from sloguard.slo_contract import SLOContract
from sloguard.trial_logger import TrialLogger
from sloguard.types import EvalResult, ServingTrialResult


def _mock_eval(config: dict, trial_id: int, rng: random.Random) -> EvalResult:
    """Mock serving evaluation."""
    if rng.random() < 0.15:
        return EvalResult(crashed=True, crash_type="oom", error_msg="Mock OOM")

    gpu_util = config.get("gpu_memory_utilization", 0.8)
    max_seqs = config.get("max_num_seqs", 32)

    # Higher gpu_util + more seqs = more throughput but higher latency
    goodput = (gpu_util * 200 + max_seqs * 3) * rng.uniform(0.8, 1.2)
    ttft = 100 + max_seqs * 5 * rng.uniform(0.5, 1.5)
    itl = 20 + max_seqs * 1.5 * rng.uniform(0.5, 1.5)
    memory = gpu_util * 16000 + max_seqs * 50

    feasible = ttft <= 500 and itl <= 100

    return EvalResult(
        objective_value=goodput if feasible else goodput * 0.2,
        constraints={
            "ttft_p99_ms": ttft,
            "itl_p99_ms": itl,
            "gpu_memory_mb": memory,
        },
        feasible=feasible,
        ttft_p99_ms=ttft,
        itl_p99_ms=itl,
        goodput_tokens_per_sec=goodput if feasible else 0,
        goodput_ratio=0.85 if feasible else 0.2,
        gpu_memory_peak_mb=memory,
        tokens_per_sec=goodput / 0.85 if feasible else goodput,
        eval_time_s=1.0,
    )


def test_full_pipeline_smoke():
    """Run optimizer -> log -> report with mock evaluation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        space = build_serving_space()
        slo = SLOContract(ttft_p99_ms=500, itl_p99_ms=100)
        constraints = slo.to_constraints_dict()

        # Run two optimizers
        for opt_name, OptClass in [
            ("random", RandomSearchOptimizer),
            ("tba-tpe", TBATPEHybrid),
        ]:
            opt = OptClass(space, constraints, budget=15, seed=42)
            logger = TrialLogger(f"{tmpdir}/{opt_name}.jsonl")
            rng = random.Random(42)

            for trial_id in range(15):
                config = opt.ask()
                result = _mock_eval(config, trial_id, rng)
                opt.tell(config, result)

                trial_result = ServingTrialResult(
                    trial_id=trial_id,
                    timestamp="2024-01-01T00:00:00Z",
                    experiment_id=f"smoke-{opt_name}",
                    config=config,
                    model_id="mock/model",
                    gpu_id="mock_gpu",
                    workload_type="interactive",
                    request_rate=4.0,
                    num_requests=100,
                    ttft_p99=result.ttft_p99_ms,
                    itl_p99=result.itl_p99_ms,
                    tokens_per_sec=result.tokens_per_sec,
                    goodput_tokens_per_sec=result.goodput_tokens_per_sec,
                    goodput_ratio=result.goodput_ratio,
                    gpu_memory_peak_mb=result.gpu_memory_peak_mb,
                    feasible=result.feasible,
                    crashed=result.crashed,
                    crash_type=result.crash_type,
                    eval_time_s=result.eval_time_s,
                    slo_ttft_p99_ms=500,
                    slo_itl_p99_ms=100,
                    optimizer_name=type(opt).__name__,
                    seed=42,
                )
                logger.log(trial_result)

            assert opt.trial_count == 15
            assert logger.count == 15

        # Generate reports
        report_gen = ReportGenerator(tmpdir)
        report_gen.load_all()
        assert len(report_gen.data) >= 1

        figures_dir = f"{tmpdir}/figures"
        report_gen.generate_all(figures_dir)

        # Check output files exist
        assert Path(f"{figures_dir}/goodput_convergence.png").exists()
        assert Path(f"{figures_dir}/latency_dist.png").exists()
        assert Path(f"{figures_dir}/crash_scatter.png").exists()
        assert Path(f"{figures_dir}/crash_waste.png").exists()


def test_optimizer_determinism():
    """Same seed produces same sequence of configs."""
    space = build_serving_space()
    constraints = {"ttft_p99_ms": 500, "itl_p99_ms": 100}

    configs_a = []
    opt_a = RandomSearchOptimizer(space, constraints, budget=10, seed=42)
    for _ in range(10):
        configs_a.append(opt_a.ask())

    configs_b = []
    opt_b = RandomSearchOptimizer(space, constraints, budget=10, seed=42)
    for _ in range(10):
        configs_b.append(opt_b.ask())

    assert configs_a == configs_b


def test_slo_contract_integration():
    """SLO contract integrates correctly with optimizer constraints."""
    slo = SLOContract(ttft_p99_ms=500, itl_p99_ms=100, gpu_memory_mb=16000)
    constraints = slo.to_constraints_dict()

    space = build_serving_space()
    opt = TBATPEHybrid(space, constraints, budget=10, seed=42)

    rng = random.Random(42)
    for _ in range(10):
        config = opt.ask()
        result = _mock_eval(config, 0, rng)
        opt.tell(config, result)

    assert opt.trial_count == 10
