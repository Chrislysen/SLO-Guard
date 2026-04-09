#!/usr/bin/env python3
"""Run a single experiment with full configuration.

Usage:
    python scripts/run_experiment.py \
        --model Qwen/Qwen2-1.5B \
        --optimizer tba-tpe \
        --workload interactive \
        --budget 30 \
        --seeds 0,1,2,3,4 \
        --output results/interactive/qwen2-1.5b/tba-tpe/
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sloguard.config_space import build_serving_space
from sloguard.experiment_runner import ExperimentRunner, create_optimizer
from sloguard.load_generator import WorkloadConfig
from sloguard.slo_contract import SLOContract


WORKLOAD_PRESETS = {
    "interactive": {
        "request_rate": 4.0,
        "num_requests": 100,
        "prompt_len_min": 128,
        "prompt_len_max": 512,
        "output_len_min": 64,
        "output_len_max": 256,
        "slo_ttft_p99_ms": 500.0,
        "slo_itl_p99_ms": 100.0,
        "slo_latency_p99_ms": 30000.0,
        "mode": "fixed",
    },
    "batch": {
        "request_rate": 16.0,
        "num_requests": 100,
        "prompt_len_min": 512,
        "prompt_len_max": 2048,
        "output_len_min": 256,
        "output_len_max": 1024,
        "slo_ttft_p99_ms": 0.0,  # no TTFT constraint
        "slo_itl_p99_ms": 0.0,   # no ITL constraint
        "slo_latency_p99_ms": 30000.0,
        "mode": "fixed",
    },
    "bursty": {
        "request_rate": 4.0,
        "num_requests": 100,
        "prompt_len_min": 128,
        "prompt_len_max": 512,
        "output_len_min": 64,
        "output_len_max": 256,
        "slo_ttft_p99_ms": 500.0,
        "slo_itl_p99_ms": 100.0,
        "slo_latency_p99_ms": 30000.0,
        "mode": "burst",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Run SLO-Guard experiment")
    parser.add_argument("--model", required=True, help="Model ID")
    parser.add_argument("--optimizer", default="tba-tpe",
                        choices=["random", "tpe", "tba", "tba-tpe", "constrained-bo"])
    parser.add_argument("--workload", default="interactive",
                        choices=["interactive", "batch", "bursty"])
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated seeds")
    parser.add_argument("--output", default="results")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-id", default="auto")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    preset = WORKLOAD_PRESETS[args.workload]
    space = build_serving_space()

    slo = SLOContract(
        ttft_p99_ms=preset["slo_ttft_p99_ms"],
        itl_p99_ms=preset["slo_itl_p99_ms"],
        request_latency_p99_ms=preset["slo_latency_p99_ms"],
    )
    constraints = slo.to_constraints_dict()

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed} | {args.optimizer} | {args.workload} | {args.model}")
        print(f"{'='*60}\n")

        opt = create_optimizer(args.optimizer, space, constraints, args.budget, seed)

        wl_config = WorkloadConfig(
            request_rate=preset["request_rate"],
            num_requests=preset["num_requests"],
            prompt_len_min=preset["prompt_len_min"],
            prompt_len_max=preset["prompt_len_max"],
            output_len_min=preset["output_len_min"],
            output_len_max=preset["output_len_max"],
            model=args.model,
        )

        workload_kwargs = {}
        if preset["mode"] == "burst":
            workload_kwargs = {
                "baseline_rate": preset["request_rate"],
                "peak_rate": preset["request_rate"] * 5,
            }

        output_dir = f"{args.output}/seed_{seed}"
        runner = ExperimentRunner(
            model=args.model,
            optimizer=opt,
            slo=slo,
            workload=wl_config,
            output_dir=output_dir,
            gpu_id=args.gpu_id,
            port=args.port,
            workload_mode=preset["mode"],
            workload_kwargs=workload_kwargs,
        )

        best = runner.run(args.budget)
        if best:
            config, result = best
            print(f"\nBest goodput: {result.goodput_tokens_per_sec or 0:.1f} tok/s")
            print(f"Crash waste: {opt.n_crashes}/{args.budget}")
        else:
            print("\nNo feasible config found.")


if __name__ == "__main__":
    main()
