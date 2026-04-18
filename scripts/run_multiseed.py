#!/usr/bin/env python3
"""Resumable multi-seed benchmark runner — uses the real ExperimentRunner.

Runs the full matrix of (optimizer, seed) pairs, interleaved trial-by-trial
so a Colab disconnect at any point leaves roughly the same amount of
progress across every pair. Writes a compact JSONL per pair that
``scripts/compute_multiseed_stats.py`` can consume directly.

Why this exists
---------------
The original multi-seed run (commit 54bddd2) bypassed LoadGenerator and
hit vLLM via curl from an inline Colab cell. Once we fixed LoadGenerator
to be truly concurrent, that path no longer exercised the code under
test. This script drives trials through ExperimentRunner so LoadGenerator,
CrashClassifier, MetricsCollector, and the utility objective all run.

Resume semantics
----------------
For each (optimizer, seed) pair, on startup we read the corresponding
results.jsonl (if present), count completed trials, and replay each
completed (config, outcome) through a fresh optimizer via ``tell()`` so
internal state (phase, blacklist, TPE warm-start) is restored before we
ask for the next config. A pair that already has ``--budget`` trials is
skipped entirely.

Usage
-----
    python scripts/run_multiseed.py \\
        --output-dir results/multiseed_concurrent/ \\
        --seeds 42 142 242 342 442 \\
        --optimizers random tba-tpe \\
        --budget 15 \\
        --model Qwen/Qwen2-1.5B
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sloguard.config_space import build_serving_space
from sloguard.experiment_runner import (
    OBJECTIVE_GOODPUT,
    OBJECTIVE_UTILITY,
    ExperimentRunner,
    compute_utility,
    create_optimizer,
)
from sloguard.load_generator import WorkloadConfig
from sloguard.slo_contract import SLOContract
from sloguard.types import EvalResult, TimeoutConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_multiseed")


# -----------------------------------------------------------------------------
# JSONL schema — kept compatible with scripts/compute_multiseed_stats.py
# -----------------------------------------------------------------------------


def _build_record(
    trial_id: int,
    seed: int,
    optimizer_name: str,
    config: dict[str, Any],
    phase: str,
    result: EvalResult,
) -> dict[str, Any]:
    """Compact per-trial JSONL record.

    Fields match what compute_multiseed_stats.py reads; utility_value is
    added as a new optional field.
    """
    if result.crashed:
        status = "crash"
    elif not result.feasible:
        status = "infeasible"
    else:
        status = "feasible"

    rec: dict[str, Any] = {
        "trial": trial_id + 1,                 # 1-indexed to match old data
        "seed": seed,
        "optimizer": optimizer_name,
        "phase": phase,
        "status": status,
        "config": config,
    }

    # Prefer true mean; fall back to p50 if mean wasn't computed (e.g. no
    # feasible latencies). For compute_multiseed_stats' fast-cluster check
    # either is fine — it thresholds at 1000ms and the bimodal gap is 1500ms+.
    lat = result.request_latency_mean_ms
    if lat is None:
        lat = result.request_latency_p50_ms
    if lat is not None:
        rec["avg_latency_ms"] = lat

    if result.goodput_tokens_per_sec is not None:
        rec["goodput_tps"] = result.goodput_tokens_per_sec
    if result.total_output_tokens is not None:
        rec["total_tokens"] = result.total_output_tokens
    if result.utility_value is not None:
        rec["utility_value"] = result.utility_value

    # Observability extras — downstream tooling can ignore.
    if result.peak_concurrency is not None:
        rec["peak_concurrency"] = result.peak_concurrency
    if result.crash_type:
        rec["crash_type"] = result.crash_type
    if result.error_msg:
        rec["error_msg"] = result.error_msg[:500]
    if result.eval_time_s:
        rec["eval_time_s"] = result.eval_time_s
    if result.server_startup_time_s is not None:
        rec["server_startup_time_s"] = result.server_startup_time_s

    return rec


def _durable_append(path: Path, line: str) -> None:
    """Append *line* to *path* and fsync — trial survives a hard-kill."""
    with open(path, "a") as f:
        f.write(line)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def _load_existing(path: Path) -> list[dict[str, Any]]:
    """Read prior records from *path*. Returns [] if the file is missing."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            logger.warning("Skipping malformed record in %s: %s", path, e)
    return records


# -----------------------------------------------------------------------------
# Replay: reconstruct optimizer state from a prior JSONL
# -----------------------------------------------------------------------------


def _record_to_eval_result(rec: dict[str, Any]) -> EvalResult:
    """Reconstruct just enough EvalResult for optimizer.tell() to advance state.

    Optimizers consume: feasible, crashed, objective_value, and the
    constraints dict for some surrogates. We don't have the raw per-
    request metrics any more, so constraints come from whatever the
    record's latency/goodput fields let us infer.
    """
    status = rec.get("status")
    crashed = status == "crash"
    feasible = status == "feasible"
    goodput = rec.get("goodput_tps")
    lat = rec.get("avg_latency_ms")

    result = EvalResult(
        feasible=feasible,
        crashed=crashed,
        crash_type=rec.get("crash_type"),
        error_msg=rec.get("error_msg"),
        goodput_tokens_per_sec=goodput,
        request_latency_mean_ms=lat,
        eval_time_s=rec.get("eval_time_s") or 0.0,
        server_startup_time_s=rec.get("server_startup_time_s"),
        utility_value=rec.get("utility_value"),
    )
    # objective_value drives optimizer.best_feasible ranking. If the record
    # was logged in utility mode we wrote utility_value; prefer that.
    if result.utility_value is not None:
        result.objective_value = result.utility_value
    elif goodput is not None:
        result.objective_value = goodput
    elif crashed:
        result.objective_value = float("-inf")
    else:
        result.objective_value = 0.0

    if lat is not None:
        result.constraints["request_latency_p99_ms"] = lat
    return result


# -----------------------------------------------------------------------------
# Per-pair state
# -----------------------------------------------------------------------------


@dataclass
class PairState:
    optimizer_name: str
    seed: int
    jsonl_path: Path
    runner: ExperimentRunner
    completed: int  # number of trials already logged
    skipped: bool   # True if this pair was already at budget on startup


def _build_pair(
    optimizer_name: str,
    seed: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> PairState:
    space = build_serving_space()
    slo = SLOContract(
        ttft_p99_ms=args.slo_ttft_p99,
        itl_p99_ms=args.slo_itl_p99,
        request_latency_p99_ms=args.slo_latency_p99,
    )
    constraints = slo.to_constraints_dict()
    optimizer = create_optimizer(
        optimizer_name, space, constraints, budget=args.budget, seed=seed,
    )
    workload = WorkloadConfig(
        request_rate=args.request_rate,
        num_requests=args.num_requests,
        prompt_len_min=args.prompt_len_min,
        prompt_len_max=args.prompt_len_max,
        output_len_min=args.output_len_min,
        output_len_max=args.output_len_max,
        model=args.model,
        max_concurrency=args.max_concurrency,
    )
    timeouts = TimeoutConfig(
        per_request_s=args.timeout_per_request_s,
        per_trial_s=args.timeout_per_trial_s,
        server_start_s=args.timeout_server_start_s,
        preflight_s=args.timeout_preflight_s,
    )
    runner = ExperimentRunner(
        model=args.model,
        optimizer=optimizer,
        slo=slo,
        workload=workload,
        output_dir=str(output_dir / f"{optimizer_name}_seed{seed}"),
        experiment_id=f"{optimizer_name}_seed{seed}",
        port=args.port,
        timeouts=timeouts,
        objective=args.objective,
        crash_penalty=args.crash_penalty,
        time_penalty=args.time_penalty,
    )

    pair_dir = output_dir / f"{optimizer_name}_seed{seed}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = pair_dir / "results.jsonl"

    # Replay prior trials so the optimizer restarts with correct state.
    prior = _load_existing(jsonl_path)
    for rec in prior:
        replay_config = rec.get("config", {})
        replay_result = _record_to_eval_result(rec)
        optimizer.tell(replay_config, replay_result)

    skipped = len(prior) >= args.budget
    return PairState(
        optimizer_name=optimizer_name,
        seed=seed,
        jsonl_path=jsonl_path,
        runner=runner,
        completed=len(prior),
        skipped=skipped,
    )


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------


def _run_one_trial(state: PairState, trial_id: int, args: argparse.Namespace) -> None:
    """Ask / evaluate / tell / log one trial for one (opt, seed) pair."""
    runner = state.runner
    config = runner.next_config()
    phase = runner.optimizer.phase

    logger.info(
        "[%s seed=%d] trial %d/%d [%s]: max_num_seqs=%s enforce_eager=%s",
        state.optimizer_name, state.seed, trial_id + 1, args.budget, phase,
        config.get("max_num_seqs"), config.get("enforce_eager"),
    )

    t0 = time.monotonic()
    result = runner.evaluate(config, trial_id)
    # Mirror ExperimentRunner.run(): always compute utility and swap in
    # utility mode so optimizer.best_feasible orders correctly.
    result.utility_value = compute_utility(
        result, crash_penalty=args.crash_penalty, time_penalty=args.time_penalty,
    )
    if args.objective == OBJECTIVE_UTILITY:
        result.objective_value = result.utility_value

    runner.optimizer.tell(config, result)

    record = _build_record(
        trial_id, state.seed, state.optimizer_name, config, phase, result,
    )
    _durable_append(state.jsonl_path, json.dumps(record) + "\n")
    state.completed += 1

    status = "CRASH" if result.crashed else ("FEASIBLE" if result.feasible else "INFEASIBLE")
    elapsed = time.monotonic() - t0
    logger.info(
        "[%s seed=%d] -> %s | latency_mean=%.0fms | goodput=%.1f tok/s "
        "| utility=%.1f | eval=%.1fs",
        state.optimizer_name, state.seed, status,
        result.request_latency_mean_ms or -1,
        result.goodput_tokens_per_sec or 0.0,
        result.utility_value if result.utility_value is not None else float("nan"),
        elapsed,
    )


def _print_summary(pairs: list[PairState], budget: int) -> None:
    print()
    print("=" * 72)
    print(f"  Run complete — {len(pairs)} pairs × {budget} trials")
    print("=" * 72)
    print(f"  {'optimizer':<12} {'seed':>5} {'done':>6} {'feasible':>10} "
          f"{'crashes':>8} {'best latency':>14} {'best goodput':>14}")
    print("  " + "-" * 68)
    for s in pairs:
        prior = _load_existing(s.jsonl_path)
        feasible = [r for r in prior if r.get("status") == "feasible"]
        crashes = sum(1 for r in prior if r.get("status") == "crash")
        lats = [r["avg_latency_ms"] for r in feasible if "avg_latency_ms" in r]
        gps = [r["goodput_tps"] for r in feasible if "goodput_tps" in r]
        best_lat = min(lats) if lats else float("nan")
        best_gp = max(gps) if gps else float("nan")
        print(f"  {s.optimizer_name:<12} {s.seed:>5} {len(prior):>6} "
              f"{len(feasible):>10} {crashes:>8} "
              f"{best_lat:>12.0f}ms {best_gp:>12.1f}tps")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to hold per-pair subdirs (results.jsonl each)")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 142, 242, 342, 442])
    parser.add_argument("--optimizers", nargs="+",
                        default=["random", "tba-tpe"],
                        choices=["random", "tpe", "tba", "tba-tpe", "constrained-bo"])
    parser.add_argument("--budget", type=int, default=15,
                        help="Trials per (optimizer, seed) pair")
    parser.add_argument("--model", default="Qwen/Qwen2-1.5B")
    parser.add_argument("--port", type=int, default=8000)
    # Workload
    parser.add_argument("--num-requests", type=int, default=5)
    parser.add_argument("--request-rate", type=float, default=2.0)
    parser.add_argument("--prompt-len-min", type=int, default=128)
    parser.add_argument("--prompt-len-max", type=int, default=512)
    parser.add_argument("--output-len-min", type=int, default=64)
    parser.add_argument("--output-len-max", type=int, default=256)
    parser.add_argument("--max-concurrency", type=int, default=50)
    # SLO
    parser.add_argument("--slo-ttft-p99", type=float, default=2000.0)
    parser.add_argument("--slo-itl-p99", type=float, default=200.0)
    parser.add_argument("--slo-latency-p99", type=float, default=30000.0)
    # Timeouts
    parser.add_argument("--timeout-per-request-s", type=float, default=60.0)
    parser.add_argument("--timeout-per-trial-s", type=float, default=180.0)
    parser.add_argument("--timeout-server-start-s", type=float, default=120.0)
    parser.add_argument("--timeout-preflight-s", type=float, default=30.0)
    # Objective
    parser.add_argument("--objective", default=OBJECTIVE_GOODPUT,
                        choices=[OBJECTIVE_GOODPUT, OBJECTIVE_UTILITY])
    parser.add_argument("--crash-penalty", type=float, default=1000.0)
    parser.add_argument("--time-penalty", type=float, default=1.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build all pairs up front so prior runs are replayed into each optimizer
    # before we issue any new trials.
    pairs: list[PairState] = []
    for opt_name in args.optimizers:
        for seed in args.seeds:
            pairs.append(_build_pair(opt_name, seed, args, args.output_dir))

    active = [p for p in pairs if not p.skipped]
    skipped = [p for p in pairs if p.skipped]
    for p in skipped:
        logger.info(
            "[%s seed=%d] already at budget (%d/%d) — skipping",
            p.optimizer_name, p.seed, p.completed, args.budget,
        )

    print("=" * 72)
    print("  SLO-Guard multi-seed runner")
    print(f"  Output:      {args.output_dir}")
    print(f"  Model:       {args.model}")
    print(f"  Optimizers:  {args.optimizers}")
    print(f"  Seeds:       {args.seeds}")
    print(f"  Budget:      {args.budget} trials/pair "
          f"({len(pairs)} pairs, {len(pairs) * args.budget} total)")
    print(f"  Resume:      {sum(p.completed for p in pairs)} trials already logged, "
          f"{sum(max(0, args.budget - p.completed) for p in pairs)} to go")
    print(f"  Objective:   {args.objective}")
    print("=" * 72)

    # Interleave: for each trial index, advance every pair that hasn't
    # reached that index yet. A disconnect leaves all pairs at roughly
    # the same trial count.
    try:
        for trial_id in range(args.budget):
            for state in active:
                if state.completed > trial_id:
                    continue  # this pair already has this trial from a prior run
                _run_one_trial(state, trial_id, args)
    except KeyboardInterrupt:
        logger.warning("Interrupted — partial results are on disk at %s", args.output_dir)
        _print_summary(pairs, args.budget)
        return 130

    _print_summary(pairs, args.budget)
    return 0


if __name__ == "__main__":
    sys.exit(main())
