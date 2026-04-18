#!/usr/bin/env python3
"""Colab-friendly benchmark runner using curl for HTTP requests.

Replaces the async Python load generator (which hangs on Colab)
with subprocess + curl for reliable non-streaming benchmarks.
Supports all SLO-Guard optimizers including tba-tpe.

Usage:
    python scripts/colab_curl_benchmark.py \
        --model Qwen/Qwen2-1.5B \
        --optimizer tba-tpe \
        --budget 15 \
        --seed 42 \
        --num-requests 10 \
        --request-rate 2.0 \
        --slo-ttft-p99 2000 \
        --slo-itl-p99 200 \
        --slo-latency-p99 30000 \
        --output results/tba_tpe_run/ \
        --port 8000
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# SLO-Guard imports
from sloguard.config_space import build_serving_space, fix_serving_config
from sloguard.crash_classifier import CrashClassifier
from sloguard.experiment_runner import create_optimizer
from sloguard.gpu_profile import (
    DEFAULT_VRAM_GB,
    detect_gpu_vram_gb,
    kv_gb_per_token_for,
    log_gpu_info,
    model_footprint_gb_for,
)
from sloguard.server_manager import VLLMServerManager
from sloguard.slo_contract import SLOContract
from sloguard.trial_logger import TrialLogger
from sloguard.types import EvalResult, ServingTrialResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("curl_benchmark")

# Timing sentinel to separate curl response body from timing data
TIMING_SEP = "\n__CURL_TIMING__"

VOCABULARY = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    "and", "then", "runs", "through", "forest", "into", "river",
    "under", "bright", "blue", "sky", "with", "warm", "gentle", "breeze",
    "across", "open", "field", "past", "old", "stone", "wall",
]


def make_prompt(rng: random.Random, target_tokens: int) -> str:
    """Generate a prompt of approximately target_tokens length."""
    words_needed = int(target_tokens / 0.75)
    words = [rng.choice(VOCABULARY) for _ in range(words_needed)]
    return "Summarize the following text: " + " ".join(words)


def curl_request(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float = 60.0,
) -> dict:
    """Send a single non-streaming request via curl.

    Returns dict with keys: success, body, time_total_s, time_starttransfer_s,
    http_code, output_tokens, error.
    """
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0.0,
    })

    write_out = TIMING_SEP + json.dumps({
        "time_total": "%{time_total}",
        "time_starttransfer": "%{time_starttransfer}",
        "http_code": "%{http_code}",
    }).replace('"', "")
    # curl write-out uses unquoted %{} — we'll parse the raw numbers

    write_out = (
        TIMING_SEP
        + '{"time_total":%{time_total},'
        + '"time_starttransfer":%{time_starttransfer},'
        + '"http_code":%{http_code}}'
    )

    try:
        proc = subprocess.run(
            [
                "curl", "-s",
                "--connect-timeout", "5",
                "--max-time", str(int(timeout)),
                "-w", write_out,
                "-H", "Content-Type: application/json",
                "-d", payload,
                url,
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 5,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "curl hard timeout", "time_total_s": timeout}

    if proc.returncode != 0:
        return {
            "success": False,
            "error": f"curl exit {proc.returncode}: {proc.stderr[:200]}",
            "time_total_s": 0,
        }

    # Split body from timing
    raw = proc.stdout
    if TIMING_SEP.strip() in raw:
        parts = raw.rsplit(TIMING_SEP.strip(), 1)
        body_str = parts[0].strip()
        timing_str = parts[1].strip() if len(parts) > 1 else "{}"
    else:
        body_str = raw.strip()
        timing_str = "{}"

    # Parse timing
    try:
        timing = json.loads(timing_str)
    except json.JSONDecodeError:
        timing = {}

    time_total = float(timing.get("time_total", 0))
    time_starttransfer = float(timing.get("time_starttransfer", 0))
    http_code = int(timing.get("http_code", 0))

    # Parse response body
    try:
        body = json.loads(body_str)
    except json.JSONDecodeError:
        return {
            "success": False,
            "error": f"Invalid JSON response: {body_str[:200]}",
            "time_total_s": time_total,
        }

    if "error" in body:
        return {
            "success": False,
            "error": str(body["error"]),
            "time_total_s": time_total,
        }

    if http_code != 200:
        return {
            "success": False,
            "error": f"HTTP {http_code}: {body_str[:200]}",
            "time_total_s": time_total,
        }

    # Extract output tokens
    usage = body.get("usage", {})
    output_tokens = usage.get("completion_tokens", 0)

    return {
        "success": True,
        "body": body,
        "time_total_s": time_total,
        "time_starttransfer_s": time_starttransfer,
        "http_code": http_code,
        "output_tokens": output_tokens,
        "error": None,
    }


def benchmark_with_curl(
    base_url: str,
    model: str,
    num_requests: int,
    request_rate: float,
    prompt_len_min: int,
    prompt_len_max: int,
    output_len_min: int,
    output_len_max: int,
    seed: int,
) -> list[dict]:
    """Run a benchmark: send num_requests via curl at target rate.

    Returns list of per-request result dicts.
    """
    rng = random.Random(seed)
    url = f"{base_url}/v1/chat/completions"
    results = []

    for i in range(num_requests):
        prompt_len = rng.randint(prompt_len_min, prompt_len_max)
        output_len = rng.randint(output_len_min, output_len_max)
        prompt = make_prompt(rng, prompt_len)

        send_time = time.monotonic()
        r = curl_request(url, model, prompt, output_len)
        r["request_id"] = i
        r["prompt_tokens"] = prompt_len
        r["send_time"] = send_time
        results.append(r)

        logger.debug(
            "  req %d/%d: %s latency=%.0fms tokens=%d",
            i + 1, num_requests,
            "OK" if r["success"] else "FAIL",
            r["time_total_s"] * 1000,
            r.get("output_tokens", 0),
        )

        # Abort early on 3 consecutive failures
        if len(results) >= 3 and all(not x["success"] for x in results[-3:]):
            logger.warning("3 consecutive failures — aborting benchmark")
            break

        # Inter-arrival delay (Poisson)
        if i < num_requests - 1 and request_rate > 0:
            delay = rng.expovariate(request_rate)
            time.sleep(delay)

    return results


def compute_metrics(
    results: list[dict],
    slo: SLOContract,
) -> EvalResult:
    """Compute EvalResult from curl benchmark results."""
    eval_result = EvalResult()

    successful = [r for r in results if r["success"]]
    if not successful:
        return eval_result

    # Request latency distribution (ms)
    latencies = [r["time_total_s"] * 1000 for r in successful]
    eval_result.request_latency_p50_ms = float(np.percentile(latencies, 50))
    eval_result.request_latency_p95_ms = float(np.percentile(latencies, 95))
    eval_result.request_latency_p99_ms = float(np.percentile(latencies, 99))

    # TTFT approximation: time_starttransfer (non-streaming: ≈ generation time)
    ttft_values = [
        r["time_starttransfer_s"] * 1000
        for r in successful
        if r.get("time_starttransfer_s", 0) > 0
    ]
    if ttft_values:
        eval_result.ttft_p50_ms = float(np.percentile(ttft_values, 50))
        eval_result.ttft_p95_ms = float(np.percentile(ttft_values, 95))
        eval_result.ttft_p99_ms = float(np.percentile(ttft_values, 99))

    # Throughput
    total_tokens = sum(r.get("output_tokens", 0) for r in successful)
    earliest = min(r["send_time"] for r in successful)
    latest_end = max(r["send_time"] + r["time_total_s"] for r in successful)
    duration = latest_end - earliest
    if duration > 0:
        eval_result.tokens_per_sec = total_tokens / duration
        eval_result.requests_per_sec = len(successful) / duration

    # Goodput: requests meeting latency SLO. Curl benchmarks have no
    # per-token ITL or true TTFT (non-streaming), so pass None for those —
    # SLOContract.check_request ignores None against active bounds.
    ratio, tps = slo.compute_goodput(
        (
            (None, None, r["time_total_s"] * 1000, r.get("output_tokens", 0))
            for r in successful
        ),
        duration_s=duration,
    )
    eval_result.goodput_ratio = ratio
    if duration > 0:
        eval_result.goodput_tokens_per_sec = tps

    eval_result.objective_value = eval_result.goodput_tokens_per_sec or 0.0

    # Constraints for optimizer
    eval_result.constraints = {}
    if eval_result.request_latency_p99_ms is not None:
        eval_result.constraints["request_latency_p99_ms"] = eval_result.request_latency_p99_ms
    if eval_result.ttft_p99_ms is not None:
        eval_result.constraints["ttft_p99_ms"] = eval_result.ttft_p99_ms

    # SLO feasibility
    eval_result.feasible = slo.check_aggregate(
        ttft_p99=eval_result.ttft_p99_ms,
        request_latency_p99=eval_result.request_latency_p99_ms,
    )

    return eval_result


def preflight_check(base_url: str, model: str) -> bool:
    """Send a single tiny request to verify the engine works."""
    r = curl_request(
        f"{base_url}/v1/chat/completions",
        model=model,
        prompt="Hi",
        max_tokens=1,
        timeout=30.0,
    )
    if not r["success"]:
        logger.warning("Pre-flight failed: %s", r.get("error"))
    return r["success"]


def run_experiment(args: argparse.Namespace) -> None:
    """Main experiment loop."""
    experiment_id = str(uuid.uuid4())[:8]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{experiment_id}.jsonl"
    trial_logger = TrialLogger(log_path)

    # Build components
    slo = SLOContract(
        ttft_p99_ms=args.slo_ttft_p99,
        itl_p99_ms=args.slo_itl_p99,
        request_latency_p99_ms=args.slo_latency_p99,
    )
    space = build_serving_space()
    constraints = slo.to_constraints_dict()
    optimizer = create_optimizer(args.optimizer, space, constraints, args.budget, args.seed)
    server = VLLMServerManager(model=args.model, port=args.port)
    classifier = CrashClassifier()  # noqa: F841 — kept for parity with experiment_runner

    # GPU + model profile so the memory guard scales to the actual hardware
    log_gpu_info(args.model)
    detected_vram = detect_gpu_vram_gb()
    vram_gb = detected_vram if detected_vram is not None else DEFAULT_VRAM_GB
    kv_gb = kv_gb_per_token_for(args.model)
    footprint_gb = model_footprint_gb_for(args.model)

    print("=" * 60)
    print("  SLO-Guard Curl Benchmark")
    print(f"  Model:     {args.model}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Budget:    {args.budget} trials")
    print(f"  Requests:  {args.num_requests} per trial @ {args.request_rate} req/s")
    print(f"  SLOs:      latency_p99<={args.slo_latency_p99}ms")
    print(f"  Output:    {log_path}")
    print("=" * 60)
    print()

    all_results = []
    total_start = time.monotonic()

    for trial_id in range(args.budget):
        trial_start = time.monotonic()
        config = optimizer.ask()
        config = fix_serving_config(
            config,
            vram_gb=vram_gb,
            kv_gb_per_token=kv_gb,
            model_footprint_gb=footprint_gb,
        )

        phase = optimizer.phase
        logger.info(
            "Trial %d/%d [%s]: max_num_seqs=%s, gpu_mem=%.2f, enforce_eager=%s, max_model_len=%s",
            trial_id + 1, args.budget, phase,
            config.get("max_num_seqs"),
            config.get("gpu_memory_utilization", 0),
            config.get("enforce_eager"),
            config.get("max_model_len"),
        )

        eval_result = EvalResult()

        # Start server
        server_started = server.start(config)
        if not server_started:
            crash_type = classifier.classify(stderr=server.stderr_output)
            eval_result.crashed = True
            eval_result.crash_type = crash_type.value
            eval_result.error_msg = server.stderr_output[:500]
            eval_result.server_startup_time_s = server.startup_time
            eval_result.eval_time_s = time.monotonic() - trial_start
            optimizer.tell(config, eval_result)
            _log_trial(
                trial_logger, trial_id, experiment_id, config,
                eval_result, args, slo, optimizer, phase,
            )
            all_results.append((config, eval_result))
            _print_status(trial_id, args.budget, eval_result, optimizer)
            continue

        eval_result.server_startup_time_s = server.startup_time

        # Pre-flight
        if not preflight_check(server.base_url, args.model):
            crash_type = classifier.classify(stderr=server.stderr_output)
            eval_result.crashed = True
            eval_result.crash_type = crash_type.value if crash_type.value != "healthy" else "startup_failure"
            eval_result.error_msg = "Pre-flight failed: engine started but cannot serve"
            eval_result.eval_time_s = time.monotonic() - trial_start
            server.stop()
            optimizer.tell(config, eval_result)
            _log_trial(
                trial_logger, trial_id, experiment_id, config,
                eval_result, args, slo, optimizer, phase,
            )
            all_results.append((config, eval_result))
            _print_status(trial_id, args.budget, eval_result, optimizer)
            continue

        # Benchmark with curl
        request_results = benchmark_with_curl(
            base_url=server.base_url,
            model=args.model,
            num_requests=args.num_requests,
            request_rate=args.request_rate,
            prompt_len_min=128,
            prompt_len_max=512,
            output_len_min=64,
            output_len_max=256,
            seed=args.seed + trial_id,
        )

        # No results = server died
        if not request_results or all(not r["success"] for r in request_results):
            eval_result.crashed = True
            eval_result.crash_type = "timeout"
            eval_result.error_msg = "No successful requests — server unresponsive"
            eval_result.eval_time_s = time.monotonic() - trial_start
            server.stop()
            optimizer.tell(config, eval_result)
            _log_trial(
                trial_logger, trial_id, experiment_id, config,
                eval_result, args, slo, optimizer, phase,
            )
            all_results.append((config, eval_result))
            _print_status(trial_id, args.budget, eval_result, optimizer)
            continue

        # Compute metrics
        eval_result = compute_metrics(request_results, slo)
        eval_result.server_startup_time_s = server.startup_time
        eval_result.eval_time_s = time.monotonic() - trial_start

        # Fetch server-side metrics via curl. Failures here are non-fatal —
        # we just lose KV-cache utilization for this trial.
        try:
            proc = subprocess.run(
                ["curl", "-s", "--max-time", "5", f"{server.base_url}/metrics"],
                capture_output=True, text=True, timeout=10,
            )
            if proc.returncode == 0 and "gpu_cache_usage_perc" in proc.stdout:
                import re
                m = re.search(r"gpu_cache_usage_perc\s+(\d+\.?\d*)", proc.stdout)
                if m:
                    eval_result.kv_cache_utilization = float(m.group(1))
        except (subprocess.SubprocessError, OSError) as e:
            logger.debug("Could not fetch /metrics: %s", e)

        server.stop()
        optimizer.tell(config, eval_result)
        _log_trial(
            trial_logger, trial_id, experiment_id, config,
            eval_result, args, slo, optimizer, phase,
        )
        all_results.append((config, eval_result))
        _print_status(trial_id, args.budget, eval_result, optimizer)

    # Summary
    elapsed = time.monotonic() - total_start
    print()
    print("=" * 60)
    print(f"  Experiment {experiment_id} complete in {elapsed:.0f}s")
    print(f"  Crashes: {optimizer.n_crashes}/{args.budget}")
    print(f"  Feasible: {optimizer.n_feasible}/{args.budget}")
    print("=" * 60)

    best = optimizer.best_feasible()
    if best:
        config, result = best
        print()
        print("BEST FEASIBLE CONFIG:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        print(f"  Goodput: {result.goodput_tokens_per_sec or 0:.1f} tok/s")
        print(f"  Latency p99: {result.request_latency_p99_ms or 0:.0f} ms")
        print(f"  Latency p50: {result.request_latency_p50_ms or 0:.0f} ms")
    else:
        print("\nNo feasible configuration found within budget.")

    print(f"\nResults saved to: {log_path}")


def _print_status(trial_id: int, budget: int, result: EvalResult, optimizer) -> None:
    if result.crashed:
        status = f"CRASH ({result.crash_type})"
    elif result.feasible:
        status = "FEASIBLE"
    else:
        status = "INFEASIBLE"

    latency = f"{result.request_latency_p50_ms:.0f}ms" if result.request_latency_p50_ms else "N/A"
    goodput = f"{result.goodput_tokens_per_sec:.1f}" if result.goodput_tokens_per_sec else "N/A"

    logger.info(
        "  -> %s | latency_p50=%s | goodput=%s | crashes=%d/%d | feasible=%d/%d",
        status, latency, goodput,
        optimizer.n_crashes, trial_id + 1,
        optimizer.n_feasible, trial_id + 1,
    )


def _log_trial(
    trial_logger: TrialLogger,
    trial_id: int,
    experiment_id: str,
    config: dict,
    result: EvalResult,
    args: argparse.Namespace,
    slo: SLOContract,
    optimizer,
    phase: str = "",
) -> None:
    """Append one trial to the JSONL log via TrialLogger (durable write)."""
    trial = ServingTrialResult(
        trial_id=trial_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        experiment_id=experiment_id,
        config=config,
        model_id=args.model,
        gpu_id="auto",
        workload_type="fixed",
        request_rate=args.request_rate,
        num_requests=args.num_requests,
        prompt_len_distribution="uniform(128,512)",
        output_len_distribution="uniform(64,256)",
        ttft_p50=result.ttft_p50_ms,
        ttft_p95=result.ttft_p95_ms,
        ttft_p99=result.ttft_p99_ms,
        itl_p50=result.itl_p50_ms,
        itl_p95=result.itl_p95_ms,
        itl_p99=result.itl_p99_ms,
        request_latency_p50=result.request_latency_p50_ms,
        request_latency_p95=result.request_latency_p95_ms,
        request_latency_p99=result.request_latency_p99_ms,
        tokens_per_sec=result.tokens_per_sec,
        requests_per_sec=result.requests_per_sec,
        goodput_tokens_per_sec=result.goodput_tokens_per_sec,
        goodput_ratio=result.goodput_ratio,
        gpu_memory_peak_mb=result.gpu_memory_peak_mb,
        gpu_memory_allocated_mb=result.gpu_memory_allocated_mb,
        kv_cache_utilization=result.kv_cache_utilization,
        feasible=result.feasible,
        crashed=result.crashed,
        crash_type=result.crash_type,
        error_msg=result.error_msg,
        server_startup_time_s=result.server_startup_time_s,
        eval_time_s=result.eval_time_s,
        slo_ttft_p99_ms=slo.ttft_p99_ms,
        slo_itl_p99_ms=slo.itl_p99_ms,
        slo_request_latency_p99_ms=slo.request_latency_p99_ms,
        slo_gpu_memory_mb=slo.gpu_memory_mb,
        optimizer_name=type(optimizer).__name__,
        optimizer_phase=phase,
        seed=args.seed,
    )
    trial_logger.log(trial)


def main():
    parser = argparse.ArgumentParser(description="SLO-Guard curl-based benchmark")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--optimizer", default="tba-tpe",
                        choices=["random", "tpe", "tba", "tba-tpe", "constrained-bo"])
    parser.add_argument("--budget", type=int, default=15, help="Number of trials")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-requests", type=int, default=10, help="Requests per trial")
    parser.add_argument("--request-rate", type=float, default=2.0, help="Requests/sec")
    parser.add_argument("--slo-ttft-p99", type=float, default=2000.0)
    parser.add_argument("--slo-itl-p99", type=float, default=200.0)
    parser.add_argument("--slo-latency-p99", type=float, default=30000.0)
    parser.add_argument("--output", default="results/curl_run/")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_experiment(args)


if __name__ == "__main__":
    main()
