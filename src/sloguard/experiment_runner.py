"""Experiment runner — the main ask/tell benchmark loop.

Orchestrates: optimizer -> server manager -> load generator -> metrics collector
-> crash classifier -> trial logger. Each trial is a complete cycle of
proposing a config, starting vLLM, benchmarking, and recording results.
"""
from __future__ import annotations

import asyncio
import logging
import multiprocessing
import queue as queue_mod
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import aiohttp

from sloguard.config_space import SearchSpace, fix_serving_config
from sloguard.crash_classifier import CrashClassifier
from sloguard.gpu_profile import (
    DEFAULT_VRAM_GB,
    detect_gpu_vram_gb,
    kv_gb_per_token_for,
    log_gpu_info,
    model_footprint_gb_for,
)
from sloguard.load_generator import WorkloadConfig, create_generator
from sloguard.metrics_collector import MetricsCollector
from sloguard.optimizer.base import BaseOptimizer
from sloguard.optimizer.optuna_tpe import OptunaColdTPE
from sloguard.optimizer.random_search import RandomSearchOptimizer
from sloguard.optimizer.tba_optimizer import TBAOptimizer
from sloguard.optimizer.tba_tpe_hybrid import TBATPEHybrid
from sloguard.server_manager import VLLMServerManager
from sloguard.slo_contract import SLOContract
from sloguard.trial_logger import TrialLogger
from sloguard.types import EvalResult, ServingTrialResult, TimeoutConfig

logger = logging.getLogger(__name__)


# Optimizer registry
OPTIMIZERS: dict[str, type[BaseOptimizer]] = {
    "random": RandomSearchOptimizer,
    "tpe": OptunaColdTPE,
    "tba": TBAOptimizer,
    "tba-tpe": TBATPEHybrid,
}

# Lazy registration for constrained BO (needs botorch)
def _get_optimizer_class(name: str) -> type[BaseOptimizer]:
    if name == "constrained-bo":
        from sloguard.optimizer.constrained_bo import ConstrainedBOOptimizer
        return ConstrainedBOOptimizer
    if name not in OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer: {name!r}. "
            f"Available: {list(OPTIMIZERS.keys()) + ['constrained-bo']}"
        )
    return OPTIMIZERS[name]


def create_optimizer(
    name: str,
    search_space: SearchSpace,
    constraints: dict[str, float],
    budget: int,
    seed: int = 42,
) -> BaseOptimizer:
    """Create an optimizer by name."""
    cls = _get_optimizer_class(name)
    return cls(
        search_space=search_space,
        constraints=constraints,
        objective="maximize_goodput",
        budget=budget,
        seed=seed,
    )


# Optimizer objective modes.
OBJECTIVE_GOODPUT = "maximize_goodput"
OBJECTIVE_UTILITY = "maximize_utility"

# Default weights for the utility score. λ₁ is the goodput-unit penalty for a
# crash; λ₂ converts tuning seconds into goodput units (1s ≈ 1 tok/s).
DEFAULT_CRASH_PENALTY = 1000.0
DEFAULT_TIME_PENALTY = 1.0


def compute_utility(
    result: EvalResult,
    crash_penalty: float = DEFAULT_CRASH_PENALTY,
    time_penalty: float = DEFAULT_TIME_PENALTY,
) -> float:
    """Wall-clock-aware utility score.

    U(θ) = goodput - λ₁·crash_flag - λ₂·(startup + eval)

    Crashed trials get U = -λ₁ - λ₂·tuning_cost, which is finite so the
    optimizer can rank "fast crash" above "slow crash" — the right
    incentive when we want the search to fail fast.

    Feasible / infeasible-but-completed trials get U = goodput - λ₂·cost
    using goodput_tokens_per_sec (0 if not measured).
    """
    startup = result.server_startup_time_s or 0.0
    eval_s = result.eval_time_s or 0.0
    time_cost = time_penalty * (startup + eval_s)
    if result.crashed:
        return -crash_penalty - time_cost
    goodput = result.goodput_tokens_per_sec or 0.0
    return goodput - time_cost


def summarize_results(results: list[EvalResult], budget: int) -> dict[str, float]:
    """Aggregate counts + wasted-seconds from a list of EvalResults.

    Crashed and infeasible trials are "wasted" budget — surfacing them
    explicitly tells you whether the search space is too wide (high crash
    rate) or the SLOs are too tight (high infeasible rate).
    """
    n_crashed = sum(1 for r in results if r.crashed)
    n_feasible = sum(1 for r in results if r.feasible and not r.crashed)
    # An "evaluated" trial is one whose result we recorded. Anything in the
    # budget that wasn't evaluated counts as infeasible too (e.g. an early
    # abort would leave gaps).
    n_evaluated = len(results)
    n_infeasible = max(0, budget - n_crashed - n_feasible)
    wasted_s = sum(r.eval_time_s for r in results if r.crashed or not r.feasible)
    return {
        "evaluated": n_evaluated,
        "feasible": n_feasible,
        "crashed": n_crashed,
        "infeasible": n_infeasible,
        "wasted_s": wasted_s,
        "crashed_pct": (n_crashed / budget * 100) if budget else 0.0,
        "infeasible_pct": (n_infeasible / budget * 100) if budget else 0.0,
    }


def _benchmark_worker(
    queue: multiprocessing.Queue,
    workload_mode: str,
    base_url: str,
    workload: Any,
    seed: int,
    workload_kwargs: dict[str, Any],
    timeouts: TimeoutConfig,
) -> None:
    """Top-level function for the benchmark subprocess.

    Must be top-level (not a method) so multiprocessing can pickle it.
    Runs the load generator and puts results into the queue.
    """
    try:
        gen = create_generator(
            mode=workload_mode,
            base_url=base_url,
            workload=workload,
            seed=seed,
            timeouts=timeouts,
            **workload_kwargs,
        )
        results = asyncio.run(gen.run(trial_timeout=timeouts.per_trial_s))
        queue.put({"results": results, "peak_concurrency": gen.peak_concurrency})
    except Exception as e:
        logging.getLogger(__name__).error("Benchmark worker error: %s", e)
        queue.put({"results": [], "peak_concurrency": 0})


class ExperimentRunner:
    """Runs a complete benchmark experiment.

    For each trial:
      1. optimizer.ask() -> proposed config
      2. Start vLLM with that config
      3. Run load generator
      4. Collect metrics
      5. Classify outcome (healthy/crash)
      6. Build EvalResult
      7. optimizer.tell(config, result)
      8. Log to JSONL
    """

    def __init__(
        self,
        model: str,
        optimizer: BaseOptimizer,
        slo: SLOContract,
        workload: WorkloadConfig,
        output_dir: str = "results",
        experiment_id: str | None = None,
        gpu_id: str = "auto",
        port: int = 8000,
        workload_mode: str = "fixed",
        workload_kwargs: dict[str, Any] | None = None,
        benchmark_duration_s: float = 60.0,
        timeouts: TimeoutConfig | None = None,
        objective: str = OBJECTIVE_GOODPUT,
        crash_penalty: float = DEFAULT_CRASH_PENALTY,
        time_penalty: float = DEFAULT_TIME_PENALTY,
    ):
        self.model = model
        self.optimizer = optimizer
        self.slo = slo
        self.workload = workload
        self.workload.model = model
        self.output_dir = output_dir
        self.experiment_id = experiment_id or str(uuid.uuid4())[:8]
        self.gpu_id = gpu_id
        self.port = port
        self.workload_mode = workload_mode
        self.workload_kwargs = workload_kwargs or {}
        self.benchmark_duration_s = benchmark_duration_s
        self.timeouts = timeouts or TimeoutConfig()
        if objective not in (OBJECTIVE_GOODPUT, OBJECTIVE_UTILITY):
            raise ValueError(f"unknown objective: {objective!r}")
        self.objective = objective
        self.crash_penalty = crash_penalty
        self.time_penalty = time_penalty

        self.server = VLLMServerManager(
            model=model, port=port, startup_timeout=self.timeouts.server_start_s,
        )
        self.classifier = CrashClassifier()
        self.metrics_collector = MetricsCollector(slo_contract=slo)

        log_path = f"{output_dir}/{self.experiment_id}.jsonl"
        self.logger = TrialLogger(log_path)

        self.results: list[tuple[dict[str, Any], EvalResult]] = []

        # Detect GPU + model profile once so the memory guard scales correctly
        # to whatever hardware we actually have.
        log_gpu_info(model)
        detected = detect_gpu_vram_gb()
        self.vram_gb = detected if detected is not None else DEFAULT_VRAM_GB
        self.kv_gb_per_token = kv_gb_per_token_for(model)
        self.model_footprint_gb = model_footprint_gb_for(model)

    def run(self, budget: int | None = None) -> tuple[dict[str, Any], EvalResult] | None:
        """Run the full experiment loop.

        Returns the best feasible (config, result) or None.
        """
        budget = budget or self.optimizer.budget
        logger.info(
            "Starting experiment %s: model=%s, optimizer=%s, budget=%d",
            self.experiment_id, self.model,
            type(self.optimizer).__name__, budget,
        )

        start_time = time.monotonic()

        for trial_id in range(budget):
            trial_start = time.monotonic()
            config = self.next_config()

            logger.info(
                "Trial %d/%d [%s]: %s",
                trial_id + 1, budget, self.optimizer.phase,
                {k: v for k, v in config.items()
                 if k in ("quantization", "max_num_seqs", "gpu_memory_utilization")},
            )

            result = self.evaluate(config, trial_id)
            # Always compute utility so we can report it alongside goodput.
            # In maximize_utility mode, also swap objective_value so the
            # optimizer's best-feasible ranking (which reads objective_value)
            # picks trials with the best utility instead of raw goodput.
            result.utility_value = compute_utility(
                result, self.crash_penalty, self.time_penalty,
            )
            if self.objective == OBJECTIVE_UTILITY:
                result.objective_value = result.utility_value
            self.optimizer.tell(config, result)
            self.results.append((config, result))

            # Build full trial result for logging
            trial_result = self._build_trial_result(
                trial_id, config, result, trial_start
            )
            self.logger.log(trial_result)

            # Progress report
            status = "CRASH" if result.crashed else ("FEASIBLE" if result.feasible else "INFEASIBLE")
            obj_str = f"{result.objective_value:.1f}" if result.objective_value > float("-inf") else "N/A"
            logger.info(
                "  -> %s | goodput=%s | crashes=%d/%d | feasible=%d/%d",
                status, obj_str,
                self.optimizer.n_crashes, trial_id + 1,
                self.optimizer.n_feasible, trial_id + 1,
            )

        elapsed = time.monotonic() - start_time
        self._log_summary(budget, elapsed)
        return self.optimizer.best_feasible()

    def _log_summary(self, budget: int, elapsed_s: float) -> None:
        """Log a one-line summary of how the budget was spent."""
        best = self.optimizer.best_feasible()
        best_goodput = (
            f"{best[1].objective_value:.1f} tok/s" if best else "none"
        )
        s = summarize_results([r for _, r in self.results], budget)
        logger.info(
            "Experiment %s complete in %.1fs | "
            "feasible=%d/%d | crashed=%d/%d (%.0f%%) | infeasible=%d/%d (%.0f%%) | "
            "wasted=%.1fs | best_goodput=%s",
            self.experiment_id, elapsed_s,
            s["feasible"], budget,
            s["crashed"], budget, s["crashed_pct"],
            s["infeasible"], budget, s["infeasible_pct"],
            s["wasted_s"], best_goodput,
        )

    def next_config(self) -> dict[str, Any]:
        """Ask the optimizer for the next config and apply fix_serving_config.

        External orchestrators (e.g. scripts/run_multiseed.py) that drive
        trials by hand can use this instead of duplicating the fix-up
        parameters (vram_gb, kv_gb_per_token, model_footprint_gb) that
        run() applies internally.
        """
        config = self.optimizer.ask()
        return fix_serving_config(
            config,
            vram_gb=self.vram_gb,
            kv_gb_per_token=self.kv_gb_per_token,
            model_footprint_gb=self.model_footprint_gb,
        )

    def evaluate(self, config: dict[str, Any], trial_id: int) -> EvalResult:
        """Evaluate a single serving configuration.

        Full lifecycle: start server -> benchmark -> collect metrics -> stop.
        Public so external orchestrators can drive the ask/tell loop
        themselves (e.g. when interleaving multiple optimizer/seed pairs).
        """
        result = EvalResult()
        eval_start = time.monotonic()

        # Start vLLM with this config
        server_started = self.server.start(config)

        if not server_started:
            # Server failed to start — classify crash
            crash_type = self.classifier.classify(
                stderr=self.server.stderr_output,
                exit_code=None,
                timed_out=False,
            )
            result.crashed = True
            result.crash_type = crash_type.value
            result.error_msg = self.server.stderr_output[:500]
            result.server_startup_time_s = self.server.startup_time
            result.eval_time_s = time.monotonic() - eval_start
            return result

        result.server_startup_time_s = self.server.startup_time

        try:
            # Pre-flight: send a single tiny request to verify the engine works
            if not self._preflight_check():
                crash_type = self.classifier.classify(
                    stderr=self.server.stderr_output,
                )
                result.crashed = True
                result.crash_type = crash_type.value if crash_type.value != "healthy" else "startup_failure"
                result.error_msg = "Pre-flight check failed: engine started but cannot serve"
                result.eval_time_s = time.monotonic() - eval_start
                self.server.stop()
                return result

            # Run benchmark in a subprocess with a hard kill timeout.
            # asyncio.wait_for can't interrupt aiohttp's streaming reads,
            # and ThreadPoolExecutor.__exit__ waits for the thread, so the
            # only reliable backstop is multiprocessing + process.kill().
            bench = self._run_benchmark_subprocess(trial_id=trial_id)
            request_results = bench["results"]
            peak_concurrency = bench["peak_concurrency"]

            # If no results at all, treat as crash
            if not request_results:
                result.crashed = True
                result.crash_type = "timeout"
                result.error_msg = "No request results — server unresponsive"
                result.eval_time_s = time.monotonic() - eval_start
                self.server.stop()
                return result

            # If all requests failed, mark infeasible (not crash) — the server
            # started but couldn't handle the workload at this config
            if all(not r.success for r in request_results):
                result.feasible = False
                result.error_msg = (
                    f"All {len(request_results)} requests failed: "
                    f"{request_results[0].error}"
                )
                result.eval_time_s = time.monotonic() - eval_start
                self.server.stop()
                return result

            # Collect metrics
            metrics = self.metrics_collector.compute(
                request_results, peak_concurrency=peak_concurrency,
            )

            # Try to get server-side metrics. fetch_server_metrics already
            # swallows network errors (returns {}); the only callers that can
            # still raise here are asyncio.run itself (RuntimeError if invoked
            # from inside a running event loop) or update_from_server.
            try:
                server_metrics = asyncio.run(
                    self.metrics_collector.fetch_server_metrics(self.server.base_url)
                )
                self.metrics_collector.update_from_server(metrics, server_metrics)
            except (RuntimeError, aiohttp.ClientError, OSError) as e:
                logger.debug("Could not fetch server-side metrics: %s", e)

            # Populate result
            result.ttft_p50_ms = metrics.ttft_p50
            result.ttft_p95_ms = metrics.ttft_p95
            result.ttft_p99_ms = metrics.ttft_p99
            result.itl_p50_ms = metrics.itl_p50
            result.itl_p95_ms = metrics.itl_p95
            result.itl_p99_ms = metrics.itl_p99
            result.request_latency_p50_ms = metrics.request_latency_p50
            result.request_latency_p95_ms = metrics.request_latency_p95
            result.request_latency_p99_ms = metrics.request_latency_p99
            result.request_latency_mean_ms = metrics.request_latency_mean
            result.tokens_per_sec = metrics.tokens_per_sec
            result.requests_per_sec = metrics.requests_per_sec
            result.total_output_tokens = metrics.total_output_tokens or None
            result.goodput_tokens_per_sec = metrics.goodput_tokens_per_sec
            result.goodput_ratio = metrics.goodput_ratio
            result.gpu_memory_peak_mb = metrics.gpu_memory_peak_mb
            result.kv_cache_utilization = metrics.kv_cache_utilization
            result.peak_concurrency = metrics.peak_concurrency

            # Set objective value (goodput)
            result.objective_value = metrics.goodput_tokens_per_sec or 0.0

            # Build constraints dict for optimizer
            result.constraints = {}
            if metrics.ttft_p99 is not None:
                result.constraints["ttft_p99_ms"] = metrics.ttft_p99
            if metrics.itl_p99 is not None:
                result.constraints["itl_p99_ms"] = metrics.itl_p99
            if metrics.request_latency_p99 is not None:
                result.constraints["request_latency_p99_ms"] = metrics.request_latency_p99
            if metrics.gpu_memory_peak_mb is not None:
                result.constraints["gpu_memory_mb"] = metrics.gpu_memory_peak_mb

            # Check SLO compliance
            result.feasible = self.slo.check_aggregate(
                ttft_p99=metrics.ttft_p99,
                itl_p99=metrics.itl_p99,
                request_latency_p99=metrics.request_latency_p99,
                gpu_memory_mb=metrics.gpu_memory_peak_mb,
            )

        except Exception as e:
            crash_type = self.classifier.classify_exception(e)
            result.crashed = True
            result.crash_type = crash_type.value
            result.error_msg = f"{type(e).__name__}: {e}"

        finally:
            self.server.stop()

        result.eval_time_s = time.monotonic() - eval_start
        return result

    def _run_benchmark_subprocess(self, trial_id: int) -> dict[str, Any]:
        """Run load generation in a subprocess that can be hard-killed.

        Returns {"results": list[RequestResult], "peak_concurrency": int}.
        On timeout / worker death, returns an empty results list and 0 concurrency.
        """
        from sloguard.load_generator import RequestResult  # noqa: F401

        trial_timeout = self.timeouts.per_trial_s
        queue: multiprocessing.Queue = multiprocessing.Queue()
        proc = multiprocessing.Process(
            target=_benchmark_worker,
            args=(
                queue,
                self.workload_mode,
                self.server.base_url,
                self.workload,
                self.optimizer.seed + trial_id,
                self.workload_kwargs,
                self.timeouts,
            ),
            daemon=True,
        )
        proc.start()
        proc.join(timeout=trial_timeout)

        if proc.is_alive():
            logger.warning("Hard-killing benchmark process after %.0fs", trial_timeout)
            proc.kill()
            proc.join(timeout=5)

        # Drain results from queue. The worker puts exactly one item — a dict
        # containing the results list and the peak in-flight concurrency.
        payload: dict[str, Any] = {"results": [], "peak_concurrency": 0}
        while True:
            try:
                payload = queue.get_nowait()
            except queue_mod.Empty:
                break

        return payload

    def _preflight_check(self) -> bool:
        """Send a single tiny request to verify the vLLM engine is functional.

        Some configs start the HTTP server but crash the inference engine,
        returning 500 on actual requests. This catches that case fast.

        Uses subprocess + curl with --max-time for a hard timeout that
        actually works (urllib's timeout doesn't cover response body reads,
        causing hangs when vLLM's EngineCore dies after health check passes).
        """
        import json as json_mod
        import subprocess

        url = f"{self.server.base_url}/v1/chat/completions"
        payload = json_mod.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1,
            "stream": False,
        })

        max_time = self.timeouts.preflight_s
        try:
            proc = subprocess.run(
                [
                    "curl", "-s",
                    "--connect-timeout", "5",
                    "--max-time", str(int(max_time)),
                    "-H", "Content-Type: application/json",
                    "-d", payload,
                    url,
                ],
                capture_output=True,
                text=True,
                # +5 so curl's own --max-time fires first and we get useful stderr
                timeout=max_time + 5,
            )
            if proc.returncode != 0:
                logger.warning("Pre-flight curl failed (exit %d): %s",
                               proc.returncode, proc.stderr[:200])
                return False

            data = json_mod.loads(proc.stdout)
            if "error" in data:
                logger.warning("Pre-flight got error: %s", data["error"])
                return False
            return True
        except subprocess.TimeoutExpired:
            logger.warning("Pre-flight hard timeout after %.0fs", max_time + 5)
            return False
        except (subprocess.SubprocessError, json_mod.JSONDecodeError, OSError) as e:
            logger.warning("Pre-flight check failed: %s", e)
            return False

    def _build_trial_result(
        self,
        trial_id: int,
        config: dict[str, Any],
        result: EvalResult,
        trial_start: float,
    ) -> ServingTrialResult:
        """Build a full ServingTrialResult for logging."""
        return ServingTrialResult(
            trial_id=trial_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            experiment_id=self.experiment_id,
            config=config,
            model_id=self.model,
            gpu_id=self.gpu_id,
            workload_type=self.workload_mode,
            request_rate=self.workload.request_rate,
            num_requests=self.workload.num_requests,
            prompt_len_distribution=f"uniform({self.workload.prompt_len_min},{self.workload.prompt_len_max})",
            output_len_distribution=f"uniform({self.workload.output_len_min},{self.workload.output_len_max})",
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
            slo_ttft_p99_ms=self.slo.ttft_p99_ms,
            slo_itl_p99_ms=self.slo.itl_p99_ms,
            slo_request_latency_p99_ms=self.slo.request_latency_p99_ms,
            slo_gpu_memory_mb=self.slo.gpu_memory_mb,
            optimizer_name=type(self.optimizer).__name__,
            optimizer_phase=self.optimizer.phase,
            seed=self.optimizer.seed,
        )
