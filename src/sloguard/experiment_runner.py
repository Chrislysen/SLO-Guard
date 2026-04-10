"""Experiment runner — the main ask/tell benchmark loop.

Orchestrates: optimizer -> server manager -> load generator -> metrics collector
-> crash classifier -> trial logger. Each trial is a complete cycle of
proposing a config, starting vLLM, benchmarking, and recording results.
"""
from __future__ import annotations

import asyncio
import logging
import multiprocessing
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from sloguard.config_space import SearchSpace, build_serving_space, fix_serving_config
from sloguard.crash_classifier import CrashClassifier, CrashType
from sloguard.load_generator import (
    BurstGenerator,
    FixedRateGenerator,
    TraceReplayGenerator,
    WorkloadConfig,
    create_generator,
)
from sloguard.metrics_collector import MetricsCollector
from sloguard.optimizer.base import BaseOptimizer
from sloguard.optimizer.optuna_tpe import OptunaColdTPE
from sloguard.optimizer.random_search import RandomSearchOptimizer
from sloguard.optimizer.tba_optimizer import TBAOptimizer
from sloguard.optimizer.tba_tpe_hybrid import TBATPEHybrid
from sloguard.server_manager import VLLMServerManager
from sloguard.slo_contract import SLOContract
from sloguard.trial_logger import TrialLogger
from sloguard.types import EvalResult, ServingTrialResult

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


def _benchmark_worker(
    queue: multiprocessing.Queue,
    workload_mode: str,
    base_url: str,
    workload: Any,
    seed: int,
    workload_kwargs: dict[str, Any],
    trial_timeout: float,
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
            **workload_kwargs,
        )
        results = asyncio.run(gen.run(trial_timeout=trial_timeout))
        queue.put(results)
    except Exception as e:
        logging.getLogger(__name__).error("Benchmark worker error: %s", e)
        queue.put([])


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

        self.server = VLLMServerManager(model=model, port=port)
        self.classifier = CrashClassifier()
        self.metrics_collector = MetricsCollector(slo_contract=slo)

        log_path = f"{output_dir}/{self.experiment_id}.jsonl"
        self.logger = TrialLogger(log_path)

        self.results: list[tuple[dict[str, Any], EvalResult]] = []

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
            config = self.optimizer.ask()
            config = fix_serving_config(config)

            logger.info(
                "Trial %d/%d: %s",
                trial_id + 1, budget,
                {k: v for k, v in config.items()
                 if k in ("quantization", "max_num_seqs", "gpu_memory_utilization")},
            )

            result = self._evaluate(config, trial_id)
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
        logger.info(
            "Experiment %s complete in %.1fs. Best feasible: %s",
            self.experiment_id, elapsed,
            self.optimizer.best_feasible() is not None,
        )

        return self.optimizer.best_feasible()

    def _evaluate(self, config: dict[str, Any], trial_id: int) -> EvalResult:
        """Evaluate a single serving configuration.

        Full lifecycle: start server -> benchmark -> collect metrics -> stop.
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
            request_results = self._run_benchmark_subprocess(
                trial_id=trial_id,
                trial_timeout=180.0,
            )

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
                result.error_msg = f"All {len(request_results)} requests failed: {request_results[0].error}"
                result.eval_time_s = time.monotonic() - eval_start
                self.server.stop()
                return result

            # Collect metrics
            metrics = self.metrics_collector.compute(request_results)

            # Try to get server-side metrics
            try:
                server_metrics = asyncio.run(
                    self.metrics_collector.fetch_server_metrics(self.server.base_url)
                )
                self.metrics_collector.update_from_server(metrics, server_metrics)
            except Exception:
                pass

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
            result.tokens_per_sec = metrics.tokens_per_sec
            result.requests_per_sec = metrics.requests_per_sec
            result.goodput_tokens_per_sec = metrics.goodput_tokens_per_sec
            result.goodput_ratio = metrics.goodput_ratio
            result.gpu_memory_peak_mb = metrics.gpu_memory_peak_mb
            result.kv_cache_utilization = metrics.kv_cache_utilization

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

    def _run_benchmark_subprocess(
        self,
        trial_id: int,
        trial_timeout: float,
    ) -> list:
        """Run load generation in a subprocess that can be hard-killed.

        Returns list of RequestResult (possibly empty on timeout).
        """
        from sloguard.load_generator import RequestResult

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
                trial_timeout,
            ),
            daemon=True,
        )
        proc.start()
        proc.join(timeout=trial_timeout)

        if proc.is_alive():
            logger.warning("Hard-killing benchmark process after %.0fs", trial_timeout)
            proc.kill()
            proc.join(timeout=5)

        # Drain results from queue
        results = []
        try:
            while not queue.empty():
                results = queue.get_nowait()
        except Exception:
            pass

        return results

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

        try:
            proc = subprocess.run(
                [
                    "curl", "-s",
                    "--connect-timeout", "5",
                    "--max-time", "30",
                    "-H", "Content-Type: application/json",
                    "-d", payload,
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=35,
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
            logger.warning("Pre-flight hard timeout after 35s")
            return False
        except Exception as e:
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
            seed=self.optimizer.seed,
        )
