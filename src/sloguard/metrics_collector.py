"""Metrics collector for LLM serving benchmarks.

Computes TTFT, ITL, request latency distributions, throughput, and goodput
from per-request results produced by the load generator.

Also queries vLLM's /metrics endpoint for GPU memory and KV cache stats.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np

from sloguard.load_generator import RequestResult
from sloguard.slo_contract import SLOContract

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics from a benchmark run."""

    # Latency distributions (ms)
    ttft_p50: float | None = None
    ttft_p95: float | None = None
    ttft_p99: float | None = None
    itl_p50: float | None = None
    itl_p95: float | None = None
    itl_p99: float | None = None
    request_latency_p50: float | None = None
    request_latency_p95: float | None = None
    request_latency_p99: float | None = None
    request_latency_mean: float | None = None  # arithmetic mean for compact logging

    # Throughput
    tokens_per_sec: float | None = None
    requests_per_sec: float | None = None

    # Goodput
    goodput_tokens_per_sec: float | None = None
    goodput_ratio: float | None = None  # fraction of requests meeting all SLOs

    # Resource usage (from /metrics endpoint)
    gpu_memory_peak_mb: float | None = None
    gpu_memory_allocated_mb: float | None = None
    kv_cache_utilization: float | None = None

    # Summary
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_output_tokens: int = 0
    duration_s: float = 0.0
    # Max number of in-flight requests observed during the run. Lets us
    # verify the load generator actually issued concurrent load rather
    # than serializing (a real bug prior to the concurrent rewrite).
    peak_concurrency: int | None = None

    # Raw distributions for plotting
    ttft_values: list[float] = field(default_factory=list)
    itl_values: list[float] = field(default_factory=list)
    latency_values: list[float] = field(default_factory=list)


class MetricsCollector:
    """Computes serving metrics from per-request results.

    Usage:
        collector = MetricsCollector(slo_contract=slo)
        metrics = collector.compute(results)
    """

    def __init__(self, slo_contract: SLOContract | None = None):
        self.slo = slo_contract

    def compute(
        self,
        results: list[RequestResult],
        peak_concurrency: int | None = None,
    ) -> BenchmarkMetrics:
        """Compute all metrics from request results."""
        metrics = BenchmarkMetrics()
        metrics.peak_concurrency = peak_concurrency

        if not results:
            return metrics

        metrics.total_requests = len(results)
        successful = [r for r in results if r.success]
        metrics.successful_requests = len(successful)
        metrics.failed_requests = metrics.total_requests - metrics.successful_requests

        if not successful:
            return metrics

        # TTFT distribution
        ttft_values = [r.ttft_ms for r in successful if r.ttft_ms is not None]
        if ttft_values:
            metrics.ttft_values = ttft_values
            metrics.ttft_p50 = float(np.percentile(ttft_values, 50))
            metrics.ttft_p95 = float(np.percentile(ttft_values, 95))
            metrics.ttft_p99 = float(np.percentile(ttft_values, 99))

        # ITL distribution (flatten all inter-token latencies)
        itl_values = []
        for r in successful:
            itl_values.extend(r.itl_ms_list)
        if itl_values:
            metrics.itl_values = itl_values
            metrics.itl_p50 = float(np.percentile(itl_values, 50))
            metrics.itl_p95 = float(np.percentile(itl_values, 95))
            metrics.itl_p99 = float(np.percentile(itl_values, 99))

        # Request latency distribution
        latency_values = [r.total_latency_ms for r in successful if r.total_latency_ms is not None]
        if latency_values:
            metrics.latency_values = latency_values
            metrics.request_latency_p50 = float(np.percentile(latency_values, 50))
            metrics.request_latency_p95 = float(np.percentile(latency_values, 95))
            metrics.request_latency_p99 = float(np.percentile(latency_values, 99))
            metrics.request_latency_mean = float(np.mean(latency_values))

        # Throughput
        total_tokens = sum(r.output_tokens for r in successful)
        metrics.total_output_tokens = total_tokens

        if successful:
            earliest = min(r.send_time for r in successful)
            latest = max(r.end_time for r in successful if r.end_time is not None)
            duration = latest - earliest
            metrics.duration_s = duration
            if duration > 0:
                metrics.tokens_per_sec = total_tokens / duration
                metrics.requests_per_sec = len(successful) / duration

        # Goodput (SLO-satisfying throughput)
        if self.slo is not None:
            self._compute_goodput(metrics, successful)

        return metrics

    def _compute_goodput(
        self, metrics: BenchmarkMetrics, successful: list[RequestResult]
    ) -> None:
        """Compute goodput: throughput of requests meeting all SLOs."""
        assert self.slo is not None
        ratio, tps = self.slo.compute_goodput(
            (
                (
                    r.ttft_ms,
                    max(r.itl_ms_list) if r.itl_ms_list else 0.0,
                    r.total_latency_ms,
                    r.output_tokens,
                )
                for r in successful
            ),
            duration_s=metrics.duration_s,
        )
        metrics.goodput_ratio = ratio
        if metrics.duration_s > 0:
            metrics.goodput_tokens_per_sec = tps

    async def fetch_server_metrics(self, base_url: str) -> dict[str, float]:
        """Fetch GPU/KV cache metrics from vLLM's Prometheus endpoint."""
        import aiohttp

        metrics: dict[str, float] = {}
        try:
            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=5)
                async with session.get(f"{base_url}/metrics", timeout=timeout) as resp:
                    if resp.status != 200:
                        return metrics
                    text = await resp.text()
                    metrics = self._parse_prometheus(text)
        except Exception as e:
            logger.debug("Failed to fetch server metrics: %s", e)

        return metrics

    def _parse_prometheus(self, text: str) -> dict[str, float]:
        """Parse the subset of vLLM's Prometheus metrics that we consume.

        Currently only ``vllm:gpu_cache_usage_perc`` (KV-cache utilization
        as a 0-1 fraction) is extracted. An earlier version of this
        parser tried to populate ``gpu_memory_peak_mb`` from the same
        metric, which was a copy-paste bug — vLLM 0.19 does not expose
        absolute peak GPU memory in MB via /metrics, so there is no
        correct source to parse. Leaving the field unset is more honest
        than fabricating a value.
        """
        metrics: dict[str, float] = {}
        for raw in text.split("\n"):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "gpu_cache_usage_perc" in line:
                match = re.search(r"(\d+\.?\d*)\s*$", line)
                if match:
                    metrics["kv_cache_utilization"] = float(match.group(1))
        return metrics

    def update_from_server(
        self, metrics: BenchmarkMetrics, server_metrics: dict[str, float],
    ) -> None:
        """Update BenchmarkMetrics with server-side metrics.

        Only kv_cache_utilization is available from vLLM 0.19's /metrics
        endpoint. gpu_memory_peak_mb stays None by design — see
        _parse_prometheus for details.
        """
        if "kv_cache_utilization" in server_metrics:
            metrics.kv_cache_utilization = server_metrics["kv_cache_utilization"]
