"""SLO contract definitions and compliance checking.

Defines Service Level Objectives for LLM serving and provides
per-request and aggregate compliance checking, plus goodput computation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SLOContract:
    """Service Level Objective contract for LLM serving.

    All latency thresholds are in milliseconds.
    Memory threshold is in megabytes.
    A value of 0 means "no constraint" for that metric.
    """

    ttft_p99_ms: float = 500.0  # Time to first token
    itl_p99_ms: float = 100.0   # Inter-token latency
    request_latency_p99_ms: float = 30000.0  # Total request latency
    gpu_memory_mb: float = 0.0  # 0 = no constraint

    def check_request(
        self,
        ttft_ms: float | None = None,
        itl_max_ms: float | None = None,
        request_latency_ms: float | None = None,
    ) -> bool:
        """Check if a single request meets all SLO thresholds.

        For per-request checking, we compare individual request metrics
        against the p99 thresholds. This is used for goodput computation:
        a request "meets the SLO" if its individual metrics are within
        the target bounds.
        """
        if self.ttft_p99_ms > 0 and ttft_ms is not None:
            if ttft_ms > self.ttft_p99_ms:
                return False

        if self.itl_p99_ms > 0 and itl_max_ms is not None:
            if itl_max_ms > self.itl_p99_ms:
                return False

        if self.request_latency_p99_ms > 0 and request_latency_ms is not None:
            if request_latency_ms > self.request_latency_p99_ms:
                return False

        return True

    def check_aggregate(
        self,
        ttft_p99: float | None = None,
        itl_p99: float | None = None,
        request_latency_p99: float | None = None,
        gpu_memory_mb: float | None = None,
    ) -> bool:
        """Check if aggregate (p99) metrics meet the SLO contract."""
        if self.ttft_p99_ms > 0 and ttft_p99 is not None:
            if ttft_p99 > self.ttft_p99_ms:
                return False

        if self.itl_p99_ms > 0 and itl_p99 is not None:
            if itl_p99 > self.itl_p99_ms:
                return False

        if self.request_latency_p99_ms > 0 and request_latency_p99 is not None:
            if request_latency_p99 > self.request_latency_p99_ms:
                return False

        if self.gpu_memory_mb > 0 and gpu_memory_mb is not None:
            if gpu_memory_mb > self.gpu_memory_mb:
                return False

        return True

    def headroom(
        self,
        ttft_p99: float | None = None,
        itl_p99: float | None = None,
        request_latency_p99: float | None = None,
        gpu_memory_mb: float | None = None,
    ) -> dict[str, float]:
        """Compute headroom: how close each metric is to violating its SLO.

        Returns dict of {metric_name: fraction_remaining}.
        1.0 = at zero, 0.0 = exactly at threshold, negative = violated.
        """
        headroom = {}

        if self.ttft_p99_ms > 0 and ttft_p99 is not None:
            headroom["ttft_p99"] = 1.0 - (ttft_p99 / self.ttft_p99_ms)

        if self.itl_p99_ms > 0 and itl_p99 is not None:
            headroom["itl_p99"] = 1.0 - (itl_p99 / self.itl_p99_ms)

        if self.request_latency_p99_ms > 0 and request_latency_p99 is not None:
            headroom["request_latency_p99"] = 1.0 - (
                request_latency_p99 / self.request_latency_p99_ms
            )

        if self.gpu_memory_mb > 0 and gpu_memory_mb is not None:
            headroom["gpu_memory"] = 1.0 - (gpu_memory_mb / self.gpu_memory_mb)

        return headroom

    def to_constraints_dict(self) -> dict[str, float]:
        """Convert to constraints dict for optimizer interface.

        Returns {constraint_name: max_allowed_value} compatible with
        the BaseOptimizer constraints format.
        """
        constraints = {}
        if self.ttft_p99_ms > 0:
            constraints["ttft_p99_ms"] = self.ttft_p99_ms
        if self.itl_p99_ms > 0:
            constraints["itl_p99_ms"] = self.itl_p99_ms
        if self.request_latency_p99_ms > 0:
            constraints["request_latency_p99_ms"] = self.request_latency_p99_ms
        if self.gpu_memory_mb > 0:
            constraints["gpu_memory_mb"] = self.gpu_memory_mb
        return constraints
