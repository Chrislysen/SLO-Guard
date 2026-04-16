"""Core data types for SLO-Guard — dataclasses for type safety."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VariableDef:
    """Definition of a single variable in the search space."""

    name: str
    var_type: str  # "categorical", "integer", "continuous"
    choices: list[Any] | None = None
    low: float | None = None
    high: float | None = None
    log_scale: bool = False
    condition: str | None = None  # python expression, e.g. "enforce_eager == True"

    def __post_init__(self):
        if self.var_type == "categorical" and not self.choices:
            raise ValueError(f"Categorical variable '{self.name}' needs choices")
        if self.var_type in ("integer", "continuous") and (self.low is None or self.high is None):
            raise ValueError(f"Numeric variable '{self.name}' needs low and high")


@dataclass
class EvalResult:
    """Result of evaluating a single serving configuration."""

    objective_value: float = float("-inf")  # goodput (tokens/s meeting SLOs)
    constraints: dict[str, float] = field(default_factory=dict)
    feasible: bool = False
    crashed: bool = False
    eval_time_s: float = 0.0
    error_msg: str | None = None
    crash_type: str | None = None  # "oom", "cuda_error", "timeout", "config_invalid", None

    # Serving-specific metrics
    ttft_p50_ms: float | None = None
    ttft_p95_ms: float | None = None
    ttft_p99_ms: float | None = None
    itl_p50_ms: float | None = None
    itl_p95_ms: float | None = None
    itl_p99_ms: float | None = None
    request_latency_p50_ms: float | None = None
    request_latency_p95_ms: float | None = None
    request_latency_p99_ms: float | None = None
    tokens_per_sec: float | None = None
    requests_per_sec: float | None = None
    goodput_tokens_per_sec: float | None = None
    goodput_ratio: float | None = None  # fraction of requests meeting all SLOs
    gpu_memory_peak_mb: float | None = None
    gpu_memory_allocated_mb: float | None = None
    kv_cache_utilization: float | None = None
    server_startup_time_s: float | None = None


@dataclass
class ServingTrialResult:
    """Full trial record for JSONL logging."""

    trial_id: int
    timestamp: str  # ISO 8601
    experiment_id: str

    # Config
    config: dict[str, Any] = field(default_factory=dict)
    model_id: str = ""
    gpu_id: str = ""

    # Workload
    workload_type: str = ""  # "interactive", "batch", "bursty_trace"
    request_rate: float = 0.0
    num_requests: int = 0
    prompt_len_distribution: str = ""
    output_len_distribution: str = ""

    # Latency metrics (ms)
    ttft_p50: float | None = None
    ttft_p95: float | None = None
    ttft_p99: float | None = None
    itl_p50: float | None = None
    itl_p95: float | None = None
    itl_p99: float | None = None
    request_latency_p50: float | None = None
    request_latency_p95: float | None = None
    request_latency_p99: float | None = None

    # Throughput
    tokens_per_sec: float | None = None
    requests_per_sec: float | None = None
    goodput_tokens_per_sec: float | None = None
    goodput_ratio: float | None = None

    # Resource usage
    gpu_memory_peak_mb: float | None = None
    gpu_memory_allocated_mb: float | None = None
    kv_cache_utilization: float | None = None

    # Outcome
    feasible: bool = False
    crashed: bool = False
    crash_type: str | None = None
    error_msg: str | None = None
    server_startup_time_s: float | None = None
    eval_time_s: float = 0.0

    # SLO contract
    slo_ttft_p99_ms: float = 0.0
    slo_itl_p99_ms: float = 0.0
    slo_request_latency_p99_ms: float = 0.0
    slo_gpu_memory_mb: float = 0.0

    # Optimizer metadata
    optimizer_name: str = ""
    optimizer_phase: str = ""  # e.g. "tba-explore", "tpe-exploit"
    seed: int = 0


@dataclass
class TrialRecord:
    """One complete trial: config + result + metadata."""

    trial_id: int
    config: dict[str, Any]
    result: EvalResult
    best_feasible_so_far: float | None
    cumulative_crashes: int
    cumulative_infeasible: int
    wall_clock_s: float
