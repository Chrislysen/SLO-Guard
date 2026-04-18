"""Tests for MetricsCollector — focused on the Prometheus parser.

Earlier versions had a dead `patterns` dict and mistakenly keyed
``gpu_memory_peak_mb`` off ``gpu_cache_usage_perc``. These tests pin the
corrected behavior:

  * ``gpu_cache_usage_perc`` → ``kv_cache_utilization``
  * ``gpu_memory_peak_mb`` is NOT populated (vLLM 0.19's /metrics
    doesn't expose absolute peak memory — we'd be fabricating it).
"""
from __future__ import annotations

from sloguard.load_generator import RequestResult
from sloguard.metrics_collector import BenchmarkMetrics, MetricsCollector

SAMPLE_METRICS = """
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model="qwen"} 0.42
# HELP vllm:num_requests_running Number of running requests.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model="qwen"} 3.0
# HELP vllm:num_requests_waiting Number of waiting requests.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model="qwen"} 0.0
"""


def test_parses_kv_cache_utilization():
    collector = MetricsCollector()
    parsed = collector._parse_prometheus(SAMPLE_METRICS)
    assert parsed["kv_cache_utilization"] == 0.42


def test_does_not_fabricate_gpu_memory_peak_mb():
    """vLLM 0.19 doesn't expose absolute peak GPU memory; the parser
    must not invent a value by reusing the KV-cache reading."""
    collector = MetricsCollector()
    parsed = collector._parse_prometheus(SAMPLE_METRICS)
    assert "gpu_memory_peak_mb" not in parsed


def test_update_from_server_only_sets_kv_cache():
    collector = MetricsCollector()
    metrics = BenchmarkMetrics()
    metrics.gpu_memory_peak_mb = None  # explicit: starts None
    collector.update_from_server(metrics, {"kv_cache_utilization": 0.77})
    assert metrics.kv_cache_utilization == 0.77
    assert metrics.gpu_memory_peak_mb is None


def test_update_from_server_ignores_stale_gpu_memory_peak_mb():
    """Even if a caller somehow supplies gpu_memory_peak_mb in the dict
    (e.g., from an older pickled result), update_from_server must not
    write it — the metric has no trustworthy source."""
    collector = MetricsCollector()
    metrics = BenchmarkMetrics()
    collector.update_from_server(
        metrics, {"kv_cache_utilization": 0.3, "gpu_memory_peak_mb": 9999.0},
    )
    assert metrics.kv_cache_utilization == 0.3
    assert metrics.gpu_memory_peak_mb is None


def test_empty_metrics_text_is_safe():
    collector = MetricsCollector()
    assert collector._parse_prometheus("") == {}
    assert collector._parse_prometheus("# just a comment\n") == {}


def test_compute_propagates_peak_concurrency():
    """Fix 1 added peak_concurrency plumbing; confirm it threads through."""
    collector = MetricsCollector()
    metrics = collector.compute(
        [
            RequestResult(
                request_id=0, prompt_tokens=10, output_tokens=5,
                send_time=0.0, first_token_time=0.05,
                token_times=[0.05, 0.1], end_time=0.1, success=True,
            ),
        ],
        peak_concurrency=7,
    )
    assert metrics.peak_concurrency == 7
