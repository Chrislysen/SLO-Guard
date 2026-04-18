"""Tests for SLO contract compliance checking and goodput computation."""
from __future__ import annotations

import pytest

from sloguard.slo_contract import SLOContract


@pytest.fixture
def slo():
    return SLOContract(
        ttft_p99_ms=500.0,
        itl_p99_ms=100.0,
        request_latency_p99_ms=30000.0,
        gpu_memory_mb=16000.0,
    )


class TestSLOContract:
    def test_request_meets_all_slos(self, slo):
        assert slo.check_request(ttft_ms=200, itl_max_ms=50, request_latency_ms=5000)

    def test_request_violates_ttft(self, slo):
        assert not slo.check_request(ttft_ms=600, itl_max_ms=50, request_latency_ms=5000)

    def test_request_violates_itl(self, slo):
        assert not slo.check_request(ttft_ms=200, itl_max_ms=150, request_latency_ms=5000)

    def test_request_violates_latency(self, slo):
        assert not slo.check_request(ttft_ms=200, itl_max_ms=50, request_latency_ms=35000)

    def test_request_none_values_pass(self, slo):
        assert slo.check_request()  # all None -> passes

    def test_aggregate_meets_slo(self, slo):
        assert slo.check_aggregate(
            ttft_p99=400, itl_p99=80, request_latency_p99=20000, gpu_memory_mb=12000
        )

    def test_aggregate_violates_memory(self, slo):
        assert not slo.check_aggregate(gpu_memory_mb=20000)

    def test_no_constraint_zero_value(self):
        slo = SLOContract(ttft_p99_ms=0.0, itl_p99_ms=0.0)
        assert slo.check_request(ttft_ms=99999, itl_max_ms=99999)

    def test_headroom_within_budget(self, slo):
        headroom = slo.headroom(ttft_p99=250, itl_p99=50)
        assert headroom["ttft_p99"] == pytest.approx(0.5, abs=0.01)
        assert headroom["itl_p99"] == pytest.approx(0.5, abs=0.01)

    def test_headroom_at_limit(self, slo):
        headroom = slo.headroom(ttft_p99=500)
        assert headroom["ttft_p99"] == pytest.approx(0.0, abs=0.01)

    def test_headroom_violated(self, slo):
        headroom = slo.headroom(ttft_p99=750)
        assert headroom["ttft_p99"] < 0

    def test_to_constraints_dict(self, slo):
        d = slo.to_constraints_dict()
        assert d["ttft_p99_ms"] == 500.0
        assert d["itl_p99_ms"] == 100.0
        assert d["request_latency_p99_ms"] == 30000.0
        assert d["gpu_memory_mb"] == 16000.0

    def test_to_constraints_dict_excludes_zero(self):
        slo = SLOContract(ttft_p99_ms=500, itl_p99_ms=0, gpu_memory_mb=0)
        d = slo.to_constraints_dict()
        assert "ttft_p99_ms" in d
        assert "itl_p99_ms" not in d
        assert "gpu_memory_mb" not in d


class TestComputeGoodput:
    def test_all_meeting(self, slo):
        # All four requests meet every SLO; total tokens = 400, duration = 4s
        requests = [(100, 50, 1000, 100)] * 4
        ratio, tps = slo.compute_goodput(requests, duration_s=4.0)
        assert ratio == 1.0
        assert tps == 100.0  # 400 tokens / 4s

    def test_partial_meeting(self, slo):
        requests = [
            (100, 50, 1000, 100),    # meets
            (600, 50, 1000, 100),    # ttft violated
            (100, 150, 1000, 100),   # itl violated
            (100, 50, 35000, 100),   # latency violated
        ]
        ratio, tps = slo.compute_goodput(requests, duration_s=10.0)
        assert ratio == pytest.approx(0.25)
        assert tps == pytest.approx(10.0)  # only 100 of 400 tokens count

    def test_empty(self, slo):
        ratio, tps = slo.compute_goodput([], duration_s=10.0)
        assert ratio == 0.0
        assert tps == 0.0

    def test_zero_duration(self, slo):
        ratio, tps = slo.compute_goodput([(100, 50, 1000, 100)], duration_s=0.0)
        assert ratio == 1.0
        assert tps == 0.0  # can't compute throughput with zero duration

    def test_none_metrics_ignored(self, slo):
        # Curl benchmark passes None for ttft and itl — should still meet SLO
        # if the latency metric is OK.
        requests = [(None, None, 1000, 100), (None, None, 35000, 100)]
        ratio, tps = slo.compute_goodput(requests, duration_s=10.0)
        assert ratio == 0.5
        assert tps == 10.0
