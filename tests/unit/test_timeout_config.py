"""Tests for TimeoutConfig dataclass and its propagation through LoadGenerator."""
from __future__ import annotations

from sloguard.load_generator import (
    BurstGenerator,
    FixedRateGenerator,
    WorkloadConfig,
    create_generator,
)
from sloguard.types import TimeoutConfig


def test_defaults_match_legacy_magic_numbers():
    """Defaults preserve the values that were previously hardcoded across
    the codebase. Anyone reading old logs should see the same caps."""
    t = TimeoutConfig()
    assert t.per_request_s == 60.0
    assert t.per_trial_s == 180.0
    assert t.server_start_s == 120.0
    assert t.preflight_s == 30.0


def test_load_generator_uses_supplied_timeouts():
    workload = WorkloadConfig(request_rate=1.0, num_requests=1, model="test")
    timeouts = TimeoutConfig(per_request_s=10.0, per_trial_s=20.0)
    gen = FixedRateGenerator("http://x", workload, seed=1, timeouts=timeouts)
    assert gen.timeouts.per_request_s == 10.0
    assert gen.timeouts.per_trial_s == 20.0


def test_load_generator_falls_back_to_defaults():
    workload = WorkloadConfig(request_rate=1.0, num_requests=1, model="test")
    gen = FixedRateGenerator("http://x", workload, seed=1)
    assert gen.timeouts.per_request_s == 60.0
    assert gen.timeouts.per_trial_s == 180.0


def test_create_generator_threads_timeouts_through():
    workload = WorkloadConfig(request_rate=1.0, num_requests=1, model="test")
    timeouts = TimeoutConfig(per_request_s=5.0)

    gen = create_generator("fixed", "http://x", workload, seed=1, timeouts=timeouts)
    assert isinstance(gen, FixedRateGenerator)
    assert gen.timeouts.per_request_s == 5.0

    gen = create_generator("burst", "http://x", workload, seed=1, timeouts=timeouts)
    assert isinstance(gen, BurstGenerator)
    assert gen.timeouts.per_request_s == 5.0


def test_burst_generator_keyword_args_still_work():
    """Adding the timeouts param shouldn't break the existing kwargs pattern."""
    workload = WorkloadConfig(request_rate=1.0, num_requests=1, model="test")
    gen = create_generator(
        "burst", "http://x", workload, seed=1,
        baseline_rate=2.0, peak_rate=10.0,
    )
    assert isinstance(gen, BurstGenerator)
    assert gen.baseline_rate == 2.0
    assert gen.peak_rate == 10.0
