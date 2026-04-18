"""Verifies compute_multiseed_stats against the published multiseed JSONLs.

Hardcoded expected values computed from the actual raw floats in
results/multiseed/ (not rounded-to-int). If these ever drift, either the
data changed or the aggregation logic regressed — both are worth a loud
test failure.

Tolerances:
  - means:   1e-6  (pure arithmetic from the files, should match exactly)
  - stds:    1e-3  (statistics.stdev may vary in the last digit across versions)
  - p-values: 5e-4 (scipy's mannwhitneyu is stable to this precision)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "multiseed"


def _load_compute_module():
    """scripts/ isn't a Python package — load compute_multiseed_stats.py directly."""
    path = REPO_ROOT / "scripts" / "compute_multiseed_stats.py"
    spec = importlib.util.spec_from_file_location("compute_multiseed_stats", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compute_multiseed_stats"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def summary():
    if not RESULTS_DIR.exists():
        pytest.skip("multiseed results not present in this checkout")
    compute = _load_compute_module()
    return compute.compute(RESULTS_DIR, fast_threshold_ms=1000.0)


def test_both_optimizers_present(summary):
    assert "random" in summary["per_optimizer"]
    assert "tba-tpe" in summary["per_optimizer"]


def test_feasibility_is_total(summary):
    r = summary["per_optimizer"]["random"]
    t = summary["per_optimizer"]["tba-tpe"]
    assert r["total_trials"] == 75 and r["total_feasible"] == 75
    assert t["total_trials"] == 75 and t["total_feasible"] == 75
    assert r["total_crashes"] == 0 and t["total_crashes"] == 0


def test_per_seed_fast_counts(summary):
    r = summary["per_optimizer"]["random"]["fast_cluster_count"]
    t = summary["per_optimizer"]["tba-tpe"]["fast_cluster_count"]
    # Order is sorted-by-seed: 42, 142, 242, 342, 442
    assert r["values"] == [9, 8, 3, 9, 8]
    assert t["values"] == [11, 9, 11, 11, 11]


def test_fast_cluster_aggregates(summary):
    r = summary["per_optimizer"]["random"]["fast_cluster_count"]
    t = summary["per_optimizer"]["tba-tpe"]["fast_cluster_count"]
    assert r["mean"] == pytest.approx(7.40, abs=1e-6)
    assert t["mean"] == pytest.approx(10.60, abs=1e-6)
    assert r["std"] == pytest.approx(2.509980, abs=1e-3)
    assert t["std"] == pytest.approx(0.894427, abs=1e-3)


def test_post_hit_aggregates(summary):
    r = summary["per_optimizer"]["random"]["post_hit_consistency"]
    t = summary["per_optimizer"]["tba-tpe"]["post_hit_consistency"]
    assert r["mean"] == pytest.approx(0.5394, abs=5e-3)
    assert t["mean"] == pytest.approx(0.8762, abs=5e-3)
    assert r["std"] == pytest.approx(0.224, abs=5e-3)
    assert t["std"] == pytest.approx(0.123, abs=5e-3)


def test_best_latency_aggregates(summary):
    """Raw floats, not rounded — should match the JSONL exactly."""
    r = summary["per_optimizer"]["random"]["best_latency_ms"]
    t = summary["per_optimizer"]["tba-tpe"]["best_latency_ms"]
    assert r["mean"] == pytest.approx(431.11, abs=5e-2)
    assert t["mean"] == pytest.approx(431.57, abs=5e-2)
    assert r["std"] == pytest.approx(1.74, abs=5e-2)
    assert t["std"] == pytest.approx(1.90, abs=5e-2)


def test_first_fast_hit_values(summary):
    r = summary["per_optimizer"]["random"]["first_fast_hit_trial"]
    t = summary["per_optimizer"]["tba-tpe"]["first_fast_hit_trial"]
    assert r["values"] == [4, 4, 3, 3, 1]
    assert t["values"] == [5, 7, 3, 3, 1]
    assert r["mean"] == pytest.approx(3.0, abs=1e-6)
    assert t["mean"] == pytest.approx(3.8, abs=1e-6)


def test_mann_whitney_pvalues(summary):
    tests = summary["mann_whitney"]
    # Fast-cluster: TBA-TPE > Random, one-sided
    assert tests["fast_cluster_count"]["p"] == pytest.approx(0.00794, abs=5e-4)
    # Post-hit: TBA-TPE > Random, one-sided
    assert tests["post_hit_consistency"]["p"] == pytest.approx(0.01039, abs=5e-4)
    # Best latency: two-sided, we're not predicting direction. Result is
    # the "indistinguishable" finding at p ≈ 0.84.
    assert tests["best_latency_ms"]["p"] == pytest.approx(0.84127, abs=5e-4)


def test_mann_whitney_direction(summary):
    """TBA-TPE's mean consistency metrics should exceed Random's."""
    r = summary["per_optimizer"]["random"]
    t = summary["per_optimizer"]["tba-tpe"]
    assert t["fast_cluster_count"]["mean"] > r["fast_cluster_count"]["mean"]
    assert t["post_hit_consistency"]["mean"] > r["post_hit_consistency"]["mean"]
