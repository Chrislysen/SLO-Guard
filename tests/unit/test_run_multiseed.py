"""Tests for the pure pieces of scripts/run_multiseed.py.

Running the full orchestrator requires vLLM + a GPU. These tests pin
the JSONL schema, the resume-from-disk parse, and the record-to-EvalResult
replay that restores optimizer state after a Colab disconnect.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from sloguard.types import EvalResult

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    """scripts/ isn't a package; load run_multiseed.py directly."""
    path = REPO_ROOT / "scripts" / "run_multiseed.py"
    spec = importlib.util.spec_from_file_location("run_multiseed", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_multiseed"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


# -----------------------------------------------------------------------------
# Record shape — must stay compatible with compute_multiseed_stats.py
# -----------------------------------------------------------------------------


def test_build_record_feasible_includes_all_expected_fields(mod):
    result = EvalResult(
        feasible=True, crashed=False,
        goodput_tokens_per_sec=220.0,
        request_latency_mean_ms=450.0,
        request_latency_p50_ms=430.0,
        total_output_tokens=500,
        utility_value=200.0,
        eval_time_s=3.0,
        server_startup_time_s=15.0,
        peak_concurrency=12,
    )
    rec = mod._build_record(
        trial_id=0, seed=42, optimizer_name="tba-tpe",
        config={"max_num_seqs": 32},
        phase="tba-explore",
        result=result,
    )
    assert rec["trial"] == 1  # 1-indexed
    assert rec["seed"] == 42
    assert rec["optimizer"] == "tba-tpe"
    assert rec["phase"] == "tba-explore"
    assert rec["status"] == "feasible"
    assert rec["avg_latency_ms"] == 450.0  # mean preferred over p50
    assert rec["goodput_tps"] == 220.0
    assert rec["total_tokens"] == 500
    assert rec["utility_value"] == 200.0
    assert rec["peak_concurrency"] == 12
    assert rec["config"] == {"max_num_seqs": 32}


def test_build_record_falls_back_to_p50_when_mean_missing(mod):
    """If mean wasn't computed (e.g. no feasible latencies), fall back to p50."""
    result = EvalResult(
        feasible=True, crashed=False,
        request_latency_mean_ms=None,
        request_latency_p50_ms=431.0,
    )
    rec = mod._build_record(0, 42, "random", {}, "", result)
    assert rec["avg_latency_ms"] == 431.0


def test_build_record_crash_has_no_latency(mod):
    result = EvalResult(
        crashed=True, crash_type="oom",
        error_msg="CUDA OOM at layer 5",
        eval_time_s=2.5,
    )
    rec = mod._build_record(3, 142, "random", {"max_num_seqs": 128}, "", result)
    assert rec["status"] == "crash"
    assert "avg_latency_ms" not in rec
    assert "goodput_tps" not in rec
    assert rec["crash_type"] == "oom"
    assert rec["error_msg"] == "CUDA OOM at layer 5"
    assert rec["eval_time_s"] == 2.5


def test_build_record_roundtrips_through_json(mod):
    """What we write must be valid JSON and deserialize to the same record."""
    result = EvalResult(
        feasible=True, crashed=False,
        goodput_tokens_per_sec=100.0,
        request_latency_mean_ms=500.0,
        total_output_tokens=500,
        utility_value=95.0,
    )
    rec = mod._build_record(0, 42, "random", {"k": "v"}, "", result)
    roundtrip = json.loads(json.dumps(rec))
    assert roundtrip == rec


# -----------------------------------------------------------------------------
# Resume: _load_existing + _record_to_eval_result
# -----------------------------------------------------------------------------


def test_load_existing_missing_file_returns_empty(mod, tmp_path):
    assert mod._load_existing(tmp_path / "nope.jsonl") == []


def test_load_existing_skips_blank_and_malformed_lines(mod, tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text(
        '{"trial":1,"status":"feasible"}\n'
        '\n'
        'this is not json\n'
        '{"trial":2,"status":"crash"}\n'
    )
    records = mod._load_existing(path)
    assert [r["trial"] for r in records] == [1, 2]


def test_record_to_eval_result_feasible(mod):
    rec = {
        "status": "feasible",
        "avg_latency_ms": 440.0,
        "goodput_tps": 225.0,
        "eval_time_s": 3.5,
        "server_startup_time_s": 12.0,
    }
    r = mod._record_to_eval_result(rec)
    assert r.feasible is True
    assert r.crashed is False
    assert r.goodput_tokens_per_sec == 225.0
    assert r.request_latency_mean_ms == 440.0
    # objective_value used for optimizer.best_feasible ranking
    assert r.objective_value == 225.0
    # latency propagated into constraints so surrogates can re-learn it
    assert r.constraints["request_latency_p99_ms"] == 440.0


def test_record_to_eval_result_crash(mod):
    rec = {"status": "crash", "crash_type": "oom", "eval_time_s": 1.0}
    r = mod._record_to_eval_result(rec)
    assert r.crashed is True
    assert r.feasible is False
    assert r.crash_type == "oom"
    assert r.objective_value == float("-inf")


def test_record_to_eval_result_infeasible(mod):
    rec = {"status": "infeasible", "avg_latency_ms": 3000.0}
    r = mod._record_to_eval_result(rec)
    assert r.crashed is False
    assert r.feasible is False
    assert r.objective_value == 0.0


def test_record_to_eval_result_prefers_utility_when_present(mod):
    """Records written in utility mode carry utility_value; replay must
    restore objective_value from it so the optimizer's best_feasible
    ranking stays self-consistent."""
    rec = {
        "status": "feasible",
        "avg_latency_ms": 440.0,
        "goodput_tps": 225.0,
        "utility_value": 200.0,
        "eval_time_s": 3.5,
    }
    r = mod._record_to_eval_result(rec)
    assert r.objective_value == 200.0  # utility beats goodput
    assert r.goodput_tokens_per_sec == 225.0  # still available for reporting


def test_record_to_eval_result_missing_everything_safe(mod):
    """A minimal record shouldn't blow up replay — optimizers should
    still be able to advance their internal trial counter."""
    r = mod._record_to_eval_result({"status": "feasible"})
    assert r.feasible is True
    # Goodput absent → objective_value = 0.0, feasible=True enough for tell()
    assert r.objective_value == 0.0


# -----------------------------------------------------------------------------
# Durable append — write must land on disk before we advance
# -----------------------------------------------------------------------------


def test_durable_append_visible_to_fresh_reader(mod, tmp_path):
    path = tmp_path / "log.jsonl"
    mod._durable_append(path, '{"trial": 1}\n')
    mod._durable_append(path, '{"trial": 2}\n')
    content = path.read_text()
    assert content == '{"trial": 1}\n{"trial": 2}\n'
