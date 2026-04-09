"""Tests for JSONL trial logger write/read roundtrip."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from sloguard.trial_logger import TrialLogger, load_experiment
from sloguard.types import ServingTrialResult


@pytest.fixture
def tmp_log_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.jsonl"


def _make_trial(trial_id: int, feasible: bool = True, crashed: bool = False) -> ServingTrialResult:
    return ServingTrialResult(
        trial_id=trial_id,
        timestamp="2024-01-01T00:00:00Z",
        experiment_id="test-001",
        config={"quantization": "fp16", "max_num_seqs": 32},
        model_id="Qwen/Qwen2-1.5B",
        gpu_id="RTX_5080",
        workload_type="interactive",
        request_rate=4.0,
        num_requests=100,
        ttft_p99=200.0 if not crashed else None,
        itl_p99=50.0 if not crashed else None,
        tokens_per_sec=500.0 if not crashed else None,
        goodput_tokens_per_sec=450.0 if feasible else None,
        goodput_ratio=0.9 if feasible else None,
        feasible=feasible,
        crashed=crashed,
        crash_type="oom" if crashed else None,
        eval_time_s=60.0,
        slo_ttft_p99_ms=500.0,
        slo_itl_p99_ms=100.0,
        optimizer_name="TBATPEHybrid",
        seed=42,
    )


class TestTrialLogger:
    def test_log_and_load_roundtrip(self, tmp_log_path):
        logger = TrialLogger(tmp_log_path)
        trial = _make_trial(0)
        logger.log(trial)

        loaded = logger.load()
        assert len(loaded) == 1
        assert loaded[0].trial_id == 0
        assert loaded[0].experiment_id == "test-001"
        assert loaded[0].model_id == "Qwen/Qwen2-1.5B"
        assert loaded[0].feasible is True

    def test_multiple_logs(self, tmp_log_path):
        logger = TrialLogger(tmp_log_path)
        logger.log(_make_trial(0))
        logger.log(_make_trial(1, feasible=False))
        logger.log(_make_trial(2, crashed=True))

        loaded = logger.load()
        assert len(loaded) == 3
        assert loaded[0].feasible is True
        assert loaded[1].feasible is False
        assert loaded[2].crashed is True

    def test_count_tracks_session(self, tmp_log_path):
        logger = TrialLogger(tmp_log_path)
        assert logger.count == 0
        logger.log(_make_trial(0))
        assert logger.count == 1
        logger.log(_make_trial(1))
        assert logger.count == 2

    def test_load_empty_file(self, tmp_log_path):
        logger = TrialLogger(tmp_log_path)
        loaded = logger.load()
        assert loaded == []

    def test_load_dicts(self, tmp_log_path):
        logger = TrialLogger(tmp_log_path)
        logger.log(_make_trial(0))

        dicts = logger.load_dicts()
        assert len(dicts) == 1
        assert dicts[0]["trial_id"] == 0

    def test_log_dict_raw(self, tmp_log_path):
        logger = TrialLogger(tmp_log_path)
        logger.log_dict({"trial_id": 99, "custom_field": "hello"})

        dicts = logger.load_dicts()
        assert len(dicts) == 1
        assert dicts[0]["custom_field"] == "hello"

    def test_load_experiment_convenience(self, tmp_log_path):
        logger = TrialLogger(tmp_log_path)
        logger.log(_make_trial(0))
        logger.log(_make_trial(1))

        loaded = load_experiment(tmp_log_path)
        assert len(loaded) == 2

    def test_jsonl_format(self, tmp_log_path):
        logger = TrialLogger(tmp_log_path)
        logger.log(_make_trial(0))

        with open(tmp_log_path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["trial_id"] == 0

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "test.jsonl"
            logger = TrialLogger(path)
            logger.log(_make_trial(0))
            assert path.exists()
