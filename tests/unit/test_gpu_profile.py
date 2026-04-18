"""Tests for GPU + model profile detection, HF probe, and registry."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sloguard import gpu_profile


@pytest.fixture
def no_hf_probe(monkeypatch: pytest.MonkeyPatch):
    """Force HF probes to return None so registry/default paths are exercised."""
    monkeypatch.setattr(gpu_profile, "kv_gb_from_hf_config", lambda _model: None)
    monkeypatch.setattr(gpu_profile, "footprint_gb_from_hf_cache", lambda _model: None)


def test_kv_gb_per_token_known_model(no_hf_probe):
    assert gpu_profile.kv_gb_per_token_for("Qwen/Qwen2-1.5B") == 0.000096
    assert gpu_profile.kv_gb_per_token_for("meta-llama/Llama-3.1-8B") == 0.000524


def test_kv_gb_per_token_unknown_model_uses_default(no_hf_probe):
    assert (
        gpu_profile.kv_gb_per_token_for("some/unknown-model")
        == gpu_profile.DEFAULT_KV_GB_PER_TOKEN
    )


def test_kv_gb_per_token_env_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SLOGUARD_KV_BYTES_PER_TOKEN", "0.0005")
    # Env override beats both probe and registry.
    assert gpu_profile.kv_gb_per_token_for("Qwen/Qwen2-1.5B") == 0.0005


def test_kv_gb_per_token_invalid_env_falls_back(
    monkeypatch: pytest.MonkeyPatch, no_hf_probe,
):
    monkeypatch.setenv("SLOGUARD_KV_BYTES_PER_TOKEN", "not-a-number")
    assert gpu_profile.kv_gb_per_token_for("Qwen/Qwen2-1.5B") == 0.000096


def test_model_footprint_known_and_unknown(no_hf_probe):
    assert gpu_profile.model_footprint_gb_for("Qwen/Qwen2-1.5B") == 4.0
    assert (
        gpu_profile.model_footprint_gb_for("some/unknown-model")
        == gpu_profile.DEFAULT_MODEL_FOOTPRINT_GB
    )


def test_detect_gpu_vram_env_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SLOGUARD_GPU_VRAM_GB", "24.0")
    assert gpu_profile.detect_gpu_vram_gb() == 24.0


def test_log_gpu_info_returns_dict_without_gpu(
    monkeypatch: pytest.MonkeyPatch, no_hf_probe,
):
    """When no GPU is detected, log_gpu_info still returns a structured dict."""
    monkeypatch.delenv("SLOGUARD_GPU_VRAM_GB", raising=False)
    monkeypatch.setattr(gpu_profile, "detect_gpu_vram_gb", lambda: None)
    monkeypatch.setattr(gpu_profile, "detect_gpu_name", lambda: None)

    info = gpu_profile.log_gpu_info("Qwen/Qwen2-1.5B")
    assert info["vram_gb"] is None
    assert info["gpu_name"] is None
    assert info["kv_gb_per_token"] == 0.000096
    assert info["model_footprint_gb"] == 4.0


# ---------------------------------------------------------------------------
# HF cache probe tests — fake a snapshot dir under a temp HF_HOME
# ---------------------------------------------------------------------------


def _make_fake_snapshot(
    tmp_path: Path,
    model_id: str,
    config: dict,
    weight_bytes: int = 0,
) -> Path:
    """Build a minimal fake HF cache layout and return the snapshot path."""
    safe = "models--" + model_id.replace("/", "--")
    snap = tmp_path / "hub" / safe / "snapshots" / "deadbeef"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps(config))
    if weight_bytes:
        (snap / "model.safetensors").write_bytes(b"\x00" * weight_bytes)
    return snap


def test_kv_gb_from_hf_config_with_explicit_head_dim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    _make_fake_snapshot(tmp_path, "fake/model", {
        "num_hidden_layers": 28,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "torch_dtype": "float16",
    })
    # 2 * 28 * 2 * 128 * 2 / 1e9 = 0.0000286 GB
    val = gpu_profile.kv_gb_from_hf_config("fake/model")
    assert val == pytest.approx(2 * 28 * 2 * 128 * 2 / 1e9)


def test_kv_gb_from_hf_config_derives_head_dim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    _make_fake_snapshot(tmp_path, "fake/model2", {
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "hidden_size": 2048,
        "torch_dtype": "bfloat16",
    })
    # head_dim = 2048 // 16 = 128, kv_heads falls back to num_attention_heads = 16
    val = gpu_profile.kv_gb_from_hf_config("fake/model2")
    assert val == pytest.approx(2 * 16 * 16 * 128 * 2 / 1e9)


def test_kv_gb_from_hf_config_missing_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert gpu_profile.kv_gb_from_hf_config("not/cached") is None


def test_kv_gb_from_hf_config_incomplete_config_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    _make_fake_snapshot(tmp_path, "fake/incomplete", {
        "num_hidden_layers": 28,
        # missing kv heads and head_dim
    })
    assert gpu_profile.kv_gb_from_hf_config("fake/incomplete") is None


def test_footprint_gb_from_hf_cache_sums_weight_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    _make_fake_snapshot(
        tmp_path, "fake/sized", {"num_hidden_layers": 1},
        weight_bytes=1_500_000_000,  # 1.5 GB on disk
    )
    val = gpu_profile.footprint_gb_from_hf_cache("fake/sized")
    # 1.5 GB * 1.4 slack = 2.1 GB
    assert val == pytest.approx(1.5 * 1.4, abs=0.01)


def test_footprint_gb_from_hf_cache_no_weights_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    _make_fake_snapshot(tmp_path, "fake/configonly", {"num_hidden_layers": 1})
    assert gpu_profile.footprint_gb_from_hf_cache("fake/configonly") is None


def test_kv_gb_per_token_prefers_probe_over_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("SLOGUARD_KV_BYTES_PER_TOKEN", raising=False)
    # Build a fake cache for a model that's also in the registry, but with
    # different dimensions so the probe and registry disagree.
    _make_fake_snapshot(tmp_path, "Qwen/Qwen2-1.5B", {
        "num_hidden_layers": 1,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "torch_dtype": "float16",
    })
    val = gpu_profile.kv_gb_per_token_for("Qwen/Qwen2-1.5B")
    # Probe wins: 2 * 1 * 1 * 64 * 2 / 1e9 — much smaller than registry 0.000096
    assert val == pytest.approx(2 * 1 * 1 * 64 * 2 / 1e9)
    assert val != 0.000096
