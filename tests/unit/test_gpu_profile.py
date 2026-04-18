"""Tests for GPU + model profile detection and registry."""
from __future__ import annotations

import pytest

from sloguard import gpu_profile


def test_kv_gb_per_token_known_model():
    assert gpu_profile.kv_gb_per_token_for("Qwen/Qwen2-1.5B") == 0.000096
    assert gpu_profile.kv_gb_per_token_for("meta-llama/Llama-3.1-8B") == 0.000524


def test_kv_gb_per_token_unknown_model_uses_default():
    assert (
        gpu_profile.kv_gb_per_token_for("some/unknown-model")
        == gpu_profile.DEFAULT_KV_GB_PER_TOKEN
    )


def test_kv_gb_per_token_env_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SLOGUARD_KV_BYTES_PER_TOKEN", "0.0005")
    # Override beats the registry — the value is for 1.5B Qwen normally
    assert gpu_profile.kv_gb_per_token_for("Qwen/Qwen2-1.5B") == 0.0005


def test_kv_gb_per_token_invalid_env_falls_back(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
):
    monkeypatch.setenv("SLOGUARD_KV_BYTES_PER_TOKEN", "not-a-number")
    val = gpu_profile.kv_gb_per_token_for("Qwen/Qwen2-1.5B")
    assert val == 0.000096


def test_model_footprint_known_and_unknown():
    assert gpu_profile.model_footprint_gb_for("Qwen/Qwen2-1.5B") == 4.0
    assert (
        gpu_profile.model_footprint_gb_for("some/unknown-model")
        == gpu_profile.DEFAULT_MODEL_FOOTPRINT_GB
    )


def test_detect_gpu_vram_env_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SLOGUARD_GPU_VRAM_GB", "24.0")
    assert gpu_profile.detect_gpu_vram_gb() == 24.0


def test_log_gpu_info_returns_dict_without_gpu(monkeypatch: pytest.MonkeyPatch):
    """When no GPU is detected, log_gpu_info still returns a structured dict."""
    monkeypatch.delenv("SLOGUARD_GPU_VRAM_GB", raising=False)
    # Force detection to fail by hiding both torch and nvidia-smi.
    monkeypatch.setattr(gpu_profile, "detect_gpu_vram_gb", lambda: None)
    monkeypatch.setattr(gpu_profile, "detect_gpu_name", lambda: None)

    info = gpu_profile.log_gpu_info("Qwen/Qwen2-1.5B")
    assert info["vram_gb"] is None
    assert info["gpu_name"] is None
    assert info["kv_gb_per_token"] == 0.000096
    assert info["model_footprint_gb"] == 4.0
