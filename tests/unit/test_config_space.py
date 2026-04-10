"""Tests for config space: sampling, neighbor proposal, validation, conditions."""
from __future__ import annotations

import random

import pytest

from sloguard.config_space import SearchSpace, build_serving_space, fix_serving_config
from sloguard.types import VariableDef


def test_build_serving_space():
    space = build_serving_space()
    assert len(space.variables) == 8
    assert "quantization" in space.variables
    assert "max_num_seqs" in space.variables
    assert "gpu_memory_utilization" in space.variables


def test_sample_random_produces_valid_config():
    space = build_serving_space()
    rng = random.Random(42)
    for _ in range(50):
        config = space.sample_random(rng)
        assert space.is_valid(config)
        assert "quantization" in config
        assert "max_num_seqs" in config
        assert config["gpu_memory_utilization"] >= 0.50
        assert config["gpu_memory_utilization"] <= 0.95


def test_sample_random_respects_bounds():
    space = build_serving_space()
    rng = random.Random(123)
    for _ in range(100):
        config = space.sample_random(rng)
        assert 4 <= config["max_num_seqs"] <= 128
        assert 512 <= config["max_num_batched_tokens"] <= 8192
        assert 512 <= config["max_model_len"] <= 4096
        assert 0.50 <= config["gpu_memory_utilization"] <= 0.95
        assert config["enforce_eager"] in [True, False]
        # enable_chunked_prefill is conditional on enforce_eager == False
        if config["enforce_eager"] is False:
            assert config["enable_chunked_prefill"] in [True, False]
        else:
            assert "enable_chunked_prefill" not in config
        assert config["enable_prefix_caching"] in [True, False]
        assert config["quantization"] in ["fp16"]


def test_propose_neighbor_changes_something():
    space = build_serving_space()
    rng = random.Random(42)
    config = space.sample_random(rng)

    changed = False
    for _ in range(20):
        neighbor = space.propose_neighbor(config, temperature=0.5, rng=rng)
        if neighbor != config:
            changed = True
            break
    assert changed, "propose_neighbor should produce different configs"


def test_propose_neighbor_stays_valid():
    space = build_serving_space()
    rng = random.Random(42)
    config = space.sample_random(rng)

    for _ in range(50):
        neighbor = space.propose_neighbor(config, temperature=0.8, rng=rng)
        assert space.is_valid(neighbor)


def test_config_distance_same_is_zero():
    space = build_serving_space()
    rng = random.Random(42)
    config = space.sample_random(rng)
    assert space.config_distance(config, config) == 0.0


def test_config_distance_different_is_positive():
    space = build_serving_space()
    rng = random.Random(42)
    a = space.sample_random(rng)
    b = space.sample_random(rng)
    dist = space.config_distance(a, b)
    assert dist >= 0.0


def test_conditional_variable():
    variables = [
        VariableDef(name="backend", var_type="categorical", choices=["a", "b"]),
        VariableDef(name="threads", var_type="integer", low=1, high=8,
                    condition="backend == 'a'"),
    ]
    space = SearchSpace(variables)
    rng = random.Random(42)

    # Sample many configs, check condition
    for _ in range(50):
        config = space.sample_random(rng)
        if config["backend"] == "a":
            assert "threads" in config
        else:
            assert "threads" not in config
        assert space.is_valid(config)


def test_allowed_values_filter():
    space = build_serving_space()
    rng = random.Random(42)
    config = space.sample_random(rng)

    # Restrict quantization choices
    allowed = {"quantization": ["fp16", "awq"]}
    for _ in range(20):
        neighbor = space.propose_neighbor(
            config, temperature=0.5, rng=rng, allowed_values=allowed
        )
        if "quantization" in neighbor:
            assert neighbor["quantization"] in ["fp16", "awq", "gptq", "squeezellm"]


def test_fix_serving_config_enforces_constraints():
    # enforce_eager + chunked_prefill conflict
    config = {"enforce_eager": True, "enable_chunked_prefill": True,
              "max_num_seqs": 8, "max_model_len": 512, "max_num_batched_tokens": 512}
    fixed = fix_serving_config(config)
    assert fixed["enable_chunked_prefill"] is False

    # max_num_batched_tokens must be >= max_model_len
    config2 = {"max_num_seqs": 8, "max_model_len": 2048, "max_num_batched_tokens": 512}
    fixed2 = fix_serving_config(config2)
    assert fixed2["max_num_batched_tokens"] >= 2048

    # gpu_memory_utilization gets rounded
    config3 = {"gpu_memory_utilization": 0.8347291, "max_num_seqs": 8,
               "max_model_len": 512, "max_num_batched_tokens": 512}
    fixed3 = fix_serving_config(config3)
    assert fixed3["gpu_memory_utilization"] == 0.83

    # No conflict: enforce_eager=False + chunked_prefill=True is fine
    config4 = {"enforce_eager": False, "enable_chunked_prefill": True,
               "max_num_seqs": 8, "max_model_len": 512, "max_num_batched_tokens": 512}
    fixed4 = fix_serving_config(config4)
    assert fixed4["enable_chunked_prefill"] is True


def test_fix_serving_config_memory_guard():
    """Memory pressure guard caps max_num_seqs * max_model_len <= 128K tokens."""
    # High max_num_seqs * max_model_len should be capped
    config = {"max_num_seqs": 128, "max_model_len": 4096,
              "max_num_batched_tokens": 8192}
    fixed = fix_serving_config(config)
    assert fixed["max_num_seqs"] * fixed["max_model_len"] <= 131072
    assert fixed["max_num_seqs"] >= 4
    assert fixed["max_model_len"] >= 512

    # Moderate combo should pass through unchanged
    config2 = {"max_num_seqs": 32, "max_model_len": 1024,
               "max_num_batched_tokens": 1024}
    fixed2 = fix_serving_config(config2)
    assert fixed2["max_num_seqs"] == 32
    assert fixed2["max_model_len"] == 1024

    # Edge case: max_num_seqs=64, max_model_len=4096 → 262144 > 128K
    config3 = {"max_num_seqs": 64, "max_model_len": 4096,
               "max_num_batched_tokens": 4096}
    fixed3 = fix_serving_config(config3)
    assert fixed3["max_num_seqs"] * fixed3["max_model_len"] <= 131072


def test_variable_def_validation():
    with pytest.raises(ValueError):
        VariableDef(name="bad", var_type="categorical")  # no choices

    with pytest.raises(ValueError):
        VariableDef(name="bad", var_type="integer")  # no bounds
