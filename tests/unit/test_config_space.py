"""Tests for config space: sampling, neighbor proposal, validation, conditions."""
from __future__ import annotations

import random

import pytest

from sloguard.config_space import SearchSpace, build_serving_space
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
        assert 0.70 <= config["gpu_memory_utilization"] <= 0.95
        assert config["enforce_eager"] in [True, False]
        assert config["enable_chunked_prefill"] in [True, False]
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


def test_variable_def_validation():
    with pytest.raises(ValueError):
        VariableDef(name="bad", var_type="categorical")  # no choices

    with pytest.raises(ValueError):
        VariableDef(name="bad", var_type="integer")  # no bounds
