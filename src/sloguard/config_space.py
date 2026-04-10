"""Hierarchical mixed-variable search space with conditional dependencies.

Ported from TBA's search_space.py, extended with vLLM serving config factory.
"""
from __future__ import annotations

import math
import random
from copy import deepcopy
from typing import Any

from sloguard.types import VariableDef


class SearchSpace:
    """Manages a hierarchical search space with categorical, integer, and continuous
    variables that may have conditional activation dependencies."""

    def __init__(self, variables: list[VariableDef]):
        self.variables = {v.name: v for v in variables}
        self._all_names = [v.name for v in variables]

    def active_variables(self, config: dict[str, Any]) -> list[str]:
        """Return names of variables whose conditions are satisfied by *config*."""
        active = []
        for name in self._all_names:
            v = self.variables[name]
            if v.condition is None or self._eval_condition(v.condition, config):
                active.append(name)
        return active

    def sample_random(self, rng: random.Random | None = None) -> dict[str, Any]:
        """Sample a valid random configuration respecting hierarchy."""
        rng = rng or random.Random()
        config: dict[str, Any] = {}

        for _ in range(10):
            changed = False
            for name in self._all_names:
                if name in config:
                    continue
                v = self.variables[name]
                if v.condition is None or self._eval_condition(v.condition, config):
                    config[name] = self._sample_variable(v, rng)
                    changed = True
            if not changed:
                break
        return config

    def propose_neighbor(
        self,
        config: dict[str, Any],
        temperature: float,
        p_structural: float = 0.3,
        rng: random.Random | None = None,
        allowed_values: dict[str, list[Any]] | None = None,
    ) -> dict[str, Any]:
        """Propose a neighboring configuration.

        With probability *p_structural*, mutate a categorical/structural variable.
        Otherwise mutate a numeric variable with step size proportional to temperature.
        """
        rng = rng or random.Random()
        neighbor = deepcopy(config)
        active = self.active_variables(config)

        structural = [n for n in active if self.variables[n].var_type == "categorical"]
        numeric = [n for n in active if self.variables[n].var_type in ("integer", "continuous")]

        if not active:
            return self.sample_random(rng)

        if structural and (not numeric or rng.random() < p_structural):
            name = rng.choice(structural)
            v = self.variables[name]
            choices = v.choices
            if allowed_values and name in allowed_values:
                choices = allowed_values[name]
            other_choices = [c for c in choices if c != config.get(name)]
            if other_choices:
                neighbor[name] = rng.choice(other_choices)
            self._resolve_hierarchy(neighbor, rng)
        elif numeric:
            name = rng.choice(numeric)
            v = self.variables[name]
            neighbor[name] = self._perturb_numeric(v, config[name], temperature, rng)

        return neighbor

    def config_distance(self, a: dict[str, Any], b: dict[str, Any]) -> float:
        """Normalized distance between two configs, handling mixed types."""
        all_names = set(self.active_variables(a)) | set(self.active_variables(b))
        if not all_names:
            return 0.0

        total = 0.0
        count = 0
        for name in all_names:
            v = self.variables[name]
            va = a.get(name)
            vb = b.get(name)
            if va is None or vb is None:
                total += 1.0
                count += 1
                continue

            if v.var_type == "categorical":
                total += 0.0 if va == vb else 1.0
            elif v.var_type in ("integer", "continuous"):
                span = v.high - v.low
                if span > 0:
                    total += abs(va - vb) / span
                else:
                    total += 0.0
            count += 1

        return total / count if count > 0 else 0.0

    def is_valid(self, config: dict[str, Any]) -> bool:
        """Check that all active variables are present and within bounds."""
        active = set(self.active_variables(config))
        for name in active:
            if name not in config:
                return False
            v = self.variables[name]
            val = config[name]
            if v.var_type == "categorical" and val not in v.choices:
                return False
            if v.var_type == "integer":
                if not isinstance(val, int) or val < v.low or val > v.high:
                    return False
            if v.var_type == "continuous":
                if not isinstance(val, (int, float)) or val < v.low or val > v.high:
                    return False
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _eval_condition(self, condition: str, config: dict[str, Any]) -> bool:
        try:
            return bool(eval(condition, {"__builtins__": {}}, config))  # noqa: S307
        except Exception:
            return False

    def _sample_variable(self, v: VariableDef, rng: random.Random) -> Any:
        if v.var_type == "categorical":
            return rng.choice(v.choices)
        elif v.var_type == "integer":
            if v.log_scale:
                log_low = math.log2(max(v.low, 1))
                log_high = math.log2(v.high)
                val = 2 ** rng.uniform(log_low, log_high)
                val = int(2 ** round(math.log2(max(val, 1))))
                return max(int(v.low), min(int(v.high), val))
            else:
                return rng.randint(int(v.low), int(v.high))
        elif v.var_type == "continuous":
            if v.log_scale:
                log_low = math.log(max(v.low, 1e-10))
                log_high = math.log(v.high)
                return math.exp(rng.uniform(log_low, log_high))
            else:
                return rng.uniform(v.low, v.high)
        raise ValueError(f"Unknown var_type: {v.var_type}")

    def _perturb_numeric(
        self, v: VariableDef, current: Any, temperature: float, rng: random.Random
    ) -> Any:
        span = v.high - v.low
        sigma = temperature * span * 0.3

        if v.var_type == "integer":
            step = max(1, int(round(rng.gauss(0, sigma))))
            new_val = current + rng.choice([-1, 1]) * step
            return max(int(v.low), min(int(v.high), int(new_val)))
        else:
            new_val = current + rng.gauss(0, sigma)
            return max(v.low, min(v.high, new_val))

    def _resolve_hierarchy(self, config: dict[str, Any], rng: random.Random) -> None:
        """After a structural change, drop inactive vars and sample newly active ones."""
        active = set(self.active_variables(config))
        to_drop = [k for k in list(config.keys()) if k not in active and k in self.variables]
        for k in to_drop:
            del config[k]
        for _ in range(10):
            changed = False
            for name in self._all_names:
                if name in config:
                    continue
                v = self.variables[name]
                if v.condition is None or self._eval_condition(v.condition, config):
                    if name in active or name not in config:
                        config[name] = self._sample_variable(v, rng)
                        changed = True
            if not changed:
                break


def fix_serving_config(config: dict[str, Any]) -> dict[str, Any]:
    """Fix known vLLM constraint violations in a config.

    vLLM 0.19 requires:
      - max_num_batched_tokens >= max_num_seqs
      - max_num_batched_tokens >= max_model_len
      - enforce_eager + enable_chunked_prefill must not both be True
      - gpu_memory_utilization must be a clean float for CLI
    """
    # --- Boolean conflict: enforce_eager + chunked_prefill ---
    # vLLM 0.19 returns internal 500s when both are True.
    if config.get("enforce_eager") and config.get("enable_chunked_prefill"):
        config["enable_chunked_prefill"] = False

    # --- Batching constraints ---
    max_seqs = config.get("max_num_seqs", 8)
    max_model_len = config.get("max_model_len", 512)
    max_batched = config.get("max_num_batched_tokens", 512)

    # max_num_batched_tokens must be >= both max_num_seqs and max_model_len
    required_min = max(max_seqs, max_model_len)
    if max_batched < required_min:
        config["max_num_batched_tokens"] = required_min

    # --- Clean up gpu_memory_utilization for CLI ---
    if "gpu_memory_utilization" in config:
        config["gpu_memory_utilization"] = round(config["gpu_memory_utilization"], 2)

    return config


def build_serving_space(
    quantization_choices: list[str] | None = None,
) -> SearchSpace:
    """Build the vLLM 0.19 serving configuration search space.

    Tunable knobs chosen to produce configs that vLLM actually accepts.
    Removed knobs that mostly cause crashes without meaningful
    performance variation (swap_space/cpu-offload, block_size).

    The remaining 7 knobs cover the key performance axes:
      - Batching: max_num_seqs, max_num_batched_tokens
      - Memory: gpu_memory_utilization, max_model_len
      - Execution: enforce_eager, enable_chunked_prefill, enable_prefix_caching

    Args:
        quantization_choices: Quantization methods available for the model.
            Defaults to ["fp16"] (safest for any model).
    """
    if quantization_choices is None:
        quantization_choices = ["fp16"]

    variables = [
        VariableDef(
            name="quantization",
            var_type="categorical",
            choices=quantization_choices,
        ),
        # Batching — the main throughput/latency tradeoff
        VariableDef(
            name="max_num_seqs",
            var_type="integer",
            low=4,
            high=128,
            log_scale=True,
        ),
        VariableDef(
            name="max_num_batched_tokens",
            var_type="integer",
            low=512,
            high=8192,
            log_scale=True,
        ),
        # Memory — how much GPU to use for KV cache.
        # Lower bound 0.50 accommodates GPUs with display/system overhead
        # (e.g. laptop GPUs with ~5 GB used by display driver).
        VariableDef(
            name="gpu_memory_utilization",
            var_type="continuous",
            low=0.50,
            high=0.95,
        ),
        # Context length — trades memory for capability
        VariableDef(
            name="max_model_len",
            var_type="integer",
            low=512,
            high=4096,
            log_scale=True,
        ),
        # Execution mode — CUDA graphs vs eager
        VariableDef(
            name="enforce_eager",
            var_type="categorical",
            choices=[True, False],
        ),
        # Chunked prefill — reduces TTFT variance on long prompts.
        # Only active when enforce_eager is False; the combination
        # enforce_eager + chunked_prefill causes internal 500s in vLLM 0.19.
        VariableDef(
            name="enable_chunked_prefill",
            var_type="categorical",
            choices=[True, False],
            condition="enforce_eager == False",
        ),
        # Prefix caching — helps with repeated prompts
        VariableDef(
            name="enable_prefix_caching",
            var_type="categorical",
            choices=[True, False],
        ),
    ]
    return SearchSpace(variables)
