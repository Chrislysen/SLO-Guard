"""Standard Optuna TPE baseline — no crash awareness.

Cold-start TPE that treats crashes as failed trials with -inf objective.
No Phase 1 feasibility mapping, no surrogate pre-filtering.
This is the "what if you just used Optuna?" comparison.
"""
from __future__ import annotations

from typing import Any

import optuna
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from sloguard.config_space import SearchSpace
from sloguard.optimizer.base import BaseOptimizer
from sloguard.types import EvalResult

optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaColdTPE(BaseOptimizer):
    """Standard Optuna TPE with constraint-aware sampling but no crash learning.

    - No feasibility boundary model
    - No Phase 1 crash-aware exploration
    - Crashes get -inf objective and high constraint violation
    - Relies entirely on TPE's internal modeling
    """

    def __init__(
        self,
        search_space: SearchSpace,
        constraints: dict[str, float],
        objective: str = "maximize_goodput",
        budget: int = 30,
        seed: int = 42,
        n_startup_trials: int = 5,
    ):
        super().__init__(search_space, constraints, objective, budget, seed)

        self._distributions = self._build_distributions()
        sampler = optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=n_startup_trials,
            constraints_func=self._constraints_func,
        )
        self._study = optuna.create_study(direction="maximize", sampler=sampler)
        self._pending_trial: optuna.trial.Trial | None = None

    def _build_distributions(self) -> dict[str, Any]:
        dists = {}
        for name in self.search_space._all_names:
            v = self.search_space.variables[name]
            if v.var_type == "categorical":
                dists[name] = CategoricalDistribution(v.choices)
            elif v.var_type == "integer":
                dists[name] = IntDistribution(int(v.low), int(v.high), log=v.log_scale)
            elif v.var_type == "continuous":
                dists[name] = FloatDistribution(v.low, v.high, log=v.log_scale)
        return dists

    def ask(self) -> dict[str, Any]:
        trial = self._study.ask()
        config = self._sample_from_trial(trial)
        self._pending_trial = trial
        return config

    def _sample_from_trial(self, trial: optuna.trial.Trial) -> dict[str, Any]:
        config: dict[str, Any] = {}
        space = self.search_space

        for name in space._all_names:
            v = space.variables[name]
            if v.condition is not None:
                if not space._eval_condition(v.condition, config):
                    continue

            if v.var_type == "categorical":
                config[name] = trial.suggest_categorical(name, v.choices)
            elif v.var_type == "integer":
                config[name] = trial.suggest_int(name, int(v.low), int(v.high), log=v.log_scale)
            elif v.var_type == "continuous":
                config[name] = trial.suggest_float(name, v.low, v.high, log=v.log_scale)

        return config

    @staticmethod
    def _constraints_func(frozen_trial: optuna.trial.FrozenTrial) -> list[float]:
        violation = frozen_trial.user_attrs.get("constraint_violation", 0.0)
        crashed = frozen_trial.user_attrs.get("crashed", False)
        return [100.0] if crashed else [violation]

    def tell(self, config: dict[str, Any], result: EvalResult) -> None:
        super().tell(config, result)

        if self._pending_trial is None:
            return

        violation = 0.0
        if result.crashed:
            violation = 100.0
        else:
            for name, cap in self.constraints.items():
                measured = result.constraints.get(name, 0.0)
                violation += max(0.0, measured - cap)

        self._pending_trial.set_user_attr("constraint_violation", violation)
        self._pending_trial.set_user_attr("crashed", result.crashed)

        value = float("-inf") if result.crashed else result.objective_value
        self._study.tell(self._pending_trial, value)
        self._pending_trial = None
