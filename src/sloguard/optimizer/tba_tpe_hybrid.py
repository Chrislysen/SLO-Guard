"""TBA-TPE Hybrid optimizer for LLM serving.

Phase 1 (first ~40% of budget): TBA feasible-first SA explores the space,
    maps crash zones, finds feasible configs.
Phase 2 (remaining ~60%): Optuna's real TPE takes over, warm-started with
    ALL Phase 1 history injected via study.add_trial().

Ported from TBA's tba_tpe_hybrid.py, adapted for serving config space.
"""
from __future__ import annotations

import math
import random
from typing import Any

import optuna
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.trial import create_trial

from sloguard.config_space import SearchSpace
from sloguard.optimizer.base import BaseOptimizer
from sloguard.optimizer.subspace_tracker import SubspaceTracker
from sloguard.types import EvalResult

optuna.logging.set_verbosity(optuna.logging.WARNING)


class TBATPEHybrid(BaseOptimizer):
    """TBA for exploration, Optuna TPE for exploitation.

    Phase 1: Feasible-first SA with adaptive temperature.
    Phase 2: Optuna TPE warm-started from Phase 1 history.

    Handoff condition (v3): requires enough feasible + crash data
    + diversity over quantization types before handing off.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        constraints: dict[str, float],
        objective: str = "maximize_goodput",
        budget: int = 30,
        seed: int = 42,
        T_init: float = 1.0,
        T_min: float = 0.001,
        patience: int = 4,
        p_structural_start: float = 0.5,
        p_structural_end: float = 0.25,
        n_initial_random: int = 3,
        enable_blacklisting: bool = True,
        max_consecutive_failures: int = 3,
        cooldown_trials: int = 8,
        min_family_diversity: int = 1,
    ):
        super().__init__(search_space, constraints, objective, budget, seed)
        self.rng = random.Random(seed)

        # Phase 1: TBA state
        self._T = T_init
        self._T_init = T_init
        self._T_min = T_min
        self._patience = patience
        self._no_improve = 0
        self._p_struct_start = p_structural_start
        self._p_struct_end = p_structural_end
        self.n_initial_random = n_initial_random

        self._current_config: dict[str, Any] | None = None
        self._current_violation: float = float("inf")
        self._current_objective: float = float("-inf")
        self._found_feasible = False
        self._tba_phase = "feasibility"

        # Subspace blacklisting (Phase 1 only)
        self._enable_blacklisting = enable_blacklisting
        self._tracker: SubspaceTracker | None = None
        if enable_blacklisting:
            cat_names = [
                v.name for v in search_space.variables.values()
                if v.var_type == "categorical"
            ]
            self._tracker = SubspaceTracker(
                categorical_names=cat_names,
                max_consecutive_failures=max_consecutive_failures,
                cooldown_trials=cooldown_trials,
            )

        # Adaptive handoff conditions — scale with budget so TPE always
        # gets ~60% of the trials for exploitation.
        self._min_tba_trials = max(3, budget // 5)          # at least 3
        self._max_tba_trials = max(5, int(budget * 0.4))    # forced handoff at 40%
        self._min_feasible_for_handoff = max(2, budget // 6)
        self._min_bad_for_handoff = max(1, budget // 10)
        self._min_family_diversity = min_family_diversity
        self._in_tpe_phase = False

        # Phase 2: Optuna (created at handoff)
        self._study: optuna.Study | None = None
        self._pending_trial: optuna.trial.Trial | None = None
        self._distributions: dict[str, Any] | None = None

    @property
    def phase(self) -> str:
        """Current optimizer phase: 'tba-explore' or 'tpe-exploit'."""
        return "tpe-exploit" if self._in_tpe_phase else "tba-explore"

    @property
    def _p_structural(self) -> float:
        progress = self.trial_count / max(self._max_tba_trials, 1)
        progress = min(progress, 1.0)
        return self._p_struct_start + (self._p_struct_end - self._p_struct_start) * progress

    def _should_handoff(self) -> bool:
        """Adaptive handoff: need enough data + diversity."""
        if self.trial_count < self._min_tba_trials:
            return False
        if self.trial_count >= self._max_tba_trials:
            return True
        n_feasible = sum(1 for _, r in self.history if r.feasible and not r.crashed)
        n_bad = sum(1 for _, r in self.history if r.crashed or not r.feasible)
        # Diversity over quantization types (serving equivalent of model families)
        n_quant_types = len({
            c.get("quantization") for c, _ in self.history if "quantization" in c
        })
        return (n_feasible >= self._min_feasible_for_handoff
                and n_bad >= self._min_bad_for_handoff
                and n_quant_types >= self._min_family_diversity)

    # ------------------------------------------------------------------
    # ask
    # ------------------------------------------------------------------

    def ask(self) -> dict[str, Any]:
        if self._in_tpe_phase:
            return self._tpe_ask()
        if self._should_handoff():
            self._handoff_to_tpe()
            self._in_tpe_phase = True
            return self._tpe_ask()
        return self._tba_ask()

    def _get_allowed_values(self) -> dict[str, list[Any]] | None:
        if not self._enable_blacklisting or self._tracker is None:
            return None
        allowed = {}
        for name, v in self.search_space.variables.items():
            if v.var_type == "categorical":
                allowed[name] = self._tracker.get_allowed_values(
                    name, v.choices, self.trial_count,
                )
        return allowed

    def _tba_ask(self) -> dict[str, Any]:
        if self.trial_count < self.n_initial_random:
            return self.search_space.sample_random(self.rng)

        if self.trial_count == self.n_initial_random:
            self._init_tba_from_history()

        if self._current_config is None:
            return self.search_space.sample_random(self.rng)

        allowed = self._get_allowed_values()
        has_combo_blacklists = (
            self._tracker is not None and self._tracker.combo_blacklisted
        )
        candidate = self.search_space.propose_neighbor(
            self._current_config,
            temperature=self._T,
            p_structural=self._p_structural,
            rng=self.rng,
            allowed_values=allowed,
        )
        if has_combo_blacklists:
            for _ in range(9):
                if not self._tracker.is_combo_blacklisted(candidate, self.trial_count):
                    break
                candidate = self.search_space.propose_neighbor(
                    self._current_config,
                    temperature=self._T,
                    p_structural=self._p_structural,
                    rng=self.rng,
                    allowed_values=allowed,
                )
        return candidate

    def _init_tba_from_history(self) -> None:
        best_feas_cfg, best_feas_obj = None, float("-inf")
        best_infeas_cfg, best_infeas_viol = None, float("inf")

        for config, result in self.history:
            if result.crashed:
                continue
            viol = self._compute_violation(result)
            if result.feasible and result.objective_value > best_feas_obj:
                best_feas_obj = result.objective_value
                best_feas_cfg = config
            elif not result.feasible and viol < best_infeas_viol:
                best_infeas_viol = viol
                best_infeas_cfg = config

        if best_feas_cfg is not None:
            self._tba_phase = "optimization"
            self._found_feasible = True
            self._current_config = best_feas_cfg
            self._current_objective = best_feas_obj
            self._current_violation = 0.0
            self._T = self._T_init * 0.7
        elif best_infeas_cfg is not None:
            self._current_config = best_infeas_cfg
            self._current_violation = best_infeas_viol
        else:
            self._current_config = self.search_space.sample_random(self.rng)

    # ------------------------------------------------------------------
    # Handoff: inject TBA history into Optuna
    # ------------------------------------------------------------------

    def _handoff_to_tpe(self) -> None:
        self._distributions = self._build_distributions()
        sampler = optuna.samplers.TPESampler(
            seed=self.seed,
            n_startup_trials=0,
            constraints_func=self._constraints_func,
        )
        self._study = optuna.create_study(direction="maximize", sampler=sampler)
        for config, result in self.history:
            self._inject_trial(config, result)

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

    def _inject_trial(self, config: dict[str, Any], result: EvalResult) -> None:
        params = {}
        active_dists = {}
        for name in self.search_space._all_names:
            if name in config:
                params[name] = config[name]
                active_dists[name] = self._distributions[name]

        violation = 0.0
        if result.crashed:
            violation = 100.0
        else:
            for cname, cap in self.constraints.items():
                measured = result.constraints.get(cname, 0.0)
                violation += max(0.0, measured - cap)

        value = float("-inf") if result.crashed else result.objective_value

        trial = create_trial(
            params=params,
            distributions=active_dists,
            values=[value],
            state=optuna.trial.TrialState.COMPLETE,
            user_attrs={
                "constraint_violation": violation,
                "crashed": result.crashed,
            },
            system_attrs={
                "constraints": [violation if not result.crashed else 100.0],
            },
        )
        self._study.add_trial(trial)

    # ------------------------------------------------------------------
    # Phase 2: Optuna TPE
    # ------------------------------------------------------------------

    def _tpe_ask(self) -> dict[str, Any]:
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

    # ------------------------------------------------------------------
    # tell
    # ------------------------------------------------------------------

    def tell(self, config: dict[str, Any], result: EvalResult) -> None:
        super().tell(config, result)
        if self._in_tpe_phase:
            self._tpe_tell(config, result)
        else:
            self._tba_tell(config, result)

    def _tpe_tell(self, config: dict[str, Any], result: EvalResult) -> None:
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

    def _tba_tell(self, config: dict[str, Any], result: EvalResult) -> None:
        if self._tracker is not None:
            if result.crashed:
                status = "crash"
            elif not result.feasible:
                status = "infeasible"
            else:
                status = "ok"
            self._tracker.record_result(config, status, self.trial_count)

        if self.trial_count <= self.n_initial_random:
            return

        if result.crashed:
            self._update_temp(improved=False)
            return

        violation = self._compute_violation(result)
        obj = result.objective_value

        if self._tba_phase == "feasibility":
            self._tba_tell_feasibility(config, result, violation, obj)
        else:
            self._tba_tell_optimization(config, result, violation, obj)

    def _tba_tell_feasibility(self, config, result, violation, obj):
        improved = violation < self._current_violation

        if result.feasible:
            self._found_feasible = True
            self._tba_phase = "optimization"
            self._current_config = config
            self._current_violation = 0.0
            self._current_objective = obj
            self._T = self._T_init * 0.7
            self._no_improve = 0
            return

        if improved:
            self._current_config = config
            self._current_violation = violation
        else:
            delta = violation - self._current_violation
            if self._T > 0 and delta > 0:
                scale = max(sum(abs(v) for v in self.constraints.values()) * 0.1
                            / max(len(self.constraints), 1), 0.1)
                if self.rng.random() < math.exp(-delta / (self._T * scale)):
                    self._current_config = config
                    self._current_violation = violation

        self._update_temp(improved)

    def _tba_tell_optimization(self, config, result, violation, obj):
        if not result.feasible:
            self._update_temp(improved=False)
            return
        improved = obj > self._current_objective
        if improved:
            self._current_config = config
            self._current_objective = obj
        else:
            delta = self._current_objective - obj
            if self._T > 0 and delta > 0:
                feasible_objs = [r.objective_value for _, r in self.history
                                 if r.feasible and not r.crashed]
                scale = max(max(feasible_objs) - min(feasible_objs), 1e-6) if len(feasible_objs) >= 2 else 1.0
                if self.rng.random() < math.exp(-delta / (self._T * scale)):
                    self._current_config = config
                    self._current_objective = obj
        self._update_temp(improved)

    def _update_temp(self, improved: bool) -> None:
        if improved:
            self._no_improve = 0
            self._T *= 0.92
        else:
            self._no_improve += 1
            if self._no_improve >= self._patience:
                self._T = min(self._T * 2.5, self._T_init * 0.8)
                self._no_improve = 0
        self._T = max(self._T, self._T_min)

    def _compute_violation(self, result: EvalResult) -> float:
        total = 0.0
        for name, cap in self.constraints.items():
            measured = result.constraints.get(name, 0.0)
            total += max(0.0, measured - cap)
        return total
