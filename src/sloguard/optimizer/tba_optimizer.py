"""TBA — Thermal Budget Annealing optimizer for LLM serving.

Feasible-first, crash-aware optimization. Two internal phases:
  Phase 1 (Feasibility): Adaptive SA to find a feasible config fast.
  Phase 2 (Optimization): KDE-based TPE within the feasible region,
      with RF surrogate as crash/constraint pre-filter.

Ported from TBA's tba_optimizer.py, adapted for serving config space.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from sloguard.config_space import SearchSpace
from sloguard.optimizer.base import BaseOptimizer
from sloguard.optimizer.feasible_tpe import FeasibleTPESampler
from sloguard.optimizer.subspace_tracker import SubspaceTracker
from sloguard.optimizer.surrogate import RFSurrogate
from sloguard.types import EvalResult


@dataclass
class _TempState:
    """Tracks adaptive temperature and reheating logic."""

    T: float
    T_init: float
    T_min: float
    no_improve_count: int = 0
    patience: int = 5

    def update(self, improved: bool, phase: str) -> None:
        if improved:
            self.no_improve_count = 0
            self.T *= 0.95 if phase == "feasibility" else 0.92
        else:
            self.no_improve_count += 1
            if self.no_improve_count >= self.patience:
                self.T = min(self.T * 2.5, self.T_init * 0.8)
                self.no_improve_count = 0
        self.T = max(self.T, self.T_min)


class TBAOptimizer(BaseOptimizer):
    """Thermal Budget Annealing optimizer for LLM serving configs.

    Phase 1: Feasible-first SA (find ANY feasible config fast)
    Phase 2: Feasible-region TPE (KDE on good/bad feasible split + RF filter)
             Falls back to annealing when < min_feasible_for_tpe observations.
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
        n_initial_random: int = 5,
        restart_interval: int = 12,
        surrogate: bool = True,
        surrogate_min_obs: int = 12,
        p_structural_start: float = 0.5,
        p_structural_end: float = 0.15,
        min_feasible_for_tpe: int = 8,
        enable_blacklisting: bool = True,
        max_consecutive_failures: int = 3,
        cooldown_trials: int = 8,
    ):
        super().__init__(search_space, constraints, objective, budget, seed)
        self.rng = random.Random(seed)
        self.temp = _TempState(T=T_init, T_init=T_init, T_min=T_min, patience=patience)
        self.n_initial_random = n_initial_random
        self.restart_interval = restart_interval

        self._p_struct_start = p_structural_start
        self._p_struct_end = p_structural_end

        # Subspace blacklisting
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

        # RF surrogate
        self.use_surrogate = surrogate
        self.surrogate_min_obs = surrogate_min_obs
        self._surrogate = RFSurrogate(search_space, seed=seed) if surrogate else None
        self._surrogate_refit_interval = 3
        self._obs_since_refit = 0

        # Feasible TPE
        self.min_feasible_for_tpe = min_feasible_for_tpe
        self._ftpe = FeasibleTPESampler(
            search_space, surrogate=self._surrogate, seed=seed,
        ) if surrogate else None
        self._ftpe_refit_interval = 3
        self._feasible_since_refit = 0

        # Phase tracking
        self._phase = "feasibility"
        self._current_config: dict[str, Any] | None = None
        self._current_violation: float = float("inf")
        self._current_objective: float = float("-inf")
        self._best_violation: float = float("inf")
        self._found_feasible = False
        self._initialized = False
        self._steps_since_restart = 0

    @property
    def _budget_progress(self) -> float:
        return self.trial_count / max(self.budget, 1)

    @property
    def _p_structural(self) -> float:
        if self.budget <= self.n_initial_random:
            return self._p_struct_start
        progress = max(0.0, (self.trial_count - self.n_initial_random)
                        / (self.budget - self.n_initial_random))
        return self._p_struct_start + (self._p_struct_end - self._p_struct_start) * progress

    def _get_feasible_history(self) -> list[tuple[dict[str, Any], EvalResult]]:
        return [(c, r) for c, r in self.history if r.feasible and not r.crashed]

    def _get_elite_restart_config(self) -> dict[str, Any]:
        feasible = self._get_feasible_history()
        if len(feasible) < 2:
            return self.search_space.sample_random(self.rng)
        feasible.sort(key=lambda cr: cr[1].objective_value, reverse=True)
        idx = self.rng.choice([1, 2]) if len(feasible) >= 3 else 1
        return feasible[idx][0]

    def _get_allowed_values(self) -> dict[str, list] | None:
        if not self._enable_blacklisting or self._tracker is None:
            return None
        allowed = {}
        for name, v in self.search_space.variables.items():
            if v.var_type == "categorical":
                allowed[name] = self._tracker.get_allowed_values(
                    name, v.choices, self.trial_count,
                )
        return allowed

    # ------------------------------------------------------------------
    # ask
    # ------------------------------------------------------------------

    def ask(self) -> dict[str, Any]:
        if self.trial_count < self.n_initial_random:
            return self.search_space.sample_random(self.rng)

        if not self._initialized:
            self._initialize_from_history()
            self._initialized = True

        if (self._steps_since_restart >= self.restart_interval
                and self._budget_progress < 0.7):
            self._steps_since_restart = 0
            return self._get_elite_restart_config()

        if self._current_config is None:
            return self.search_space.sample_random(self.rng)

        if self._phase == "optimization":
            feasible_hist = self._get_feasible_history()
            if (self.use_surrogate
                    and self._ftpe is not None
                    and len(feasible_hist) >= self.min_feasible_for_tpe):
                return self._feasible_tpe_ask(feasible_hist)

        allowed = self._get_allowed_values()
        has_combo_blacklists = (
            self._tracker is not None and self._tracker.combo_blacklisted
        )
        candidate = self.search_space.propose_neighbor(
            self._current_config,
            temperature=self.temp.T,
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
                    temperature=self.temp.T,
                    p_structural=self._p_structural,
                    rng=self.rng,
                    allowed_values=allowed,
                )
        return candidate

    def _feasible_tpe_ask(self, feasible_hist: list[tuple[dict, EvalResult]]) -> dict[str, Any]:
        if self._obs_since_refit >= self._surrogate_refit_interval:
            self._surrogate.set_history_ref(self.history)
            self._surrogate.fit(self.history)
            self._obs_since_refit = 0

        if self._feasible_since_refit >= self._ftpe_refit_interval:
            self._ftpe.fit(feasible_hist)
            self._feasible_since_refit = 0

        if not self._ftpe._fitted:
            self._ftpe.fit(feasible_hist)
            if not self._ftpe._fitted:
                return self.search_space.propose_neighbor(
                    self._current_config,
                    temperature=self.temp.T,
                    p_structural=self._p_structural,
                    rng=self.rng,
                )

        return self._ftpe.sample()

    # ------------------------------------------------------------------
    # tell
    # ------------------------------------------------------------------

    def tell(self, config: dict[str, Any], result: EvalResult) -> None:
        super().tell(config, result)
        self._steps_since_restart += 1
        self._obs_since_refit += 1

        if result.feasible and not result.crashed:
            self._feasible_since_refit += 1

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
            self.temp.update(improved=False, phase=self._phase)
            return

        violation = self._compute_violation(result)
        obj = result.objective_value

        if self._phase == "feasibility":
            self._tell_feasibility_phase(config, result, violation, obj)
        else:
            self._tell_optimization_phase(config, result, violation, obj)

    def _initialize_from_history(self) -> None:
        best_feasible_config = None
        best_feasible_obj = float("-inf")
        best_infeasible_config = None
        best_infeasible_violation = float("inf")

        for config, result in self.history:
            if result.crashed:
                continue
            violation = self._compute_violation(result)
            if result.feasible:
                if result.objective_value > best_feasible_obj:
                    best_feasible_obj = result.objective_value
                    best_feasible_config = config
            else:
                if violation < best_infeasible_violation:
                    best_infeasible_violation = violation
                    best_infeasible_config = config

        if best_feasible_config is not None:
            self._phase = "optimization"
            self._found_feasible = True
            self._current_config = best_feasible_config
            self._current_objective = best_feasible_obj
            self._current_violation = 0.0
            self.temp.T = self.temp.T_init * 0.7
        elif best_infeasible_config is not None:
            self._current_config = best_infeasible_config
            self._current_violation = best_infeasible_violation
        else:
            self._current_config = self.search_space.sample_random(self.rng)
            self._current_violation = float("inf")

    def _tell_feasibility_phase(self, config, result, violation, obj):
        improved = violation < self._current_violation

        if result.feasible:
            self._found_feasible = True
            self._phase = "optimization"
            self._current_config = config
            self._current_violation = 0.0
            self._current_objective = obj
            self.temp.T = self.temp.T_init * 0.7
            self.temp.no_improve_count = 0
            return

        if improved:
            self._current_config = config
            self._current_violation = violation
            self._best_violation = min(self._best_violation, violation)
        else:
            delta = violation - self._current_violation
            if self.temp.T > 0 and delta > 0:
                accept_prob = math.exp(-delta / (self.temp.T * self._violation_scale()))
                if self.rng.random() < accept_prob:
                    self._current_config = config
                    self._current_violation = violation

        self.temp.update(improved=improved, phase="feasibility")

    def _tell_optimization_phase(self, config, result, violation, obj):
        if not result.feasible:
            self.temp.update(improved=False, phase="optimization")
            return

        improved = obj > self._current_objective

        if improved:
            self._current_config = config
            self._current_objective = obj
        else:
            delta = self._current_objective - obj
            if self.temp.T > 0 and delta > 0:
                scale = self._objective_scale()
                accept_prob = math.exp(-delta / (self.temp.T * scale))
                if self.rng.random() < accept_prob:
                    self._current_config = config
                    self._current_objective = obj

        self.temp.update(improved=improved, phase="optimization")

        snap_prob = 0.4 * (1.0 - self._budget_progress)
        bf = self.best_feasible()
        if bf is not None and self._current_objective < bf[1].objective_value:
            if self.rng.random() < snap_prob:
                self._current_config = bf[0]
                self._current_objective = bf[1].objective_value

    def _compute_violation(self, result: EvalResult) -> float:
        total = 0.0
        for name, cap in self.constraints.items():
            measured = result.constraints.get(name, 0.0)
            total += max(0.0, measured - cap)
        return total

    def _violation_scale(self) -> float:
        cap_vals = [abs(v) for v in self.constraints.values()]
        s = sum(cap_vals) / max(len(cap_vals), 1)
        return max(s * 0.1, 0.1)

    def _objective_scale(self) -> float:
        feasible_objs = [
            r.objective_value for _, r in self.history if r.feasible and not r.crashed
        ]
        if len(feasible_objs) < 2:
            return 1.0
        return max(max(feasible_objs) - min(feasible_objs), 1e-6)
