"""Constrained Bayesian Optimization baseline.

GP-based BO with separate GPs for the objective and each constraint.
Uses Expected Improvement with Constraints (EIC) acquisition.

Falls back to random search if botorch is not installed, so the
rest of the pipeline doesn't break on CPU-only environments.
"""
from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np

from sloguard.config_space import SearchSpace
from sloguard.optimizer.base import BaseOptimizer
from sloguard.types import EvalResult

logger = logging.getLogger(__name__)

try:
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.optim import optimize_acqf
    from botorch.acquisition.analytic import ExpectedImprovement
    from gpytorch.mlls import ExactMarginalLogLikelihood

    BOTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False


class ConstrainedBOOptimizer(BaseOptimizer):
    """GP-based constrained BO (Gelbart-style unknown constraints).

    Models the objective with a GP and each constraint with a separate GP.
    Acquisition: EI(x) * Product_i(P(c_i(x) <= threshold_i)).

    If botorch is not available, falls back to random search with a warning.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        constraints: dict[str, float],
        objective: str = "maximize_goodput",
        budget: int = 30,
        seed: int = 42,
        n_initial_random: int = 5,
        refit_interval: int = 1,
    ):
        super().__init__(search_space, constraints, objective, budget, seed)
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.n_initial_random = n_initial_random
        self.refit_interval = refit_interval

        # Variable metadata for encoding
        self._var_names = list(search_space._all_names)
        self._var_defs = [search_space.variables[n] for n in self._var_names]
        self._n_dims = len(self._var_names)

        if not BOTORCH_AVAILABLE:
            logger.warning(
                "botorch not installed — ConstrainedBO falls back to random search. "
                "Install with: pip install 'sloguard[bo]'"
            )

    def ask(self) -> dict[str, Any]:
        # Initial random exploration
        if self.trial_count < self.n_initial_random or not BOTORCH_AVAILABLE:
            return self.search_space.sample_random(self.rng)

        # Need at least some non-crashed observations
        valid = [(c, r) for c, r in self.history if not r.crashed]
        if len(valid) < self.n_initial_random:
            return self.search_space.sample_random(self.rng)

        try:
            return self._bo_ask(valid)
        except Exception as e:
            logger.debug("BO acquisition failed (%s), falling back to random", e)
            return self.search_space.sample_random(self.rng)

    def _bo_ask(self, valid_history: list[tuple[dict[str, Any], EvalResult]]) -> dict[str, Any]:
        """Fit GP and optimize acquisition function."""
        # Encode observations
        X_list = []
        Y_obj_list = []

        for config, result in valid_history:
            x = self._encode_normalized(config)
            X_list.append(x)
            Y_obj_list.append(result.objective_value)

        X = torch.tensor(np.array(X_list), dtype=torch.float64)
        Y_obj = torch.tensor(Y_obj_list, dtype=torch.float64).unsqueeze(-1)

        # Standardize objective
        y_mean = Y_obj.mean()
        y_std = Y_obj.std().clamp(min=1e-6)
        Y_obj_std = (Y_obj - y_mean) / y_std

        # Fit objective GP
        obj_model = SingleTaskGP(X, Y_obj_std)
        obj_mll = ExactMarginalLogLikelihood(obj_model.likelihood, obj_model)
        fit_gpytorch_mll(obj_mll)

        # Fit constraint GPs
        constraint_models = {}
        for cname, threshold in self.constraints.items():
            Y_c_list = []
            for _, result in valid_history:
                measured = result.constraints.get(cname, 0.0)
                # Positive = violation, negative = satisfied
                Y_c_list.append(measured - threshold)
            Y_c = torch.tensor(Y_c_list, dtype=torch.float64).unsqueeze(-1)
            c_model = SingleTaskGP(X, Y_c)
            c_mll = ExactMarginalLogLikelihood(c_model.likelihood, c_model)
            fit_gpytorch_mll(c_mll)
            constraint_models[cname] = c_model

        # Best feasible objective (standardized)
        feasible_mask = torch.tensor([
            r.feasible and not r.crashed for _, r in valid_history
        ])
        if feasible_mask.any():
            best_f = Y_obj_std[feasible_mask].max()
        else:
            best_f = Y_obj_std.min()

        # EI acquisition
        ei = ExpectedImprovement(model=obj_model, best_f=best_f)

        # Optimize by generating random candidates and scoring
        n_candidates = 500
        candidates = []
        for _ in range(n_candidates):
            config = self.search_space.sample_random(self.rng)
            x = self._encode_normalized(config)
            candidates.append((config, x))

        X_cand = torch.tensor(
            np.array([x for _, x in candidates]), dtype=torch.float64
        ).unsqueeze(1)

        # EI scores
        with torch.no_grad():
            ei_scores = ei(X_cand)

            # Constraint satisfaction probabilities
            feas_prob = torch.ones(n_candidates, dtype=torch.float64)
            for cname, c_model in constraint_models.items():
                c_pred = c_model.posterior(X_cand.squeeze(1))
                # P(constraint <= 0) = P(satisfied)
                c_mean = c_pred.mean.squeeze(-1)
                c_var = c_pred.variance.squeeze(-1).clamp(min=1e-6)
                # Standard normal CDF approximation
                z = -c_mean / c_var.sqrt()
                p_sat = 0.5 * (1 + torch.erf(z / (2**0.5)))
                feas_prob *= p_sat

            # Combined: EI * P(feasible)
            acq_values = ei_scores * feas_prob

        best_idx = acq_values.argmax().item()
        return candidates[best_idx][0]

    def _encode_normalized(self, config: dict[str, Any]) -> np.ndarray:
        """Encode config to [0, 1]^d for GP."""
        x = np.zeros(self._n_dims)
        for i, (name, v) in enumerate(zip(self._var_names, self._var_defs)):
            val = config.get(name)
            if val is None:
                x[i] = 0.5
            elif v.var_type == "categorical":
                idx = v.choices.index(val) if val in v.choices else 0
                x[i] = idx / max(len(v.choices) - 1, 1)
            elif v.var_type in ("integer", "continuous"):
                span = v.high - v.low
                x[i] = (float(val) - v.low) / span if span > 0 else 0.5
        return x
