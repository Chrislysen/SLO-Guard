"""Feasible-region TPE sampler.

Splits feasible history into good (top 30%) and bad (bottom 70%) by objective,
fits per-variable KDEs on each group, and samples from the good distribution
weighted by l(x)/g(x). RF surrogate filters out predicted-infeasible candidates.

Ported from TBA's feasible_tpe.py.
"""
from __future__ import annotations

import math
import random as stdlib_random
from typing import Any

import numpy as np
from scipy.stats import gaussian_kde

from sloguard.config_space import SearchSpace
from sloguard.optimizer.surrogate import RFSurrogate
from sloguard.types import EvalResult


class FeasibleTPESampler:
    """KDE-based sampler operating only within the feasible region.

    Splits feasible observations into good/bad by objective, fits marginal
    KDEs per variable, and samples from l(x)/g(x).
    """

    def __init__(
        self,
        search_space: SearchSpace,
        surrogate: RFSurrogate | None,
        seed: int = 42,
        gamma: float = 0.3,
        n_candidates: int = 24,
        bandwidth_factor: float = 1.0,
    ):
        self.space = search_space
        self.surrogate = surrogate
        self.rng = stdlib_random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.gamma = gamma
        self.n_candidates = n_candidates
        self.bw_factor = bandwidth_factor

        self._var_names = list(search_space._all_names)
        self._var_defs = [search_space.variables[n] for n in self._var_names]

        self._good_data: dict[str, np.ndarray] = {}
        self._bad_data: dict[str, np.ndarray] = {}
        self._good_kdes: dict[str, gaussian_kde | None] = {}
        self._bad_kdes: dict[str, gaussian_kde | None] = {}
        self._fitted = False

    def fit(self, feasible_history: list[tuple[dict[str, Any], EvalResult]]) -> bool:
        """Fit KDEs on feasible history split into good/bad."""
        if len(feasible_history) < 4:
            self._fitted = False
            return False

        sorted_hist = sorted(
            feasible_history, key=lambda cr: cr[1].objective_value, reverse=True
        )
        n_good = max(2, int(len(sorted_hist) * self.gamma))
        good = sorted_hist[:n_good]
        bad = sorted_hist[n_good:]
        if len(bad) < 2:
            bad = sorted_hist[n_good - 1:]

        self._good_data = {}
        self._bad_data = {}
        self._good_kdes = {}
        self._bad_kdes = {}

        for name, vdef in zip(self._var_names, self._var_defs):
            g_vals = self._extract_values(good, name, vdef)
            b_vals = self._extract_values(bad, name, vdef)

            self._good_data[name] = g_vals
            self._bad_data[name] = b_vals
            self._good_kdes[name] = self._fit_kde(g_vals, vdef)
            self._bad_kdes[name] = self._fit_kde(b_vals, vdef)

        self._fitted = True
        return True

    def sample(self) -> dict[str, Any]:
        """Sample a config from the good distribution, filtered by surrogate."""
        if not self._fitted:
            return self.space.sample_random(self.rng)

        candidates = []
        for _ in range(self.n_candidates):
            config = self._sample_from_good()
            score = self._tpe_score(config)
            candidates.append((config, score))

        if self.surrogate is not None and self.surrogate.is_ready:
            scored = []
            for config, tpe_score in candidates:
                _, p_feas = self.surrogate.predict(config)
                combined = tpe_score * (0.3 + 0.7 * p_feas)
                scored.append((config, combined))
            candidates = scored

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def _extract_values(
        self,
        history: list[tuple[dict[str, Any], EvalResult]],
        name: str,
        vdef: Any,
    ) -> np.ndarray:
        vals = []
        for config, _ in history:
            v = config.get(name)
            if v is None:
                continue
            if vdef.var_type == "categorical":
                if v in vdef.choices:
                    vals.append(float(vdef.choices.index(v)))
            else:
                vals.append(float(v))
        return np.array(vals) if vals else np.array([])

    def _fit_kde(self, vals: np.ndarray, vdef: Any) -> gaussian_kde | None:
        if len(vals) < 2:
            return None
        if np.std(vals) < 1e-10:
            vals = vals + self.np_rng.normal(0, 1e-6, size=len(vals))
        try:
            bw = "scott" if len(vals) >= 5 else "silverman"
            kde = gaussian_kde(vals, bw_method=bw)
            kde.set_bandwidth(kde.factor * self.bw_factor)
            return kde
        except Exception:
            return None

    def _sample_from_good(self) -> dict[str, Any]:
        config: dict[str, Any] = {}
        for name, vdef in zip(self._var_names, self._var_defs):
            if vdef.condition is not None:
                if not self.space._eval_condition(vdef.condition, config):
                    continue
            kde = self._good_kdes.get(name)
            if kde is not None and len(self._good_data.get(name, [])) >= 2:
                val = self._sample_var_from_kde(kde, vdef)
            else:
                val = self._sample_var_uniform(vdef)
            config[name] = val
        return config

    def _sample_var_from_kde(self, kde: gaussian_kde, vdef: Any) -> Any:
        for _ in range(20):
            raw = float(kde.resample(1, seed=self.np_rng.randint(0, 2**31))[0, 0])
            if vdef.var_type == "categorical":
                idx = max(0, min(len(vdef.choices) - 1, int(round(raw))))
                return vdef.choices[idx]
            elif vdef.var_type == "integer":
                val = int(round(raw))
                if vdef.low <= val <= vdef.high:
                    return val
            else:
                if vdef.low <= raw <= vdef.high:
                    return raw
        return self._sample_var_uniform(vdef)

    def _sample_var_uniform(self, vdef: Any) -> Any:
        if vdef.var_type == "categorical":
            return self.rng.choice(vdef.choices)
        elif vdef.var_type == "integer":
            return self.rng.randint(int(vdef.low), int(vdef.high))
        else:
            return self.rng.uniform(vdef.low, vdef.high)

    def _tpe_score(self, config: dict[str, Any]) -> float:
        log_l = 0.0
        log_g = 0.0

        for name, vdef in zip(self._var_names, self._var_defs):
            v = config.get(name)
            if v is None:
                continue
            if vdef.var_type == "categorical":
                num_val = float(vdef.choices.index(v)) if v in vdef.choices else 0.0
            else:
                num_val = float(v)

            l_kde = self._good_kdes.get(name)
            g_kde = self._bad_kdes.get(name)

            l_density = float(l_kde.evaluate([num_val])[0]) if l_kde is not None else 1e-10
            g_density = float(g_kde.evaluate([num_val])[0]) if g_kde is not None else 1e-10

            log_l += math.log(max(l_density, 1e-10))
            log_g += math.log(max(g_density, 1e-10))

        return log_l - log_g
