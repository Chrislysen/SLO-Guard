"""Random Forest surrogate for pre-screening candidates.

Predicts objective value and probability of feasibility. OOB-gated:
only trusts its own predictions when OOB score exceeds threshold.

Ported from TBA's surrogate.py.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sloguard.config_space import SearchSpace
from sloguard.types import EvalResult

OOB_THRESHOLD = 0.3


class RFSurrogate:
    """Lightweight RF surrogate that predicts objective and feasibility.

    Encodes configs as flat feature vectors (ordinal for categoricals,
    raw for numerics, midpoint-fill for inactive conditionals).

    Self-gating: checks OOB score before trusting predictions.
    """

    def __init__(self, search_space: SearchSpace, seed: int = 42):
        self.space = search_space
        self.seed = seed

        self._var_names = list(search_space._all_names)
        self._var_defs = [search_space.variables[n] for n in self._var_names]
        self._n_features = len(self._var_names)

        self._midpoints: list[float] = []
        for v in self._var_defs:
            if v.var_type == "categorical":
                self._midpoints.append(len(v.choices) / 2.0)
            else:
                self._midpoints.append((v.low + v.high) / 2.0)

        self._obj_model: RandomForestRegressor | None = None
        self._feas_model: RandomForestClassifier | None = None
        self._feas_classifier_fitted = False
        self._feas_constant = 0.5
        self._history_ref: list = []

        self._obj_oob_score: float = -1.0
        self._feas_oob_score: float = -1.0
        self._trustworthy = False

    def encode(self, config: dict[str, Any]) -> np.ndarray:
        """Encode a config dict into a feature vector."""
        x = np.zeros(self._n_features)
        for i, (name, v) in enumerate(zip(self._var_names, self._var_defs)):
            val = config.get(name)
            if val is None:
                x[i] = self._midpoints[i]
            elif v.var_type == "categorical":
                x[i] = float(v.choices.index(val)) if val in v.choices else self._midpoints[i]
            else:
                x[i] = float(val)
        return x

    def fit(self, history: list[tuple[dict[str, Any], EvalResult]]) -> None:
        """Fit RF models on observed data with OOB scoring."""
        if len(history) < 3:
            return

        X_all, Y_obj, Y_feas = [], [], []
        for config, result in history:
            X_all.append(self.encode(config))
            Y_feas.append(1 if (result.feasible and not result.crashed) else 0)
            Y_obj.append(-1e4 if result.crashed else result.objective_value)

        X = np.array(X_all)
        Y_obj_arr = np.array(Y_obj)
        Y_feas_arr = np.array(Y_feas)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self._obj_model = RandomForestRegressor(
                n_estimators=50, max_depth=8, min_samples_leaf=2,
                oob_score=True, random_state=self.seed,
            )
            self._obj_model.fit(X, Y_obj_arr)
            self._obj_oob_score = self._obj_model.oob_score_

            if len(set(Y_feas_arr)) > 1:
                self._feas_model = RandomForestClassifier(
                    n_estimators=50, max_depth=8, min_samples_leaf=2,
                    oob_score=True, random_state=self.seed,
                )
                self._feas_model.fit(X, Y_feas_arr)
                self._feas_oob_score = self._feas_model.oob_score_
                self._feas_classifier_fitted = True
            else:
                self._feas_model = None
                self._feas_classifier_fitted = False
                self._feas_constant = float(Y_feas_arr[0])
                self._feas_oob_score = -1.0

        self._trustworthy = self._obj_oob_score >= OOB_THRESHOLD

    @property
    def is_ready(self) -> bool:
        return self._obj_model is not None

    @property
    def is_trustworthy(self) -> bool:
        return self.is_ready and self._trustworthy

    def predict(self, config: dict[str, Any]) -> tuple[float, float]:
        """Return (predicted_objective, probability_of_feasible)."""
        if not self.is_ready:
            return 0.0, 0.5

        x = self.encode(config).reshape(1, -1)
        pred_obj = float(self._obj_model.predict(x)[0])

        if self._feas_classifier_fitted and self._feas_model is not None:
            proba = self._feas_model.predict_proba(x)[0]
            classes = list(self._feas_model.classes_)
            p_feas = float(proba[classes.index(1)]) if 1 in classes else 0.0
        else:
            p_feas = self._feas_constant

        return pred_obj, p_feas

    def score_candidate(self, config: dict[str, Any]) -> float:
        """Combined acquisition score: feasibility-gated predicted improvement."""
        pred_obj, p_feas = self.predict(config)

        obj_vals = [
            r.objective_value for _, r in self._history_ref if not r.crashed
        ] if self._history_ref else []

        if obj_vals and max(obj_vals) > min(obj_vals):
            obj_min, obj_max = min(obj_vals), max(obj_vals)
            norm_obj = (pred_obj - obj_min) / (obj_max - obj_min)
        else:
            norm_obj = pred_obj

        return p_feas * 0.6 + norm_obj * p_feas * 0.4

    def set_history_ref(self, history: list[tuple[dict[str, Any], EvalResult]]) -> None:
        self._history_ref = history
