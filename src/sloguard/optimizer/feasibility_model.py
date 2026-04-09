"""Feasibility model for cross-config/GPU/model crash prediction.

Wraps the RF surrogate with serving-specific features: GPU embeddings,
model embeddings, and three-class prediction (feasible/infeasible/crash).
"""
from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sloguard.config_space import SearchSpace
from sloguard.types import EvalResult

logger = logging.getLogger(__name__)

# Known GPU profiles for embedding
GPU_PROFILES: dict[str, dict[str, float]] = {
    "RTX_5080_16GB": {"vram_gb": 16, "compute_cap": 10.0, "bandwidth_gbps": 960},
    "RTX_4090_24GB": {"vram_gb": 24, "compute_cap": 8.9, "bandwidth_gbps": 1008},
    "RTX_4080_16GB": {"vram_gb": 16, "compute_cap": 8.9, "bandwidth_gbps": 717},
    "RTX_3090_24GB": {"vram_gb": 24, "compute_cap": 8.6, "bandwidth_gbps": 936},
    "A100_40GB": {"vram_gb": 40, "compute_cap": 8.0, "bandwidth_gbps": 1555},
    "A100_80GB": {"vram_gb": 80, "compute_cap": 8.0, "bandwidth_gbps": 2039},
    "L4_24GB": {"vram_gb": 24, "compute_cap": 8.9, "bandwidth_gbps": 300},
    "T4_16GB": {"vram_gb": 16, "compute_cap": 7.5, "bandwidth_gbps": 300},
}

# Known model profiles for embedding
MODEL_PROFILES: dict[str, dict[str, float]] = {
    "Qwen/Qwen2-1.5B": {"params_b": 1.5, "layers": 28, "hidden": 1536},
    "microsoft/phi-2": {"params_b": 2.7, "layers": 32, "hidden": 2560},
    "mistralai/Mistral-7B-v0.1": {"params_b": 7.0, "layers": 32, "hidden": 4096},
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {"params_b": 1.1, "layers": 22, "hidden": 2048},
    "meta-llama/Llama-3.2-1B": {"params_b": 1.0, "layers": 16, "hidden": 2048},
}


class FeasibilityModel:
    """Predicts feasibility of serving configs across GPUs and models.

    Three-class prediction: feasible (0), infeasible (1), crash (2).
    Features: config encoding + GPU profile + model profile.
    """

    def __init__(self, search_space: SearchSpace, seed: int = 42):
        self.space = search_space
        self.seed = seed

        self._var_names = list(search_space._all_names)
        self._var_defs = [search_space.variables[n] for n in self._var_names]
        self._n_config_features = len(self._var_names)
        # config features + 3 GPU features + 3 model features
        self._n_features = self._n_config_features + 3 + 3

        self._model: RandomForestClassifier | None = None
        self._fitted = False
        self._oob_score: float = -1.0

    def _encode_config(self, config: dict[str, Any]) -> np.ndarray:
        """Encode config dict to feature vector."""
        x = np.zeros(self._n_config_features)
        for i, (name, v) in enumerate(zip(self._var_names, self._var_defs)):
            val = config.get(name)
            if val is None:
                x[i] = 0.0
            elif v.var_type == "categorical":
                x[i] = float(v.choices.index(val)) if val in v.choices else 0.0
            else:
                x[i] = float(val)
        return x

    def _encode_gpu(self, gpu_id: str) -> np.ndarray:
        """Encode GPU identifier to feature vector."""
        profile = GPU_PROFILES.get(gpu_id, {"vram_gb": 16, "compute_cap": 8.0, "bandwidth_gbps": 500})
        return np.array([profile["vram_gb"], profile["compute_cap"], profile["bandwidth_gbps"]])

    def _encode_model(self, model_id: str) -> np.ndarray:
        """Encode model identifier to feature vector."""
        profile = MODEL_PROFILES.get(
            model_id, {"params_b": 3.0, "layers": 32, "hidden": 2048}
        )
        return np.array([profile["params_b"], profile["layers"], profile["hidden"]])

    def _encode_full(
        self, config: dict[str, Any], gpu_id: str, model_id: str
    ) -> np.ndarray:
        cfg = self._encode_config(config)
        gpu = self._encode_gpu(gpu_id)
        mdl = self._encode_model(model_id)
        return np.concatenate([cfg, gpu, mdl])

    def fit(
        self,
        trials: list[tuple[dict[str, Any], EvalResult, str, str]],
    ) -> None:
        """Fit the feasibility model.

        Args:
            trials: List of (config, result, gpu_id, model_id) tuples.
        """
        if len(trials) < 5:
            return

        X_list, Y_list = [], []
        for config, result, gpu_id, model_id in trials:
            x = self._encode_full(config, gpu_id, model_id)
            X_list.append(x)
            if result.crashed:
                Y_list.append(2)  # crash
            elif not result.feasible:
                Y_list.append(1)  # infeasible
            else:
                Y_list.append(0)  # feasible

        X = np.array(X_list)
        Y = np.array(Y_list)

        if len(set(Y)) < 2:
            logger.debug("Only one class in training data, skipping fit")
            return

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=2,
                oob_score=True,
                random_state=self.seed,
                class_weight="balanced",
            )
            self._model.fit(X, Y)
            self._oob_score = self._model.oob_score_
            self._fitted = True

        logger.info("FeasibilityModel fitted: OOB accuracy=%.3f", self._oob_score)

    def predict_feasibility(
        self, config: dict[str, Any], gpu_id: str, model_id: str,
    ) -> tuple[float, float]:
        """Predict P(feasible) and P(crash) for a config.

        Returns:
            (p_feasible, p_crash) tuple.
        """
        if not self._fitted or self._model is None:
            return 0.5, 0.25

        x = self._encode_full(config, gpu_id, model_id).reshape(1, -1)
        proba = self._model.predict_proba(x)[0]
        classes = list(self._model.classes_)

        p_feasible = float(proba[classes.index(0)]) if 0 in classes else 0.0
        p_crash = float(proba[classes.index(2)]) if 2 in classes else 0.0

        return p_feasible, p_crash

    def should_skip(
        self,
        config: dict[str, Any],
        gpu_id: str,
        model_id: str,
        crash_threshold: float = 0.8,
    ) -> bool:
        """Return True if this config is predicted to crash with high probability."""
        _, p_crash = self.predict_feasibility(config, gpu_id, model_id)
        return p_crash >= crash_threshold

    def feature_importance(self) -> dict[str, float]:
        """Return feature importance scores."""
        if not self._fitted or self._model is None:
            return {}

        importances = self._model.feature_importances_
        names = list(self._var_names) + ["gpu_vram", "gpu_compute", "gpu_bandwidth",
                                          "model_params", "model_layers", "model_hidden"]
        return dict(zip(names, importances))

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def oob_accuracy(self) -> float:
        return self._oob_score
