"""Random search optimizer — simplest baseline."""
from __future__ import annotations

import random
from typing import Any

from sloguard.config_space import SearchSpace
from sloguard.optimizer.base import BaseOptimizer


class RandomSearchOptimizer(BaseOptimizer):
    """Uniform random sampling over the config space.

    No learning, no crash awareness. Surprisingly competitive in
    high-dimensional spaces — the baseline every method must beat.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        constraints: dict[str, float],
        objective: str = "maximize_goodput",
        budget: int = 30,
        seed: int = 42,
    ):
        super().__init__(search_space, constraints, objective, budget, seed)
        self.rng = random.Random(seed)

    def ask(self) -> dict[str, Any]:
        return self.search_space.sample_random(self.rng)
