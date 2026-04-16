"""Base optimizer interface — all optimizers implement this ask/tell protocol."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sloguard.types import EvalResult
from sloguard.config_space import SearchSpace


class BaseOptimizer(ABC):
    """Ask/tell interface shared by every optimizer."""

    def __init__(
        self,
        search_space: SearchSpace,
        constraints: dict[str, float],
        objective: str,
        budget: int,
        seed: int = 42,
    ):
        self.search_space = search_space
        self.constraints = constraints
        self.objective = objective
        self.budget = budget
        self.seed = seed

        self.history: list[tuple[dict[str, Any], EvalResult]] = []
        self._best_feasible: tuple[dict[str, Any], EvalResult] | None = None
        self.trial_count = 0

    @abstractmethod
    def ask(self) -> dict[str, Any]:
        """Return next configuration to evaluate."""
        ...

    def tell(self, config: dict[str, Any], result: EvalResult) -> None:
        """Report evaluation result."""
        self.history.append((config, result))
        self.trial_count += 1

        if result.feasible and not result.crashed:
            if (
                self._best_feasible is None
                or result.objective_value > self._best_feasible[1].objective_value
            ):
                self._best_feasible = (config, result)

    def best_feasible(self) -> tuple[dict[str, Any], EvalResult] | None:
        """Return best (config, result) that was feasible, or None."""
        return self._best_feasible

    @property
    def phase(self) -> str:
        """Current optimizer phase (for logging). Override in subclasses."""
        return "single"

    @property
    def n_crashes(self) -> int:
        return sum(1 for _, r in self.history if r.crashed)

    @property
    def n_infeasible(self) -> int:
        return sum(1 for _, r in self.history if not r.feasible and not r.crashed)

    @property
    def n_feasible(self) -> int:
        return sum(1 for _, r in self.history if r.feasible and not r.crashed)
