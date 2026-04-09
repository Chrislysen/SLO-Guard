"""JSONL trial logger for structured experiment logging.

Each trial is logged as a single JSON line, enabling streaming writes
and easy post-hoc analysis. Schema follows ServingTrialResult.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sloguard.types import ServingTrialResult

logger = logging.getLogger(__name__)


class TrialLogger:
    """Writes ServingTrialResult entries as JSONL.

    Usage:
        logger = TrialLogger("results/experiment_001.jsonl")
        logger.log(trial_result)
        # ...
        all_results = logger.load()
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._count = 0

    def log(self, result: ServingTrialResult) -> None:
        """Append a trial result as a JSONL line."""
        data = asdict(result)
        # Clean up None values for compactness
        data = {k: v for k, v in data.items() if v is not None}

        with open(self.path, "a") as f:
            f.write(json.dumps(data, default=str) + "\n")

        self._count += 1
        logger.debug("Logged trial %d to %s", result.trial_id, self.path)

    def log_dict(self, data: dict[str, Any]) -> None:
        """Log a raw dict as a JSONL line."""
        with open(self.path, "a") as f:
            f.write(json.dumps(data, default=str) + "\n")
        self._count += 1

    @property
    def count(self) -> int:
        """Number of trials logged in this session."""
        return self._count

    def load(self) -> list[ServingTrialResult]:
        """Load all trial results from the JSONL file."""
        if not self.path.exists():
            return []

        results = []
        with open(self.path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    result = _dict_to_trial_result(data)
                    results.append(result)
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.warning("Skipping malformed line %d: %s", line_num, e)

        return results

    def load_dicts(self) -> list[dict[str, Any]]:
        """Load all trial results as raw dicts."""
        if not self.path.exists():
            return []

        results = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        return results


def _dict_to_trial_result(data: dict[str, Any]) -> ServingTrialResult:
    """Convert a dict back to ServingTrialResult, handling missing fields."""
    # Get all field names from the dataclass
    import dataclasses

    fields = {f.name: f for f in dataclasses.fields(ServingTrialResult)}
    kwargs = {}
    for name, f in fields.items():
        if name in data:
            kwargs[name] = data[name]
    return ServingTrialResult(**kwargs)


def load_experiment(path: str | Path) -> list[ServingTrialResult]:
    """Convenience function to load trial results from a JSONL file."""
    return TrialLogger(path).load()
