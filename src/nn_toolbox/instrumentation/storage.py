"""
Bounded storage buffer for diagnostic metrics and time-series records.
Prevents memory exhaustion during long diagnostic runs.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterator, List, Optional


class RollingMetricsStorage:
    """Ring-buffer storage for per-step training metrics, gradient norms, and diagnostic snapshots."""

    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self._records: deque = deque(maxlen=max_history)

    def append(self, step: int, metrics: Dict[str, Any]) -> None:
        """Store metrics dictionary tagged with step number."""
        self._records.append({"step": step, **metrics})

    def clear(self) -> None:
        """Clear all stored entries."""
        self._records.clear()

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Return the most recently appended metrics entry."""
        if not self._records:
            return None
        return self._records[-1]

    def to_list(self) -> List[Dict[str, Any]]:
        """Return all stored entries as a list."""
        return list(self._records)

    def get_metric_series(self, key: str) -> List[Any]:
        """Extract a single metric time-series across all stored steps."""
        return [r.get(key) for r in self._records if key in r]

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self._records)
