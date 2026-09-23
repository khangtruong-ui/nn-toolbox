"""
Base class and interface for diagnostic detectors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from nn_toolbox.core.finding import DiagnosticFinding


class BaseDetector(ABC):
    """Abstract base class for diagnostic detectors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Detector identifier."""
        pass

    @property
    @abstractmethod
    def category(self) -> str:
        """FindingCategory value."""
        pass

    @abstractmethod
    def detect(self, context: Dict[str, Any]) -> List[DiagnosticFinding]:
        """Examine collected diagnostic metrics and formulate findings."""
        pass
