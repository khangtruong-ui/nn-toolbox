"""
Core diagnostic finding and evidence structures.
Distinguishes observations from interpretations and hypotheses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class FindingCategory(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    OPTIMIZATION = "optimization"
    MEMORIZATION = "memorization"
    TRAIN_EVAL = "train_eval"
    DATA = "data"
    STABILITY = "stability"
    ARCHITECTURE = "architecture"
    INITIALIZATION = "initialization"
    GENERAL = "general"


@dataclass
class DiagnosticFinding:
    """A structured finding representing an empirical observation,

    its cautious interpretation, underlying quantitative evidence,
    and possible causal hypotheses.
    """

    category: str
    severity: str  # "info" | "warning" | "critical"
    observation: str
    interpretation: str = ""
    module: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    hypotheses: List[str] = field(default_factory=list)
    confidence: str = "medium"  # "low" | "medium" | "high"
    suggested_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert finding to standard dictionary representation."""
        return {
            "category": str(self.category),
            "severity": str(self.severity),
            "module": self.module,
            "observation": self.observation,
            "interpretation": self.interpretation,
            "evidence": self.evidence,
            "hypotheses": self.hypotheses,
            "confidence": self.confidence,
            "suggested_actions": self.suggested_actions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DiagnosticFinding:
        """Instantiate finding from a dictionary."""
        return cls(
            category=data.get("category", "general"),
            severity=data.get("severity", "info"),
            observation=data.get("observation", ""),
            interpretation=data.get("interpretation", ""),
            module=data.get("module"),
            evidence=data.get("evidence", {}),
            hypotheses=data.get("hypotheses", []),
            confidence=data.get("confidence", "medium"),
            suggested_actions=data.get("suggested_actions", []),
        )

    def is_actionable(self) -> bool:
        """Returns True if the finding warrants user attention (warning or critical)."""
        return self.severity in (Severity.WARNING.value, Severity.CRITICAL.value, "warning", "critical")
