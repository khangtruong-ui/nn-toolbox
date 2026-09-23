"""
Detectors for activation saturation and dead neurons (e.g. dead ReLU, saturated sigmoid/tanh).
"""

from __future__ import annotations

from typing import Any, Dict, List
from nn_toolbox.core.finding import DiagnosticFinding, FindingCategory, Severity
from nn_toolbox.detectors.base import BaseDetector


class SaturationDetector(BaseDetector):
    """Detects layers with high fractions of dead or saturated activations."""

    @property
    def name(self) -> str:
        return "saturation"

    @property
    def category(self) -> str:
        return FindingCategory.FORWARD.value

    def detect(self, context: Dict[str, Any]) -> List[DiagnosticFinding]:
        findings: List[DiagnosticFinding] = []
        act_stats = context.get("activation_stats", {})

        for mod_name, stats in act_stats.items():
            zero_frac = stats.get("zero_fraction", 0.0)
            if zero_frac > 0.85:
                findings.append(
                    DiagnosticFinding(
                        category=FindingCategory.FORWARD.value,
                        severity=Severity.WARNING.value,
                        module=mod_name,
                        observation=f"Module '{mod_name}' has {zero_frac*100:.1f}% zero activations (dead units).",
                        interpretation="A substantial portion of features are inactive/zeroed out.",
                        evidence={"zero_fraction": zero_frac, "stats": stats},
                        hypotheses=[
                            "Dying ReLU syndrome caused by large negative biases or high initial learning rate",
                            "High dropout rate or aggressive thresholding",
                            "Overly aggressive zero-padding propagation",
                        ],
                        confidence="medium",
                        suggested_actions=[
                            "Try LeakyReLU, GELU, or SiLU instead of standard ReLU",
                            "Inspect bias initialization in preceding linear/conv layer",
                        ],
                    )
                )

        return findings
