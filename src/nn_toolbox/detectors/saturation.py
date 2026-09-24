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
        param_details = context.get("parameter_info", {}).get("param_details", {})

        trainable_prefixes = set()
        if param_details:
            for p_name, p_info in param_details.items():
                if p_info.get("requires_grad", False):
                    parts = p_name.split(".")
                    for i in range(1, len(parts) + 1):
                        trainable_prefixes.add(".".join(parts[:i]))

        for mod_name, stats in act_stats.items():
            zero_frac = stats.get("zero_fraction", 0.0)
            if zero_frac > 0.85:
                is_frozen = False
                if param_details:
                    mod_parts = mod_name.split(".")
                    is_frozen = not any(".".join(mod_parts[:i]) in trainable_prefixes for i in range(1, len(mod_parts) + 1))

                if is_frozen:
                    severity = Severity.INFO.value
                    obs = f"Module '{mod_name}' [FROZEN] has {zero_frac*100:.1f}% zero activations (unconditioned or inactive in frozen backbone)."
                    hypotheses = [
                        "Unconditioned cross-attention (e.g. empty/zero text conditioning in diffusion model)",
                        "Expected structural sparsity in frozen pretrained backbone",
                    ]
                    suggested_actions = [
                        "No action needed for frozen foundation backbone modules.",
                    ]
                else:
                    severity = Severity.WARNING.value
                    obs = f"Module '{mod_name}' has {zero_frac*100:.1f}% zero activations (dead units)."
                    hypotheses = [
                        "Dying ReLU syndrome caused by large negative biases or high initial learning rate",
                        "High dropout rate or aggressive thresholding",
                        "Overly aggressive zero-padding propagation",
                    ]
                    suggested_actions = [
                        "Try LeakyReLU, GELU, or SiLU instead of standard ReLU",
                        "Inspect bias initialization in preceding linear/conv layer",
                    ]

                findings.append(
                    DiagnosticFinding(
                        category=FindingCategory.FORWARD.value,
                        severity=severity,
                        module=mod_name,
                        observation=obs,
                        interpretation="A substantial portion of features are inactive/zeroed out.",
                        evidence={"zero_fraction": zero_frac, "stats": stats, "is_frozen": is_frozen},
                        hypotheses=hypotheses,
                        confidence="medium",
                        suggested_actions=suggested_actions,
                    )
                )

        return findings
