"""
Detectors for training instability, gradient oscillation, and excessive update-to-weight ratios.
"""

from __future__ import annotations

from typing import Any, Dict, List
from nn_toolbox.core.finding import DiagnosticFinding, FindingCategory, Severity
from nn_toolbox.detectors.base import BaseDetector


class InstabilityDetector(BaseDetector):
    """Detects optimization oscillation, destabilizing updates, and loss spikes."""

    @property
    def name(self) -> str:
        return "instability"

    @property
    def category(self) -> str:
        return FindingCategory.STABILITY.value

    def detect(self, context: Dict[str, Any]) -> List[DiagnosticFinding]:
        findings: List[DiagnosticFinding] = []

        # 1. Gradient direction oscillation
        bwd_analysis = context.get("backward_analysis", {})
        recent_cos = bwd_analysis.get("recent_cosine_similarity")
        if recent_cos is not None and recent_cos < -0.7:
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.STABILITY.value,
                    severity=Severity.WARNING.value,
                    observation=f"Gradient direction is oscillating strongly across consecutive steps (cos similarity: {recent_cos:.3f}).",
                    interpretation="The optimizer appears to be bouncing back and forth across a valley in the loss landscape.",
                    evidence={"cosine_similarity": recent_cos},
                    hypotheses=[
                        "Learning rate is too high for the local curvature",
                        "Momentum buffer is overshooting",
                        "Mini-batch size is too small, inducing alternating sample gradients",
                    ],
                    confidence="medium",
                    suggested_actions=[
                        "Decrease the learning rate by 2-5×",
                        "Increase batch size or gradient accumulation steps",
                    ],
                )
            )

        # 2. Update-to-weight ratio check (||Delta theta|| / ||theta||)
        update_info = context.get("update_stats", {})
        global_update_ratio = update_info.get("global_update_ratio", 0.0)

        if global_update_ratio > 0.3:
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.OPTIMIZATION.value,
                    severity=Severity.WARNING.value,
                    observation=f"Global parameter update-to-weight ratio is excessively high ({global_update_ratio:.2f}).",
                    interpretation="A single optimizer step alters more than 30% of total parameter magnitude.",
                    evidence={"global_update_ratio": global_update_ratio},
                    hypotheses=[
                        "Learning rate is too large",
                        "Gradient clipping is missing or clip threshold is too high",
                    ],
                    confidence="high",
                    suggested_actions=[
                        "Lower learning rate",
                        "Enable torch.nn.utils.clip_grad_norm_ with max_norm <= 1.0",
                    ],
                )
            )

        # Per-module update-to-weight ratio anomalies
        per_param_updates = update_info.get("updates", {})
        for name, up_entry in per_param_updates.items():
            u_ratio = up_entry.get("update_ratio", 0.0)
            p_norm = up_entry.get("param_norm", 0.0)
            d_norm = up_entry.get("delta_norm", 0.0)
            # Avoid false positives on zero-initialized biases or negligible parameter displacements
            if u_ratio > 0.5 and (p_norm >= 0.05 or d_norm >= 0.05):
                findings.append(
                    DiagnosticFinding(
                        category=FindingCategory.OPTIMIZATION.value,
                        severity=Severity.WARNING.value,
                        module=name,
                        observation=f"Parameter '{name}' update-to-weight ratio is very high ({u_ratio:.2f}).",
                        interpretation=f"Weights in '{name}' are being violently overwritten in a single step.",
                        evidence=up_entry,
                        hypotheses=["High gradient scale or small parameter magnitude"],
                        confidence="medium",
                        suggested_actions=[f"Check learning rate or weight decay on '{name}'"],
                    )
                )

        return findings
