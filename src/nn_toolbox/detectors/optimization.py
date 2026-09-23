"""
Detectors for optimizer dysfunction, stagnant updates, and parameter freezing.
"""

from __future__ import annotations

from typing import Any, Dict, List
from nn_toolbox.core.finding import DiagnosticFinding, FindingCategory, Severity
from nn_toolbox.detectors.base import BaseDetector


class OptimizationDetector(BaseDetector):
    """Detects issues where parameters fail to update, updates are infinitesimally small, or weights are frozen."""

    @property
    def name(self) -> str:
        return "optimization"

    @property
    def category(self) -> str:
        return FindingCategory.OPTIMIZATION.value

    def detect(self, context: Dict[str, Any]) -> List[DiagnosticFinding]:
        findings: List[DiagnosticFinding] = []

        # 1. Check parameter trainability
        param_info = context.get("parameter_info", {})
        total_p = param_info.get("total_parameters", 0)
        trainable_p = param_info.get("trainable_parameters", 0)

        if total_p > 0 and trainable_p == 0:
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.OPTIMIZATION.value,
                    severity=Severity.CRITICAL.value,
                    observation="All model parameters are frozen (requires_grad=False).",
                    interpretation="No parameter optimization can occur.",
                    evidence=param_info,
                    hypotheses=["Parameters frozen during initialization or backbone freezing without unfreezing head"],
                    confidence="high",
                    suggested_actions=["Call p.requires_grad = True on target parameters."],
                )
            )

        # 2. Check for stagnant updates (gradients exist, but updates are zero)
        update_info = context.get("update_stats", {})
        global_u_ratio = update_info.get("global_update_ratio", 0.0)
        bwd_analysis = context.get("backward_analysis", {})
        has_grads = bwd_analysis.get("params_with_gradients", 0) > 0

        if has_grads and global_u_ratio == 0.0 and update_info.get("global_update_norm", 0.0) == 0.0:
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.OPTIMIZATION.value,
                    severity=Severity.CRITICAL.value,
                    observation="Gradients were calculated, but parameter values did not change after optimizer.step().",
                    interpretation="The optimizer step produced zero displacement in parameter space.",
                    evidence=update_info,
                    hypotheses=[
                        "Optimizer learning rate is set to 0.0",
                        "Optimizer param_groups do not include model parameters",
                        "GradScaler skipped optimizer step due to inf/nan gradients",
                    ],
                    confidence="high",
                    suggested_actions=["Check optimizer.param_groups[0]['lr']", "Check if GradScaler skipped the step"],
                )
            )
        elif has_grads and global_u_ratio < 1e-6 and global_u_ratio > 0:
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.OPTIMIZATION.value,
                    severity=Severity.WARNING.value,
                    observation=f"Update-to-weight ratio is extremely small ({global_u_ratio:.2e}).",
                    interpretation="Parameter updates are near the noise floor, which may cause training to stall.",
                    evidence=update_info,
                    hypotheses=[
                        "Learning rate is too small",
                        "Gradients are severely attenuated",
                    ],
                    confidence="medium",
                    suggested_actions=["Try increasing the learning rate by 5-10×."],
                )
            )

        return findings
