"""
Detector for inter-branch gradient scale balance and layer dominance.
Measures scale-invariant RMS gradient distributions across distinct sub-networks
and architectural branches to detect gradient starvation and branch dominance.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Dict, List
import torch

from nn_toolbox.core.finding import DiagnosticFinding, FindingCategory, Severity
from nn_toolbox.detectors.base import BaseDetector


class GradientBalanceDetector(BaseDetector):
    """Detects gradient scale imbalance across sub-networks and branches."""

    @property
    def name(self) -> str:
        return "gradient_balance"

    @property
    def category(self) -> str:
        return FindingCategory.BACKWARD.value

    def detect(self, context: Dict[str, Any]) -> List[DiagnosticFinding]:
        findings: List[DiagnosticFinding] = []
        bwd_analysis = context.get("backward_analysis", {})
        param_stats = context.get("parameter_gradient_stats", {})

        if not param_stats and "step_param_stats" in bwd_analysis:
            param_stats = bwd_analysis["step_param_stats"]

        # If not directly passed, extract from bwd_analysis if available
        if not param_stats:
            return findings

        # Group by top-level architectural branch (e.g. encoder, decoder, head, classifier)
        branch_rms_values: Dict[str, List[float]] = defaultdict(list)
        branch_param_counts: Dict[str, int] = defaultdict(int)

        for name, stats in param_stats.items():
            if not stats.get("has_grad", False) or not stats.get("is_finite", True):
                continue
            rms = stats.get("grad_rms", 0.0)
            if rms <= 0.0:
                continue

            parts = name.split(".")
            branch = parts[0] if len(parts) > 1 else "root"
            branch_rms_values[branch].append(rms)
            branch_param_counts[branch] += 1

        if len(branch_rms_values) < 2:
            return findings

        # Compute geometric or median RMS per branch
        branch_summary: Dict[str, float] = {}
        for branch, rms_list in branch_rms_values.items():
            if rms_list:
                median_rms = float(torch.tensor(rms_list).median().item())
                branch_summary[branch] = median_rms

        if not branch_summary:
            return findings

        max_branch, max_rms = max(branch_summary.items(), key=lambda item: item[1])
        min_branch, min_rms = min(branch_summary.items(), key=lambda item: item[1])

        if min_rms > 0:
            disparity_ratio = max_rms / min_rms
        else:
            disparity_ratio = float("inf")

        # Disparity ratio > 500x indicates severe branch starvation/dominance
        if disparity_ratio > 500.0 and max_rms > 1e-4:
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.BACKWARD.value,
                    severity=Severity.WARNING.value,
                    module=min_branch,
                    observation=(
                        f"Severe gradient scale disparity across model branches ({disparity_ratio:.1f}×): "
                        f"branch '{max_branch}' (RMS: {max_rms:.2e}) heavily dominates branch '{min_branch}' (RMS: {min_rms:.2e})."
                    ),
                    interpretation=(
                        f"Updates are concentrated overwhelmingly in '{max_branch}', while '{min_branch}' receives "
                        "severely attenuated gradient signals and risks stagnation."
                    ),
                    evidence={
                        "branch_medians": branch_summary,
                        "dominant_branch": max_branch,
                        "starved_branch": min_branch,
                        "disparity_ratio": disparity_ratio,
                    },
                    hypotheses=[
                        "Unbalanced composite loss weights (e.g. aux loss or reconstruction loss scale)",
                        "Bridge layers between branches lack gradient normalization (e.g. GroupNorm or RMSNorm)",
                        "Pretrained backbone requires distinct learning rate from task-specific decoder",
                    ],
                    confidence="medium",
                    suggested_actions=[
                        f"Consider applying separate optimizer parameter groups with higher lr for '{min_branch}'",
                        "Verify loss weighting between tasks/heads",
                        "Inspect normalization at the branch interface",
                    ],
                )
            )

        return findings
