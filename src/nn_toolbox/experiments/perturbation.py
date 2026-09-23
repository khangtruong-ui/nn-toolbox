"""
Perturbation and empirical sensitivity diagnostic.
Measures output divergence under input, intermediate activation, or parameter noise.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Union
import torch
import torch.nn as nn

from nn_toolbox.analysis.sensitivity import compute_empirical_sensitivity
from nn_toolbox.core.finding import DiagnosticFinding, FindingCategory, Severity


def perturbation_test(
    model: nn.Module,
    sample_input: torch.Tensor,
    epsilons: Optional[List[float]] = None,
    num_trials: int = 3,
) -> Dict[str, Any]:
    """Test model output stability under varying scales of input perturbations."""
    if epsilons is None:
        epsilons = [1e-4, 1e-3, 1e-2]

    sensitivity = compute_empirical_sensitivity(
        model,
        sample_input,
        epsilons=epsilons,
        num_trials=num_trials,
    )

    findings: List[DiagnosticFinding] = []

    max_rel = sensitivity.get("max_relative_gain", 0.0)
    if max_rel > 100.0:
        findings.append(
            DiagnosticFinding(
                category=FindingCategory.STABILITY.value,
                severity=Severity.WARNING.value,
                observation=f"Model output exhibits high perturbation sensitivity (relative gain: {max_rel:.1f}×).",
                interpretation="Small perturbations in the input space produce disproportionately large shifts in the output.",
                evidence=sensitivity,
                hypotheses=[
                    "Exploding activations in intermediate layers",
                    "Missing LayerNorm / spectral normalization",
                    "Near-singular weights or large condition number",
                ],
                confidence="medium",
                suggested_actions=["Inspect forward signal propagation and activation variance."],
            )
        )
    else:
        findings.append(
            DiagnosticFinding(
                category=FindingCategory.STABILITY.value,
                severity=Severity.INFO.value,
                observation=f"Local perturbation sensitivity is moderate (relative gain: {max_rel:.2f}×).",
                interpretation="The model exhibits well-conditioned local response to input noise.",
                evidence=sensitivity,
                hypotheses=["Local Lipschitz response is stable."],
                confidence="high",
            )
        )

    return {
        "sensitivity": sensitivity,
        "findings": findings,
    }
