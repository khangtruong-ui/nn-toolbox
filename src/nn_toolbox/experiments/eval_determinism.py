"""
Evaluation Mode Determinism and Stochastic Drift Diagnostic.
Runs repeated forward passes on identical inputs in model.eval() to isolate
uncontrolled stochasticity, unseeded random perturbations, and non-deterministic layers.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from nn_toolbox.core.finding import DiagnosticFinding, FindingCategory, Severity


def eval_determinism_test(
    model: nn.Module,
    sample_input: Any,
    num_passes: int = 3,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Test whether model.eval() produces deterministic, reproducible outputs across repeated invocations.

    Args:
        model: PyTorch model.
        sample_input: Input tensor or batch.
        num_passes: Number of repeated evaluation forward passes.
        atol: Absolute tolerance for determinism.
        rtol: Relative tolerance for determinism.
        device: Device to run evaluation on.

    Returns:
        Dictionary with determinism status, max relative difference, and diagnostic findings.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    if torch.is_tensor(sample_input):
        x = sample_input.to(device)
    else:
        x = sample_input

    was_training = model.training
    model.eval()

    outputs: List[torch.Tensor] = []
    findings: List[DiagnosticFinding] = []

    try:
        with torch.no_grad():
            for pass_idx in range(num_passes):
                out = model(x)
                if isinstance(out, (tuple, list)):
                    out_t = out[0]
                elif isinstance(out, dict):
                    out_t = next(iter(out.values()))
                else:
                    out_t = out
                outputs.append(out_t.detach().float().cpu())
    finally:
        if was_training:
            model.train()

    if len(outputs) < 2:
        return {
            "is_deterministic": True,
            "max_difference": 0.0,
            "relative_drift": 0.0,
            "findings": findings,
        }

    base_out = outputs[0]
    base_norm = float(torch.linalg.norm(base_out).item())

    max_diff = 0.0
    for i in range(1, len(outputs)):
        diff = float(torch.linalg.norm(outputs[i] - base_out).item())
        if diff > max_diff:
            max_diff = diff

    rel_drift = max_diff / (base_norm + 1e-12) if base_norm > 0 else 0.0
    is_deterministic = rel_drift < rtol and max_diff < atol

    if is_deterministic:
        findings.append(
            DiagnosticFinding(
                category=FindingCategory.STABILITY.value,
                severity=Severity.INFO.value,
                observation=f"Evaluation mode is strictly deterministic across {num_passes} passes (max diff: {max_diff:.2e}).",
                interpretation="Repeated inferences on identical inputs produce bitwise or near-identical reproducible results.",
                evidence={"num_passes": num_passes, "max_diff": max_diff, "rel_drift": rel_drift},
                confidence="high",
            )
        )
    else:
        severity = Severity.WARNING.value if rel_drift < 0.2 else Severity.CRITICAL.value
        findings.append(
            DiagnosticFinding(
                category=FindingCategory.STABILITY.value,
                severity=severity,
                observation=(
                    f"Repeated eval() calls on identical inputs produced non-deterministic outputs "
                    f"(relative drift: {rel_drift*100:.2f}%, absolute diff: {max_diff:.2e})."
                ),
                interpretation=(
                    "The model contains active stochastic operations or unseeded random generation in eval() mode. "
                    "Inference results will vary unpredictably across consecutive calls."
                ),
                evidence={"num_passes": num_passes, "max_diff": max_diff, "rel_drift": rel_drift},
                hypotheses=[
                    "Unseeded random noise generation (e.g. torch.randn) active without 'if self.training:' guard",
                    "Dropout module not respecting model.eval() state",
                    "Stochastic sampling head (e.g. VAE reparameterization trick) active during deterministic inference",
                ],
                confidence="high",
                suggested_actions=[
                    "Wrap stochastic noise generation with 'if self.training:' or provide fixed generator/seed during eval",
                    "Use deterministic mode (e.g. posterior.mode() or fixed noise schedule) during evaluation",
                ],
            )
        )

    return {
        "is_deterministic": is_deterministic,
        "max_difference": max_diff,
        "relative_drift": rel_drift,
        "findings": findings,
    }
