"""
Train/eval consistency diagnostic.
Runs identical inputs through model.train() and model.eval() to isolate
stochasticity, normalization shifts, and train/eval mode bugs.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from nn_toolbox.core.finding import DiagnosticFinding, FindingCategory, Severity


def train_eval_test(
    model: nn.Module,
    sample_input: Any,
    atol: float = 1e-4,
    rtol: float = 1e-2,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Compare forward outputs of model in train() vs eval() mode on the same input."""
    if device is None:
        device = next(model.parameters()).device

    if torch.is_tensor(sample_input):
        x = sample_input.to(device)
    else:
        x = sample_input

    # Detect present stochastic or mode-dependent layers
    has_dropout = any(isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)) for m in model.modules())
    has_batchnorm = any(
        isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))
        for m in model.modules()
    )

    findings: List[DiagnosticFinding] = []

    was_training = model.training

    try:
        # Run in train mode
        model.train()
        with torch.no_grad():
            out_train = model(x)
            if isinstance(out_train, (tuple, list)):
                out_train = out_train[0]

        # Run in eval mode
        model.eval()
        with torch.no_grad():
            out_eval = model(x)
            if isinstance(out_eval, (tuple, list)):
                out_eval = out_eval[0]

        diff = (out_train - out_eval).float()
        l2_diff = float(torch.linalg.norm(diff).item())
        eval_norm = float(torch.linalg.norm(out_eval.float()).item())
        rel_diff = l2_diff / (eval_norm + 1e-12) if eval_norm > 0 else 0.0

        is_identical = rel_diff < 1e-5

        # Also run eval twice to verify deterministic behavior in eval mode
        with torch.no_grad():
            out_eval_2 = model(x)
            if isinstance(out_eval_2, (tuple, list)):
                out_eval_2 = out_eval_2[0]

        eval_rep_diff = float(torch.linalg.norm((out_eval - out_eval_2).float()).item())
        is_eval_deterministic = eval_rep_diff < 1e-6

        if not is_eval_deterministic:
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.TRAIN_EVAL.value,
                    severity=Severity.WARNING.value,
                    observation=f"Repeated eval() calls on identical inputs produced non-deterministic outputs (difference: {eval_rep_diff:.2e}).",
                    interpretation="The model exhibits stochasticity or unseeded random state even in evaluation mode.",
                    evidence={"eval_repeat_diff": eval_rep_diff},
                    hypotheses=[
                        "Dropout or random sampling active during eval()",
                        "Custom module not respecting model.eval() or self.training flag",
                    ],
                    confidence="high",
                    suggested_actions=["Check custom layers for 'if self.training:' guards."],
                )
            )

        if not is_identical:
            if not has_dropout and not has_batchnorm:
                # Discrepancy without obvious dropout/batchnorm
                findings.append(
                    DiagnosticFinding(
                        category=FindingCategory.TRAIN_EVAL.value,
                        severity=Severity.WARNING.value,
                        observation=f"Output differs between train() and eval() (relative difference: {rel_diff * 100:.1f}%) without standard Dropout/BatchNorm.",
                        interpretation="A custom layer or normalization module has different behavior between training and evaluation.",
                        evidence={
                            "relative_difference": rel_diff,
                            "has_dropout": has_dropout,
                            "has_batchnorm": has_batchnorm,
                        },
                        hypotheses=[
                            "Custom normalization or augmentations inside model.forward",
                            "Feature masking or stochastic drop path",
                        ],
                        confidence="medium",
                        suggested_actions=["Verify custom layers implementing distinct forward paths."],
                    )
                )
            else:
                findings.append(
                    DiagnosticFinding(
                        category=FindingCategory.TRAIN_EVAL.value,
                        severity=Severity.INFO.value,
                        observation=f"Train vs Eval output difference observed ({rel_diff * 100:.1f}%), consistent with present {['Dropout' if has_dropout else 'BatchNorm']}.",
                        interpretation="The output shift matches expected behavior of stochastic or running-statistics modules.",
                        evidence={"relative_difference": rel_diff},
                        hypotheses=["Normal train/eval mode separation."],
                        confidence="high",
                    )
                )
        else:
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.TRAIN_EVAL.value,
                    severity=Severity.INFO.value,
                    observation="Model outputs are identical between train() and eval() mode.",
                    interpretation="No stochastic layers or training-specific behaviors active on this input.",
                    evidence={"relative_difference": rel_diff},
                    hypotheses=["Deterministic architecture."],
                    confidence="high",
                )
            )

        return {
            "relative_difference": rel_diff,
            "absolute_l2_difference": l2_diff,
            "is_identical": is_identical,
            "is_eval_deterministic": is_eval_deterministic,
            "has_dropout": has_dropout,
            "has_batchnorm": has_batchnorm,
            "findings": findings,
        }

    finally:
        if was_training:
            model.train()
        else:
            model.eval()
