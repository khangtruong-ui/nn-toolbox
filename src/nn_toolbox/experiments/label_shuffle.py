"""
Label shuffle experiment.
Compares training loss descent on true targets vs randomized permutation of targets.
Sanity check against trivial label leakage or memorization artifacts.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from nn_toolbox.core.finding import DiagnosticFinding, FindingCategory, Severity


def label_shuffle_test(
    model: nn.Module,
    sample_input: torch.Tensor,
    sample_target: torch.Tensor,
    loss_fn: Callable[[Any, Any], torch.Tensor],
    optimizer_factory: Optional[Callable[[Any], torch.optim.Optimizer]] = None,
    steps: int = 25,
    learning_rate: float = 1e-3,
) -> Dict[str, Any]:
    """Train copy of model on true targets vs permuted/randomized targets to check label sensitivity."""
    orig_state = copy.deepcopy(model.state_dict())

    # Create shuffled targets (permute batch dimension)
    perm = torch.randperm(sample_target.size(0))
    shuffled_target = sample_target[perm]

    def _train_run(y_data: torch.Tensor) -> List[float]:
        model.load_state_dict(orig_state)
        model.train()
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = optimizer_factory(trainable) if optimizer_factory else torch.optim.AdamW(trainable, lr=learning_rate)

        losses = []
        for _ in range(steps):
            opt.zero_grad()
            out = model(sample_input)
            loss = loss_fn(out, y_data)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        return losses

    try:
        true_losses = _train_run(sample_target)
        shuffled_losses = _train_run(shuffled_target)

        true_drop = (true_losses[0] - true_losses[-1]) / (abs(true_losses[0]) + 1e-12)
        shuffled_drop = (shuffled_losses[0] - shuffled_losses[-1]) / (abs(shuffled_losses[0]) + 1e-12)

        findings: List[DiagnosticFinding] = []

        if true_drop > 0.3 and shuffled_drop < 0.05:
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.DATA.value,
                    severity=Severity.INFO.value,
                    observation=f"Model learned true targets (loss drop: {true_drop*100:.1f}%) significantly faster than shuffled targets ({shuffled_drop*100:.1f}%).",
                    interpretation="The model relies on genuine input-target relationships rather than pure unconditional memorization capacity.",
                    evidence={"true_drop_pct": true_drop * 100, "shuffled_drop_pct": shuffled_drop * 100},
                    hypotheses=["Legitimate pattern learning."],
                    confidence="high",
                )
            )
        elif abs(true_drop - shuffled_drop) < 0.05 and true_drop > 0.4:
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.DATA.value,
                    severity=Severity.WARNING.value,
                    observation=f"Model fits shuffled random targets almost as quickly ({shuffled_drop*100:.1f}%) as true targets ({true_drop*100:.1f}%).",
                    interpretation="The model possesses massive capacity to memorize arbitrary patterns or labels on this batch size.",
                    evidence={"true_drop_pct": true_drop * 100, "shuffled_drop_pct": shuffled_drop * 100},
                    hypotheses=[
                        "High model capacity relative to sample size",
                        "Weak inductive bias requiring strong regularization or more data",
                    ],
                    confidence="medium",
                )
            )

        return {
            "true_losses": true_losses,
            "shuffled_losses": shuffled_losses,
            "true_drop": true_drop,
            "shuffled_drop": shuffled_drop,
            "findings": findings,
        }

    finally:
        model.load_state_dict(orig_state)
