"""
Lightweight learning-rate sensitivity experiment.
Characterizes optimization response across logarithmic scales:
stagnation, useful learning, instability, divergence, or NaNs.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Dict, List, Optional, Union
import torch
import torch.nn as nn

from nn_toolbox.core.finding import DiagnosticFinding, FindingCategory, Severity


def lr_sweep(
    model: nn.Module,
    sample_input: Any,
    sample_target: Optional[Any],
    loss_fn: Callable[[Any, Any], torch.Tensor],
    optimizer_factory: Optional[Callable[[Any, float], torch.optim.Optimizer]] = None,
    lr_range: Optional[List[float]] = None,
    steps_per_lr: int = 15,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Test response of model across a range of learning rates on a fixed sample batch."""
    if lr_range is None:
        lr_range = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    if device is None:
        device = next(model.parameters()).device

    orig_state = copy.deepcopy(model.state_dict())

    # Prepare batch on device
    if torch.is_tensor(sample_input):
        x = sample_input.to(device)
    else:
        x = sample_input

    def _to_device(t: Any) -> Any:
        if t is None:
            return None
        if torch.is_tensor(t):
            return t.to(device)
        if isinstance(t, dict):
            return {k: _to_device(v) for k, v in t.items()}
        if isinstance(t, (tuple, list)):
            return [_to_device(v) for v in t]
        return t

    y = _to_device(sample_target)

    sweep_results: List[Dict[str, Any]] = []
    findings: List[DiagnosticFinding] = []

    try:
        for lr in lr_range:
            model.load_state_dict(orig_state)
            model.train()

            trainable_params = [p for p in model.parameters() if p.requires_grad]
            if not trainable_params:
                break

            if optimizer_factory is not None:
                opt = optimizer_factory(trainable_params, lr)
            else:
                opt = torch.optim.AdamW(trainable_params, lr=lr)

            initial_loss = None
            final_loss = None
            losses: List[float] = []
            status = "unknown"

            for step in range(steps_per_lr):
                opt.zero_grad()
                out = model(x)
                if y is not None:
                    loss = loss_fn(out, y)
                else:
                    loss = loss_fn(out)

                loss_val = float(loss.item())
                if initial_loss is None:
                    initial_loss = loss_val
                losses.append(loss_val)

                if math.isnan(loss_val) or math.isinf(loss_val):
                    status = "nan_or_inf"
                    final_loss = loss_val
                    break

                loss.backward()
                opt.step()
                final_loss = loss_val

            if status != "nan_or_inf" and initial_loss is not None and final_loss is not None:
                rel_change = (final_loss - initial_loss) / (abs(initial_loss) + 1e-12)
                if final_loss > initial_loss * 3.0:
                    status = "divergent"
                elif max(losses) > min(losses) * 2.0 and final_loss > min(losses) * 1.5:
                    status = "unstable"
                elif abs(rel_change) < 0.01:
                    status = "too_little_movement"
                elif rel_change < -0.05:
                    status = "useful_learning"
                else:
                    status = "marginal_movement"

            sweep_results.append({
                "lr": lr,
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "losses": losses,
                "status": status,
            })

        # Summarize sweep
        useful_lrs = [r["lr"] for r in sweep_results if r["status"] == "useful_learning"]
        divergent_lrs = [r["lr"] for r in sweep_results if r["status"] in ("divergent", "nan_or_inf")]
        stagnant_lrs = [r["lr"] for r in sweep_results if r["status"] == "too_little_movement"]

        if not useful_lrs and sweep_results:
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.OPTIMIZATION.value,
                    severity=Severity.WARNING.value,
                    observation=f"No learning rate in test range {lr_range} demonstrated clear loss decrease.",
                    interpretation=(
                        "The loss remained either stagnant or became unstable across all tested learning rates. "
                        "This may indicate gradient scale issues, poor loss landscape, or frozen parameters."
                    ),
                    evidence={"sweep_results": sweep_results},
                    hypotheses=[
                        "Loss function produces zero or detached gradients",
                        "Gradient scale is mismatched with typical optimizer learning rates",
                        "Model architecture is saturated or dead",
                    ],
                    confidence="medium",
                    suggested_actions=["Inspect parameter gradients and run numerical gradient check."],
                )
            )
        elif useful_lrs:
            min_useful = min(useful_lrs)
            max_useful = max(useful_lrs)
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.OPTIMIZATION.value,
                    severity=Severity.INFO.value,
                    observation=f"Useful learning observed in learning rate window: [{min_useful:.1e}, {max_useful:.1e}].",
                    interpretation="Model responds favorably to standard gradient steps within this range.",
                    evidence={"useful_lrs": useful_lrs, "divergent_lrs": divergent_lrs},
                    hypotheses=["Optimization responds as expected."],
                    confidence="high",
                )
            )

    finally:
        model.load_state_dict(orig_state)

    return {
        "sweep_results": sweep_results,
        "findings": findings,
    }
