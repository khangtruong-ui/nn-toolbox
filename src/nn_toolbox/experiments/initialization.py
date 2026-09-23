"""
Initialization diagnostic using synthetic inputs (x ~ Normal(0, 1)).
Inspects signal propagation at initialization to separate architecture/init problems
from later optimization dynamics.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from nn_toolbox.core.finding import DiagnosticFinding, FindingCategory, Severity
from nn_toolbox.instrumentation.activations import ActivationMonitor
from nn_toolbox.instrumentation.gradients import GradientMonitor


def initialization_diagnostic(
    model: nn.Module,
    sample_shape_or_input: Union[Tuple[int, ...], torch.Tensor],
    loss_fn: Optional[Callable[[Any], torch.Tensor]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Test forward and backward signal propagation at initialization using synthetic standard normal inputs."""
    if device is None:
        device = next(model.parameters()).device

    # Generate synthetic input x ~ Normal(0, 1)
    if isinstance(sample_shape_or_input, torch.Tensor):
        x = torch.randn_like(sample_shape_or_input, device=device)
    else:
        x = torch.randn(sample_shape_or_input, device=device)

    act_monitor = ActivationMonitor(model)
    grad_monitor = GradientMonitor(model)

    was_training = model.training
    model.train()

    findings: List[DiagnosticFinding] = []

    try:
        act_monitor.attach()
        grad_monitor.attach()

        model.zero_grad()
        out = model(x)

        if isinstance(out, (tuple, list)):
            out_tensor = out[0]
        elif isinstance(out, dict):
            out_tensor = next(iter(out.values()))
        else:
            out_tensor = out

        # Forward propagation analysis
        fwd_analysis = act_monitor.analyze_signal_propagation(step=0)

        # Backward pass
        if loss_fn is not None:
            loss = loss_fn(out_tensor)
        else:
            loss = out_tensor.float().sum()

        loss.backward()
        bwd_analysis = grad_monitor.analyze_backward_propagation()

        # Check for forward explosions/attenuations
        amp_events = fwd_analysis.get("amplification_events", [])
        att_events = fwd_analysis.get("attenuation_events", [])

        if amp_events:
            first_amp = amp_events[0]
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.INITIALIZATION.value,
                    severity=Severity.WARNING.value,
                    module=first_amp["module"],
                    observation=(
                        f"At initialization, activation scale amplified significantly at '{first_amp['module']}' "
                        f"(std: {first_amp['std']:.2f}, {first_amp['ratio_to_prev']:.1f}× previous layer)."
                    ),
                    interpretation="Initial weight scale or missing normalization is causing signal explosion during forward propagation.",
                    evidence={"amplification_events": amp_events},
                    hypotheses=[
                        "Weight initialization variance is too large (e.g. standard Normal instead of He/Xavier)",
                        "Unscaled residual additions without normalization",
                        "Missing activation scaling in attention or convolution blocks",
                    ],
                    confidence="medium",
                    suggested_actions=["Check initialization scheme (e.g. nn.init.kaiming_normal_)", "Verify LayerNorm/BatchNorm placement"],
                )
            )

        if att_events:
            first_att = att_events[0]
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.INITIALIZATION.value,
                    severity=Severity.WARNING.value,
                    module=first_att["module"],
                    observation=(
                        f"At initialization, activation scale collapsed at '{first_att['module']}' "
                        f"(std: {first_att['std']:.2e}, {first_att['ratio_to_prev']:.2e}× previous layer)."
                    ),
                    interpretation="Initial forward signal is vanishing early in the network.",
                    evidence={"attenuation_events": att_events},
                    hypotheses=[
                        "Weight initialization is too small",
                        "Premature activation death (e.g. ReLU dead with large negative bias)",
                    ],
                    confidence="medium",
                    suggested_actions=["Check weight initialization gain and bias initialization."],
                )
            )

        # Check backward signals
        bwd_vanishing = bwd_analysis.get("vanishing_params", [])
        bwd_exploding = bwd_analysis.get("exploding_params", [])

        if bwd_exploding:
            first_exp = bwd_exploding[0]
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.BACKWARD.value,
                    severity=Severity.WARNING.value,
                    module=first_exp["param"],
                    observation=f"At initialization, parameter '{first_exp['param']}' produced exploding gradients (norm: {first_exp['grad_norm']:.2e}).",
                    interpretation="Gradient signal is exploding at initialization prior to any parameter updates.",
                    evidence={"exploding_params": bwd_exploding},
                    hypotheses=["Unscaled backward path", "Initialization variance mismatch"],
                    confidence="medium",
                    suggested_actions=["Verify gradient clipping and initialization scaling."],
                )
            )

        if not amp_events and not att_events and not bwd_exploding:
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.INITIALIZATION.value,
                    severity=Severity.INFO.value,
                    observation="Forward and backward signals propagate stably at initialization with unit-variance inputs.",
                    interpretation="Initial signal scaling is well-behaved across network depth.",
                    evidence={"median_fwd_std": fwd_analysis.get("median_std", 1.0)},
                    hypotheses=["Initialization is stable."],
                    confidence="high",
                )
            )

        return {
            "forward_analysis": fwd_analysis,
            "backward_analysis": bwd_analysis,
            "findings": findings,
        }

    finally:
        act_monitor.detach()
        grad_monitor.detach()
        model.zero_grad()
        if not was_training:
            model.eval()
