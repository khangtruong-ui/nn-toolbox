"""
Detectors for vanishing forward signals and vanishing/zero gradients.
"""

from __future__ import annotations

from typing import Any, Dict, List
from nn_toolbox.core.finding import DiagnosticFinding, FindingCategory, Severity
from nn_toolbox.detectors.base import BaseDetector


class VanishingDetector(BaseDetector):
    """Detects vanishing forward activation signals and vanishing or dead backward gradients."""

    @property
    def name(self) -> str:
        return "vanishing"

    @property
    def category(self) -> str:
        return FindingCategory.BACKWARD.value

    def detect(self, context: Dict[str, Any]) -> List[DiagnosticFinding]:
        findings: List[DiagnosticFinding] = []

        # 1. Forward signal attenuation
        fwd_analysis = context.get("forward_analysis", {})
        att_events = fwd_analysis.get("attenuation_events", [])
        for att in att_events:
            mod_name = att["module"]
            std_val = att["std"]
            ratio = att["ratio_to_prev"]

            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.FORWARD.value,
                    severity=Severity.WARNING.value,
                    module=mod_name,
                    observation=f"Module '{mod_name}' activation scale collapsed (std: {std_val:.2e}, {ratio:.2e}× previous layer).",
                    interpretation="Forward signal is severely attenuated, potentially causing downstream layers to receive nearly constant inputs.",
                    evidence=att,
                    hypotheses=[
                        "Dead activation functions (e.g. all-negative inputs to ReLU)",
                        "Excessive regularization or shrinkage weights",
                        "Multiplication by near-zero scaling factor",
                    ],
                    confidence="medium",
                    suggested_actions=[
                        f"Inspect input distribution and biases in '{mod_name}'",
                        "Consider LeakyReLU or GELU instead of standard ReLU",
                    ],
                )
            )

        # 2. Vanishing gradients
        bwd_analysis = context.get("backward_analysis", {})
        vanishing_params = bwd_analysis.get("vanishing_params", [])
        for van in vanishing_params:
            param_name = van["param"]
            gnorm = van["grad_norm"]

            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.BACKWARD.value,
                    severity=Severity.WARNING.value,
                    module=param_name,
                    observation=f"Parameter '{param_name}' gradient norm is near zero ({gnorm:.2e}).",
                    interpretation="Backward gradient signal is vanishing before reaching this layer.",
                    evidence=van,
                    hypotheses=[
                        "Deep un-skip-connected architecture with saturating activations",
                        "Saturated sigmoid/tanh functions killing backward gradients",
                        "Zero loss sensitivity with respect to this layer's output",
                    ],
                    confidence="medium",
                    suggested_actions=[
                        "Add residual skip connections",
                        "Verify activation functions are non-saturating",
                    ],
                )
            )

        # 3. Completely missing gradients on trainable parameters
        zero_grad_params = bwd_analysis.get("zero_grad_params", [])
        if zero_grad_params:
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.BACKWARD.value,
                    severity=Severity.CRITICAL.value,
                    observation=f"{len(zero_grad_params)} trainable parameter(s) received NO gradient during backward pass: {zero_grad_params[:5]}{'...' if len(zero_grad_params)>5 else ''}.",
                    interpretation="The computation graph between the loss and these parameters appears detached or unused.",
                    evidence={"zero_grad_params": zero_grad_params},
                    hypotheses=[
                        "Accidental .detach() call in forward method",
                        "Parameter defined in __init__ but never invoked in forward()",
                        "Condition or branching logic bypassing layer execution",
                    ],
                    confidence="high",
                    suggested_actions=[
                        "Verify that every sublayer's output connects to the final loss output",
                        "Search model forward implementation for .detach() or numpy conversions",
                    ],
                )
            )

        return findings
