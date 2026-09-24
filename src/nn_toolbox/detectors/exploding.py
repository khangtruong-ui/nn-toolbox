"""
Detectors for forward activation explosions and backward gradient explosions.
"""

from __future__ import annotations

from typing import Any, Dict, List
from nn_toolbox.core.finding import DiagnosticFinding, FindingCategory, Severity
from nn_toolbox.detectors.base import BaseDetector


class ExplodingDetector(BaseDetector):
    """Detects exploding activations during forward passes and exploding gradients during backward passes."""

    @property
    def name(self) -> str:
        return "exploding"

    @property
    def category(self) -> str:
        return FindingCategory.FORWARD.value

    def detect(self, context: Dict[str, Any]) -> List[DiagnosticFinding]:
        findings: List[DiagnosticFinding] = []

        # 1. Forward activation explosion checks
        fwd_analysis = context.get("forward_analysis", {})
        amp_events = fwd_analysis.get("amplification_events", [])
        param_details = context.get("parameter_info", {}).get("param_details", {})

        trainable_prefixes = set()
        if param_details:
            for p_name, p_info in param_details.items():
                if p_info.get("requires_grad", False):
                    parts = p_name.split(".")
                    for i in range(1, len(parts) + 1):
                        trainable_prefixes.add(".".join(parts[:i]))

        for event in amp_events:
            mod_name = event["module"]
            ratio = event["ratio_to_prev"]
            ratio_med = event["ratio_to_median"]
            std_val = event["std"]

            is_frozen = False
            if param_details:
                mod_parts = mod_name.split(".")
                is_frozen = not any(".".join(mod_parts[:i]) in trainable_prefixes for i in range(1, len(mod_parts) + 1))

            if is_frozen:
                severity = Severity.WARNING.value if ratio > 20.0 else Severity.INFO.value
                obs = f"Module '{mod_name}' [FROZEN] has activation std {ratio:.1f}× larger than the previous layer (std: {std_val:.2f}, {ratio_med:.1f}× median)."
                hypotheses = [
                    "Structural feature scaling within frozen pretrained foundation backbone",
                    "Input range mismatch for frozen backbone (e.g. unnormalized input tensor)",
                ]
                suggested_actions = [
                    "Verify input tensor normalization matches pretrained backbone expectations.",
                ]
            else:
                severity = Severity.CRITICAL.value if (ratio > 20.0 or std_val > 1e4) else Severity.WARNING.value
                obs = f"Module '{mod_name}' has activation std {ratio:.1f}× larger than the previous layer (std: {std_val:.2f}, {ratio_med:.1f}× median)."
                hypotheses = [
                    "Unscaled residual connection addition",
                    "Missing LayerNorm / BatchNorm / GroupNorm layer",
                    "Large weight initialization scaling",
                    "Unbounded activation function without clipping",
                ]
                suggested_actions = [
                    f"Check normalization before or after '{mod_name}'",
                    "Verify residual branch scale factor (e.g. 1/sqrt(2) or learnable gamma)",
                ]

            event_with_frozen = dict(event)
            event_with_frozen["is_frozen"] = is_frozen

            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.FORWARD.value,
                    severity=severity,
                    module=mod_name,
                    observation=obs,
                    interpretation="This pattern is consistent with possible forward-signal amplification or missing normalization.",
                    evidence=event_with_frozen,
                    hypotheses=hypotheses,
                    confidence="medium" if severity == Severity.WARNING.value else "high",
                    suggested_actions=suggested_actions,
                )
            )

        # 2. Check for non-finite activations
        latest_act_stats = context.get("activation_stats", {})
        for mod, stats in latest_act_stats.items():
            if stats.get("nan_fraction", 0.0) > 0 or stats.get("inf_fraction", 0.0) > 0:
                findings.append(
                    DiagnosticFinding(
                        category=FindingCategory.FORWARD.value,
                        severity=Severity.CRITICAL.value,
                        module=mod,
                        observation=f"Activation in module '{mod}' contains non-finite values (NaN: {stats.get('nan_fraction', 0.0)*100:.1f}%, Inf: {stats.get('inf_fraction', 0.0)*100:.1f}%).",
                        interpretation="Numerical overflow or division by zero occurred during forward evaluation.",
                        evidence=stats,
                        hypotheses=[
                            "Numerical overflow in exponential/log/division op",
                            "Unstable normalization with zero variance and small epsilon",
                            "FP16 underflow/overflow",
                        ],
                        confidence="high",
                        suggested_actions=[
                            "Inspect layer operations for epsilons in division/sqrt",
                            "Consider using bfloat16 or float32 for sensitive normalization",
                        ],
                    )
                )

        # 3. Backward gradient explosion checks
        bwd_analysis = context.get("backward_analysis", {})
        exploding_params = bwd_analysis.get("exploding_params", [])
        for exp in exploding_params:
            param_name = exp["param"]
            gnorm = exp["grad_norm"]
            ratio = exp.get("ratio_to_median", 1.0)

            severity = Severity.CRITICAL.value if (gnorm > 1e4 or (ratio > 50.0 and gnorm > 100.0)) else Severity.WARNING.value
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.BACKWARD.value,
                    severity=severity,
                    module=param_name,
                    observation=f"Parameter '{param_name}' gradient norm is unusually large ({gnorm:.2e}, {ratio:.1f}× model median).",
                    interpretation="This pattern indicates localized or global gradient explosion.",
                    evidence=exp,
                    hypotheses=[
                        "Steep loss landscape or extreme loss weighting",
                        "Recurrent or deep unnormalized backward chain",
                        "Exploding upstream activations propagating to gradients",
                    ],
                    confidence="medium",
                    suggested_actions=[
                        "Enable or reduce gradient clipping threshold",
                        "Review learning rate and loss scaling",
                    ],
                )
            )

        return findings
