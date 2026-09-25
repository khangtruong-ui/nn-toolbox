"""
Detectors for Bootstrapping v1.0 kickstarting phase.
Evaluates parameter isolation, loss trajectory, sample fitting, and transition readiness.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
from nn_toolbox.core.finding import DiagnosticFinding, FindingCategory, Severity
from nn_toolbox.detectors.base import BaseDetector


class BootstrapDetector(BaseDetector):
    """Diagnoses and verifies the success of the Bootstrapping v1.0 kickstarting phase."""

    @property
    def name(self) -> str:
        return "bootstrap"

    @property
    def category(self) -> str:
        return FindingCategory.BOOTSTRAP.value

    def detect(self, context: Dict[str, Any]) -> List[DiagnosticFinding]:
        findings: List[DiagnosticFinding] = []

        boot_info = context.get("bootstrapping") or context.get("bootstrap_metrics")
        if not boot_info or not isinstance(boot_info, dict):
            return findings

        # Check if bootstrapping was actually executed
        if not boot_info.get("enabled", True) and not boot_info.get("run_bootstrap", True):
            return findings

        initial_loss = float(boot_info.get("initial_loss", 0.0))
        final_loss = float(boot_info.get("final_loss", 0.0))
        best_score = float(boot_info.get("best_score", 0.0))
        target_score = boot_info.get("target_score", None)
        min_loss_drop = float(boot_info.get("min_loss_drop", 0.10))
        epochs = int(boot_info.get("bootstrap_epochs", boot_info.get("epochs", 1)))
        num_samples = int(boot_info.get("bootstrap_examples", boot_info.get("num_samples", 0)))
        grad_leak = bool(boot_info.get("frozen_param_grad_leak", False))
        acceptable_fit = bool(boot_info.get("acceptable_fit", False))

        # Calculate relative loss reduction
        if initial_loss > 1e-8:
            loss_drop = (initial_loss - final_loss) / initial_loss
        else:
            loss_drop = 0.0

        # 1. Check for gradient leakage into frozen parameters
        if grad_leak:
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.BOOTSTRAP.value,
                    severity=Severity.CRITICAL.value,
                    observation="Bootstrapping v1.0 isolation violation: Non-zero gradients detected on frozen parameters during kickstart.",
                    interpretation="Parameters designated as frozen received backpropagation updates, which undermines kickstart isolation.",
                    evidence=boot_info,
                    hypotheses=[
                        "requires_grad was not cleared on frozen modules",
                        "Optimizer included frozen parameter groups during kickstart",
                    ],
                    confidence="high",
                    suggested_actions=[
                        "Ensure p.requires_grad = False on frozen modules before bootstrap optimizer step",
                        "Filter optimizer parameters: [p for p in model.parameters() if p.requires_grad]",
                    ],
                )
            )

        # 2. Check for numeric divergence / explosion
        if math.isnan(final_loss) or math.isinf(final_loss) or (initial_loss > 0 and final_loss > initial_loss * 1.15):
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.BOOTSTRAP.value,
                    severity=Severity.CRITICAL.value,
                    observation=f"Bootstrapping v1.0 divergence detected: Loss increased during kickstart phase ({initial_loss:.4f} -> {final_loss:.4f}, change: {loss_drop*100:+.1f}%).",
                    interpretation="The kickstart phase destabilized the model instead of fitting the sample subset.",
                    evidence=boot_info,
                    hypotheses=[
                        "Bootstrap learning rate is excessively large",
                        "Improper initialization produced exploding activations",
                        "Numerical instability in loss calculation",
                    ],
                    confidence="high",
                    suggested_actions=[
                        "Reduce bootstrap_lr by an order of magnitude",
                        "Use kaiming_normal initialization on unfrozen head layers",
                        "Enable gradient clipping during bootstrapping",
                    ],
                )
            )
            return findings

        # 3. Check for lack of acceptable fit / stagnation
        has_target_score = target_score is not None and float(target_score) > 0.0
        score_passed = (best_score >= float(target_score)) if has_target_score else True
        loss_passed = loss_drop >= min_loss_drop

        if not acceptable_fit and not (loss_passed and score_passed):
            score_msg = f" with score {best_score:.4f} (target: {target_score})" if has_target_score else f" (score: {best_score:.4f})"
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.BOOTSTRAP.value,
                    severity=Severity.WARNING.value,
                    observation=f"Bootstrapping v1.0 did not achieve acceptable fit: Relative loss reduction was {loss_drop*100:.1f}% (target >= {min_loss_drop*100:.1f}%){score_msg}.",
                    interpretation="The kickstart phase failed to adequately adapt the unfrozen components to the training distribution before full model release.",
                    evidence=boot_info,
                    hypotheses=[
                        "Bootstrap epochs too few to achieve convergence on sample subset",
                        "Bootstrap learning rate too small for rapid kickstart adaptation",
                        "Unfrozen module capacity or initialization mismatch",
                    ],
                    confidence="medium",
                    suggested_actions=[
                        "Increase bootstrap_epochs (e.g. from 3 to 5 or 10)",
                        "Increase bootstrap_lr or use AdamW with weight decay 1e-4",
                        "Verify that head initialization matches the activation functions",
                    ],
                )
            )
            return findings

        # 4. Verified Healthy Bootstrapping v1.0 Confirmation
        score_detail = f", score: {best_score:.4f}" if best_score > 0 else ""
        strategy = boot_info.get("strategy", "channel_stream")
        if strategy in ("channel_stream", "dimension_stream"):
            stream_ratio = boot_info.get("stream_ratio", None)
            ratio_str = f" (stream ratio: {float(stream_ratio):.0%})" if stream_ratio is not None else ""
            stream_obs = f"End-to-end channel stream computation{ratio_str} confirmed with zero gradient leakage into frozen tail channels."
        else:
            stream_obs = "Frozen parameters remained isolated and model is primed for full release."

        findings.append(
            DiagnosticFinding(
                category=FindingCategory.BOOTSTRAP.value,
                severity=Severity.INFO.value,
                observation=(
                    f"Bootstrapping v1.0 verified successful: Model achieved acceptable kickstart fit on {num_samples} samples "
                    f"({epochs} epochs; loss: {initial_loss:.4f} -> {final_loss:.4f}, drop: {loss_drop*100:.1f}%{score_detail}). "
                    f"{stream_obs}"
                ),
                interpretation="The kickstarting phase succeeded in fitting the core stream representations to the data manifold, preventing gradient shock upon full parameter release.",
                evidence=boot_info,
                confidence="high",
            )
        )

        return findings
