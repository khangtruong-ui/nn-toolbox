"""
Bootstrapping v1.0 verification experiment.
Tests parameter freeze isolation, kickstart convergence, loss reduction, and release readiness.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Dict, List, Optional, Union
import torch
import torch.nn as nn

from nn_toolbox.core.finding import DiagnosticFinding, FindingCategory, Severity
from nn_toolbox.detectors.bootstrap import BootstrapDetector


def verify_bootstrapping(
    model: nn.Module,
    sample_batch_or_loader: Optional[Any] = None,
    loss_fn: Optional[Callable[..., torch.Tensor]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    bootstrap_epochs: int = 3,
    freeze_param_names: Optional[List[str]] = None,
    target_score: Optional[float] = None,
    min_loss_drop: float = 0.10,
    bootstrap_results: Optional[Dict[str, Any]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Verify whether Bootstrapping v1.0 kickstarting works effectively.

    Can either:
    1. Assess an already completed bootstrap session via `bootstrap_results`.
    2. Execute an active kickstart verification experiment on sample data.

    Returns:
        Dict containing:
            - findings: List[DiagnosticFinding]
            - metrics: Dict[str, Any]
            - acceptable_fit: bool
    """
    detector = BootstrapDetector()

    # Case 1: Caller provided completed bootstrap run metrics
    if bootstrap_results is not None:
        context = {"bootstrapping": bootstrap_results}
        findings = detector.detect(context)
        acceptable = bool(bootstrap_results.get("acceptable_fit", False))
        if not acceptable:
            init_l = float(bootstrap_results.get("initial_loss", 0.0))
            fin_l = float(bootstrap_results.get("final_loss", 0.0))
            drop = (init_l - fin_l) / init_l if init_l > 1e-8 else 0.0
            acceptable = drop >= min_loss_drop and not bool(bootstrap_results.get("frozen_param_grad_leak", False))

        return {
            "findings": findings,
            "metrics": bootstrap_results,
            "acceptable_fit": acceptable,
        }

    # Case 2: Active kickstart experiment on sample data
    if sample_batch_or_loader is None or loss_fn is None:
        return {
            "findings": [],
            "metrics": {},
            "acceptable_fit": False,
        }

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    # Snapshot original state
    was_training = model.training
    orig_state = copy.deepcopy(model.state_dict())
    orig_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}

    model.train()

    try:
        # Determine frozen parameters
        frozen_names = set()
        if freeze_param_names:
            for name, p in model.named_parameters():
                if any(fp in name for fp in freeze_param_names):
                    p.requires_grad = False
                    frozen_names.add(name)
        else:
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    frozen_names.add(name)

        # Ensure at least some parameters remain trainable
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            return {
                "findings": [
                    DiagnosticFinding(
                        category=FindingCategory.BOOTSTRAP.value,
                        severity=Severity.CRITICAL.value,
                        observation="Cannot execute bootstrapping experiment: all model parameters are frozen.",
                        interpretation="At least one module/head must remain trainable during kickstarting.",
                    )
                ],
                "metrics": {},
                "acceptable_fit": False,
            }

        # Setup optimizer if not provided
        if optimizer is None:
            opt = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)
        else:
            opt = optimizer

        # Unpack input/target
        if isinstance(sample_batch_or_loader, (tuple, list)):
            x = sample_batch_or_loader[0].to(device)
            y = sample_batch_or_loader[1].to(device) if len(sample_batch_or_loader) > 1 else None
        elif isinstance(sample_batch_or_loader, dict):
            x = sample_batch_or_loader.get("image", sample_batch_or_loader.get("input", sample_batch_or_loader.get("x"))).to(device)
            y = sample_batch_or_loader.get("mask", sample_batch_or_loader.get("label", sample_batch_or_loader.get("y")))
            if isinstance(y, torch.Tensor):
                y = y.to(device)
        else:
            # First item from iterable / DataLoader
            first_b = next(iter(sample_batch_or_loader))
            if isinstance(first_b, (tuple, list)):
                x = first_b[0].to(device)
                y = first_b[1].to(device) if len(first_b) > 1 else None
            elif isinstance(first_b, dict):
                x = first_b.get("image", first_b.get("input", first_b.get("x"))).to(device)
                y = first_b.get("mask", first_b.get("label", first_b.get("y")))
                if isinstance(y, torch.Tensor):
                    y = y.to(device)
            else:
                x = first_b.to(device)
                y = None

        losses = []
        grad_leak = False

        for step in range(bootstrap_epochs):
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y) if y is not None else loss_fn(out)
            if isinstance(loss, (tuple, list)):
                loss = loss[0]
            loss_val = float(loss.item())
            losses.append(loss_val)

            loss.backward()

            # Check isolation: do frozen parameters have gradients?
            for name, p in model.named_parameters():
                if name in frozen_names and p.grad is not None:
                    if float(p.grad.abs().sum().item()) > 1e-9:
                        grad_leak = True

            opt.step()

        initial_loss = losses[0] if losses else 0.0
        final_loss = losses[-1] if losses else 0.0
        loss_drop = (initial_loss - final_loss) / initial_loss if initial_loss > 1e-8 else 0.0
        acceptable_fit = (loss_drop >= min_loss_drop) and not grad_leak and not math.isnan(final_loss)

        metrics = {
            "enabled": True,
            "run_bootstrap": True,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_drop": loss_drop,
            "best_score": 1.0 - (final_loss / max(initial_loss, 1.0)) if initial_loss > 0 else 0.5,
            "target_score": target_score,
            "min_loss_drop": min_loss_drop,
            "bootstrap_epochs": bootstrap_epochs,
            "bootstrap_examples": x.shape[0] if hasattr(x, "shape") else 1,
            "frozen_param_grad_leak": grad_leak,
            "frozen_param_count": len(frozen_names),
            "trainable_param_count": len(trainable_params),
            "acceptable_fit": acceptable_fit,
            "losses": losses,
        }

        findings = detector.detect({"bootstrapping": metrics})

        return {
            "findings": findings,
            "metrics": metrics,
            "acceptable_fit": acceptable_fit,
        }

    finally:
        # Non-destructive: restore original weights and requires_grad states
        model.load_state_dict(orig_state)
        for name, p in model.named_parameters():
            if name in orig_requires_grad:
                p.requires_grad = orig_requires_grad[name]
        model.zero_grad(set_to_none=True)
        model.train(was_training)
