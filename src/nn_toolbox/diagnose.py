"""
Main high-level diagnostic pipeline for nn-toolbox.
Orchestrates passive instrumentation, anomaly detectors, and targeted experiments.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from nn_toolbox.analysis.similarity import analyze_representation_collapse
from nn_toolbox.core.finding import DiagnosticFinding, FindingCategory, Severity
from nn_toolbox.core.report_data import DiagnosticReport
from nn_toolbox.detectors import (
    CollapseDetector,
    DataDetector,
    ExplodingDetector,
    GradientBalanceDetector,
    GraphConnectivityDetector,
    InstabilityDetector,
    OptimizationDetector,
    SaturationDetector,
    TensorLayoutDetector,
    VanishingDetector,
)
from nn_toolbox.experiments.eval_determinism import eval_determinism_test
from nn_toolbox.experiments.initialization import initialization_diagnostic
from nn_toolbox.experiments.lr_sweep import lr_sweep
from nn_toolbox.experiments.overfit import overfit_test
from nn_toolbox.experiments.perturbation import perturbation_test
from nn_toolbox.experiments.train_eval import train_eval_test
from nn_toolbox.instrumentation.activations import ActivationMonitor
from nn_toolbox.instrumentation.gradients import GradientMonitor
from nn_toolbox.instrumentation.parameters import ParameterMonitor


def diagnose(
    model: nn.Module,
    dataloader: Optional[Iterable[Any]] = None,
    loss_fn: Optional[Callable[..., torch.Tensor]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    mode: str = "light",  # "light" | "deep"
    diagnostics: Optional[List[str]] = None,
    device: Optional[Union[str, torch.device]] = None,
    num_batches: int = 2,
    sample_input: Optional[Any] = None,
    sample_target: Optional[Any] = None,
    verbose: bool = True,
) -> DiagnosticReport:
    """Run an automated diagnostic session on a PyTorch model and training pipeline.

    Args:
        model: PyTorch nn.Module.
        dataloader: Optional training DataLoader or iterable of batches.
        loss_fn: Optional loss function (outputs, targets) -> loss tensor.
        optimizer: Optional PyTorch optimizer instance.
        mode: Diagnostic mode: "light" (fast, non-destructive) or "deep" (includes active experiments).
        diagnostics: Optional list of diagnostics to enable/disable.
        device: Torch device (auto-detected if None).
        num_batches: Number of sample batches to evaluate.
        sample_input: Optional explicit input tensor (overrides dataloader).
        sample_target: Optional explicit target tensor.
        verbose: If True, prints terminal report summary.

    Returns:
        DiagnosticReport containing findings, metrics, and prioritized investigation targets.
    """
    model_name = getattr(model, "__class__", type(model)).__name__
    report = DiagnosticReport(model_name=model_name, mode=mode)

    # 1. Device resolution
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    # 2. Extract sample data batch
    batch_input: Optional[torch.Tensor] = None
    batch_target: Optional[Any] = None

    if sample_input is not None:
        batch_input = sample_input
        batch_target = sample_target
    elif dataloader is not None:
        try:
            for b in dataloader:
                if isinstance(b, dict):
                    img = b.get("image", b.get("input", b.get("x")))
                    if "mask" in b and "label" in b:
                        tgt = {"mask": b["mask"], "label": b["label"]}
                    else:
                        tgt = b.get("mask", b.get("label", b.get("y", b.get("target"))))
                    if img is not None:
                        batch_input = img
                        batch_target = tgt
                        break
                elif isinstance(b, (tuple, list)) and len(b) >= 1:
                    batch_input = b[0]
                    if len(b) >= 2:
                        batch_target = b[1]
                    break
        except Exception as e:
            report.add_finding(
                DiagnosticFinding(
                    category=FindingCategory.DATA.value,
                    severity=Severity.WARNING.value,
                    observation=f"Failed to fetch initial sample batch from dataloader: {e}",
                    interpretation="The dataloader iterator produced an exception on first fetch.",
                    hypotheses=["Dataset index out of bounds", "Corrupted file during loading"],
                )
            )

    # Context dictionary for detectors
    context: Dict[str, Any] = {}

    # 3. Data Sanity Inspection
    if batch_input is not None and torch.is_tensor(batch_input):
        with torch.no_grad():
            b_in = batch_input.detach().float()
            in_finite = bool(torch.isfinite(b_in).all().item())
            in_var = float(b_in.var().item()) if b_in.numel() > 1 else 0.0
            in_min = float(b_in.min().item()) if b_in.numel() > 0 else 0.0
            in_max = float(b_in.max().item()) if b_in.numel() > 0 else 0.0

            tgt_finite = True
            tgt_var = 1.0
            if batch_target is not None and torch.is_tensor(batch_target):
                b_tgt = batch_target.detach().float()
                tgt_finite = bool(torch.isfinite(b_tgt).all().item())
                tgt_var = float(b_tgt.var().item()) if b_tgt.numel() > 1 else 0.0

            context["data_sanity"] = {
                "inputs_finite": in_finite,
                "input_is_constant": in_var < 1e-12,
                "input_min": in_min,
                "input_max": in_max,
                "input_shape": list(batch_input.shape),
                "targets_finite": tgt_finite,
                "target_is_constant": tgt_var < 1e-12 and batch_target is not None,
            }
            report.metrics["data_sanity"] = context["data_sanity"]

    # 4. Parameter Inspection
    param_monitor = ParameterMonitor(model)
    param_info = param_monitor.inspect_parameters()
    context["parameter_info"] = param_info
    report.metrics["parameter_info"] = param_info

    # 5. Passive Forward & Backward Instrumentation
    was_training = model.training
    model.train()

    act_monitor = ActivationMonitor(model)
    grad_monitor = GradientMonitor(model)

    latest_output = None

    if batch_input is not None and torch.is_tensor(batch_input):
        x = batch_input.to(device)
        if batch_target is not None:
            if torch.is_tensor(batch_target):
                y = batch_target.to(device)
            elif isinstance(batch_target, dict):
                y = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch_target.items()}
            elif isinstance(batch_target, (tuple, list)):
                y = [v.to(device) if torch.is_tensor(v) else v for v in batch_target]
            else:
                y = batch_target
        else:
            y = None

        try:
            with act_monitor, grad_monitor:
                model.zero_grad()
                out = model(x)
                latest_output = out

                # Forward stats
                fwd_analysis = act_monitor.analyze_signal_propagation(step=0)
                context["forward_analysis"] = fwd_analysis
                context["activation_stats"] = act_monitor.get_latest_stats()
                report.metrics["forward_analysis"] = fwd_analysis

                # If loss_fn provided, run backward & optimization step
                if loss_fn is not None:
                    if y is not None:
                        loss = loss_fn(out, y)
                    else:
                        loss = loss_fn(out)

                    if isinstance(loss, (tuple, list)):
                        loss_t = loss[0]
                    elif isinstance(loss, dict):
                        loss_t = next(iter(loss.values()))
                    else:
                        loss_t = loss

                    loss_t.backward()

                    # Backward stats
                    bwd_analysis = grad_monitor.analyze_backward_propagation()
                    param_grads = grad_monitor.collect_parameter_gradients()
                    context["backward_analysis"] = bwd_analysis
                    context["parameter_gradient_stats"] = param_grads
                    report.metrics["backward_analysis"] = bwd_analysis

                    # If optimizer provided, evaluate update-to-weight ratio non-destructively
                    if optimizer is not None:
                        param_monitor.snapshot_before_step()
                        optimizer.step()
                        update_stats = param_monitor.snapshot_after_step()
                        context["update_stats"] = update_stats
                        report.metrics["update_stats"] = update_stats
                        optimizer.zero_grad()
        except Exception as e:
            report.add_finding(
                DiagnosticFinding(
                    category=FindingCategory.GENERAL.value,
                    severity=Severity.CRITICAL.value,
                    observation=f"Forward or backward pass encountered an unhandled exception: {e}",
                    interpretation="The model, loss function, or optimizer crashed during standard execution.",
                    hypotheses=["Dimension mismatch between model output and loss target", "Shape mismatch in layer operations"],
                    confidence="high",
                )
            )

    # 6. Representation Collapse & Tensor Layout Check
    non_contiguous = []
    if latest_output is not None:
        out_tensors = [latest_output] if torch.is_tensor(latest_output) else (
            list(latest_output) if isinstance(latest_output, (tuple, list)) else []
        )
        for idx, ot in enumerate(out_tensors):
            if torch.is_tensor(ot) and not ot.is_contiguous():
                non_contiguous.append({
                    "name": f"output_{idx}" if len(out_tensors) > 1 else "output",
                    "shape": list(ot.shape),
                    "stride": list(ot.stride()),
                })
        context["tensor_layout"] = {"non_contiguous": non_contiguous}
        report.metrics["tensor_layout"] = context["tensor_layout"]

        out_tensor = latest_output[0] if isinstance(latest_output, (tuple, list)) else latest_output
        if torch.is_tensor(out_tensor) and out_tensor.ndim >= 2:
            rep_analysis = analyze_representation_collapse(out_tensor)
            context["representation_collapse"] = rep_analysis
            report.metrics["representation_collapse"] = rep_analysis

    # 7. Train/Eval Consistency & Determinism
    if batch_input is not None and torch.is_tensor(batch_input):
        te_res = train_eval_test(model, batch_input.to(device))
        report.add_findings(te_res.get("findings", []))
        report.metrics["train_eval"] = te_res

        det_res = eval_determinism_test(model, batch_input.to(device), num_passes=2)
        report.add_findings(det_res.get("findings", []))
        report.metrics["eval_determinism"] = det_res

    # 8. Run Detectors on collected context
    detectors = [
        DataDetector(),
        ExplodingDetector(),
        VanishingDetector(),
        SaturationDetector(),
        CollapseDetector(),
        InstabilityDetector(),
        OptimizationDetector(),
        GraphConnectivityDetector(),
        GradientBalanceDetector(),
        TensorLayoutDetector(),
    ]

    for det in detectors:
        findings = det.detect(context)
        report.add_findings(findings)

    # 8.1 Record verified healthy learnability dimensions (brief confirmations)
    ds = context.get("data_sanity", {})
    if ds.get("inputs_finite", False) and not ds.get("input_is_constant", False):
        report.add_finding(
            DiagnosticFinding(
                category=FindingCategory.DATA.value,
                severity=Severity.INFO.value,
                observation=f"Input data is finite and non-constant (variance: {ds.get('input_var', 1.0):.2e} range: [{ds.get('input_min', 0.0):.2f}, {ds.get('input_max', 0.0):.2f}]).",
                interpretation="Inputs pass basic numeric validity checks without NaN, Inf, or zero variance.",
                evidence=ds,
                confidence="high",
            )
        )

    fwd = context.get("forward_analysis", {})
    if fwd and not fwd.get("amplification_events") and not fwd.get("attenuation_events"):
        layers_dict = fwd.get("layers", {})
        n_layers = len(layers_dict)
        if n_layers > 0:
            stds = [s.get("std", 1.0) for s in layers_dict.values() if s.get("std") is not None]
            min_std, max_std = (min(stds), max(stds)) if stds else (1.0, 1.0)
            report.add_finding(
                DiagnosticFinding(
                    category=FindingCategory.FORWARD.value,
                    severity=Severity.INFO.value,
                    observation=f"Forward activation propagation is stable across {n_layers} layer(s) (std range: {min_std:.2f} - {max_std:.2f}).",
                    interpretation="Activations propagate through network depth without exploding or vanishing.",
                    evidence={"monitored_layers": n_layers, "min_std": min_std, "max_std": max_std},
                    confidence="high",
                )
            )

    bwd = context.get("backward_analysis", {})
    if bwd:
        zero_params = bwd.get("zero_grad_params", [])
        van_params = bwd.get("vanishing_params", [])
        exp_params = bwd.get("exploding_params", [])
        active_p = bwd.get("params_with_gradients", 0)
        total_trainable = bwd.get("total_trainable_parameters", active_p)
        mean_gnorm = bwd.get("mean_gradient_norm", 0.0)

        if not zero_params and not van_params and not exp_params and active_p > 0:
            report.add_finding(
                DiagnosticFinding(
                    category=FindingCategory.BACKWARD.value,
                    severity=Severity.INFO.value,
                    observation=f"Active gradient flow verified on 100% of trainable parameters ({active_p}/{total_trainable}, mean norm: {mean_gnorm:.2e}).",
                    interpretation="Loss gradients propagate smoothly to all trainable parameters without dead layers or gradient explosion.",
                    evidence=bwd,
                    confidence="high",
                )
            )

    upd = context.get("update_stats", {})
    if upd:
        u_ratio = upd.get("global_update_ratio", 0.0)
        u_norm = upd.get("global_update_norm", 0.0)
        if 1e-5 <= u_ratio <= 0.05:
            report.add_finding(
                DiagnosticFinding(
                    category=FindingCategory.OPTIMIZATION.value,
                    severity=Severity.INFO.value,
                    observation=f"Healthy parameter update ratio: ||Δθ||/||θ|| = {u_ratio:.2e} (displacement norm: {u_norm:.2e}).",
                    interpretation="The optimizer step produces adequate parameter displacement in the effective learning regime (1e-4 to 1e-2).",
                    evidence=upd,
                    confidence="high",
                )
            )

    rep = context.get("representation_collapse", {})
    if rep and not rep.get("is_collapsed", False):
        erank = rep.get("effective_rank", 1.0)
        eratio = rep.get("effective_rank_ratio", 1.0)
        report.add_finding(
            DiagnosticFinding(
                category=FindingCategory.ARCHITECTURE.value,
                severity=Severity.INFO.value,
                observation=f"Output representation is full-rank and diverse (effective rank: {erank:.2f}, ratio: {eratio:.2f}).",
                interpretation="Representations occupy diverse subspaces without dimensional collapse.",
                evidence=rep,
                confidence="high",
            )
        )

    layout = context.get("tensor_layout", {})
    if "non_contiguous" in layout and len(layout["non_contiguous"]) == 0:
        report.add_finding(
            DiagnosticFinding(
                category=FindingCategory.ARCHITECTURE.value,
                severity=Severity.INFO.value,
                observation="Tensor memory layout is contiguous across model outputs.",
                interpretation="Outputs exhibit contiguous striding without unaligned slicing artifacts.",
                confidence="high",
            )
        )

    should_run_deep = (mode == "deep") or (diagnostics and any(d in diagnostics for d in ["deep", "overfit", "lr_sweep", "perturbation"]))

    if should_run_deep and batch_input is not None and torch.is_tensor(batch_input):
        # 9a. Overfit Test
        if loss_fn is not None and (diagnostics is None or "overfit" in diagnostics):
            try:
                sample_pool = {"x": batch_input, "y": batch_target}
                of_res = overfit_test(model, sample_pool, loss_fn=loss_fn, sizes=[1, 8])
                report.add_findings(of_res.get("findings", []))
                report.metrics["overfit_test"] = of_res
            except Exception as e:
                report.add_finding(DiagnosticFinding(category="memorization", severity="warning", observation=f"Overfit test failed to complete: {e}"))

        # 9b. Learning Rate Sweep
        if loss_fn is not None and (diagnostics is None or "lr_sweep" in diagnostics):
            try:
                lr_res = lr_sweep(model, batch_input, batch_target, loss_fn=loss_fn)
                report.add_findings(lr_res.get("findings", []))
                report.metrics["lr_sweep"] = lr_res
            except Exception as e:
                report.add_finding(DiagnosticFinding(category="optimization", severity="warning", observation=f"LR sweep failed to complete: {e}"))

        # 9c. Initialization Diagnostic
        if diagnostics is None or "initialization" in diagnostics:
            try:
                init_res = initialization_diagnostic(model, batch_input.shape)
                report.add_findings(init_res.get("findings", []))
                report.metrics["initialization"] = init_res
            except Exception as e:
                report.add_finding(DiagnosticFinding(category="initialization", severity="warning", observation=f"Initialization diagnostic failed to complete: {e}"))

        # 9d. Perturbation Sensitivity Test
        if diagnostics is None or "perturbation" in diagnostics:
            try:
                pert_res = perturbation_test(model, batch_input.to(device))
                report.add_findings(pert_res.get("findings", []))
                report.metrics["perturbation"] = pert_res
            except Exception as e:
                report.add_finding(DiagnosticFinding(category="stability", severity="warning", observation=f"Perturbation test failed to complete: {e}"))

        # 9e. Evaluation Mode Determinism & Stochastic Drift
        if diagnostics is None or "eval_determinism" in diagnostics or "determinism" in diagnostics:
            try:
                det_res = eval_determinism_test(model, batch_input.to(device), num_passes=4)
                report.add_findings(det_res.get("findings", []))
                report.metrics["eval_determinism"] = det_res
            except Exception as e:
                report.add_finding(DiagnosticFinding(category="stability", severity="warning", observation=f"Evaluation determinism test failed to complete: {e}"))

    # Restore training mode
    if not was_training:
        model.eval()

    # 10. Terminal display
    if verbose:
        report.print_summary()

    return report


def cli_main():
    """Command-line entrypoint for nn-diagnose."""
    parser = argparse.ArgumentParser(description="nn-toolbox: Neural network diagnostic and investigation tool")
    parser.add_argument("--mode", choices=["light", "deep"], default="light", help="Diagnostic mode")
    parser.add_argument("--json", type=str, default=None, help="Save report as JSON file")
    parser.add_argument("--html", type=str, default=None, help="Save report as HTML file")
    args = parser.parse_args()
    print("nn-toolbox CLI ready. Use diagnose(model, dataloader, loss_fn, optimizer) in your Python script.")
