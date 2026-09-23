"""
Numerical gradient check (finite difference vs autograd).
Verifies custom layers, loss functions, and backward implementations.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Union
import torch
import torch.nn as nn

from nn_toolbox.core.finding import DiagnosticFinding, FindingCategory, Severity


def gradient_check(
    module_or_model: nn.Module,
    sample_input: torch.Tensor,
    loss_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    parameters: Optional[List[str]] = None,
    max_elements_per_param: int = 50,
    eps: float = 1e-4,
    rtol: float = 1e-2,
    atol: float = 1e-4,
) -> Dict[str, Any]:
    """Perform finite-difference numerical gradient checks on selected parameters or modules."""
    module_or_model.zero_grad()
    was_training = module_or_model.training
    module_or_model.train()

    findings: List[DiagnosticFinding] = []
    param_results: Dict[str, Dict[str, Any]] = {}

    try:
        # Determine parameters to test
        target_params: Dict[str, nn.Parameter] = {}
        for name, p in module_or_model.named_parameters():
            if not p.requires_grad:
                continue
            if parameters is not None and not any(pat in name for pat in parameters):
                continue
            target_params[name] = p

        if not target_params:
            return {
                "success": False,
                "error": "No matching trainable parameters found for gradient check.",
                "param_results": {},
                "findings": [],
            }

        # 1. Forward and analytical backward pass
        out = module_or_model(sample_input)
        if isinstance(out, (tuple, list)):
            out_tensor = out[0]
        else:
            out_tensor = out

        if loss_fn is not None:
            loss = loss_fn(out_tensor)
        else:
            loss = out_tensor.float().sum()

        loss.backward()

        # Cache analytical gradients
        analytical_grads: Dict[str, torch.Tensor] = {}
        for name, p in target_params.items():
            if p.grad is not None:
                analytical_grads[name] = p.grad.detach().clone()
            else:
                analytical_grads[name] = None

        # 2. Numerical finite-difference computation
        for name, p in target_params.items():
            ana_grad = analytical_grads.get(name)
            if ana_grad is None:
                param_results[name] = {
                    "passed": False,
                    "max_relative_error": float("nan"),
                    "status": "NO_AUTOGRAD_GRADIENT",
                }
                findings.append(
                    DiagnosticFinding(
                        category=FindingCategory.BACKWARD.value,
                        severity=Severity.CRITICAL.value,
                        module=name,
                        observation=f"Parameter '{name}' did not receive any gradient during autograd backward pass.",
                        interpretation="The computation graph between output and this parameter is disconnected.",
                        evidence={"param": name},
                        hypotheses=["Detached tensor (.detach()) in forward pass", "Parameter not used in computation"],
                        confidence="high",
                    )
                )
                continue

            num_elements = p.numel()
            # Select subset of indices to check
            if num_elements > max_elements_per_param:
                step = max(1, num_elements // max_elements_per_param)
                indices = list(range(0, num_elements, step))[:max_elements_per_param]
            else:
                indices = list(range(num_elements))

            flat_p = p.data.view(-1)
            flat_ana = ana_grad.view(-1)
            max_rel_err = 0.0

            for idx in indices:
                orig_val = float(flat_p[idx].item())

                # f(theta + eps)
                flat_p[idx] = orig_val + eps
                out_plus = module_or_model(sample_input)
                if isinstance(out_plus, (tuple, list)):
                    out_plus = out_plus[0]
                loss_plus = float(loss_fn(out_plus).item()) if loss_fn else float(out_plus.sum().item())

                # f(theta - eps)
                flat_p[idx] = orig_val - eps
                out_minus = module_or_model(sample_input)
                if isinstance(out_minus, (tuple, list)):
                    out_minus = out_minus[0]
                loss_minus = float(loss_fn(out_minus).item()) if loss_fn else float(out_minus.sum().item())

                # Restore
                flat_p[idx] = orig_val

                # Central difference: (f(x+eps) - f(x-eps)) / (2*eps)
                num_grad = (loss_plus - loss_minus) / (2.0 * eps)
                ana_val = float(flat_ana[idx].item())

                denom = max(abs(num_grad) + abs(ana_val), 1e-8)
                rel_err = abs(num_grad - ana_val) / denom
                max_rel_err = max(max_rel_err, rel_err)

            passed = max_rel_err <= rtol or max_rel_err * denom <= atol

            param_results[name] = {
                "passed": passed,
                "max_relative_error": max_rel_err,
                "status": "PASS" if passed else "FAIL",
                "elements_checked": len(indices),
            }

            if not passed:
                findings.append(
                    DiagnosticFinding(
                        category=FindingCategory.BACKWARD.value,
                        severity=Severity.CRITICAL.value,
                        module=name,
                        observation=(
                            f"Numerical gradient mismatch on '{name}': "
                            f"relative error {max_rel_err:.2e} exceeds tolerance {rtol:.2e}."
                        ),
                        interpretation="Analytical autograd gradient disagrees with empirical finite-difference slope.",
                        evidence={"param": name, "relative_error": max_rel_err, "rtol": rtol},
                        hypotheses=[
                            "Incorrect custom torch.autograd.Function backward formula",
                            "In-place operation corrupting forward state needed by backward",
                            "Non-smooth or non-differentiable operation near evaluation point",
                        ],
                        confidence="high",
                        suggested_actions=["Check custom autograd backward implementations and in-place tensor operations."],
                    )
                )

        all_passed = all(r.get("passed", False) for r in param_results.values())
        if all_passed:
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.BACKWARD.value,
                    severity=Severity.INFO.value,
                    observation=f"Numerical gradient check passed across all {len(param_results)} checked parameter(s).",
                    interpretation="Autograd analytical gradients agree with empirical finite-difference gradients.",
                    confidence="high",
                )
            )

        return {
            "success": True,
            "all_passed": all_passed,
            "param_results": param_results,
            "findings": findings,
        }

    finally:
        module_or_model.zero_grad()
        if not was_training:
            module_or_model.eval()
