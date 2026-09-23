"""
Layer and module ablation diagnostic.
Safely tests bypassing candidate modules (where input/output dimensions match)
to localize anomalous behavior.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
import torch
import torch.nn as nn

from nn_toolbox.core.finding import DiagnosticFinding, FindingCategory, Severity


class IdentityBypass(nn.Module):
    """Identity bypass module."""
    def forward(self, x: Any, *args, **kwargs) -> Any:
        return x


def ablation_test(
    model: nn.Module,
    sample_input: torch.Tensor,
    loss_fn: Optional[Callable[[Any], torch.Tensor]] = None,
    candidate_modules: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Test the effect of temporarily replacing candidate submodules with an identity pass-through.

    Only bypasses modules where forward pass succeeds without shape incompatibility.
    """
    findings: List[DiagnosticFinding] = []
    ablation_results: Dict[str, Dict[str, Any]] = {}

    with torch.no_grad():
        try:
            base_out = model(sample_input)
            if isinstance(base_out, (tuple, list)):
                base_out = base_out[0]
            base_loss = float(loss_fn(base_out).item()) if loss_fn else float(base_out.norm().item())
        except Exception as e:
            return {"success": False, "error": f"Base model evaluation failed: {e}"}

    # Discover candidate submodules if not provided
    if candidate_modules is None:
        candidate_modules = []
        for name, m in model.named_modules():
            if name and len(list(m.children())) == 0 and not isinstance(m, (nn.Dropout,)):
                candidate_modules.append(name)

    # Limit to at most 10 modules for performance
    target_list = candidate_modules[:10]

    for mod_name in target_list:
        # Navigate to parent and attribute
        parts = mod_name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p, None)
            if parent is None:
                break
        if parent is None or not hasattr(parent, parts[-1]):
            continue

        attr_name = parts[-1]
        original_module = getattr(parent, attr_name)

        # Attempt replacement with Identity
        setattr(parent, attr_name, IdentityBypass())
        try:
            with torch.no_grad():
                abl_out = model(sample_input)
                if isinstance(abl_out, (tuple, list)):
                    abl_out = abl_out[0]
                abl_loss = float(loss_fn(abl_out).item()) if loss_fn else float(abl_out.norm().item())

            loss_delta = abl_loss - base_loss
            ablation_results[mod_name] = {
                "compatible": True,
                "base_metric": base_loss,
                "ablated_metric": abl_loss,
                "delta": loss_delta,
            }
        except Exception:
            ablation_results[mod_name] = {
                "compatible": False,
                "reason": "Shape mismatch or interface incompatibility",
            }
        finally:
            # Always restore original module
            setattr(parent, attr_name, original_module)

    return {
        "success": True,
        "ablation_results": ablation_results,
        "findings": findings,
    }
