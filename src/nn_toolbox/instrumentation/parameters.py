"""
Parameter monitoring and update-to-weight ratio tracking.
Distinguishes raw gradient magnitude from actual parameter updates.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from nn_toolbox.analysis.statistics import compute_tensor_stats


class ParameterMonitor:
    """Monitors model parameters and tracks exact parameter updates across optimizer steps."""

    def __init__(self, model: nn.Module, sample_limit: int = 4096):
        self.model = model
        self.sample_limit = sample_limit
        self._pre_step_params: Dict[str, torch.Tensor] = {}
        self.update_history: List[Dict[str, Any]] = []

    def inspect_parameters(self) -> Dict[str, Any]:
        """Compute structural and numerical properties of all model parameters."""
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        total_norm_sq = 0.0

        param_details: Dict[str, Dict[str, Any]] = {}

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                numel = param.numel()
                total_params += numel
                if param.requires_grad:
                    trainable_params += numel
                else:
                    frozen_params += numel

                stats = compute_tensor_stats(
                    param,
                    sample_limit=self.sample_limit,
                    compute_quantiles=False,
                    compute_higher_moments=False,
                )

                l2_norm = stats["l2_norm"]
                total_norm_sq += l2_norm**2

                param_details[name] = {
                    "shape": list(param.shape),
                    "numel": numel,
                    "requires_grad": param.requires_grad,
                    "norm": l2_norm,
                    "rms": stats["rms"],
                    "mean": stats["mean"],
                    "min": stats["min"],
                    "max": stats["max"],
                    "zero_fraction": stats["zero_fraction"],
                    "nan_fraction": stats["nan_fraction"],
                    "inf_fraction": stats["inf_fraction"],
                    "is_finite": stats["is_finite"],
                }

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": frozen_params,
            "trainable_fraction": trainable_params / total_params if total_params > 0 else 0.0,
            "global_param_norm": math.sqrt(total_norm_sq),
            "parameters": param_details,
        }

    def snapshot_before_step(self) -> None:
        """Cache current parameter values before calling optimizer.step()."""
        self._pre_step_params.clear()
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    # Detach to CPU or keep on device; clone to avoid in-place update modification
                    self._pre_step_params[name] = param.detach().clone()

    def snapshot_after_step(self) -> Dict[str, Any]:
        """Compute delta between current parameters and pre-step snapshot."""
        if not self._pre_step_params:
            return {"updates": {}, "global_update_norm": 0.0, "global_update_ratio": 0.0}

        updates: Dict[str, Dict[str, Any]] = {}
        total_delta_sq = 0.0
        total_param_sq = 0.0

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name not in self._pre_step_params:
                    continue

                prev_p = self._pre_step_params[name]
                cur_p = param.detach()

                delta = cur_p - prev_p
                delta_norm = float(torch.linalg.norm(delta.float()).item())
                p_norm = float(torch.linalg.norm(cur_p.float()).item())

                total_delta_sq += delta_norm**2
                total_param_sq += p_norm**2

                update_ratio = delta_norm / (p_norm + 1e-12) if p_norm > 0 else 0.0

                updates[name] = {
                    "delta_norm": delta_norm,
                    "param_norm": p_norm,
                    "update_ratio": update_ratio,
                    "zero_update": delta_norm == 0.0,
                }

        global_delta_norm = math.sqrt(total_delta_sq)
        global_param_norm = math.sqrt(total_param_sq)
        global_update_ratio = (
            global_delta_norm / (global_param_norm + 1e-12) if global_param_norm > 0 else 0.0
        )

        step_result = {
            "updates": updates,
            "global_update_norm": global_delta_norm,
            "global_update_ratio": global_update_ratio,
        }

        self.update_history.append(step_result)
        if len(self.update_history) > 20:
            self.update_history.pop(0)

        self._pre_step_params.clear()
        return step_result
