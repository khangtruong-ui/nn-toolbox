"""
Gradient monitoring and backward signal propagation instrumentation.
Tracks gradient norms, sparsity, cosine similarity across steps, and layer-to-layer scaling.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from nn_toolbox.analysis.similarity import compute_vector_cosine_similarity
from nn_toolbox.analysis.statistics import compute_tensor_stats
from nn_toolbox.instrumentation.hooks import HookManager


class GradientMonitor:
    """Monitors parameter and activation gradients, computes gradient statistics,

    and analyzes backward signal propagation.
    """

    def __init__(
        self,
        model: nn.Module,
        hook_manager: Optional[HookManager] = None,
        sample_limit: int = 4096,
    ):
        self.model = model
        self.sample_limit = sample_limit

        self._own_hook_manager = hook_manager is None
        self.hook_manager = hook_manager or HookManager(model)

        self.step_param_stats: Dict[int, Dict[str, Dict[str, Any]]] = {}
        self.step_layer_stats: Dict[int, Dict[str, Dict[str, Any]]] = {}
        self.gradient_vectors: List[torch.Tensor] = []
        self.cosine_similarities: List[float] = []

        self._current_step: int = 0
        self._attached: bool = False

    def attach(self, register_module_hooks: bool = False) -> None:
        """Attach backward monitoring.

        By default, relies on native parameter gradients collected post-backward to eliminate
        fragile PyTorch BackwardHookFunction view+inplace autograd collisions.
        """
        if self._attached:
            return
        if register_module_hooks:
            self.hook_manager.register_backward_hook(self._on_backward)
        self._attached = True

    def detach(self) -> None:
        """Remove backward hooks."""
        if self._attached and self._own_hook_manager:
            self.hook_manager.remove_hooks()
            self._attached = False

    def reset(self) -> None:
        """Clear recorded gradient statistics."""
        self.step_param_stats.clear()
        self.step_layer_stats.clear()
        self.gradient_vectors.clear()
        self.cosine_similarities.clear()
        self._current_step = 0

    def step(self) -> None:
        """Advance to next step."""
        self._current_step += 1

    def _extract_tensor(self, grad: Any) -> Optional[torch.Tensor]:
        if isinstance(grad, torch.Tensor):
            return grad
        if isinstance(grad, (tuple, list)):
            for item in grad:
                if isinstance(item, torch.Tensor):
                    return item
        return None

    def _on_backward(self, module_name: str, module: nn.Module, grad_in: Any, grad_out: Any) -> None:
        """Hook called during backward pass for intermediate module output gradients."""
        tensor = self._extract_tensor(grad_out)
        if tensor is None:
            tensor = self._extract_tensor(grad_in)
        if tensor is None:
            return

        stats = compute_tensor_stats(
            tensor,
            sample_limit=self.sample_limit,
            compute_quantiles=False,
            compute_higher_moments=False,
        )

        if self._current_step not in self.step_layer_stats:
            self.step_layer_stats[self._current_step] = {}
        self.step_layer_stats[self._current_step][module_name] = stats

    def collect_parameter_gradients(self) -> Dict[str, Dict[str, Any]]:
        """Collect current gradient statistics from model.named_parameters().

        Also flattens and updates global gradient trajectory for cosine similarity.
        """
        param_stats: Dict[str, Dict[str, Any]] = {}
        flat_grads = []

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue

                p_norm = float(torch.linalg.norm(param.detach().float()).item())
                if param.grad is None:
                    param_stats[name] = {
                        "has_grad": False,
                        "grad_norm": 0.0,
                        "grad_rms": 0.0,
                        "grad_to_param_ratio": 0.0,
                        "param_norm": p_norm,
                        "is_finite": True,
                        "nan_fraction": 0.0,
                        "inf_fraction": 0.0,
                        "zero_fraction": 1.0,
                    }
                    continue

                g = param.grad.detach()
                g_stats = compute_tensor_stats(
                    g,
                    sample_limit=self.sample_limit,
                    compute_quantiles=False,
                    compute_higher_moments=False,
                )

                g_norm = g_stats["l2_norm"]
                ratio = g_norm / (p_norm + 1e-12) if p_norm > 0 else 0.0

                param_stats[name] = {
                    "has_grad": True,
                    "grad_norm": g_norm,
                    "grad_rms": g_stats["rms"],
                    "grad_to_param_ratio": ratio,
                    "param_norm": p_norm,
                    "min": g_stats["min"],
                    "max": g_stats["max"],
                    "is_finite": g_stats["is_finite"],
                    "nan_fraction": g_stats["nan_fraction"],
                    "inf_fraction": g_stats["inf_fraction"],
                    "zero_fraction": g_stats["zero_fraction"],
                }

                # Downsample gradient vector if extremely large to prevent OOM
                flat_g = g.flatten().float()
                if flat_g.numel() > 10000:
                    indices = torch.linspace(0, flat_g.numel() - 1, 10000, device=flat_g.device).long()
                    flat_grads.append(flat_g[indices].cpu())
                else:
                    flat_grads.append(flat_g.cpu())

            if flat_grads:
                global_vec = torch.cat(flat_grads)
                if self.gradient_vectors:
                    prev_vec = self.gradient_vectors[-1]
                    sim = compute_vector_cosine_similarity(global_vec, prev_vec)
                    self.cosine_similarities.append(sim)
                # Keep last 5 gradient vectors
                self.gradient_vectors.append(global_vec)
                if len(self.gradient_vectors) > 5:
                    self.gradient_vectors.pop(0)

        if self._current_step not in self.step_param_stats:
            self.step_param_stats[self._current_step] = {}
        self.step_param_stats[self._current_step] = param_stats

        return param_stats

    def analyze_backward_propagation(self, step: Optional[int] = None) -> Dict[str, Any]:
        """Analyze gradient scale across model layers to detect vanishing or exploding gradients."""
        target_step = step if step is not None else (
            max(self.step_param_stats.keys()) if self.step_param_stats else 0
        )
        param_stats = self.step_param_stats.get(target_step, {})
        if not param_stats:
            param_stats = self.collect_parameter_gradients()

        valid_norms = [s["grad_norm"] for s in param_stats.values() if s.get("has_grad", False) and s.get("is_finite", True)]
        median_norm = float(torch.tensor(valid_norms).median().item()) if valid_norms else 0.0

        vanishing_layers = []
        exploding_layers = []
        zero_grad_layers = []

        for name, s in param_stats.items():
            if not s.get("has_grad", False):
                zero_grad_layers.append(name)
                continue

            g_norm = s["grad_norm"]
            ratio_to_median = g_norm / (median_norm + 1e-12) if median_norm > 0 else 1.0

            if g_norm < 1e-8 or (median_norm > 1e-5 and ratio_to_median < 0.05):
                vanishing_layers.append({"param": name, "grad_norm": g_norm, "ratio_to_median": ratio_to_median})
            elif ratio_to_median > 20.0 or g_norm > 1e4:
                exploding_layers.append({"param": name, "grad_norm": g_norm, "ratio_to_median": ratio_to_median})

        recent_cos_sim = self.cosine_similarities[-1] if self.cosine_similarities else None
        is_oscillating = recent_cos_sim is not None and recent_cos_sim < -0.7

        return {
            "median_grad_norm": median_norm,
            "params_with_gradients": len(valid_norms),
            "zero_grad_params": zero_grad_layers,
            "vanishing_params": vanishing_layers,
            "exploding_params": exploding_layers,
            "recent_cosine_similarity": recent_cos_sim,
            "is_oscillating": is_oscillating,
        }

    def __enter__(self) -> GradientMonitor:
        self.attach()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.detach()
