"""
Perturbation and local sensitivity analysis.
Estimates empirical Lipschitz factor without constructing full Jacobians.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Union
import torch


def compute_empirical_sensitivity(
    model_or_fn: Callable[[torch.Tensor], Any],
    input_tensor: torch.Tensor,
    epsilons: Optional[List[float]] = None,
    num_trials: int = 3,
) -> Dict[str, Any]:
    """Compute local empirical sensitivity of a model or layer output to input perturbations.

    Measures ||f(x + delta) - f(x)|| / ||delta|| across several noise scales.
    """
    if epsilons is None:
        epsilons = [1e-4, 1e-3, 1e-2]

    results: Dict[str, Any] = {"epsilons": epsilons, "trials": []}

    was_training = getattr(model_or_fn, "training", False)
    if hasattr(model_or_fn, "eval"):
        model_or_fn.eval()

    try:
        with torch.no_grad():
            x = input_tensor.detach()
            x_norm = float(torch.linalg.norm(x.float()).item())
            if x_norm == 0.0:
                x_norm = 1.0

            # Base output
            y0 = model_or_fn(x)
            if isinstance(y0, (tuple, list)):
                y0 = y0[0]

            y0_norm = float(torch.linalg.norm(y0.float()).item())
            if y0_norm == 0.0:
                y0_norm = 1.0

            sensitivity_ratios = []

            for eps in epsilons:
                trial_ratios = []
                for _ in range(num_trials):
                    noise = torch.randn_like(x)
                    noise_norm = float(torch.linalg.norm(noise.float()).item())
                    if noise_norm > 0:
                        delta = noise / noise_norm * (eps * x_norm)
                    else:
                        delta = torch.zeros_like(x)

                    x_pert = x + delta
                    actual_delta_norm = float(torch.linalg.norm(delta.float()).item())

                    y_pert = model_or_fn(x_pert)
                    if isinstance(y_pert, (tuple, list)):
                        y_pert = y_pert[0]

                    diff = (y_pert - y0).float()
                    diff_norm = float(torch.linalg.norm(diff).item())

                    # Relative amplification: (||dy|| / ||y||) / (||dx|| / ||x||)
                    if actual_delta_norm > 0:
                        abs_ratio = diff_norm / actual_delta_norm
                        rel_ratio = (diff_norm / y0_norm) / (actual_delta_norm / x_norm)
                    else:
                        abs_ratio = 0.0
                        rel_ratio = 0.0

                    trial_ratios.append({
                        "absolute_gain": abs_ratio,
                        "relative_gain": rel_ratio,
                    })

                mean_abs = sum(t["absolute_gain"] for t in trial_ratios) / len(trial_ratios)
                mean_rel = sum(t["relative_gain"] for t in trial_ratios) / len(trial_ratios)

                sensitivity_ratios.append({
                    "epsilon": eps,
                    "mean_absolute_gain": mean_abs,
                    "mean_relative_gain": mean_rel,
                })

            max_rel_gain = max(r["mean_relative_gain"] for r in sensitivity_ratios) if sensitivity_ratios else 0.0
            is_unusually_sensitive = max_rel_gain > 50.0

            return {
                "is_unusually_sensitive": is_unusually_sensitive,
                "max_relative_gain": max_rel_gain,
                "per_epsilon_results": sensitivity_ratios,
            }

    finally:
        if hasattr(model_or_fn, "train") and was_training:
            model_or_fn.train()
