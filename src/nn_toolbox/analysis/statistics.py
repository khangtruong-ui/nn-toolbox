"""
Efficient streaming and batch statistics for tensors without large memory allocations.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple
import torch


class WelfordAccumulator:
    """Welford's online algorithm for computing mean and variance in a single pass

    with numerical stability.
    """

    def __init__(self):
        self.count: int = 0
        self.mean: float = 0.0
        self.m2: float = 0.0
        self.min_val: float = float("inf")
        self.max_val: float = float("-inf")
        self.nan_count: int = 0
        self.inf_count: int = 0
        self.zero_count: int = 0
        self.l1_sum: float = 0.0
        self.l2_sq_sum: float = 0.0

    def update(self, tensor: torch.Tensor) -> None:
        """Update running statistics with a new tensor batch."""
        if tensor is None or tensor.numel() == 0:
            return

        with torch.no_grad():
            t = tensor.detach().float()
            numel = t.numel()

            # Check for non-finite elements
            is_nan = torch.isnan(t)
            is_inf = torch.isinf(t)
            n_nan = int(is_nan.sum().item())
            n_inf = int(is_inf.sum().item())

            self.nan_count += n_nan
            self.inf_count += n_inf

            if n_nan + n_inf == numel:
                return

            finite_mask = ~(is_nan | is_inf)
            finite_t = t[finite_mask]
            k = finite_t.numel()
            if k == 0:
                return

            self.zero_count += int((finite_t == 0).sum().item())
            self.l1_sum += float(finite_t.abs().sum().item())
            self.l2_sq_sum += float((finite_t * finite_t).sum().item())

            t_min = float(finite_t.min().item())
            t_max = float(finite_t.max().item())
            self.min_val = min(self.min_val, t_min)
            self.max_val = max(self.max_val, t_max)

            batch_mean = float(finite_t.mean().item())
            batch_var = float(finite_t.var(unbiased=False).item()) if k > 1 else 0.0

            # Parallel Welford merge
            new_count = self.count + k
            delta = batch_mean - self.mean
            self.mean += delta * k / new_count
            self.m2 += batch_var * k + delta**2 * self.count * k / new_count
            self.count = new_count

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return math.sqrt(max(0.0, self.variance))

    @property
    def rms(self) -> float:
        if self.count == 0:
            return 0.0
        return math.sqrt(max(0.0, self.l2_sq_sum / self.count))


def compute_tensor_stats(
    tensor: torch.Tensor,
    sample_limit: int = 8192,
    compute_quantiles: bool = True,
    compute_higher_moments: bool = True,
) -> Dict[str, Any]:
    """Compute rich summary statistics for a tensor efficiently.

    Avoids full GPU-to-CPU copies and minimizes allocations.
    """
    if tensor is None or tensor.numel() == 0:
        return {
            "numel": 0,
            "mean": 0.0,
            "std": 0.0,
            "variance": 0.0,
            "rms": 0.0,
            "l1_norm": 0.0,
            "l2_norm": 0.0,
            "min": 0.0,
            "max": 0.0,
            "zero_fraction": 0.0,
            "nan_fraction": 0.0,
            "inf_fraction": 0.0,
            "is_finite": True,
        }

    with torch.no_grad():
        t = tensor.detach()
        numel = t.numel()

        is_nan = torch.isnan(t)
        is_inf = torch.isinf(t)
        n_nan = int(is_nan.sum().item())
        n_inf = int(is_inf.sum().item())
        nan_fraction = n_nan / numel
        inf_fraction = n_inf / numel
        is_finite = (n_nan == 0) and (n_inf == 0)

        if not is_finite:
            # Handle non-finite tensor
            finite_mask = ~(is_nan | is_inf)
            finite_t = t[finite_mask].float()
        else:
            finite_t = t.float()

        finite_numel = finite_t.numel()

        if finite_numel == 0:
            return {
                "numel": numel,
                "mean": float("nan"),
                "std": float("nan"),
                "variance": float("nan"),
                "rms": float("nan"),
                "l1_norm": float("nan"),
                "l2_norm": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "zero_fraction": 0.0,
                "nan_fraction": nan_fraction,
                "inf_fraction": inf_fraction,
                "is_finite": False,
            }

        mean_val = float(finite_t.mean().item())
        std_val = float(finite_t.std(unbiased=False).item()) if finite_numel > 1 else 0.0
        var_val = std_val * std_val
        l1_norm = float(finite_t.abs().sum().item())
        l2_sq = float((finite_t * finite_t).sum().item())
        l2_norm = math.sqrt(max(0.0, l2_sq))
        rms_val = math.sqrt(max(0.0, l2_sq / finite_numel))

        min_val = float(finite_t.min().item())
        max_val = float(finite_t.max().item())
        zero_count = int((finite_t == 0).sum().item())
        zero_fraction = zero_count / finite_numel

        stats: Dict[str, Any] = {
            "numel": numel,
            "mean": mean_val,
            "std": std_val,
            "variance": var_val,
            "rms": rms_val,
            "l1_norm": l1_norm,
            "l2_norm": l2_norm,
            "min": min_val,
            "max": max_val,
            "zero_fraction": zero_fraction,
            "nan_fraction": nan_fraction,
            "inf_fraction": inf_fraction,
            "is_finite": is_finite,
        }

        # Subsample for percentiles and higher moments if tensor is large
        if compute_quantiles or compute_higher_moments:
            if finite_numel > sample_limit:
                indices = torch.randint(0, finite_numel, (sample_limit,), device=finite_t.device)
                sample = finite_t.flatten()[indices]
            else:
                sample = finite_t.flatten()

            if compute_quantiles:
                try:
                    q_vals = torch.quantile(
                        sample,
                        torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95], device=sample.device),
                    )
                    stats["percentiles"] = {
                        "p5": float(q_vals[0].item()),
                        "p25": float(q_vals[1].item()),
                        "p50": float(q_vals[2].item()),
                        "p75": float(q_vals[3].item()),
                        "p95": float(q_vals[4].item()),
                    }
                except Exception:
                    stats["percentiles"] = {}

            if compute_higher_moments and sample.numel() > 2 and std_val > 1e-12:
                centered = sample - mean_val
                m3 = float((centered**3).mean().item())
                m4 = float((centered**4).mean().item())
                stats["skewness"] = m3 / (std_val**3)
                stats["kurtosis"] = (m4 / (std_val**4)) - 3.0  # Excess kurtosis
            else:
                stats["skewness"] = 0.0
                stats["kurtosis"] = 0.0

        return stats
