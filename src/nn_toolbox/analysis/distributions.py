"""
Distribution analysis, dead unit detection, and saturation metrics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import torch


def compute_approximate_histogram(
    tensor: torch.Tensor,
    bins: int = 10,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    sample_limit: int = 8192,
) -> Dict[str, Any]:
    """Compute histogram counts and bin edges for a tensor efficiently."""
    if tensor is None or tensor.numel() == 0:
        return {"counts": [], "bin_edges": []}

    with torch.no_grad():
        t = tensor.detach().float()
        is_finite = torch.isfinite(t)
        if not is_finite.any():
            return {"counts": [], "bin_edges": []}

        flat = t[is_finite].flatten()
        if flat.numel() > sample_limit:
            idx = torch.randint(0, flat.numel(), (sample_limit,), device=flat.device)
            flat = flat[idx]

        lo = min_val if min_val is not None else float(flat.min().item())
        hi = max_val if max_val is not None else float(flat.max().item())

        if lo == hi:
            return {"counts": [flat.numel()], "bin_edges": [lo, hi]}

        hist = torch.histc(flat, bins=bins, min=lo, max=hi)
        step = (hi - lo) / bins
        edges = [lo + i * step for i in range(bins + 1)]

        return {
            "counts": [int(c.item()) for c in hist],
            "bin_edges": [float(e) for e in edges],
        }


def analyze_saturation(
    tensor: torch.Tensor,
    act_type: str = "auto",
    threshold: float = 0.95,
) -> Dict[str, Any]:
    """Analyze whether activations are saturated in extreme regions (e.g. sigmoid or tanh)."""
    if tensor is None or tensor.numel() == 0:
        return {"saturated_fraction": 0.0, "type": act_type}

    with torch.no_grad():
        t = tensor.detach().float()
        finite_t = t[torch.isfinite(t)]
        if finite_t.numel() == 0:
            return {"saturated_fraction": 0.0, "type": act_type}

        min_val = float(finite_t.min().item())
        max_val = float(finite_t.max().item())

        detected_type = act_type
        if act_type == "auto":
            if min_val >= -0.05 and max_val <= 1.05:
                detected_type = "sigmoid"
            elif min_val >= -1.05 and max_val <= 1.05 and min_val < -0.1:
                detected_type = "tanh"
            else:
                detected_type = "relu"

        saturated_count = 0
        total = finite_t.numel()

        if detected_type == "sigmoid":
            # Saturated when close to 0 or 1
            sat_mask = (finite_t < (1.0 - threshold)) | (finite_t > threshold)
            saturated_count = int(sat_mask.sum().item())
        elif detected_type == "tanh":
            # Saturated when close to -1 or +1
            sat_mask = (finite_t < -threshold) | (finite_t > threshold)
            saturated_count = int(sat_mask.sum().item())
        elif detected_type == "relu":
            # Saturated when <= 0 (dead ReLU)
            sat_mask = finite_t <= 0.0
            saturated_count = int(sat_mask.sum().item())

        sat_fraction = saturated_count / total if total > 0 else 0.0

        return {
            "type": detected_type,
            "saturated_fraction": sat_fraction,
            "min": min_val,
            "max": max_val,
        }


def detect_dead_features(
    feature_tensor: torch.Tensor,
    threshold: float = 1e-7,
) -> Dict[str, Any]:
    """Detect dead feature channels / neurons across a batch.

    feature_tensor shape is expected to have shape (N, C, ...) or (N, D).
    """
    if feature_tensor is None or feature_tensor.ndim < 2:
        return {"dead_fraction": 0.0, "dead_indices": [], "total_features": 0}

    with torch.no_grad():
        t = feature_tensor.detach().float()
        # Flatten spatial dimensions: (N, C, *) -> (N, C, -1) -> max over spatial -> (N, C)
        n = t.shape[0]
        c = t.shape[1]
        reshaped = t.view(n, c, -1)

        # Max absolute value per feature across the batch
        max_abs = reshaped.abs().amax(dim=(0, 2))  # shape: (C,)
        is_dead = (max_abs < threshold) | torch.isnan(max_abs)
        dead_indices = torch.nonzero(is_dead).squeeze(-1).tolist()
        if isinstance(dead_indices, int):
            dead_indices = [dead_indices]

        dead_fraction = len(dead_indices) / c if c > 0 else 0.0

        return {
            "dead_fraction": dead_fraction,
            "dead_count": len(dead_indices),
            "dead_indices": dead_indices,
            "total_features": c,
        }
