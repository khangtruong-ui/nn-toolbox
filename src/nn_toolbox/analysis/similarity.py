"""
Similarity and representation collapse metrics: cosine similarity, pairwise distances, and effective rank.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch


def compute_vector_cosine_similarity(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute cosine similarity between two 1D vectors or flattened tensors."""
    if v1 is None or v2 is None or v1.numel() == 0 or v2.numel() == 0:
        return 0.0

    with torch.no_grad():
        f1 = v1.detach().flatten().float()
        f2 = v2.detach().flatten().float()

        if f1.numel() != f2.numel():
            min_len = min(f1.numel(), f2.numel())
            f1 = f1[:min_len]
            f2 = f2[:min_len]

        norm1 = float(torch.linalg.norm(f1).item())
        norm2 = float(torch.linalg.norm(f2).item())

        if norm1 < eps or norm2 < eps:
            return 0.0

        dot = float(torch.dot(f1, f2).item())
        sim = dot / (norm1 * norm2)
        return float(max(-1.0, min(1.0, sim)))


def compute_gradient_cosine_similarity(
    grads1: Union[List[torch.Tensor], Dict[str, torch.Tensor], torch.Tensor],
    grads2: Union[List[torch.Tensor], Dict[str, torch.Tensor], torch.Tensor],
) -> float:
    """Compute overall cosine similarity between two sets of model gradients."""
    with torch.no_grad():
        def _flatten(g: Any) -> torch.Tensor:
            if isinstance(g, torch.Tensor):
                return g.flatten()
            if isinstance(g, dict):
                parts = [p.flatten() for p in g.values() if p is not None and torch.is_tensor(p)]
                return torch.cat(parts) if parts else torch.tensor([], dtype=torch.float32)
            if isinstance(g, (list, tuple)):
                parts = [p.flatten() for p in g if p is not None and torch.is_tensor(p)]
                return torch.cat(parts) if parts else torch.tensor([], dtype=torch.float32)
            return torch.tensor([], dtype=torch.float32)

        v1 = _flatten(grads1)
        v2 = _flatten(grads2)
        return compute_vector_cosine_similarity(v1, v2)


def compute_pairwise_cosine_similarity(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute pairwise cosine similarity matrix for a batch of sample representations.

    features shape: (N, D) where N is number of samples.
    """
    with torch.no_grad():
        f = features.detach().float()
        if f.ndim > 2:
            f = f.reshape(f.shape[0], -1)

        norms = torch.linalg.norm(f, dim=1, keepdim=True).clamp_min(eps)
        normalized = f / norms
        sim_matrix = torch.matmul(normalized, normalized.t())
        return sim_matrix.clamp(-1.0, 1.0)


def compute_effective_rank(features: torch.Tensor, eps: float = 1e-12) -> Dict[str, Any]:
    """Compute effective rank (Roy & Vetterli, 2007) of a feature matrix.

    Effective rank = exp(-sum p_i ln(p_i)) where p_i = sigma_i / sum(sigma).
    Measures dimensional collapse: rank close to 1 indicates representations
    have collapsed into a single 1D subspace.
    """
    with torch.no_grad():
        f = features.detach().float()
        if f.ndim > 2:
            f = f.reshape(f.shape[0], -1)

        n, d = f.shape
        max_possible_rank = min(n, d)
        if max_possible_rank <= 1:
            return {
                "effective_rank": 1.0,
                "max_possible_rank": max_possible_rank,
                "rank_ratio": 1.0,
                "condition_number": 1.0,
            }

        # Center features
        f_centered = f - f.mean(dim=0, keepdim=True)
        try:
            s = torch.linalg.svdvals(f_centered)
            s_sum = s.sum().item()
            if s_sum < eps:
                return {
                    "effective_rank": 0.0,
                    "max_possible_rank": max_possible_rank,
                    "rank_ratio": 0.0,
                    "condition_number": float("inf"),
                }

            p = (s / s_sum).clamp_min(eps)
            entropy = -float((p * torch.log(p)).sum().item())
            eff_rank = math.exp(entropy)
            cond = float((s[0] / s[-1].clamp_min(eps)).item())

            return {
                "effective_rank": eff_rank,
                "max_possible_rank": max_possible_rank,
                "rank_ratio": eff_rank / max_possible_rank,
                "condition_number": cond,
                "singular_values": [float(val.item()) for val in s[:10]],
            }
        except Exception:
            return {
                "effective_rank": float("nan"),
                "max_possible_rank": max_possible_rank,
                "rank_ratio": float("nan"),
                "condition_number": float("nan"),
            }


def analyze_representation_collapse(features: torch.Tensor) -> Dict[str, Any]:
    """Perform comprehensive representation collapse analysis on intermediate or latent representations.

    Returns pairwise similarity distribution, feature variance, and effective rank.
    """
    if features is None or features.numel() == 0:
        return {"collapsed": False, "mean_similarity": 0.0}

    with torch.no_grad():
        f = features.detach().float()
        if f.ndim > 2:
            f = f.reshape(f.shape[0], -1)

        n, d = f.shape
        if n < 2:
            return {"collapsed": False, "mean_similarity": 0.0, "sample_count": n}

        sim_matrix = compute_pairwise_cosine_similarity(f)
        # Extract off-diagonal elements
        mask = ~torch.eye(n, dtype=torch.bool, device=f.device)
        off_diag = sim_matrix[mask]

        mean_sim = float(off_diag.mean().item()) if off_diag.numel() > 0 else 0.0
        max_sim = float(off_diag.max().item()) if off_diag.numel() > 0 else 0.0
        min_sim = float(off_diag.min().item()) if off_diag.numel() > 0 else 0.0
        sim_std = float(off_diag.std().item()) if off_diag.numel() > 1 else 0.0

        feature_vars = f.var(dim=0, unbiased=False)
        mean_feature_var = float(feature_vars.mean().item())
        zero_var_features = int((feature_vars < 1e-8).sum().item())

        rank_info = compute_effective_rank(f)

        is_collapsed = (mean_sim > 0.98 and sim_std < 0.02) or (
            rank_info.get("rank_ratio", 1.0) < 0.05 and n >= 4
        )

        return {
            "is_collapsed": is_collapsed,
            "mean_pairwise_similarity": mean_sim,
            "min_pairwise_similarity": min_sim,
            "max_pairwise_similarity": max_sim,
            "similarity_std": sim_std,
            "mean_feature_variance": mean_feature_var,
            "collapsed_features_count": zero_var_features,
            "total_features": d,
            "effective_rank": rank_info.get("effective_rank", float("nan")),
            "rank_ratio": rank_info.get("rank_ratio", float("nan")),
            "sample_count": n,
        }
