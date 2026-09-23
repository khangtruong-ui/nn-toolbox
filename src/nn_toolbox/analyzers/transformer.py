"""
Architecture-aware analyzer for Transformers and Attention mechanisms.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from nn_toolbox.core.finding import DiagnosticFinding, FindingCategory, Severity


class TransformerAnalyzer:
    """Analyzes Transformer-specific structures, query-key scaling, and attention entropy."""

    def __init__(self, model: nn.Module):
        self.model = model

    def analyze_attention_weights(self, attention_matrix: torch.Tensor) -> Dict[str, Any]:
        """Analyze an attention weight matrix of shape (B, num_heads, seq_len, seq_len) or (B, seq_len, seq_len)."""
        findings: List[DiagnosticFinding] = []

        with torch.no_grad():
            att = attention_matrix.detach().float()
            seq_len = att.shape[-1]
            max_entropy = math.log(max(seq_len, 2))

            # Shannon entropy per distribution (sum over last dimension)
            eps = 1e-12
            entropy = - (att * torch.log(att.clamp_min(eps))).sum(dim=-1)
            mean_entropy = float(entropy.mean().item())
            entropy_ratio = mean_entropy / max_entropy if max_entropy > 0 else 1.0

            # Attention collapse: entropy near 0 means attending to single token strictly everywhere
            if entropy_ratio < 0.05 and seq_len > 4:
                findings.append(
                    DiagnosticFinding(
                        category=FindingCategory.ARCHITECTURE.value,
                        severity=Severity.WARNING.value,
                        observation=f"Attention entropy is extremely low ({mean_entropy:.2f} / max {max_entropy:.2f}, ratio: {entropy_ratio:.2f}).",
                        interpretation="Attention weights have collapsed into a one-hot-like distribution onto a single token.",
                        evidence={"mean_entropy": mean_entropy, "entropy_ratio": entropy_ratio},
                        hypotheses=["Query-Key dot products too large without 1/sqrt(d_k) scaling", "Severe sink token phenomenon"],
                        confidence="medium",
                        suggested_actions=["Verify 1/sqrt(d_k) attention scaling"],
                    )
                )

            return {
                "mean_entropy": mean_entropy,
                "max_possible_entropy": max_entropy,
                "entropy_ratio": entropy_ratio,
                "findings": findings,
            }
