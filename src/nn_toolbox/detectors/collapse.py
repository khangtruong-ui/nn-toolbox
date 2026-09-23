"""
Detectors for representation collapse and dimensional collapse in latent/feature embeddings.
"""

from __future__ import annotations

from typing import Any, Dict, List
from nn_toolbox.core.finding import DiagnosticFinding, FindingCategory, Severity
from nn_toolbox.detectors.base import BaseDetector


class CollapseDetector(BaseDetector):
    """Detects when distinct samples produce nearly identical representations or low-rank collapse."""

    @property
    def name(self) -> str:
        return "collapse"

    @property
    def category(self) -> str:
        return FindingCategory.ARCHITECTURE.value

    def detect(self, context: Dict[str, Any]) -> List[DiagnosticFinding]:
        findings: List[DiagnosticFinding] = []
        collapse_info = context.get("representation_collapse", {})

        if not collapse_info:
            return findings

        if collapse_info.get("is_collapsed", False):
            mean_sim = collapse_info.get("mean_pairwise_similarity", 0.0)
            rank_ratio = collapse_info.get("rank_ratio", 1.0)
            eff_rank = collapse_info.get("effective_rank", 1.0)

            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.ARCHITECTURE.value,
                    severity=Severity.CRITICAL.value,
                    observation=(
                        f"Representation collapse detected: mean pairwise cosine similarity is {mean_sim:.3f} "
                        f"and effective rank is {eff_rank:.1f} (rank ratio: {rank_ratio:.3f})."
                    ),
                    interpretation="The model maps distinct inputs into nearly the same representation point or 1D ray.",
                    evidence=collapse_info,
                    hypotheses=[
                        "Contrastive or self-supervised loss missing negative sample repulsion",
                        "Batch normalization running stats collapse",
                        "Loss function trivial solution (e.g. constant output minimizes loss)",
                    ],
                    confidence="high",
                    suggested_actions=[
                        "Inspect loss terms for variance/covariance regularization (e.g. VICReg, Barlow Twins)",
                        "Verify temperature parameter or negative pair sampling",
                    ],
                )
            )

        return findings
