"""
Detector for tensor memory layout and contiguity anomalies.
Detects non-contiguous intermediate and output representations that impair GPU
kernel fusion or trigger downstream view/reshape exceptions.
"""

from __future__ import annotations

from typing import Any, Dict, List
import torch
import torch.nn as nn

from nn_toolbox.core.finding import DiagnosticFinding, FindingCategory, Severity
from nn_toolbox.detectors.base import BaseDetector


class TensorLayoutDetector(BaseDetector):
    """Detects non-contiguous tensor layouts and stride incompatibilities."""

    @property
    def name(self) -> str:
        return "tensor_layout"

    @property
    def category(self) -> str:
        return FindingCategory.ARCHITECTURE.value

    def detect(self, context: Dict[str, Any]) -> List[DiagnosticFinding]:
        findings: List[DiagnosticFinding] = []
        layout_info = context.get("tensor_layout", {})

        non_contiguous_tensors = layout_info.get("non_contiguous", [])
        for entry in non_contiguous_tensors:
            name = entry.get("name", "output")
            shape = entry.get("shape", [])
            stride = entry.get("stride", [])

            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.ARCHITECTURE.value,
                    severity=Severity.WARNING.value,
                    module=name,
                    observation=(
                        f"Tensor '{name}' has non-contiguous memory layout (shape: {shape}, stride: {stride})."
                    ),
                    interpretation=(
                        "Non-contiguous tensors cause subsequent .view() calls to fail with RuntimeError, "
                        "and cause suboptimal memory bandwidth / redundant copies in PyTorch CUDA kernels."
                    ),
                    evidence=entry,
                    hypotheses=[
                        "Tensor was sliced along spatial or channel dimensions (e.g. [:, :, :H, :W])",
                        "Tensor was permuted or transposed (e.g. permute(0, 2, 3, 1)) without .contiguous()",
                    ],
                    confidence="high",
                    suggested_actions=[
                        f"Add .contiguous() after slicing or transpose in module producing '{name}'",
                        "Use .reshape(...) instead of .view(...) for dimension restructuring",
                    ],
                )
            )

        return findings
