"""
Detectors for data pipeline anomalies: NaNs, constant tensors, channel ordering, range mismatches.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import torch

from nn_toolbox.core.finding import DiagnosticFinding, FindingCategory, Severity
from nn_toolbox.detectors.base import BaseDetector


class DataDetector(BaseDetector):
    """Detects invalid or suspicious data tensors fed into the model."""

    @property
    def name(self) -> str:
        return "data"

    @property
    def category(self) -> str:
        return FindingCategory.DATA.value

    def detect(self, context: Dict[str, Any]) -> List[DiagnosticFinding]:
        findings: List[DiagnosticFinding] = []
        data_info = context.get("data_sanity", {})

        if not data_info:
            return findings

        # 1. Non-finite values in inputs/targets
        if not data_info.get("inputs_finite", True):
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.DATA.value,
                    severity=Severity.CRITICAL.value,
                    observation="Input tensors contain NaN or Inf values.",
                    interpretation="Corrupt data or division by zero in data preprocessing pipeline.",
                    evidence=data_info,
                    hypotheses=[
                        "Corrupted image/sample files",
                        "Unstable transform or division by std with zero standard deviation",
                        "Missing fill_na or unhandled null values",
                    ],
                    confidence="high",
                    suggested_actions=["Check data loader augmentations and dataset files for corrupt samples."],
                )
            )

        if not data_info.get("targets_finite", True):
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.DATA.value,
                    severity=Severity.CRITICAL.value,
                    observation="Target/label tensors contain NaN or Inf values.",
                    interpretation="Target labels contain invalid numbers.",
                    evidence=data_info,
                    hypotheses=["Corrupt ground truth labels in dataset"],
                    confidence="high",
                    suggested_actions=["Filter invalid labels during dataset __getitem__."],
                )
            )

        # 2. Constant inputs
        if data_info.get("input_is_constant", False):
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.DATA.value,
                    severity=Severity.CRITICAL.value,
                    observation="Input tensor is completely constant (zero variance across all elements).",
                    interpretation="The model is receiving blank/dummy inputs.",
                    evidence=data_info,
                    hypotheses=["All-black images or zeroed out tensors in transform pipeline"],
                    confidence="high",
                    suggested_actions=["Verify image loading and normalization pipeline."],
                )
            )

        # 3. Constant targets
        if data_info.get("target_is_constant", False):
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.DATA.value,
                    severity=Severity.WARNING.value,
                    observation="Target tensor in current batch is completely constant (e.g. all empty masks or single class).",
                    interpretation="The batch has no target diversity, which can cause gradient collapse or metric oscillation.",
                    evidence=data_info,
                    hypotheses=["Severe class imbalance or empty segmentation masks"],
                    confidence="medium",
                    suggested_actions=["Check batch sampling strategy and mask parsing logic."],
                )
            )

        # 4. Range anomalies (e.g. unnormalized [0, 255] pixels)
        in_min = data_info.get("input_min")
        in_max = data_info.get("input_max")
        if in_max is not None and in_min is not None and in_min >= -0.1 and in_max > 5.0 and in_max <= 256.0:
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.DATA.value,
                    severity=Severity.WARNING.value,
                    observation=f"Input tensor values range up to {in_max:.1f} (exceeding standard [0, 1] or [-1, 1] range).",
                    interpretation="Inputs appear to be unnormalized raw uint8/float [0, 255] pixels.",
                    evidence=data_info,
                    hypotheses=[
                        "Missing ToTensor() or / 255.0 normalization step",
                        "Raw byte values passed directly to model",
                    ],
                    confidence="medium",
                    suggested_actions=["Ensure image transforms divide by 255.0 or apply standard Normalize."],
                )
            )

        # 5. Channel ordering: e.g. NHWC instead of NCHW
        in_shape = data_info.get("input_shape", [])
        if len(in_shape) == 4 and in_shape[-1] in (1, 3, 4) and in_shape[1] > 4:
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.DATA.value,
                    severity=Severity.WARNING.value,
                    observation=f"Input tensor shape {in_shape} resembles NHWC (channel-last) format.",
                    interpretation="PyTorch standard Conv2d expects NCHW (channel-first) format.",
                    evidence=data_info,
                    hypotheses=["Missing permute(0, 3, 1, 2) after numpy image loading"],
                    confidence="medium",
                    suggested_actions=["Permute tensor to (batch, channels, height, width)."],
                )
            )

        return findings
