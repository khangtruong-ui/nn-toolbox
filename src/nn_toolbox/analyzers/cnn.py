"""
Architecture-aware analyzer for Convolutional Neural Networks (CNNs).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from nn_toolbox.core.finding import DiagnosticFinding, FindingCategory, Severity


class CNNAnalyzer:
    """Analyzes CNN-specific structures, spatial dimension changes, and channel progression."""

    def __init__(self, model: nn.Module):
        self.model = model

    def analyze_spatial_progression(self, sample_input: torch.Tensor) -> Dict[str, Any]:
        """Trace spatial dimensions (H, W) through convolution layers and check for early collapse."""
        conv_layers: List[Dict[str, Any]] = []
        findings: List[DiagnosticFinding] = []

        handles = []

        def _make_hook(name: str):
            def _hook(m: nn.Module, inp: Any, out: Any):
                if torch.is_tensor(out) and out.ndim == 4:
                    conv_layers.append({
                        "name": name,
                        "shape": list(out.shape),
                        "channels": out.shape[1],
                        "height": out.shape[2],
                        "width": out.shape[3],
                    })
            return _hook

        for name, m in self.model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                handles.append(m.register_forward_hook(_make_hook(name)))

        was_training = self.model.training
        self.model.eval()

        try:
            with torch.no_grad():
                self.model(sample_input)

            # Check for premature spatial collapse
            for i, layer in enumerate(conv_layers):
                h, w = layer["height"], layer["width"]
                if (h <= 1 or w <= 1) and i < len(conv_layers) - 1:
                    findings.append(
                        DiagnosticFinding(
                            category=FindingCategory.ARCHITECTURE.value,
                            severity=Severity.WARNING.value,
                            module=layer["name"],
                            observation=f"Feature map spatially collapsed to {h}x{w} prematurely at layer '{layer['name']}'.",
                            interpretation="Downsampling occurred too rapidly, destroying spatial information for subsequent layers.",
                            evidence=layer,
                            hypotheses=["Excessive stride or pooling layers in early blocks"],
                            confidence="high",
                            suggested_actions=["Reduce stride or pooling operations in early stages."],
                        )
                    )
                    break

            return {
                "conv_layers_count": len(conv_layers),
                "progression": conv_layers,
                "findings": findings,
            }

        finally:
            for h in handles:
                h.remove()
            if was_training:
                self.model.train()
