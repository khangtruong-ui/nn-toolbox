"""
Detector for structural graph connectivity and disconnected trainable parameters.
Identifies parameters marked requires_grad=True that are structurally disconnected
from model output or loss computation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List
import torch
import torch.nn as nn

from nn_toolbox.core.finding import DiagnosticFinding, FindingCategory, Severity
from nn_toolbox.detectors.base import BaseDetector


class GraphConnectivityDetector(BaseDetector):
    """Detects trainable parameters and submodules that are structurally disconnected from the loss graph."""

    @property
    def name(self) -> str:
        return "graph_connectivity"

    @property
    def category(self) -> str:
        return FindingCategory.ARCHITECTURE.value

    def detect(self, context: Dict[str, Any]) -> List[DiagnosticFinding]:
        findings: List[DiagnosticFinding] = []
        bwd_analysis = context.get("backward_analysis", {})
        param_info = context.get("parameter_info", {})
        trainable_names = set(param_info.get("trainable_param_names", []))

        zero_grad_params = bwd_analysis.get("zero_grad_params", [])
        if not zero_grad_params:
            return findings

        # Group zero-grad parameters by top-level or immediate parent module
        module_groups: Dict[str, List[str]] = defaultdict(list)
        for p_name in zero_grad_params:
            parts = p_name.split(".")
            if len(parts) > 1:
                # Group by top 2 levels if deep (e.g. vae.decoder or classifier_head)
                parent = ".".join(parts[:2]) if len(parts) > 2 else parts[0]
            else:
                parent = "root"
            module_groups[parent].append(p_name)

        for parent_module, params in module_groups.items():
            count = len(params)
            sample_names = params[:4]
            sample_str = ", ".join(f"'{p}'" for p in sample_names)
            if count > 4:
                sample_str += f", ... (+{count - 4} more)"

            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.ARCHITECTURE.value,
                    severity=Severity.CRITICAL.value,
                    module=parent_module if parent_module != "root" else None,
                    observation=(
                        f"Module '{parent_module}' contains {count} trainable parameter(s) structurally "
                        f"disconnected from backward loss computation ({sample_str})."
                    ),
                    interpretation=(
                        "These parameters are marked requires_grad=True, consuming optimizer memory and state, "
                        "but have no computational connection to the loss output."
                    ),
                    evidence={
                        "module": parent_module,
                        "disconnected_count": count,
                        "disconnected_parameters": params,
                    },
                    hypotheses=[
                        "Submodule was instantiated as trainable but is omitted from model.forward()",
                        "Intermediate tensor was detached via .detach(), .item(), or numpy conversion",
                        "Auxiliary loss component was omitted or loss weight set to 0.0",
                    ],
                    confidence="high",
                    suggested_actions=[
                        f"If '{parent_module}' is a frozen or auxiliary component, set param.requires_grad = False",
                        f"Verify that outputs of '{parent_module}' contribute to the loss function",
                    ],
                )
            )

        return findings
