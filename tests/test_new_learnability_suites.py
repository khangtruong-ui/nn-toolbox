"""
Tests for new learnability and stability suites:
- GraphConnectivityDetector
- GradientBalanceDetector
- TensorLayoutDetector
- eval_determinism_test
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nn_toolbox.detectors.connectivity import GraphConnectivityDetector
from nn_toolbox.detectors.gradient_balance import GradientBalanceDetector
from nn_toolbox.detectors.layout import TensorLayoutDetector
from nn_toolbox.experiments.eval_determinism import eval_determinism_test
from nn_toolbox.diagnose import diagnose


def test_graph_connectivity_detector():
    det = GraphConnectivityDetector()
    assert det.name == "graph_connectivity"
    assert det.category == "architecture"

    # Case 1: Disconnected parameters detected
    context = {
        "backward_analysis": {
            "zero_grad_params": [
                "vae.decoder.conv1.weight",
                "vae.decoder.conv1.bias",
                "vae.decoder.norm.weight",
            ]
        },
        "parameter_info": {
            "trainable_param_names": [
                "vae.encoder.conv.weight",
                "vae.decoder.conv1.weight",
                "vae.decoder.conv1.bias",
                "vae.decoder.norm.weight",
            ]
        },
    }
    findings = det.detect(context)
    assert len(findings) == 1
    f = findings[0]
    assert f.severity == "critical"
    assert f.module == "vae.decoder"
    assert "vae.decoder" in f.observation
    assert f.evidence["disconnected_count"] == 3

    # Case 2: Clean case with no zero grad params
    assert len(det.detect({"backward_analysis": {"zero_grad_params": []}})) == 0


def test_gradient_balance_detector():
    det = GradientBalanceDetector()
    assert det.name == "gradient_balance"
    assert det.category == "backward"

    # Case 1: Severe inter-branch gradient disparity
    context = {
        "parameter_gradient_stats": {
            "encoder.conv1.weight": {"has_grad": True, "is_finite": True, "grad_rms": 0.5},
            "encoder.conv2.weight": {"has_grad": True, "is_finite": True, "grad_rms": 0.4},
            "aux_head.linear.weight": {"has_grad": True, "is_finite": True, "grad_rms": 2e-5},
            "aux_head.linear.bias": {"has_grad": True, "is_finite": True, "grad_rms": 1e-5},
        }
    }
    findings = det.detect(context)
    assert len(findings) == 1
    f = findings[0]
    assert f.severity == "warning"
    assert f.evidence["dominant_branch"] == "encoder"
    assert f.evidence["starved_branch"] == "aux_head"
    assert f.evidence["disparity_ratio"] > 500.0

    # Case 2: Balanced branches
    balanced_context = {
        "parameter_gradient_stats": {
            "encoder.conv1.weight": {"has_grad": True, "is_finite": True, "grad_rms": 0.2},
            "decoder.conv1.weight": {"has_grad": True, "is_finite": True, "grad_rms": 0.15},
        }
    }
    assert len(det.detect(balanced_context)) == 0


def test_tensor_layout_detector():
    det = TensorLayoutDetector()
    assert det.name == "tensor_layout"
    assert det.category == "architecture"

    context = {
        "tensor_layout": {
            "non_contiguous": [
                {
                    "name": "mask_logits",
                    "shape": [2, 1, 256, 256],
                    "stride": [131072, 65536, 512, 1],
                }
            ]
        }
    }
    findings = det.detect(context)
    assert len(findings) == 1
    f = findings[0]
    assert f.severity == "warning"
    assert f.module == "mask_logits"
    assert "non-contiguous" in f.observation
    assert "reshape" in f.suggested_actions[1]


def test_eval_determinism_test():
    # Deterministic model
    det_model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    )
    x = torch.randn(4, 8)
    res = eval_determinism_test(det_model, x, num_passes=3)
    assert res["is_deterministic"] is True
    assert res["max_difference"] < 1e-5
    assert len(res["findings"]) == 1
    assert res["findings"][0].severity == "info"

    # Non-deterministic model in eval
    class StochasticEvalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 2)

        def forward(self, x):
            # injects random noise even in eval mode
            return self.fc(x) + torch.randn(x.size(0), 2) * 5.0

    stoch_model = StochasticEvalModel()
    res_stoch = eval_determinism_test(stoch_model, x, num_passes=3)
    assert res_stoch["is_deterministic"] is False
    assert res_stoch["max_difference"] > 0.1
    assert len(res_stoch["findings"]) == 1
    assert res_stoch["findings"][0].severity in ["warning", "critical"]


def test_diagnose_integration_with_new_features():
    class ModelWithDisconnectedBranch(nn.Module):
        def __init__(self):
            super().__init__()
            self.active = nn.Linear(4, 2)
            self.dormant = nn.Linear(4, 2)  # trainable but omitted from forward

        def forward(self, x):
            return self.active(x)

    model = ModelWithDisconnectedBranch()
    x = torch.randn(8, 4)
    y = torch.randn(8, 2)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    report = diagnose(
        model=model,
        dataloader=loader,
        loss_fn=nn.MSELoss(),
        num_batches=2,
        mode="deep",
        verbose=False,
    )

    # Check that disconnected branch is detected and reported
    finding_modules = [f.module for f in report.findings]
    assert any(m == "dormant" for m in finding_modules)

    # Check that eval determinism metric was recorded
    assert "eval_determinism" in report.metrics
    assert report.metrics["eval_determinism"]["is_deterministic"] is True
