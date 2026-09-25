"""
Tests for Bootstrapping v1.0 detector and verification experiment in nn-toolbox.
"""

import copy
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nn_toolbox.core.finding import FindingCategory, Severity
from nn_toolbox.detectors.bootstrap import BootstrapDetector
from nn_toolbox.experiments.bootstrapping import verify_bootstrapping
from nn_toolbox.diagnose import diagnose


class ToyModelWithBackboneAndHead(nn.Module):
    """Toy neural network with an encoder backbone and prediction head."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.head(features)


def test_bootstrap_detector_successful():
    """Verify that a successful kickstarting phase yields an INFO confirmation finding."""
    detector = BootstrapDetector()
    assert detector.name == "bootstrap"
    assert detector.category == "bootstrap"

    context = {
        "bootstrapping": {
            "enabled": True,
            "run_bootstrap": True,
            "initial_loss": 1.25,
            "final_loss": 0.45,
            "best_score": 0.72,
            "target_score": 0.50,
            "min_loss_drop": 0.15,
            "bootstrap_epochs": 5,
            "bootstrap_examples": 512,
            "acceptable_fit": True,
            "frozen_param_grad_leak": False,
        }
    }

    findings = detector.detect(context)
    assert len(findings) == 1
    f = findings[0]
    assert f.severity == Severity.INFO.value
    assert f.category == FindingCategory.BOOTSTRAP.value
    assert "Bootstrapping v1.0 verified successful" in f.observation
    assert "512 samples" in f.observation
    assert "5 epochs" in f.observation
    assert "drop: 64.0%" in f.observation
    assert f.confidence == "high"


def test_bootstrap_detector_isolation_violation():
    """Verify that gradient leakage into frozen parameters generates a CRITICAL finding."""
    detector = BootstrapDetector()

    context = {
        "bootstrapping": {
            "enabled": True,
            "initial_loss": 1.0,
            "final_loss": 0.5,
            "best_score": 0.6,
            "bootstrap_epochs": 3,
            "bootstrap_examples": 256,
            "acceptable_fit": False,
            "frozen_param_grad_leak": True,
        }
    }

    findings = detector.detect(context)
    critical_findings = [f for f in findings if f.severity == Severity.CRITICAL.value]
    assert len(critical_findings) >= 1
    assert "isolation violation" in critical_findings[0].observation.lower()
    assert "leak" in critical_findings[0].observation.lower() or "frozen" in critical_findings[0].observation.lower()


def test_bootstrap_detector_divergence():
    """Verify that loss divergence generates a CRITICAL finding."""
    detector = BootstrapDetector()

    context = {
        "bootstrapping": {
            "enabled": True,
            "initial_loss": 0.8,
            "final_loss": 2.5,
            "best_score": 0.1,
            "bootstrap_epochs": 3,
            "bootstrap_examples": 512,
            "acceptable_fit": False,
            "frozen_param_grad_leak": False,
        }
    }

    findings = detector.detect(context)
    assert len(findings) == 1
    f = findings[0]
    assert f.severity == Severity.CRITICAL.value
    assert "divergence" in f.observation.lower()


def test_bootstrap_detector_insufficient_fit():
    """Verify that insufficient loss drop or score generates a WARNING finding."""
    detector = BootstrapDetector()

    context = {
        "bootstrapping": {
            "enabled": True,
            "initial_loss": 1.0,
            "final_loss": 0.96,  # only 4% drop, target >= 15%
            "best_score": 0.20,
            "target_score": 0.50,
            "min_loss_drop": 0.15,
            "bootstrap_epochs": 3,
            "bootstrap_examples": 512,
            "acceptable_fit": False,
            "frozen_param_grad_leak": False,
        }
    }

    findings = detector.detect(context)
    assert len(findings) == 1
    f = findings[0]
    assert f.severity == Severity.WARNING.value
    assert "did not achieve acceptable fit" in f.observation.lower()


def test_bootstrap_detector_disabled():
    """Verify that if bootstrapping is disabled, no findings are formulated."""
    detector = BootstrapDetector()
    context = {"bootstrapping": {"enabled": False, "run_bootstrap": False}}
    assert len(detector.detect(context)) == 0
    assert len(detector.detect({})) == 0


def test_verify_bootstrapping_active_experiment():
    """Test active kickstart verification experiment on a toy model."""
    model = ToyModelWithBackboneAndHead()
    # Freeze encoder
    for p in model.encoder.parameters():
        p.requires_grad = False

    x = torch.randn(32, 8)
    y = torch.randn(32, 1)

    result = verify_bootstrapping(
        model=model,
        sample_batch_or_loader=(x, y),
        loss_fn=nn.MSELoss(),
        bootstrap_epochs=4,
        freeze_param_names=["encoder"],
        min_loss_drop=0.05,
    )

    assert "findings" in result
    assert "metrics" in result
    assert "acceptable_fit" in result
    metrics = result["metrics"]
    assert metrics["frozen_param_grad_leak"] is False
    assert metrics["loss_drop"] > 0
    assert metrics["trainable_param_count"] > 0
    assert metrics["frozen_param_count"] > 0

    # Ensure non-destructive state restoration
    for name, p in model.named_parameters():
        if "encoder" in name:
            assert p.requires_grad is False
        if "head" in name:
            assert p.requires_grad is True


def test_diagnose_integration_with_bootstrapping():
    """Verify that diagnose() seamlessly incorporates bootstrapping metrics and detector."""
    model = ToyModelWithBackboneAndHead()
    x = torch.randn(16, 8)
    y = torch.randn(16, 1)
    dataloader = DataLoader(TensorDataset(x, y), batch_size=8)

    bootstrap_data = {
        "enabled": True,
        "initial_loss": 0.95,
        "final_loss": 0.35,
        "best_score": 0.68,
        "target_score": 0.50,
        "min_loss_drop": 0.15,
        "bootstrap_epochs": 5,
        "bootstrap_examples": 512,
        "acceptable_fit": True,
        "frozen_param_grad_leak": False,
    }

    report = diagnose(
        model=model,
        dataloader=dataloader,
        loss_fn=nn.MSELoss(),
        bootstrapping=bootstrap_data,
        mode="light",
        verbose=False,
    )

    assert "bootstrapping" in report.metrics
    bootstrap_findings = [f for f in report.findings if f.category == "bootstrap"]
    assert len(bootstrap_findings) == 1
    assert bootstrap_findings[0].severity == "info"
    assert "Bootstrapping v1.0 verified successful" in bootstrap_findings[0].observation


def test_verify_bootstrapping_channel_stream():
    """Test active kickstart verification experiment using end-to-end channel stream masking."""
    model = ToyModelWithBackboneAndHead()
    x = torch.randn(32, 8)
    y = torch.randn(32, 1)

    result = verify_bootstrapping(
        model=model,
        sample_batch_or_loader=(x, y),
        loss_fn=nn.MSELoss(),
        bootstrap_epochs=4,
        strategy="channel_stream",
        stream_ratio=0.5,
        min_loss_drop=0.01,
    )

    assert "findings" in result
    assert "metrics" in result
    metrics = result["metrics"]
    assert metrics["strategy"] == "channel_stream"
    assert metrics["stream_ratio"] == 0.5
    assert metrics["active_stream_channels"] > 0
    assert metrics["frozen_tail_channels"] > 0
    assert metrics["frozen_param_grad_leak"] is False
    assert metrics["loss_drop"] >= 0.0
