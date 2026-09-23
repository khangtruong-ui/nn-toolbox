"""
Pathological cases and architecture verification test suite for nn-toolbox.
Validates detection of failure symptoms across diverse model topologies and error modes.
"""

import math
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nn_toolbox.analyzers.cnn import CNNAnalyzer
from nn_toolbox.analyzers.transformer import TransformerAnalyzer
from nn_toolbox.diagnose import diagnose
from nn_toolbox.experiments.gradient_check import gradient_check
from nn_toolbox.experiments.overfit import overfit_test


# 1. Standard Architectures: MLP, CNN, Transformer, BatchNorm, Dropout
class StandardMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )

    def forward(self, x):
        return self.net(x)


class StandardCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        x = torch.relu(self.bn(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.global_pool(x).flatten(1)
        return self.fc(x)


class SmallTransformer(nn.Module):
    def __init__(self, d_model=16, nhead=2, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(8, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=32, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x):
        h = self.embedding(x)
        out = self.encoder(h)
        return self.head(out[:, 0])


# 2. Pathological Networks
class ExplodingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 32, bias=False)
        self.fc2 = nn.Linear(32, 32, bias=False)
        self.fc3 = nn.Linear(32, 2, bias=False)
        # Intentionally huge weights
        with torch.no_grad():
            self.fc1.weight.fill_(5.0)
            self.fc2.weight.fill_(5.0)
            self.fc3.weight.fill_(5.0)

    def forward(self, x):
        return self.fc3(self.fc2(self.fc1(x)))


class VanishingNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for _ in range(8):
            fc = nn.Linear(16, 16)
            # Small weights with saturating tanh
            nn.init.constant_(fc.weight, 0.01)
            nn.init.constant_(fc.bias, 0.0)
            layers.append(fc)
            layers.append(nn.Tanh())
        layers.append(nn.Linear(16, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DetachedGraphNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        # Intentionally detached computation graph
        h_detached = h.detach()
        return self.fc2(h_detached)


class CustomWrongGradientFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return torch.matmul(x, w)

    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        # Intentionally wrong gradient computation!
        grad_x = torch.matmul(grad_output, w.t())
        wrong_grad_w = torch.ones_like(w) * 999.0  # Incorrect formula
        return grad_x, wrong_grad_w


class CustomWrongGradientLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(4, 2))

    def forward(self, x):
        return CustomWrongGradientFunction.apply(x, self.w)


def test_standard_architectures_diagnose():
    # 1. MLP
    mlp = StandardMLP()
    x_mlp = torch.randn(8, 16)
    y_mlp = torch.randn(8, 8)
    opt_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    rep_mlp = diagnose(mlp, sample_input=x_mlp, sample_target=y_mlp, loss_fn=loss_fn, optimizer=opt_mlp, mode="light", verbose=False)
    assert len(rep_mlp.critical_findings) == 0

    # 2. CNN
    cnn = StandardCNN()
    x_cnn = torch.randn(4, 3, 32, 32)
    y_cnn = torch.randn(4, 2)
    opt_cnn = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    rep_cnn = diagnose(cnn, sample_input=x_cnn, sample_target=y_cnn, loss_fn=loss_fn, optimizer=opt_cnn, mode="light", verbose=False)
    assert "data_sanity" in rep_cnn.metrics

    # 3. Transformer
    tf = SmallTransformer()
    x_tf = torch.randn(4, 10, 8)
    y_tf = torch.randn(4, 2)
    opt_tf = torch.optim.Adam(tf.parameters(), lr=1e-3)
    rep_tf = diagnose(tf, sample_input=x_tf, sample_target=y_tf, loss_fn=loss_fn, optimizer=opt_tf, mode="light", verbose=False)
    assert len(rep_tf.critical_findings) == 0


def test_pathology_exploding_network():
    net = ExplodingNet()
    x = torch.randn(4, 8)
    y = torch.randn(4, 2)
    loss_fn = nn.MSELoss()
    opt = torch.optim.SGD(net.parameters(), lr=0.01)

    rep = diagnose(net, sample_input=x, sample_target=y, loss_fn=loss_fn, optimizer=opt, mode="light", verbose=False)
    # Exploding activations should be flagged
    assert any(
        f.category == "forward" and f.is_actionable() for f in rep.findings
    ) or any("amplification" in f.interpretation for f in rep.findings)


def test_pathology_vanishing_network():
    net = VanishingNet()
    x = torch.randn(4, 16)
    y = torch.randn(4, 2)
    loss_fn = nn.MSELoss()
    opt = torch.optim.SGD(net.parameters(), lr=0.01)

    rep = diagnose(net, sample_input=x, sample_target=y, loss_fn=loss_fn, optimizer=opt, mode="light", verbose=False)
    # Should catch vanishing signal or small updates
    assert any("vanishing" in f.observation.lower() or "attenuated" in f.interpretation.lower() for f in rep.findings)


def test_pathology_detached_graph():
    net = DetachedGraphNet()
    x = torch.randn(4, 8)
    y = torch.randn(4, 2)
    loss_fn = nn.MSELoss()
    opt = torch.optim.SGD(net.parameters(), lr=0.01)

    rep = diagnose(net, sample_input=x, sample_target=y, loss_fn=loss_fn, optimizer=opt, mode="light", verbose=False)
    # fc1 should receive NO gradient
    actionables = rep.actionable_findings
    assert any(
        "zero gradient" in f.observation.lower() or "detached" in f.interpretation.lower()
        for f in actionables
    )


def test_pathology_frozen_parameters():
    net = StandardMLP()
    for p in net.parameters():
        p.requires_grad = False

    x = torch.randn(4, 16)
    y = torch.randn(4, 8)
    loss_fn = nn.MSELoss()

    rep = diagnose(net, sample_input=x, sample_target=y, loss_fn=loss_fn, mode="light", verbose=False)
    assert any("frozen" in f.observation.lower() for f in rep.findings)


def test_pathology_incorrect_input_normalization():
    net = StandardCNN()
    # Unnormalized 0-255 image input
    x_bad = torch.rand(4, 3, 32, 32) * 255.0
    y = torch.randn(4, 2)
    loss_fn = nn.MSELoss()

    rep = diagnose(net, sample_input=x_bad, sample_target=y, loss_fn=loss_fn, mode="light", verbose=False)
    assert any("255" in f.observation or "unnormalized" in f.interpretation.lower() for f in rep.findings)


def test_pathology_cannot_memorize_tiny_dataset():
    # Model that always outputs constant zero regardless of input
    class BrokenConstantModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return torch.zeros(x.size(0), 1)

    broken = BrokenConstantModel()
    x = torch.randn(4, 4)
    y = torch.ones(4, 1) * 10.0
    loss_fn = nn.MSELoss()

    of_res = overfit_test(broken, {"x": x, "y": y}, loss_fn=loss_fn, sizes=[1, 4], max_steps=20)
    assert any(f.severity == "critical" for f in of_res["findings"])


def test_pathology_custom_layer_wrong_gradient():
    layer = CustomWrongGradientLayer()
    x = torch.randn(2, 4)
    res = gradient_check(layer, x, parameters=["w"])
    assert res["all_passed"] is False
    assert res["param_results"]["w"]["status"] == "FAIL"
    assert any(f.severity == "critical" for f in res["findings"])


def test_cnn_and_transformer_analyzers():
    # CNN Analyzer
    cnn = StandardCNN()
    cnn_analyzer = CNNAnalyzer(cnn)
    x = torch.randn(2, 3, 16, 16)
    cnn_res = cnn_analyzer.analyze_spatial_progression(x)
    assert cnn_res["conv_layers_count"] >= 2

    # Transformer Analyzer
    tf_analyzer = TransformerAnalyzer(cnn)
    att_weights = torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)
    tf_res = tf_analyzer.analyze_attention_weights(att_weights)
    assert tf_res["mean_entropy"] > 0
