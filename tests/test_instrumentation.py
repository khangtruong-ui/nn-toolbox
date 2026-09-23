"""Tests for instrumentation: hooks, monitors, statistics, and storage."""

import torch
import torch.nn as nn
from nn_toolbox.analysis.statistics import WelfordAccumulator, compute_tensor_stats
from nn_toolbox.instrumentation.activations import ActivationMonitor
from nn_toolbox.instrumentation.gradients import GradientMonitor
from nn_toolbox.instrumentation.hooks import HookManager
from nn_toolbox.instrumentation.parameters import ParameterMonitor
from nn_toolbox.instrumentation.storage import RollingMetricsStorage


class SimpleToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def test_welford_accumulator():
    acc = WelfordAccumulator()
    t1 = torch.tensor([1.0, 2.0, 3.0])
    t2 = torch.tensor([4.0, 5.0, 6.0])
    acc.update(t1)
    acc.update(t2)

    assert acc.count == 6
    assert abs(acc.mean - 3.5) < 1e-5
    assert abs(acc.min_val - 1.0) < 1e-5
    assert abs(acc.max_val - 6.0) < 1e-5
    assert acc.std > 0


def test_compute_tensor_stats():
    t = torch.randn(100, 20)
    stats = compute_tensor_stats(t)
    assert stats["numel"] == 2000
    assert stats["is_finite"] is True
    assert "percentiles" in stats
    assert "p50" in stats["percentiles"]
    assert stats["zero_fraction"] < 0.1

    # NaN / Inf handling
    t_bad = torch.tensor([1.0, float("nan"), float("inf")])
    bad_stats = compute_tensor_stats(t_bad)
    assert bad_stats["is_finite"] is False
    assert bad_stats["nan_fraction"] > 0
    assert bad_stats["inf_fraction"] > 0


def test_hook_manager_lifecycle():
    net = SimpleToyNet()
    called = []

    def hook_fn(name, m, inp, out):
        called.append(name)

    with HookManager(net) as hm:
        hm.register_forward_hook(hook_fn)
        x = torch.randn(2, 8)
        _ = net(x)
        assert len(called) > 0

    # Ensure hooks are removed after exiting context
    called.clear()
    _ = net(x)
    assert len(called) == 0


def test_activation_monitor_propagation():
    net = SimpleToyNet()
    with ActivationMonitor(net) as monitor:
        x = torch.randn(4, 8)
        _ = net(x)
        prop = monitor.analyze_signal_propagation()
        assert "layers" in prop
        assert len(prop["layers"]) >= 2
        assert "fc1" in prop["layers"] or any("fc1" in k for k in prop["layers"])


def test_gradient_and_parameter_monitor():
    net = SimpleToyNet()
    param_mon = ParameterMonitor(net)
    p_info = param_mon.inspect_parameters()
    assert p_info["total_parameters"] > 0
    assert p_info["trainable_fraction"] == 1.0

    grad_mon = GradientMonitor(net)
    with grad_mon:
        x = torch.randn(4, 8)
        out = net(x)
        loss = out.sum()
        loss.backward()

        g_stats = grad_mon.collect_parameter_gradients()
        assert len(g_stats) > 0
        bwd_prop = grad_mon.analyze_backward_propagation()
        assert bwd_prop["params_with_gradients"] > 0

    # Optimizer step tracking
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    param_mon.snapshot_before_step()
    opt.step()
    updates = param_mon.snapshot_after_step()
    assert updates["global_update_norm"] > 0
    assert updates["global_update_ratio"] > 0


def test_rolling_metrics_storage():
    storage = RollingMetricsStorage(max_history=5)
    for i in range(10):
        storage.append(i, {"loss": 1.0 / (i + 1)})
    assert len(storage) == 5
    assert storage.get_latest()["step"] == 9
    series = storage.get_metric_series("loss")
    assert len(series) == 5
