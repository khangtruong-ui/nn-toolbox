"""Tests for targeted experiments in nn_toolbox."""

import torch
import torch.nn as nn
from nn_toolbox.experiments.ablation import ablation_test
from nn_toolbox.experiments.gradient_check import gradient_check
from nn_toolbox.experiments.initialization import initialization_diagnostic
from nn_toolbox.experiments.label_shuffle import label_shuffle_test
from nn_toolbox.experiments.lr_sweep import lr_sweep
from nn_toolbox.experiments.overfit import overfit_test
from nn_toolbox.experiments.perturbation import perturbation_test
from nn_toolbox.experiments.train_eval import train_eval_test


class LinearTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


class ModelWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 12)
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(12, 2)

    def forward(self, x):
        return self.fc2(self.drop(torch.relu(self.fc1(x))))


def test_overfit_experiment_success():
    model = LinearTestModel()
    x = torch.randn(8, 4)
    y = torch.randn(8, 2)
    loss_fn = nn.MSELoss()

    res = overfit_test(model, {"x": x, "y": y}, loss_fn=loss_fn, sizes=[1, 4], max_steps=60, learning_rate=0.02)
    assert res["success"] is True
    assert len(res["results"]) == 2
    assert any(r["memorized"] for r in res["results"])


def test_lr_sweep_experiment():
    model = LinearTestModel()
    x = torch.randn(8, 4)
    y = torch.randn(8, 2)
    loss_fn = nn.MSELoss()

    res = lr_sweep(model, x, y, loss_fn=loss_fn, lr_range=[1e-4, 1e-2, 1e2], steps_per_lr=5)
    assert len(res["sweep_results"]) == 3
    assert any(r["status"] in ("useful_learning", "too_little_movement") for r in res["sweep_results"])


def test_initialization_diagnostic():
    model = LinearTestModel()
    res = initialization_diagnostic(model, (4, 4))
    assert "forward_analysis" in res
    assert "backward_analysis" in res
    assert len(res["findings"]) > 0


def test_train_eval_experiment():
    # Without dropout -> identical
    net_det = LinearTestModel()
    x = torch.randn(4, 4)
    res_det = train_eval_test(net_det, x)
    assert res_det["is_identical"] is True
    assert res_det["is_eval_deterministic"] is True

    # With dropout -> differs
    net_drop = ModelWithDropout()
    x6 = torch.randn(4, 6)
    res_drop = train_eval_test(net_drop, x6)
    assert res_drop["has_dropout"] is True
    assert res_drop["is_eval_deterministic"] is True


def test_gradient_check_experiment():
    model = LinearTestModel()
    x = torch.randn(2, 4)
    loss_fn = lambda out: out.sum()

    res = gradient_check(model, x, loss_fn=loss_fn, parameters=["fc.weight", "fc.bias"])
    assert res["success"] is True
    assert res["all_passed"] is True
    for p_name, r in res["param_results"].items():
        assert r["status"] == "PASS"


def test_perturbation_experiment():
    model = LinearTestModel()
    x = torch.randn(4, 4)
    res = perturbation_test(model, x, epsilons=[1e-4, 1e-2])
    assert "sensitivity" in res
    assert res["sensitivity"]["max_relative_gain"] >= 0


def test_ablation_experiment():
    model = ModelWithDropout()
    x = torch.randn(4, 6)
    res = ablation_test(model, x)
    assert res["success"] is True
    assert "ablation_results" in res


def test_label_shuffle_experiment():
    model = LinearTestModel()
    x = torch.randn(8, 4)
    y = torch.randn(8, 2)
    loss_fn = nn.MSELoss()
    res = label_shuffle_test(model, x, y, loss_fn=loss_fn, steps=10)
    assert "true_drop" in res
    assert "shuffled_drop" in res
