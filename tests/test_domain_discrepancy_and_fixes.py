"""Tests for dict-output bug fixes, diagnose metric recording, and cross-domain discrepancy diagnostics."""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from nn_toolbox import (
    ablation_test,
    batchnorm_domain_shift_test,
    diagnose,
    eval_determinism_test,
    inspect_pretrained_weight_health,
    layer_representation_discrepancy_test,
    train_eval_test,
)
from nn_toolbox.analysis.sensitivity import compute_empirical_sensitivity


class _DictOutputSegmenter(nn.Module):
    """Model returning dict with non-tensor metadata first (mirrors HF/FOCAL outputs)."""

    def __init__(self, in_ch: int = 3, mid_ch: int = 8):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1)
        self.block = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Conv2d(mid_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        h = torch.relu(self.stem(x))
        h = h + self.block(h)
        logits = self.head(h)
        return {
            "skip_mask_loss": False,
            "mask_logits": logits,
            "features": h,
        }


def test_diagnose_records_actual_input_var_and_rank_ratio():
    torch.manual_seed(0)
    # Create a model whose output is strictly rank-1 across batch samples
    model = nn.Linear(4, 16, bias=False)
    with torch.no_grad():
        direction = torch.ones(1, 16)
        model.weight.copy_(direction.t().expand(16, 4) * 0.25)

    # Non-unit input variance (~9.0 instead of default 1.0)
    x = torch.randn(8, 4) * 3.0
    expected_var = float(x.var().item())

    report = diagnose(model, sample_input=x, verbose=False)
    ds = report.metrics["data_sanity"]
    assert "input_var" in ds
    assert abs(ds["input_var"] - expected_var) < 1e-4

    rep = report.metrics["representation_collapse"]
    assert rep["rank_ratio"] < 0.2
    # When not flagged collapsed by rank_ratio check or if rank_ratio checked directly,
    # ensure the stored healthy observation never hardcodes missing 1.00 when rank_ratio < 0.99.
    full_rank_model = nn.Sequential(nn.Linear(8, 8))
    x_fr = torch.randn(8, 8) * 2.5
    rep_fr_report = diagnose(full_rank_model, sample_input=x_fr, verbose=False)
    arch_obs = [f.observation for f in rep_fr_report.findings if f.category == "architecture" and "effective rank" in f.observation]
    actual_ratio = rep_fr_report.metrics["representation_collapse"]["rank_ratio"]
    if arch_obs and actual_ratio < 0.99:
        assert f"ratio: {actual_ratio:.2f}" in arch_obs[0]


def test_dict_output_models_supported_across_diagnostics():
    torch.manual_seed(42)
    model = _DictOutputSegmenter()
    x = torch.randn(4, 3, 16, 16)

    def _dict_loss(out, tgt=None):
        return out["mask_logits"].pow(2).mean()

    report = diagnose(model, sample_input=x, loss_fn=_dict_loss, verbose=False)
    assert "representation_collapse" in report.metrics
    assert "tensor_layout" in report.metrics
    assert not any(
        f.severity == "critical" and "unhandled exception" in f.observation.lower()
        for f in report.findings
    )

    te = train_eval_test(model, x)
    assert te["is_identical"] is True
    assert te["is_eval_deterministic"] is True

    det = eval_determinism_test(model, x, num_passes=3)
    assert det["is_deterministic"] is True

    abl = ablation_test(model, x, candidate_modules=["block"])
    assert abl["success"] is True
    assert abl["ablation_results"]["block"]["compatible"] is True

    sens = compute_empirical_sensitivity(model, x, num_trials=2)
    assert sens["max_relative_gain"] >= 0.0


def test_inspect_pretrained_weight_health_catches_all_five_pathologies():
    class _PathologicalNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.dead_conv = nn.Conv2d(4, 4, 3, padding=1)
            self.subnormal_bn = nn.BatchNorm2d(4)
            self.spiked_proj = nn.Linear(32, 16, bias=False)
            self.collinear_head = nn.Conv2d(8, 2, 1)
            self.ln = nn.LayerNorm(16)

        def forward(self, x):
            return x

    net = _PathologicalNet()
    with torch.no_grad():
        # 1. Near-zero collapsed conv weight
        net.dead_conv.weight.fill_(1e-6)
        # 2. Subnormal BN running_var
        net.subnormal_bn.running_var.fill_(5.605e-45)
        net.subnormal_bn.running_mean.fill_(5.605e-45)
        # 3. Spiked projection matrix (rank-1 spike + high condition number)
        u = torch.randn(16, 1)
        v = torch.randn(1, 32)
        ortho_noise = torch.linalg.svd(torch.randn(16, 32), full_matrices=False)
        low_spec = ortho_noise.U @ torch.diag(torch.linspace(0.05, 0.2, 16)) @ ortho_noise.Vh
        net.spiked_proj.weight.copy_(50.0 * (u @ v) + low_spec)
        # 4. Anti-correlated rank-1 2-channel head (cos ~ -1.0)
        row = torch.tensor([0.3, -0.4, 0.2, 0.1, -0.5, 0.2, 0.1, 0.3]).view(1, 8, 1, 1)
        net.collinear_head.weight[0].copy_(row.squeeze(0))
        net.collinear_head.weight[1].copy_(-row.squeeze(0) + 1e-3)
        # 5. Negative LayerNorm weight + large bias offset
        net.ln.weight.fill_(0.1)
        net.ln.weight[0] = -0.05
        net.ln.bias.fill_(1.5)

    res = inspect_pretrained_weight_health(net)
    assert res["bn_health"]["subnormal_bn_layers_count"] == 1
    assert any(d["layer"] == "dead_conv" for d in res["dead_weights"])
    assert any(d["layer"] == "collinear_head" and d["signed_cosine"] < -0.99 for d in res["collinear_heads"])
    assert any(d["layer"] == "spiked_proj" for d in res["spectral_anomalies"])
    assert any(d["layer"] == "ln" and d["min_weight"] < 0.0 for d in res["norm_bias_anomalies"])
    assert any(f.severity == "warning" for f in res["findings"])


def test_batchnorm_domain_shift_test_detects_ood_shift_and_skips_dead_bn():
    class _BNNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
            self.bn_active = nn.BatchNorm2d(8)
            self.bn_dead = nn.BatchNorm2d(8)
            self.head = nn.Conv2d(8, 1, 1)

        def forward(self, x):
            h = self.bn_active(self.conv1(x))
            h = h + 0.0 * self.bn_dead(h)
            return self.head(torch.relu(h))

    torch.manual_seed(7)
    net = _BNNet()
    ref_x = torch.randn(6, 3, 16, 16)
    ood_x = torch.randn(6, 3, 16, 16) * 4.5 + 3.0

    # Calibrate running stats on reference distribution
    net.train()
    with torch.no_grad():
        for _ in range(5):
            net(torch.randn(8, 3, 16, 16))
        # Collapse bn_dead to subnormal floor
        net.bn_dead.running_var.fill_(5.605e-45)
        net.bn_dead.running_mean.fill_(5.605e-45)

    shift_res = batchnorm_domain_shift_test(net, ref_x, ood_x)
    assert shift_res["num_bn_layers"] == 2
    assert shift_res["skipped_dead_layers"] == 1
    assert shift_res["active_bn_layers"] == 1
    assert not math.isnan(shift_res["ood_log_var_ratio"])
    assert shift_res["ood_log_var_ratio"] > shift_res["ref_log_var_ratio"] + 0.5
    assert shift_res["ood_mean_z"] > shift_res["ref_mean_z"] + 0.5
    assert any(f.severity == "warning" for f in shift_res["findings"])


def test_layer_representation_discrepancy_detects_ood_projection_bottleneck():
    class _BottleneckModel(nn.Module):
        """Intermediate stage encodes both ref & ood signal; final neck amplifies only ref."""

        def __init__(self):
            super().__init__()
            self.stage = nn.Identity()
            self.neck = nn.Conv2d(8, 8, kernel_size=1, bias=False)

        def forward(self, x):
            h = self.stage(x)
            return self.neck(h)

    model = _BottleneckModel()
    with torch.no_grad():
        # Neck keeps only channels 0..1 (ref forensic + ref semantic), zeroes channels 4..5 (ood forensic)
        w = torch.zeros(8, 8, 1, 1)
        w[0, 0, 0, 0] = 5.0
        w[1, 1, 0, 0] = 1.0
        w[2, 2, 0, 0] = 1.0  # ood semantic nuisance stays strong!
        w[3, 3, 0, 0] = 1.0
        model.neck.weight.copy_(w)

    B, H, W = 3, 16, 16
    mask = torch.zeros(B, 1, H, W)
    mask[:, :, 4:8, 4:8] = 1.0

    # Ref: forensic signal in ch 0, tiny nuisance in ch 2
    ref_x = torch.randn(B, 8, H, W) * 0.05
    ref_x[:, 0, :, :] = torch.where(mask[:, 0] > 0.5, 2.0, -2.0)
    ref_x[:, 2, :8, :] += 0.3

    # OOD: forensic signal sits in ch 4 (killed by neck!), strong semantic background in ch 2
    ood_x = torch.randn(B, 8, H, W) * 0.05
    ood_x[:, 4, :, :] = torch.where(mask[:, 0] > 0.5, 3.0, -3.0)
    ood_x[:, 2, :8, :] += 1.5

    res = layer_representation_discrepancy_test(
        model,
        ref_x,
        ood_x,
        ref_masks=mask,
        ood_masks=mask,
        module_names=["stage", "neck"],
        bottleneck_drop_threshold=0.2,
    )
    assert res["probed_layers"] == ["stage", "neck"]
    stage_ood_auc = res["per_layer"]["stage"]["ood"]["cv_auroc"]
    neck_ood_auc = res["per_layer"]["neck"]["ood"]["cv_auroc"]
    assert stage_ood_auc > 0.95
    assert neck_ood_auc < stage_ood_auc - 0.15
    assert res["per_layer"]["stage"]["ood"]["gt_to_km_ratio"] > res["per_layer"]["neck"]["ood"]["gt_to_km_ratio"] + 0.2
    assert any(f.severity == "warning" and "bottleneck" in f.observation.lower() for f in res["findings"])
