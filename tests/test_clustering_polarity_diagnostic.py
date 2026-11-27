"""Tests for clustering_polarity_diagnostic in nn_toolbox."""

from __future__ import annotations

import torch
import torch.nn as nn

from nn_toolbox import clustering_polarity_diagnostic


class _SyntheticFeatureNet(nn.Module):
    """Test helper returning fixed spatial feature maps directly."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_clustering_polarity_diagnostic_detects_inversion_and_border_recovery():
    torch.manual_seed(0)
    b, c, h, w = 4, 16, 32, 32

    # Domain 1: Standard minority interior tampered patch (no polarity inversion)
    f_std = torch.zeros(b, c, h, w)
    f_std[:, 0, :, :] = 1.0
    m_std = torch.zeros(b, 1, h, w)
    f_std[:, 0, 10:18, 10:18] = 0.0
    f_std[:, 1, 10:18, 10:18] = 1.0
    m_std[:, 0, 10:18, 10:18] = 1.0

    # Domain 2: Majority interior tampered (>50% area, pristine border) -> minority flips!
    f_inv = torch.zeros(b, c, h, w)
    f_inv[:, 0, :, :] = 1.0
    m_inv = torch.zeros(b, 1, h, w)
    f_inv[:, 0, 4:28, 4:28] = 0.0
    f_inv[:, 1, 4:28, 4:28] = 1.0
    m_inv[:, 0, 4:28, 4:28] = 1.0  # 24*24 / 32*32 = 56.25% tampered

    net = _SyntheticFeatureNet()
    res = clustering_polarity_diagnostic(
        net,
        domain_batches={"standard": (f_std, m_std), "inverted": (f_inv, m_inv)},
        feature_extractor=lambda _m, x: x,
        border_weight=0.75,
        area_gate=0.25,
    )
    std_m = res["per_domain"]["standard"]
    inv_m = res["per_domain"]["inverted"]

    assert std_m["minority_iou"] > 0.95
    assert std_m["border_polarity_iou"] > 0.95
    assert std_m["polarity_flip_rate"] == 0.0

    assert inv_m["polarity_flip_rate"] == 1.0
    assert inv_m["minority_iou"] < 0.05
    assert inv_m["border_polarity_iou"] > 0.95
    assert inv_m["oracle_polarity_iou"] > 0.95
    assert any("POLARITY_INVERSION" in f.observation and f.module == "inverted" for f in res["findings"])


def test_clustering_polarity_diagnostic_detects_semantic_hijacking():
    torch.manual_seed(1)
    b, c, h, w = 4, 16, 32, 32
    f_hij = torch.zeros(b, c, h, w)
    # Huge left-right semantic background split
    f_hij[:, 0, :, :16] = 2.0
    f_hij[:, 1, :, 16:] = 2.0
    # Tiny orthogonal tampered signal
    f_hij[:, 2, 12:16, 12:16] = 0.25
    m_hij = torch.zeros(b, 1, h, w)
    m_hij[:, 0, 12:16, 12:16] = 1.0

    net = _SyntheticFeatureNet()
    res = clustering_polarity_diagnostic(
        net,
        domain_batches={"hijacked": (f_hij, m_hij)},
        feature_extractor=lambda _m, x: x,
        hijack_ratio_threshold=0.40,
    )
    hm = res["per_domain"]["hijacked"]
    assert hm["gt_to_km_ratio"] < 0.40
    assert any("SEMANTIC_HIJACKING" in f.observation and f.module == "hijacked" for f in res["findings"])
