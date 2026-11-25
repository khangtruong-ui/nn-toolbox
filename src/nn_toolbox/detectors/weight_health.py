"""Pretrained weight and parameter structural health inspection.

Detects subtle structural pathologies in pretrained or adapted checkpoints:
- Dead or IEEE-754 subnormal-collapsed BatchNorm running statistics
- Near-zero collapsed convolutional/linear layers
- Spiked / ill-conditioned projection matrices (spectral top-1 dominance)
- Rank-1 collinear multi-channel output heads (e.g. anti-correlated affine channels)
- Normalization layers with negative affine scales or large constant bias offsets
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from nn_toolbox.core.finding import DiagnosticFinding, FindingCategory, Severity


def _is_batchnorm(module: nn.Module) -> bool:
    return isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))


def _is_non_batch_norm_layer(module: nn.Module) -> bool:
    if _is_batchnorm(module):
        return False
    if isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
        return True
    cls_name = type(module).__name__.lower()
    return ("layernorm" in cls_name or "groupnorm" in cls_name) and hasattr(module, "weight") and hasattr(module, "bias")


def _compute_matrix_spectrum(weight: torch.Tensor) -> Optional[Dict[str, float]]:
    w_2d = weight.detach().float().reshape(weight.shape[0], -1)
    m, n = w_2d.shape
    min_dim, max_dim = min(m, n), max(m, n)
    if min_dim < 8 or min_dim > 1024 or max_dim > 4096:
        return None
    norm_val = float(w_2d.norm().item())
    if norm_val <= 1e-12:
        return None

    sv = torch.linalg.svdvals(w_2d)
    sv_pos = sv[sv > 1e-12]
    if sv_pos.numel() < 2:
        return None
    p = sv_pos / sv_pos.sum()
    erank = float(torch.exp(-(p * p.log()).sum()).item())
    sv0 = float(sv[0].item())
    sv1 = float(sv[1].item())
    sv_last = float(sv[-1].item())
    return {
        "rows": int(m),
        "cols": int(n),
        "norm": norm_val,
        "sv_max": sv0,
        "sv_second": sv1,
        "sv_min": sv_last,
        "condition_number": sv0 / max(sv_last, 1e-12),
        "top1_spike_ratio": sv0 / max(sv1, 1e-12),
        "effective_rank": erank,
        "effective_rank_ratio": erank / float(min_dim),
    }


def _compute_head_collinearity(weight: torch.Tensor) -> Optional[Dict[str, Any]]:
    out_c = int(weight.shape[0])
    if out_c < 2 or out_c > 32:
        return None
    rows = weight.detach().float().reshape(out_c, -1)
    norms = rows.norm(dim=1)
    active_idx = torch.nonzero(norms > 1e-12, as_tuple=False).flatten()
    if active_idx.numel() < 2:
        return None

    active_rows = rows[active_idx] / norms[active_idx].unsqueeze(1)
    cos_mat = active_rows @ active_rows.t()
    n_act = active_rows.shape[0]
    triu_r, triu_c = torch.triu_indices(n_act, n_act, offset=1, device=cos_mat.device)
    pair_cos = cos_mat[triu_r, triu_c]
    abs_cos = pair_cos.abs()
    best_pos = int(abs_cos.argmax().item())
    return {
        "out_channels": out_c,
        "active_channels": int(n_act),
        "max_abs_cosine": float(abs_cos[best_pos].item()),
        "signed_cosine": float(pair_cos[best_pos].item()),
        "pair": [int(active_idx[triu_r[best_pos]].item()), int(active_idx[triu_c[best_pos]].item())],
        "row_norms": [round(float(x), 6) for x in norms.tolist()],
    }


def inspect_pretrained_weight_health(
    model: nn.Module,
    dead_var_threshold: float = 1e-6,
    subnormal_threshold: float = 1e-30,
    dead_conv_norm_threshold: float = 1e-3,
    head_collinearity_threshold: float = 0.95,
    spike_ratio_threshold: float = 2.0,
    cond_threshold: float = 40.0,
    norm_bias_ratio_threshold: float = 0.35,
) -> Dict[str, Any]:
    """Audit parameter tensors and BatchNorm running buffers for structural degeneration.

    Args:
        model: PyTorch module to inspect (read-only).
        dead_var_threshold: BatchNorm running variance threshold below which a channel is dead.
        subnormal_threshold: IEEE-754 near-underflow running variance threshold.
        dead_conv_norm_threshold: Frobenius norm threshold for non-zero collapsed Conv/Linear weights.
        head_collinearity_threshold: Absolute inter-row cosine threshold for compact output heads.
        spike_ratio_threshold: Minimum ``sv_max / sv_second`` ratio to flag spectral spike projections.
        cond_threshold: Minimum condition number required alongside ``spike_ratio_threshold``.
        norm_bias_ratio_threshold: Constant bias dominance threshold for LayerNorm/GroupNorm layers.

    Returns:
        Dictionary with structured inspection metrics and ``DiagnosticFinding`` entries.
    """
    bn_dead_layers: List[Dict[str, Any]] = []
    subnormal_bn_layers: List[Dict[str, Any]] = []
    total_bn_layers = 0
    total_bn_dead_channels = 0

    dead_weights: List[Dict[str, Any]] = []
    zero_init_weights: List[Dict[str, Any]] = []
    spectral_anomalies: List[Dict[str, Any]] = []
    collinear_heads: List[Dict[str, Any]] = []
    norm_bias_anomalies: List[Dict[str, Any]] = []

    for name, mod in model.named_modules():
        if _is_batchnorm(mod) and getattr(mod, "track_running_stats", False) and mod.running_var is not None:
            total_bn_layers += 1
            rv = mod.running_var.detach().float()
            n_ch = int(rv.numel())
            n_dead = int((rv < dead_var_threshold).sum().item())
            n_sub = int((rv < subnormal_threshold).sum().item())
            if n_dead > 0:
                total_bn_dead_channels += n_dead
                entry = {
                    "layer": name,
                    "dead_channels": n_dead,
                    "subnormal_channels": n_sub,
                    "total_channels": n_ch,
                    "min_running_var": float(rv.min().item()),
                    "max_running_var": float(rv.max().item()),
                }
                bn_dead_layers.append(entry)
                if n_sub > 0:
                    subnormal_bn_layers.append(entry)
            continue

        if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            w = mod.weight
            w_norm = float(w.detach().float().norm().item())
            if w_norm == 0.0:
                zero_init_weights.append({"layer": name, "shape": list(w.shape)})
            elif w_norm < dead_conv_norm_threshold:
                dead_weights.append({
                    "layer": name,
                    "shape": list(w.shape),
                    "frobenius_norm": w_norm,
                })

            col_info = _compute_head_collinearity(w)
            if col_info is not None and col_info["max_abs_cosine"] >= head_collinearity_threshold:
                collinear_heads.append({"layer": name, **col_info})

            spec = _compute_matrix_spectrum(w)
            if (
                spec is not None
                and spec["top1_spike_ratio"] >= spike_ratio_threshold
                and spec["condition_number"] >= cond_threshold
            ):
                spectral_anomalies.append({"layer": name, **spec})
            continue

        if _is_non_batch_norm_layer(mod):
            w = getattr(mod, "weight", None)
            b = getattr(mod, "bias", None)
            if w is not None and torch.is_tensor(w) and w.numel() > 0:
                wf = w.detach().float()
                min_w = float(wf.min().item())
                mean_abs_w = float(wf.abs().mean().item())
                b_norm = float(b.detach().float().norm().item()) if (b is not None and torch.is_tensor(b)) else 0.0
                bias_ratio = b_norm / (mean_abs_w * math.sqrt(max(1, wf.numel())) + 1e-8)
                if min_w < 0.0 or bias_ratio > norm_bias_ratio_threshold:
                    norm_bias_anomalies.append({
                        "layer": name,
                        "channels": int(wf.numel()),
                        "min_weight": min_w,
                        "max_weight": float(wf.max().item()),
                        "mean_abs_weight": mean_abs_w,
                        "bias_norm": b_norm,
                        "bias_scale_ratio": bias_ratio,
                    })

    findings: List[DiagnosticFinding] = []

    if subnormal_bn_layers:
        sample_names = [e["layer"] for e in subnormal_bn_layers[:5]]
        findings.append(
            DiagnosticFinding(
                category=FindingCategory.ARCHITECTURE.value,
                severity=Severity.WARNING.value,
                module=sample_names[0],
                observation=(
                    f"Detected {len(subnormal_bn_layers)} BatchNorm layer(s) with IEEE-754 subnormal-collapsed "
                    f"running_var (< {subnormal_threshold:.0e}): {', '.join(sample_names)}."
                ),
                interpretation=(
                    "Near-zero conv inputs combined with BatchNorm weight decay drove running statistics into subnormal "
                    "floating-point floor, collapsing those residual blocks to near-constant bias additions."
                ),
                evidence={
                    "subnormal_layers": subnormal_bn_layers,
                    "dead_bn_layers_count": len(bn_dead_layers),
                    "total_bn_dead_channels": total_bn_dead_channels,
                },
                hypotheses=[
                    "Pretrained residual branch collapsed under weight decay or unscaled BN affine weights",
                    "Dead channels cannot respond to out-of-distribution inputs without explicit re-training or replacement",
                ],
                confidence="high",
                suggested_actions=[
                    "Inspect or bypass collapsed residual blocks rather than relying on them as active adapters.",
                ],
            )
        )
    elif bn_dead_layers:
        findings.append(
            DiagnosticFinding(
                category=FindingCategory.ARCHITECTURE.value,
                severity=Severity.WARNING.value,
                module=bn_dead_layers[0]["layer"],
                observation=(
                    f"Detected {len(bn_dead_layers)}/{total_bn_layers} BatchNorm layer(s) with dead channels "
                    f"({total_bn_dead_channels} total channels with running_var < {dead_var_threshold:.0e})."
                ),
                interpretation="Zeroed or pruned channels yield degenerate BatchNorm normalization statistics.",
                evidence={"dead_bn_layers": bn_dead_layers},
                confidence="high",
            )
        )

    if dead_weights:
        sample_names = [d["layer"] for d in dead_weights[:5]]
        findings.append(
            DiagnosticFinding(
                category=FindingCategory.ARCHITECTURE.value,
                severity=Severity.WARNING.value,
                module=sample_names[0],
                observation=(
                    f"Detected {len(dead_weights)} non-zero Conv/Linear layer(s) with near-zero weight norm "
                    f"(< {dead_conv_norm_threshold:.0e}): {', '.join(sample_names)}."
                ),
                interpretation="Weights decayed nearly to zero, turning affected residual branches into pass-throughs.",
                evidence={"dead_weights": dead_weights},
                confidence="high",
            )
        )

    if collinear_heads:
        top = collinear_heads[0]
        findings.append(
            DiagnosticFinding(
                category=FindingCategory.ARCHITECTURE.value,
                severity=Severity.WARNING.value,
                module=top["layer"],
                observation=(
                    f"Compact output head '{top['layer']}' collapsed to rank-1 collinearity "
                    f"(cos={top['signed_cosine']:+.4f} between channels {top['pair'][0]} and {top['pair'][1]})."
                ),
                interpretation=(
                    "Multi-channel output rows point along a single axis (or exact opposite directions), reducing "
                    "expressive capacity to 1 degree of freedom (e.g., tied bias and scale adjustments)."
                ),
                evidence={"collinear_heads": collinear_heads},
                confidence="high",
            )
        )

    if spectral_anomalies:
        top = sorted(spectral_anomalies, key=lambda d: d["condition_number"], reverse=True)[0]
        findings.append(
            DiagnosticFinding(
                category=FindingCategory.ARCHITECTURE.value,
                severity=Severity.WARNING.value,
                module=top["layer"],
                observation=(
                    f"Projection layer '{top['layer']}' exhibits a dominant spectral spike "
                    f"(sv1/sv2={top['top1_spike_ratio']:.2f}, cond={top['condition_number']:.1f}, "
                    f"erank_ratio={top['effective_rank_ratio']:.3f})."
                ),
                interpretation=(
                    "A single singular direction amplifies one task-specific mode far more strongly than residual dimensions, "
                    "which can attenuate weaker out-of-domain signal directions."
                ),
                evidence={"spectral_anomalies": spectral_anomalies},
                confidence="high",
            )
        )

    if norm_bias_anomalies:
        top = sorted(norm_bias_anomalies, key=lambda d: d["bias_scale_ratio"], reverse=True)[0]
        findings.append(
            DiagnosticFinding(
                category=FindingCategory.ARCHITECTURE.value,
                severity=Severity.WARNING.value,
                module=top["layer"],
                observation=(
                    f"Normalization layer '{top['layer']}' shows extreme bias-to-scale shift "
                    f"(ratio={top['bias_scale_ratio']:.2f}, min_weight={top['min_weight']:.4f})."
                ),
                interpretation=(
                    "Strong normalization bias combined with compressed or negative affine scales shifts post-norm "
                    "embeddings into a narrow off-center cone."
                ),
                evidence={"norm_bias_anomalies": norm_bias_anomalies},
                confidence="high",
            )
        )

    if not findings:
        findings.append(
            DiagnosticFinding(
                category=FindingCategory.ARCHITECTURE.value,
                severity=Severity.INFO.value,
                observation=(
                    f"Pretrained weight health scan clean across {total_bn_layers} BatchNorm layer(s); "
                    "no subnormal collapse, rank-1 heads, or extreme spectral spikes detected."
                ),
                interpretation="Parameter spectra and normalization statistics lie in healthy operating bounds.",
                confidence="high",
            )
        )

    return {
        "bn_health": {
            "total_bn_layers": total_bn_layers,
            "dead_bn_layers_count": len(bn_dead_layers),
            "subnormal_bn_layers_count": len(subnormal_bn_layers),
            "total_dead_channels": total_bn_dead_channels,
            "dead_bn_layers": bn_dead_layers,
            "subnormal_bn_layers": subnormal_bn_layers,
        },
        "dead_weights": dead_weights,
        "zero_init_weights": zero_init_weights,
        "spectral_anomalies": spectral_anomalies,
        "collinear_heads": collinear_heads,
        "norm_bias_anomalies": norm_bias_anomalies,
        "findings": findings,
    }
