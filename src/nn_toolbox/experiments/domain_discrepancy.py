"""Cross-domain discrepancy diagnostics for pretrained & adapted networks.

Compares in-domain reference batches against out-of-distribution (OOD) batches:
1. ``batchnorm_domain_shift_test``: Quantifies per-layer BatchNorm covariate shift
   (standardized mean z-shift and |log variance ratio|) over active channels.
2. ``layer_representation_discrepancy_test``: Measures layer-by-layer strict split-half
   cross-validated centroid AUROC, ground-truth vs. 2-means separation ratio (GT/KM),
   clustering axis alignment, effective rank ratio, and feature cone concentration.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nn_toolbox.core.finding import DiagnosticFinding, FindingCategory, Severity
from nn_toolbox.core.tensor_utils import extract_primary_tensor


def _resolve_model_device(model: nn.Module, device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _compute_exact_auroc_1d(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Exact Mann-Whitney U AUROC with tie mid-ranks for 1D tensors."""
    n = scores.numel()
    n_pos = int((labels >= 0.5).sum().item())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    s = scores.float()
    t = (labels >= 0.5).to(torch.float64)
    sorted_s, sort_idx = torch.sort(s)
    sorted_t = t[sort_idx]

    diff = sorted_s[1:] != sorted_s[:-1]
    if not diff.any():
        return 0.5

    split_idx = torch.nonzero(diff, as_tuple=False).flatten() + 1
    zeros = torch.zeros(1, device=s.device, dtype=torch.long)
    total = torch.tensor([n], device=s.device, dtype=torch.long)
    starts = torch.cat([zeros, split_idx])
    ends = torch.cat([split_idx, total])
    midranks = (starts + ends + 1).to(torch.float64) / 2.0
    sorted_ranks = torch.repeat_interleave(midranks, ends - starts)
    pos_rank_sum = float((sorted_ranks * sorted_t).sum().item())
    u_stat = pos_rank_sum - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / float(n_pos * n_neg))


def _normalize_to_bchw(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    """Convert 4D NCHW/NHWC or 3D token tensors into float32 (B, C, H, W)."""
    if tensor.ndim == 4:
        b, d1, d2, d3 = tensor.shape
        # Detect NHWC (e.g. ViT blocks [B, H, W, C] where d1 == d2 != d3 and not small C)
        if d1 == d2 and d2 != d3 and d1 >= 4 and d3 >= 16:
            return tensor.permute(0, 3, 1, 2).contiguous().float()
        return tensor.contiguous().float()

    if tensor.ndim == 3:
        b, n, c = tensor.shape
        side = int(math.isqrt(n))
        if side * side == n and side > 1:
            return tensor.reshape(b, side, side, c).permute(0, 3, 1, 2).contiguous().float()
        return tensor.permute(0, 2, 1).unsqueeze(2).contiguous().float()

    return None


def _subsample_2means_sep(fs: torch.Tensor, num_iters: int = 6) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """Deterministic global-mean + farthest-point spherical 2-means on unit vectors [N, C]."""
    c1 = F.normalize(fs.mean(dim=0, keepdim=True), p=2, dim=-1)
    d1 = 1.0 - (fs @ c1.t()).squeeze(-1)
    k_far = max(1, fs.shape[0] // 50)
    far_idx = torch.topk(d1, k=k_far, dim=-1).indices
    c2 = F.normalize(fs[far_idx].mean(dim=0, keepdim=True), p=2, dim=-1)
    cent = torch.cat([c1, c2], dim=0)

    for _ in range(num_iters):
        assign = (fs @ cent.t()).argmax(dim=-1)
        for k in (0, 1):
            mk = assign == k
            if mk.any():
                cent[k] = F.normalize(fs[mk].mean(dim=0), p=2, dim=0)

    km_sep = float((1.0 - (cent[0] * cent[1]).sum()).clamp(min=0.0).item())
    return km_sep, cent[0], cent[1]


@torch.no_grad()
def _compute_layer_forensic_metrics(
    feat_bchw: torch.Tensor,
    mask_tensor: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    b, c, h, w = feat_bchw.shape
    f_raw = feat_bchw.float()
    fn = F.normalize(f_raw, p=2, dim=1)

    gt_grid: Optional[torch.Tensor] = None
    if mask_tensor is not None:
        m = mask_tensor.float()
        if m.ndim == 3:
            m = m.unsqueeze(1)
        gt_grid = F.interpolate(m.to(f_raw.device), size=(h, w), mode="area") >= 0.5

    stds = [float(f_raw.std().item())]
    eranks: List[float] = []
    cone_coss: List[float] = []
    km_seps: List[float] = []
    gt_seps: List[float] = []
    cv_aurocs: List[float] = []
    axis_als: List[float] = []

    for i in range(b):
        fi = fn[i].permute(1, 2, 0).reshape(-1, c)
        n_px = fi.shape[0]
        if n_px < 2:
            continue
        step = max(1, n_px // 512)
        fs = fi[::step]

        cone_coss.append(float(fs.mean(dim=0).pow(2).sum().item()))

        fc = fs - fs.mean(dim=0, keepdim=True)
        try:
            sv = torch.linalg.svdvals(fc)
            sv = sv[sv > 1e-8]
            if sv.numel() > 0:
                p = sv / sv.sum()
                erank = float(torch.exp(-(p * p.log()).sum()).item())
                eranks.append(erank / float(max(1, min(fs.shape))))
        except Exception:
            pass

        km_sep, km_c0, km_c1 = _subsample_2means_sep(fs)
        km_seps.append(km_sep)

        if gt_grid is not None:
            gi = gt_grid[i, 0].reshape(-1)
            pos_idx = torch.nonzero(gi, as_tuple=False).squeeze(-1)
            neg_idx = torch.nonzero(~gi, as_tuple=False).squeeze(-1)
            if pos_idx.numel() >= 4 and neg_idx.numel() >= 4:
                p_tr, p_te = pos_idx[0::2], pos_idx[1::2]
                n_tr, n_te = neg_idx[0::2], neg_idx[1::2]
                mu_p = F.normalize(fi[p_tr].mean(dim=0, keepdim=True), p=2, dim=-1)
                mu_n = F.normalize(fi[n_tr].mean(dim=0, keepdim=True), p=2, dim=-1)
                te_idx = torch.cat([p_te, n_te], dim=0)
                te_lbl = torch.cat([
                    torch.ones(p_te.numel(), device=fi.device),
                    torch.zeros(n_te.numel(), device=fi.device),
                ])
                sc = ((fi[te_idx] @ mu_p.t()) - (fi[te_idx] @ mu_n.t())).squeeze(-1)
                auc = _compute_exact_auroc_1d(sc, te_lbl)
                if not math.isnan(auc):
                    cv_aurocs.append(auc)

                mu_pf = F.normalize(fi[pos_idx].mean(dim=0), p=2, dim=0)
                mu_nf = F.normalize(fi[neg_idx].mean(dim=0), p=2, dim=0)
                gt_seps.append(float((1.0 - (mu_pf * mu_nf).sum()).clamp(min=0.0).item()))

                u_gt = mu_pf - mu_nf
                u_km = km_c0 - km_c1
                ng, nk = float(u_gt.norm().item()), float(u_km.norm().item())
                if ng > 1e-8 and nk > 1e-8:
                    axis_als.append(float(((u_gt / ng) * (u_km / nk)).sum().abs().item()))

    def _safe_mean(vals: List[float]) -> float:
        return float(np.mean(vals)) if vals else float("nan")

    gt_mean = _safe_mean(gt_seps)
    km_mean = _safe_mean(km_seps)
    gt_km_ratio = (
        float(gt_mean / (km_mean + 1e-8))
        if not (math.isnan(gt_mean) or math.isnan(km_mean))
        else float("nan")
    )
    return {
        "std": _safe_mean(stds),
        "erank_ratio": _safe_mean(eranks),
        "cone_cos": _safe_mean(cone_coss),
        "cv_auroc": _safe_mean(cv_aurocs),
        "gt_sep": gt_mean,
        "km_sep": km_mean,
        "gt_to_km_ratio": gt_km_ratio,
        "axis_alignment": _safe_mean(axis_als),
    }


def batchnorm_domain_shift_test(
    model: nn.Module,
    ref_inputs: torch.Tensor,
    ood_inputs: torch.Tensor,
    active_var_threshold: float = 1e-6,
    shift_delta_threshold: float = 0.4,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Compare BatchNorm running statistics against live reference vs OOD activations.

    Excludes dead/subnormal channels (``running_var <= active_var_threshold``) to avoid
    division underflow / NaN contamination.
    """
    dev = _resolve_model_device(model, device)
    was_training = model.training
    model.eval()

    bn_modules = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))
        and getattr(mod, "track_running_stats", False)
        and mod.running_mean is not None
        and mod.running_var is not None
    ]

    if not bn_modules:
        model.train(was_training)
        return {
            "num_bn_layers": 0,
            "ref_mean_z": 0.0,
            "ood_mean_z": 0.0,
            "ref_log_var_ratio": 0.0,
            "ood_log_var_ratio": 0.0,
            "max_shift_delta": 0.0,
            "top_shifted_layers": [],
            "findings": [],
        }

    stats: Dict[str, Dict[str, List[Tuple[float, float]]]] = {
        "ref": {name: [] for name, _ in bn_modules},
        "ood": {name: [] for name, _ in bn_modules},
    }
    domain_holder = ["ref"]
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(layer_name: str, bn_mod: nn.Module):
        rm = bn_mod.running_mean.detach().float()
        rv = bn_mod.running_var.detach().float()
        eps = float(bn_mod.eps)
        active = rv > active_var_threshold

        def _hook(_mod: nn.Module, inp: Tuple[Any, ...], _out: Any) -> None:
            if not inp or not torch.is_tensor(inp[0]) or not active.any():
                return
            x_in = inp[0].detach().float()
            reduce_dims = (0, *range(2, x_in.ndim))
            b_mean = x_in.mean(dim=reduce_dims)
            b_var = x_in.var(dim=reduce_dims, unbiased=False)
            mz = float(((b_mean[active] - rm[active]).abs() / torch.sqrt(rv[active] + eps)).mean().item())
            lvr = float((torch.log((b_var[active] + eps) / (rv[active] + eps)).abs()).mean().item())
            stats[domain_holder[0]][layer_name].append((mz, lvr))

        return _hook

    try:
        for name, mod in bn_modules:
            handles.append(mod.register_forward_hook(_make_hook(name, mod)))
        with torch.no_grad():
            domain_holder[0] = "ref"
            model(ref_inputs.to(dev))
            domain_holder[0] = "ood"
            model(ood_inputs.to(dev))
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    layer_rows: List[Dict[str, Any]] = []
    skipped_dead_layers = 0
    for name, _ in bn_modules:
        r_vals = stats["ref"][name]
        o_vals = stats["ood"][name]
        if not r_vals or not o_vals:
            skipped_dead_layers += 1
            continue
        ref_mz = float(np.mean([v[0] for v in r_vals]))
        ref_lvr = float(np.mean([v[1] for v in r_vals]))
        ood_mz = float(np.mean([v[0] for v in o_vals]))
        ood_lvr = float(np.mean([v[1] for v in o_vals]))
        delta = (ood_mz - ref_mz) + (ood_lvr - ref_lvr)
        layer_rows.append({
            "layer": name,
            "ref_mean_z": ref_mz,
            "ood_mean_z": ood_mz,
            "ref_log_var_ratio": ref_lvr,
            "ood_log_var_ratio": ood_lvr,
            "shift_delta": delta,
        })

    layer_rows.sort(key=lambda r: r["shift_delta"], reverse=True)
    ref_mean_z = float(np.mean([r["ref_mean_z"] for r in layer_rows])) if layer_rows else 0.0
    ood_mean_z = float(np.mean([r["ood_mean_z"] for r in layer_rows])) if layer_rows else 0.0
    ref_lvr_all = float(np.mean([r["ref_log_var_ratio"] for r in layer_rows])) if layer_rows else 0.0
    ood_lvr_all = float(np.mean([r["ood_log_var_ratio"] for r in layer_rows])) if layer_rows else 0.0
    max_delta = layer_rows[0]["shift_delta"] if layer_rows else 0.0

    findings: List[DiagnosticFinding] = []
    if max_delta >= shift_delta_threshold or (ood_lvr_all - ref_lvr_all) >= 0.15:
        top = layer_rows[0] if layer_rows else {}
        findings.append(
            DiagnosticFinding(
                category=FindingCategory.DATA.value,
                severity=Severity.WARNING.value,
                module=top.get("layer"),
                observation=(
                    f"Elevated BatchNorm OOD covariate shift across {len(layer_rows)} active BN layer(s): "
                    f"|log_var_ratio| ref={ref_lvr_all:.3f} -> ood={ood_lvr_all:.3f} "
                    f"(peak layer '{top.get('layer', 'N/A')}' delta={max_delta:+.3f})."
                ),
                interpretation=(
                    "Frozen BatchNorm running statistics diverge substantially on OOD target activations, "
                    "mismatching standardization at normalization boundaries."
                ),
                evidence={
                    "ref_mean_z": ref_mean_z,
                    "ood_mean_z": ood_mean_z,
                    "ref_log_var_ratio": ref_lvr_all,
                    "ood_log_var_ratio": ood_lvr_all,
                    "skipped_dead_layers": skipped_dead_layers,
                    "top_shifted_layers": layer_rows[:5],
                },
                confidence="high",
                suggested_actions=[
                    "Test BatchNorm running-stat recalibration / Test-Time BN adaptation or LayerNorm transition layers.",
                ],
            )
        )
    else:
        findings.append(
            DiagnosticFinding(
                category=FindingCategory.DATA.value,
                severity=Severity.INFO.value,
                observation=(
                    f"BatchNorm activations are well aligned across domains ({len(layer_rows)} active layers, "
                    f"ref |log_var_ratio|={ref_lvr_all:.3f}, ood={ood_lvr_all:.3f})."
                ),
                interpretation="Frozen BatchNorm buffers generalize stably between reference and target batches.",
                confidence="high",
            )
        )

    return {
        "num_bn_layers": len(bn_modules),
        "active_bn_layers": len(layer_rows),
        "skipped_dead_layers": skipped_dead_layers,
        "ref_mean_z": ref_mean_z,
        "ood_mean_z": ood_mean_z,
        "ref_log_var_ratio": ref_lvr_all,
        "ood_log_var_ratio": ood_lvr_all,
        "max_shift_delta": max_delta,
        "top_shifted_layers": layer_rows[:10],
        "findings": findings,
    }


def _select_candidate_modules(model: nn.Module, max_modules: int = 12) -> List[str]:
    candidates = [
        name
        for name, mod in model.named_modules()
        if name and isinstance(mod, (nn.Conv2d, nn.BatchNorm2d, nn.LayerNorm, nn.MultiheadAttention))
    ]
    if not candidates:
        candidates = [name for name, mod in model.named_modules() if name and len(list(mod.children())) == 0]
    if len(candidates) <= max_modules:
        return candidates
    indices = np.linspace(0, len(candidates) - 1, num=max_modules, dtype=int)
    seen: set[str] = set()
    selected: List[str] = []
    for idx in indices:
        nm = candidates[int(idx)]
        if nm not in seen:
            seen.add(nm)
            selected.append(nm)
    return selected


def layer_representation_discrepancy_test(
    model: nn.Module,
    ref_inputs: torch.Tensor,
    ood_inputs: torch.Tensor,
    ref_masks: Optional[torch.Tensor] = None,
    ood_masks: Optional[torch.Tensor] = None,
    module_names: Optional[Sequence[str]] = None,
    bottleneck_drop_threshold: float = 0.25,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Audit per-layer representation geometry and split-half separability across domains.

    Detects representation bottlenecks where intermediate layers separate OOD foreground/background
    well (high ``gt_to_km_ratio`` / ``cv_auroc``) while later projection/neck layers collapse OOD
    signal while retaining reference-domain accuracy.
    """
    dev = _resolve_model_device(model, device)
    was_training = model.training
    model.eval()

    named_map = dict(model.named_modules())
    target_names = list(module_names) if module_names is not None else _select_candidate_modules(model)
    valid_names = [n for n in target_names if n in named_map]

    captured: Dict[str, Dict[str, List[Dict[str, float]]]] = {"ref": {}, "ood": {}}
    cur_domain = ["ref"]
    cur_mask: List[Optional[torch.Tensor]] = [ref_masks]
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(mod_name: str):
        def _hook(_mod: nn.Module, _inp: Tuple[Any, ...], out: Any) -> None:
            t = extract_primary_tensor(out)
            if t is None:
                return
            feat = _normalize_to_bchw(t)
            if feat is None or feat.shape[1] < 2:
                return
            m = _compute_layer_forensic_metrics(feat, cur_mask[0])
            captured[cur_domain[0]].setdefault(mod_name, []).append(m)

        return _hook

    try:
        for nm in valid_names:
            handles.append(named_map[nm].register_forward_hook(_make_hook(nm)))
        with torch.no_grad():
            cur_domain[0] = "ref"
            cur_mask[0] = ref_masks
            model(ref_inputs.to(dev))

            cur_domain[0] = "ood"
            cur_mask[0] = ood_masks
            model(ood_inputs.to(dev))
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    per_layer: Dict[str, Dict[str, Dict[str, float]]] = {}
    ordered_layers: List[str] = []
    for nm in valid_names:
        r_list = captured["ref"].get(nm)
        o_list = captured["ood"].get(nm)
        if not r_list or not o_list:
            continue
        ordered_layers.append(nm)
        per_layer[nm] = {
            "ref": {k: float(np.nanmean([x[k] for x in r_list])) for k in r_list[0]},
            "ood": {k: float(np.nanmean([x[k] for x in o_list])) for k in o_list[0]},
        }

    findings: List[DiagnosticFinding] = []
    bottleneck_summary: Dict[str, Any] = {}

    if len(ordered_layers) >= 2:
        final_name = ordered_layers[-1]
        final_ref = per_layer[final_name]["ref"]
        final_ood = per_layer[final_name]["ood"]

        # 1. Check OOD projection bottleneck collapse across layers
        ood_ratios = [
            (nm, per_layer[nm]["ood"]["gt_to_km_ratio"])
            for nm in ordered_layers[:-1]
            if not math.isnan(per_layer[nm]["ood"]["gt_to_km_ratio"])
        ]
        final_ood_ratio = final_ood["gt_to_km_ratio"]
        final_ref_ratio = final_ref["gt_to_km_ratio"]

        if ood_ratios and not math.isnan(final_ood_ratio):
            peak_name, peak_ood_ratio = max(ood_ratios, key=lambda kv: kv[1])
            peak_ref_ratio = per_layer[peak_name]["ref"]["gt_to_km_ratio"]
            ood_drop = peak_ood_ratio - final_ood_ratio
            ref_drop = (peak_ref_ratio - final_ref_ratio) if not (math.isnan(peak_ref_ratio) or math.isnan(final_ref_ratio)) else 0.0
            bottleneck_summary = {
                "peak_ood_layer": peak_name,
                "peak_ood_gt_to_km": peak_ood_ratio,
                "final_layer": final_name,
                "final_ood_gt_to_km": final_ood_ratio,
                "ood_drop": ood_drop,
                "ref_drop": ref_drop,
            }
            if ood_drop >= bottleneck_drop_threshold and (ref_drop < bottleneck_drop_threshold * 0.5 or final_ref_ratio >= 0.7):
                findings.append(
                    DiagnosticFinding(
                        category=FindingCategory.ARCHITECTURE.value,
                        severity=Severity.WARNING.value,
                        module=final_name,
                        observation=(
                            f"Detected OOD representation bottleneck at '{final_name}': GT/KM ratio drops from "
                            f"{peak_ood_ratio:.3f} at '{peak_name}' down to {final_ood_ratio:.3f} "
                            f"(Δ={ood_drop:+.3f}), while reference-domain GT/KM stays preserved ({final_ref_ratio:.3f})."
                        ),
                        interpretation=(
                            "Late projection/neck layers selectively amplify reference-domain task modes while attenuating "
                            "or entangling OOD forensic cues present in earlier hidden blocks."
                        ),
                        evidence=bottleneck_summary,
                        confidence="high",
                        suggested_actions=[
                            f"Tap earlier hidden features around '{peak_name}' or replace/re-train projection head '{final_name}'.",
                        ],
                    )
                )

        # 2. Check severe cross-domain CV AUROC gap at final layer
        r_auc, o_auc = final_ref["cv_auroc"], final_ood["cv_auroc"]
        if not (math.isnan(r_auc) or math.isnan(o_auc)) and (r_auc - o_auc) >= 0.15:
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.ARCHITECTURE.value,
                    severity=Severity.WARNING.value,
                    module=final_name,
                    observation=(
                        f"Large split-half CV AUROC cross-domain drop at '{final_name}': "
                        f"ref={r_auc:.3f} vs ood={o_auc:.3f} (Δ={r_auc - o_auc:.3f})."
                    ),
                    interpretation="Final representations separate reference foreground cleanly but degrade on OOD inputs.",
                    evidence={"ref_cv_auroc": r_auc, "ood_cv_auroc": o_auc},
                    confidence="high",
                )
            )

        # 3. Check extreme positive-cone / bias concentration on OOD
        ood_cone = final_ood["cone_cos"]
        if not math.isnan(ood_cone) and ood_cone >= 0.70:
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.ARCHITECTURE.value,
                    severity=Severity.WARNING.value,
                    module=final_name,
                    observation=(
                        f"Narrow feature cone concentration on OOD inputs at '{final_name}' "
                        f"(mean pairwise cosine E[f_i·f_j] = {ood_cone:.3f})."
                    ),
                    interpretation=(
                        "Unit-normalized spatial tokens concentrate in a narrow positive cone, collapsing cosine "
                        "distances and destabilizing soft-assignment clustering."
                    ),
                    evidence={"ood_cone_cos": ood_cone, "ref_cone_cos": final_ref["cone_cos"]},
                    confidence="high",
                )
            )

    if not findings:
        findings.append(
            DiagnosticFinding(
                category=FindingCategory.ARCHITECTURE.value,
                severity=Severity.INFO.value,
                observation=f"Layer representation discrepancy check completed across {len(ordered_layers)} layer(s) without bottleneck collapse.",
                interpretation="Reference and target representations stay well conditioned across probed depth.",
                confidence="high",
            )
        )

    return {
        "probed_layers": ordered_layers,
        "per_layer": per_layer,
        "bottleneck_summary": bottleneck_summary,
        "findings": findings,
    }


def _compute_binary_iou_1d(pred_bin: torch.Tensor, target_bin: torch.Tensor) -> float:
    inter = float((pred_bin & target_bin).sum().item())
    union = float((pred_bin | target_bin).sum().item())
    if union <= 0.0:
        return 1.0 if inter == 0.0 else 0.0
    return inter / union


@torch.no_grad()
def clustering_polarity_diagnostic(
    model: nn.Module,
    domain_batches: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    feature_extractor: Optional[Any] = None,
    device: Optional[Union[str, torch.device]] = None,
    border_weight: float = 0.75,
    area_gate: float = 0.25,
    temperature: float = 0.1,
    polarity_flip_warn_threshold: float = 0.25,
    hijack_ratio_threshold: float = 0.40,
) -> Dict[str, Any]:
    """Quantify 2-means polarity inversion vs. semantic variance hijacking across domains.

    For each domain's ``(images, masks)`` pair, extracts normalized spatial features,
    runs flip-equivariant spherical 2-means clustering, and compares:
      - ``minority`` polarity (standard smaller-cluster-is-tampered heuristic),
      - ``border_aware`` polarity (penalizes border-touching clusters when min area > ``area_gate``),
      - ``oracle`` polarity (best GT-aligned cluster assignment).

    Args:
        model: Neural network module to evaluate (switched temporarily to eval mode).
        domain_batches: Mapping ``domain_name -> (images [B, C, H, W], masks [B, 1, H, W])``.
        feature_extractor: Optional callable ``(model, images) -> features``. Defaults to
            ``model.extract_features(images)`` when present, otherwise ``model(images)``.
        device: Optional target device.
        border_weight: Weight on perimeter fraction in border-aware polarity scoring.
        area_gate: Minimum cluster area fraction required before border override activates.
        temperature: Cosine similarity logit temperature.
        polarity_flip_warn_threshold: Flip rate threshold triggering a polarity-inversion finding.
        hijack_ratio_threshold: ``gt_to_km_ratio`` threshold triggering a semantic-hijacking finding.

    Returns:
        Dictionary with ``per_domain`` metrics and ``DiagnosticFinding`` list.
    """
    target_device = _resolve_model_device(model, device)
    was_training = model.training
    model.eval()
    per_domain: Dict[str, Dict[str, float]] = {}
    findings: List[DiagnosticFinding] = []

    try:
        for dom_name, (images, masks) in domain_batches.items():
            imgs = images.to(target_device)
            msks = masks.to(target_device).float()
            if msks.ndim == 3:
                msks = msks.unsqueeze(1)

            if feature_extractor is not None:
                raw_feats = feature_extractor(model, imgs)
            elif hasattr(model, "extract_features"):
                raw_feats = model.extract_features(imgs)
            else:
                raw_feats = extract_primary_tensor(model(imgs))

            feat_bchw = _normalize_to_bchw(extract_primary_tensor(raw_feats))
            if feat_bchw is None:
                raise ValueError(f"Expected 3D/4D spatial features for domain '{dom_name}', got {tuple(raw_feats.shape)}.")

            b, c, h_f, w_f = feat_bchw.shape
            n_px = h_f * w_f
            f_norm = F.normalize(feat_bchw.float(), p=2, dim=1)
            f_img = f_norm.permute(0, 2, 3, 1).reshape(b, n_px, c)
            gt_full = (msks[:, 0] >= 0.5)
            orig_h, orig_w = msks.shape[2], msks.shape[3]
            gt_grid = (F.interpolate(msks, size=(h_f, w_f), mode="area")[:, 0] >= 0.5)

            bmask = torch.zeros((h_f, w_f), dtype=torch.bool, device=target_device)
            bmask[0, :] = True
            bmask[-1, :] = True
            bmask[:, 0] = True
            bmask[:, -1] = True
            b_flat = bmask.view(1, n_px).expand(b, -1)
            n_border = max(1.0, float(bmask.sum().item()))

            # Flip-equivariant mean + top-k farthest-point initialization
            c_mean = F.normalize(f_img.mean(dim=1, keepdim=True), p=2, dim=-1)
            k_far = max(1, n_px // 50)
            d1 = 1.0 - torch.bmm(f_img, c_mean.transpose(1, 2)).squeeze(-1)
            far1 = torch.topk(d1, k=k_far, dim=-1).indices
            c2 = F.normalize(torch.gather(f_img, 1, far1.unsqueeze(-1).expand(-1, -1, c)).mean(dim=1, keepdim=True), p=2, dim=-1)
            d2 = 1.0 - torch.bmm(f_img, c2.transpose(1, 2)).squeeze(-1)
            far2 = torch.topk(d2, k=k_far, dim=-1).indices
            c1 = F.normalize(torch.gather(f_img, 1, far2.unsqueeze(-1).expand(-1, -1, c)).mean(dim=1, keepdim=True), p=2, dim=-1)
            centroids = torch.cat([c1, c2], dim=1)

            for _ in range(8):
                sims = torch.bmm(f_img, centroids.transpose(1, 2))
                assign = torch.argmax(sims, dim=-1)
                w0 = (assign == 0).unsqueeze(-1).float()
                w1 = (assign == 1).unsqueeze(-1).float()
                cnt0 = w0.sum(dim=1, keepdim=True)
                cnt1 = w1.sum(dim=1, keepdim=True)
                s0 = torch.bmm(w0.transpose(1, 2), f_img)
                s1 = torch.bmm(w1.transpose(1, 2), f_img)
                nc0 = F.normalize(torch.where(cnt0 > 0, s0 / cnt0.clamp(min=1.0), centroids[:, 0:1, :]), p=2, dim=-1)
                nc1 = F.normalize(torch.where(cnt1 > 0, s1 / cnt1.clamp(min=1.0), centroids[:, 1:2, :]), p=2, dim=-1)
                centroids = torch.cat([nc0, nc1], dim=1)

            sims = torch.bmm(f_img, centroids.transpose(1, 2))
            labels = torch.argmax(sims, dim=-1)
            c1_cnt = labels.sum(dim=-1, keepdim=True).float()
            c0_cnt = float(n_px) - c1_cnt

            area0 = c0_cnt / float(max(1, n_px))
            area1 = c1_cnt / float(max(1, n_px))
            bf1 = ((labels == 1) & b_flat).float().sum(dim=-1, keepdim=True) / n_border
            bf0 = 1.0 - bf1
            cond_min = c1_cnt > c0_cnt
            cond_border = (area1 + border_weight * bf1) > (area0 + border_weight * bf0)
            if area_gate > 0.0:
                cond_border = torch.where(torch.minimum(area0, area1) > area_gate, cond_border, cond_min)

            min_ious: List[float] = []
            brd_ious: List[float] = []
            orc_ious: List[float] = []
            min_aucs: List[float] = []
            brd_aucs: List[float] = []
            orc_aucs: List[float] = []
            flips = 0
            pos_imgs = 0
            axis_aligns: List[float] = []
            gt_km_ratios: List[float] = []

            for i in range(b):
                gt_i_full = gt_full[i]
                if not gt_i_full.any():
                    continue
                pos_imgs += 1
                gt_i_grid = gt_grid[i].reshape(n_px)

                def _eval_cond(c0_is_tamp: bool) -> Tuple[float, float]:
                    diff_i = ((sims[i, :, 0] - sims[i, :, 1]) if c0_is_tamp else (sims[i, :, 1] - sims[i, :, 0])) / max(1e-4, float(temperature))
                    logit_up = F.interpolate(
                        diff_i.view(1, 1, h_f, w_f), size=(orig_h, orig_w), mode="bilinear", align_corners=False
                    )[0, 0]
                    pred_i = logit_up >= 0.0
                    return (
                        _compute_binary_iou_1d(pred_i, gt_i_full),
                        _compute_exact_auroc_1d(logit_up.reshape(-1), gt_i_full.reshape(-1).float()),
                    )

                iou_0, auc_0 = _eval_cond(True)
                iou_1, auc_1 = _eval_cond(False)
                oracle_c0 = iou_0 >= iou_1
                min_c0 = bool(cond_min[i, 0].item())
                brd_c0 = bool(cond_border[i, 0].item())

                if min_c0 != oracle_c0:
                    flips += 1

                min_ious.append(iou_0 if min_c0 else iou_1)
                min_aucs.append(auc_0 if min_c0 else auc_1)
                brd_ious.append(iou_0 if brd_c0 else iou_1)
                brd_aucs.append(auc_0 if brd_c0 else auc_1)
                orc_ious.append(max(iou_0, iou_1))
                orc_aucs.append(auc_0 if oracle_c0 else auc_1)

                if gt_i_grid.any() and (~gt_i_grid).any():
                    mu_p = F.normalize(f_img[i][gt_i_grid].mean(dim=0), p=2, dim=0)
                    mu_n = F.normalize(f_img[i][~gt_i_grid].mean(dim=0), p=2, dim=0)
                    gt_sep = float((1.0 - (mu_p * mu_n).sum()).clamp(min=0.0).item())
                    km_sep = float((1.0 - (centroids[i, 0] * centroids[i, 1]).sum()).clamp(min=0.0).item())
                    gt_km_ratios.append(gt_sep / max(km_sep, 1e-6))
                    gt_axis = F.normalize(mu_p - mu_n, p=2, dim=0)
                    km_axis = F.normalize(centroids[i, 0] - centroids[i, 1], p=2, dim=0)
                    axis_aligns.append(float((gt_axis * km_axis).sum().abs().item()))

            dom_metrics = {
                "samples": float(b),
                "positive_samples": float(pos_imgs),
                "minority_iou": float(np.nanmean(min_ious)) if min_ious else float("nan"),
                "border_polarity_iou": float(np.nanmean(brd_ious)) if brd_ious else float("nan"),
                "oracle_polarity_iou": float(np.nanmean(orc_ious)) if orc_ious else float("nan"),
                "minority_auroc": float(np.nanmean(min_aucs)) if min_aucs else float("nan"),
                "border_polarity_auroc": float(np.nanmean(brd_aucs)) if brd_aucs else float("nan"),
                "oracle_polarity_auroc": float(np.nanmean(orc_aucs)) if orc_aucs else float("nan"),
                "polarity_flip_rate": float(flips / max(1, pos_imgs)) if pos_imgs > 0 else 0.0,
                "axis_alignment": float(np.nanmean(axis_aligns)) if axis_aligns else float("nan"),
                "gt_to_km_ratio": float(np.nanmean(gt_km_ratios)) if gt_km_ratios else float("nan"),
            }
            per_domain[dom_name] = dom_metrics

            flip_rate = dom_metrics["polarity_flip_rate"]
            orc_gap = dom_metrics["oracle_polarity_iou"] - dom_metrics["minority_iou"]
            if pos_imgs > 0 and flip_rate >= polarity_flip_warn_threshold and not math.isnan(orc_gap) and orc_gap >= 0.05:
                findings.append(
                    DiagnosticFinding(
                        category=FindingCategory.ARCHITECTURE.value,
                        severity=Severity.WARNING.value,
                        module=dom_name,
                        observation=(
                            f"POLARITY_INVERSION on '{dom_name}': minority-cluster heuristic flipped "
                            f"{flip_rate * 100:.1f}% of samples (minority IoU={dom_metrics['minority_iou']:.3f} vs "
                            f"oracle={dom_metrics['oracle_polarity_iou']:.3f}; border-aware={dom_metrics['border_polarity_iou']:.3f})."
                        ),
                        interpretation="Tampered regions occupy >=50% area or background wraps image edges, breaking smaller-cluster assumption.",
                        evidence=dom_metrics,
                        confidence="high",
                        suggested_actions=[
                            "Replace minority-cluster polarity with border-aware or learned antisymmetric cluster polarity.",
                        ],
                    )
                )

            gk = dom_metrics["gt_to_km_ratio"]
            if not math.isnan(gk) and gk <= hijack_ratio_threshold:
                findings.append(
                    DiagnosticFinding(
                        category=FindingCategory.ARCHITECTURE.value,
                        severity=Severity.WARNING.value,
                        module=dom_name,
                        observation=(
                            f"SEMANTIC_HIJACKING on '{dom_name}': GT/KM separation ratio is {gk:.3f} "
                            f"(<= {hijack_ratio_threshold:.2f}; oracle IoU={dom_metrics['oracle_polarity_iou']:.3f})."
                        ),
                        interpretation="Non-task background semantics dominate 2-means variance over the target signal.",
                        evidence=dom_metrics,
                        confidence="high",
                        suggested_actions=[
                            "Fuse intermediate/pre-neck representations or adapt the feature extractor on paired target-domain cues.",
                        ],
                    )
                )

        if not findings:
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.ARCHITECTURE.value,
                    severity=Severity.INFO.value,
                    observation=f"Clustering polarity diagnostic completed across {len(per_domain)} domain(s) with no inversion or hijacking alerts.",
                    interpretation="2-means cluster axes and polarity align with ground-truth masks.",
                    confidence="high",
                )
            )
    finally:
        if was_training:
            model.train()

    return {"per_domain": per_domain, "findings": findings}
