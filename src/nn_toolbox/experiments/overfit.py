"""
Automated memorization experiment on tiny dataset subsets (1, 2, 8, 32 samples).
Evaluates whether the optimization pipeline is fundamentally capable of memorizing data.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from nn_toolbox.core.finding import DiagnosticFinding, FindingCategory, Severity


def overfit_test(
    model: nn.Module,
    dataset_or_batch: Any,
    loss_fn: Callable[[Any, Any], torch.Tensor],
    optimizer_factory: Optional[Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer]] = None,
    sizes: Optional[List[int]] = None,
    max_steps: int = 30,
    target_loss: float = 0.01,
    learning_rate: float = 1e-3,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Train a copy of the model repeatedly on tiny subsets of data to test memorization capacity."""
    if sizes is None:
        sizes = [1, 2, 8]

    # Resolve target device
    if device is None:
        device = next(model.parameters()).device

    # Extract sample pool
    inputs_pool = []
    targets_pool = []

    # Handle dataset, dataloader, list, or batch dict/tuple
    def _slice_target(t: Any, idx: int) -> Any:
        if t is None:
            return None
        if torch.is_tensor(t):
            return t[idx : idx + 1]
        if isinstance(t, dict):
            return {k: _slice_target(v, idx) for k, v in t.items()}
        if isinstance(t, (tuple, list)):
            return [_slice_target(v, idx) for v in t]
        return t

    if isinstance(dataset_or_batch, dict):
        img = dataset_or_batch.get("image", dataset_or_batch.get("input", dataset_or_batch.get("x")))
        tgt = dataset_or_batch.get("mask", dataset_or_batch.get("label", dataset_or_batch.get("y", dataset_or_batch.get("target"))))
        if img is not None:
            n_avail = len(img)
            for i in range(n_avail):
                inputs_pool.append(img[i : i + 1])
                if tgt is not None:
                    targets_pool.append(_slice_target(tgt, i))
    elif isinstance(dataset_or_batch, (tuple, list)) and len(dataset_or_batch) >= 2 and torch.is_tensor(dataset_or_batch[0]):
        x_all, y_all = dataset_or_batch[0], dataset_or_batch[1]
        n_avail = len(x_all)
        for i in range(n_avail):
            inputs_pool.append(x_all[i : i + 1])
            targets_pool.append(_slice_target(y_all, i))
    elif hasattr(dataset_or_batch, "__iter__"):
        # Take up to max(sizes) samples from iterator/loader
        needed = max(sizes) if sizes else 32
        for batch in dataset_or_batch:
            if isinstance(batch, dict):
                img = batch.get("image", batch.get("input", batch.get("x")))
                tgt = batch.get("mask", batch.get("label", batch.get("y", batch.get("target"))))
                if img is not None:
                    for i in range(len(img)):
                        inputs_pool.append(img[i : i + 1])
                        if tgt is not None:
                            targets_pool.append(_slice_target(tgt, i))
            elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
                x_b, y_b = batch[0], batch[1]
                for i in range(len(x_b)):
                    inputs_pool.append(x_b[i : i + 1])
                    targets_pool.append(_slice_target(y_b, i))
            if len(inputs_pool) >= needed:
                break

    if not inputs_pool:
        return {
            "success": False,
            "error": "Could not extract sample inputs from the provided dataset or batch.",
            "results": [],
            "findings": [],
        }

    results: List[Dict[str, Any]] = []
    findings: List[DiagnosticFinding] = []

    # Store initial state dict to reset cleanly
    orig_state = copy.deepcopy(model.state_dict())

    def _collate_targets(pool_slice: List[Any], dev: torch.device) -> Any:
        if not pool_slice or pool_slice[0] is None:
            return None
        first = pool_slice[0]
        if torch.is_tensor(first):
            return torch.cat(pool_slice, dim=0).to(dev)
        if isinstance(first, dict):
            return {
                k: _collate_targets([t[k] for t in pool_slice if isinstance(t, dict) and k in t], dev)
                for k in first
            }
        if isinstance(first, (tuple, list)):
            return [
                _collate_targets([t[idx] for t in pool_slice if isinstance(t, (tuple, list)) and len(t) > idx], dev)
                for idx in range(len(first))
            ]
        return first

    try:
        for size in sizes:
            actual_size = min(size, len(inputs_pool))
            if actual_size == 0:
                continue

            sub_x = torch.cat(inputs_pool[:actual_size], dim=0).to(device)
            sub_y = _collate_targets(targets_pool[:actual_size], device) if targets_pool else None

            # Reset model parameters
            model.load_state_dict(orig_state)
            model.train()

            trainable_params = [p for p in model.parameters() if p.requires_grad]
            if not trainable_params:
                findings.append(
                    DiagnosticFinding(
                        category=FindingCategory.OPTIMIZATION.value,
                        severity=Severity.CRITICAL.value,
                        observation="Model has zero parameters with requires_grad=True.",
                        interpretation="The optimizer cannot update any weights because all parameters are frozen.",
                        hypotheses=["Parameters frozen by mistake", "requires_grad flags disabled"],
                        suggested_actions=["Check model.requires_grad_() or unfreeze target layers."],
                    )
                )
                break

            if optimizer_factory is not None:
                opt = optimizer_factory(trainable_params)
            else:
                opt = torch.optim.AdamW(trainable_params, lr=learning_rate)

            initial_loss = float("inf")
            final_loss = float("inf")
            loss_history: List[float] = []
            grad_norm_history: List[float] = []
            memorized = False

            for step_idx in range(max_steps):
                opt.zero_grad()
                out = model(sub_x)

                if sub_y is not None:
                    loss = loss_fn(out, sub_y)
                else:
                    loss = loss_fn(out)

                loss_val = float(loss.item())
                if step_idx == 0:
                    initial_loss = loss_val

                loss_history.append(loss_val)

                if math.isnan(loss_val) or math.isinf(loss_val):
                    break

                try:
                    loss.backward()
                except Exception:
                    # Detached graph, missing grad_fn, or backward failure
                    final_loss = loss_val
                    break

                # Measure gradient norm
                total_gnorm_sq = sum(
                    float(torch.linalg.norm(p.grad.float()).item()) ** 2
                    for p in trainable_params
                    if p.grad is not None
                )
                gnorm = math.sqrt(total_gnorm_sq)
                grad_norm_history.append(gnorm)

                opt.step()

                curr_reduction = (
                    (initial_loss - loss_val) / (abs(initial_loss) + 1e-12)
                    if not math.isnan(initial_loss) and abs(initial_loss) > 1e-12
                    else 0.0
                )

                if loss_val <= target_loss or (step_idx >= 5 and curr_reduction >= 0.6):
                    memorized = True
                    final_loss = loss_val
                    break

                final_loss = loss_val

            loss_reduction = (
                (initial_loss - final_loss) / (abs(initial_loss) + 1e-12)
                if not math.isnan(initial_loss) and not math.isnan(final_loss)
                else 0.0
            )

            res_entry = {
                "size": actual_size,
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "loss_reduction_pct": loss_reduction * 100.0,
                "steps_taken": len(loss_history),
                "memorized": memorized or (final_loss <= target_loss) or (loss_reduction > 0.6),
                "loss_history": loss_history[:10] + ([loss_history[-1]] if len(loss_history) > 10 else []),
                "final_grad_norm": grad_norm_history[-1] if grad_norm_history else 0.0,
            }
            results.append(res_entry)

        # Formulate findings based on pattern across sizes
        all_memorized = all(r["memorized"] for r in results)
        none_memorized = all(not r["memorized"] for r in results)

        if none_memorized:
            single_sample_res = results[0] if results else None
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.MEMORIZATION.value,
                    severity=Severity.CRITICAL.value,
                    observation=(
                        f"Model failed to overfit even a tiny dataset of {single_sample_res['size'] if single_sample_res else 1} sample(s). "
                        f"Final loss: {single_sample_res['final_loss'] if single_sample_res else 'N/A'} (reduction: {single_sample_res['loss_reduction_pct']:.1f}%)."
                        if single_sample_res
                        else "Model failed tiny overfit test."
                    ),
                    interpretation="The model architecture, loss function, or optimization setup cannot fit a trivial amount of data.",
                    evidence={"results": results},
                    hypotheses=[
                        "Bug in loss function or target labels (e.g. inverted targets, incorrect loss sign)",
                        "Broken gradient flow or detached computation graph",
                        "Learning rate too small or inappropriate optimizer",
                        "Missing parameter updates or severe activation saturation",
                    ],
                    confidence="high",
                    suggested_actions=[
                        "Verify loss_fn signature and target dimensions",
                        "Check gradient check and backward signal propagation",
                        "Test with a simpler optimizer (e.g. AdamW lr=1e-3)",
                    ],
                )
            )
        elif not all_memorized:
            # Succeeded on small, failed on larger
            ok_sizes = [r["size"] for r in results if r["memorized"]]
            failed_sizes = [r["size"] for r in results if not r["memorized"]]
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.MEMORIZATION.value,
                    severity=Severity.WARNING.value,
                    observation=f"Model successfully memorizes {ok_sizes} sample(s) but failed on {failed_sizes} sample(s).",
                    interpretation=(
                        "The optimization pipeline is functionally capable of learning, "
                        "but capacity, regularization, or optimization speed limits fitting larger batches."
                    ),
                    evidence={"ok_sizes": ok_sizes, "failed_sizes": failed_sizes, "results": results},
                    hypotheses=[
                        "Model capacity may be limited for the task complexity",
                        "Learning rate or step count may need tuning for larger batch sizes",
                        "Strong implicit regularization or normalization constraints",
                    ],
                    confidence="medium",
                    suggested_actions=[
                        "Inspect model capacity and width/depth",
                        "Perform learning rate sweep across batch sizes",
                    ],
                )
            )
        else:
            findings.append(
                DiagnosticFinding(
                    category=FindingCategory.MEMORIZATION.value,
                    severity=Severity.INFO.value,
                    observation=f"Model successfully memorized all tiny datasets: {[r['size'] for r in results]} samples.",
                    interpretation="The fundamental forward, backward, loss, and optimization pathways are working properly.",
                    evidence={"results": results},
                    hypotheses=["Core model and optimization pipeline are functional."],
                    confidence="high",
                )
            )

    finally:
        # Always restore original weights
        model.load_state_dict(orig_state)

    return {
        "success": True,
        "results": results,
        "findings": findings,
    }
