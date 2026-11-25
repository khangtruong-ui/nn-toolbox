"""Shared tensor extraction utilities for heterogeneous model outputs."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple
import torch

DEFAULT_PRIMARY_OUTPUT_KEYS: Tuple[str, ...] = (
    "mask_logits",
    "logits",
    "pred_masks",
    "masks",
    "output",
    "out",
    "pred",
)


def extract_primary_tensor(
    output: Any,
    preferred_keys: Sequence[str] = DEFAULT_PRIMARY_OUTPUT_KEYS,
) -> Optional[torch.Tensor]:
    """Extract the primary tensor from Tensor, tuple/list, or dict model outputs.

    Checks ``preferred_keys`` first when ``output`` is a ``dict`` so non-tensor metadata
    or auxiliary boolean fields (e.g., ``skip_mask_loss``) never shadow primary logits.
    """
    if torch.is_tensor(output):
        return output

    if isinstance(output, dict):
        for key in preferred_keys:
            val = output.get(key)
            if torch.is_tensor(val):
                return val
        for val in output.values():
            if torch.is_tensor(val):
                return val
        return None

    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item
        return None

    return None


def extract_output_tensors(output: Any) -> List[Tuple[str, torch.Tensor]]:
    """Collect all ``(name, Tensor)`` items from a Tensor, tuple/list, or dict output."""
    if torch.is_tensor(output):
        return [("output", output)]

    if isinstance(output, dict):
        return [(str(k), v) for k, v in output.items() if torch.is_tensor(v)]

    if isinstance(output, (tuple, list)):
        items = [t for t in output if torch.is_tensor(t)]
        if len(items) == 1:
            return [("output", items[0])]
        return [(f"output_{idx}", t) for idx, t in enumerate(items)]

    return []
