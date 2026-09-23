"""
Activation monitoring and forward signal propagation instrumentation.
Computes streaming statistics without caching full intermediate tensors.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from nn_toolbox.analysis.statistics import WelfordAccumulator, compute_tensor_stats
from nn_toolbox.instrumentation.hooks import HookManager


class ActivationMonitor:
    """Monitors intermediate module activations during forward passes

    and analyzes forward signal propagation dynamics.
    """

    def __init__(
        self,
        model: nn.Module,
        hook_manager: Optional[HookManager] = None,
        sample_limit: int = 4096,
        compute_quantiles: bool = True,
    ):
        self.model = model
        self.sample_limit = sample_limit
        self.compute_quantiles = compute_quantiles

        self._own_hook_manager = hook_manager is None
        self.hook_manager = hook_manager or HookManager(model)

        self.step_stats: Dict[int, Dict[str, Dict[str, Any]]] = {}
        self.accumulators: Dict[str, WelfordAccumulator] = {}
        self.layer_order: List[str] = []
        self._current_step: int = 0
        self._attached: bool = False

    def attach(self) -> None:
        """Attach forward hooks to target modules."""
        if self._attached:
            return
        self.hook_manager.register_forward_hook(self._on_forward)
        self._attached = True

    def detach(self) -> None:
        """Remove forward hooks."""
        if self._attached and self._own_hook_manager:
            self.hook_manager.remove_hooks()
            self._attached = False

    def reset(self) -> None:
        """Clear recorded statistics."""
        self.step_stats.clear()
        self.accumulators.clear()
        self.layer_order.clear()
        self._current_step = 0

    def step(self) -> None:
        """Advance to the next recording step."""
        self._current_step += 1

    def _extract_tensor(self, output: Any) -> Optional[torch.Tensor]:
        """Unpack tensor from tuple/list/dict module return values."""
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)):
            for item in output:
                if isinstance(item, torch.Tensor):
                    return item
        if isinstance(output, dict):
            for val in output.values():
                if isinstance(val, torch.Tensor):
                    return val
        return None

    def _on_forward(self, module_name: str, module: nn.Module, inputs: Any, outputs: Any) -> None:
        """Forward hook callback that calculates stats and discards output tensor."""
        tensor = self._extract_tensor(outputs)
        if tensor is None:
            return

        if module_name not in self.layer_order:
            self.layer_order.append(module_name)

        if module_name not in self.accumulators:
            self.accumulators[module_name] = WelfordAccumulator()

        self.accumulators[module_name].update(tensor)

        stats = compute_tensor_stats(
            tensor,
            sample_limit=self.sample_limit,
            compute_quantiles=self.compute_quantiles,
            compute_higher_moments=True,
        )

        if self._current_step not in self.step_stats:
            self.step_stats[self._current_step] = {}
        self.step_stats[self._current_step][module_name] = stats

    def get_latest_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return activation statistics for the most recent forward step."""
        if not self.step_stats:
            return {}
        latest_idx = max(self.step_stats.keys())
        return self.step_stats[latest_idx]

    def analyze_signal_propagation(self, step: Optional[int] = None) -> Dict[str, Any]:
        """Analyze activation scale progression across model depth.

        Calculates layer-to-layer ratios and deviations from the model median.
        """
        if not self.step_stats:
            return {"layers": {}, "amplification_events": [], "attenuation_events": []}

        target_step = step if step is not None else max(self.step_stats.keys())
        layer_stats = self.step_stats.get(target_step, {})
        if not layer_stats:
            return {"layers": {}, "amplification_events": [], "attenuation_events": []}

        ordered_layers = [name for name in self.layer_order if name in layer_stats]
        stds = [layer_stats[name]["std"] for name in ordered_layers if not math.isnan(layer_stats[name]["std"])]
        median_std = float(torch.tensor(stds).median().item()) if stds else 1.0
        if median_std <= 0:
            median_std = 1.0

        propagation: Dict[str, Dict[str, Any]] = {}
        amplifications = []
        attenuations = []

        prev_std: Optional[float] = None
        for name in ordered_layers:
            s = layer_stats[name]
            cur_std = s["std"]
            cur_rms = s["rms"]

            ratio_to_prev = (cur_std / prev_std) if (prev_std is not None and prev_std > 1e-8) else 1.0
            ratio_to_median = cur_std / median_std if median_std > 0 else 1.0

            prop_entry = {
                "std": cur_std,
                "rms": cur_rms,
                "ratio_to_prev": ratio_to_prev,
                "ratio_to_median": ratio_to_median,
                "zero_fraction": s["zero_fraction"],
                "nan_fraction": s["nan_fraction"],
                "inf_fraction": s["inf_fraction"],
                "is_finite": s["is_finite"],
            }
            propagation[name] = prop_entry

            # Detect relative anomalies (cautious thresholds)
            if ratio_to_prev > 5.0 or ratio_to_median > 10.0:
                amplifications.append({
                    "module": name,
                    "ratio_to_prev": ratio_to_prev,
                    "ratio_to_median": ratio_to_median,
                    "std": cur_std,
                })
            elif (ratio_to_prev < 0.1 and prev_std is not None and prev_std > 1e-4) or ratio_to_median < 0.05:
                attenuations.append({
                    "module": name,
                    "ratio_to_prev": ratio_to_prev,
                    "ratio_to_median": ratio_to_median,
                    "std": cur_std,
                })

            prev_std = cur_std

        return {
            "median_std": median_std,
            "layers": propagation,
            "amplification_events": amplifications,
            "attenuation_events": attenuations,
        }

    def __enter__(self) -> ActivationMonitor:
        self.attach()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.detach()
