"""
Safe forward and backward hook management for PyTorch models.
Ensures hooks are completely removed without memory leaks or graph retention.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Type
import torch
import torch.nn as nn


class HookManager:
    """Manages forward and backward hooks on PyTorch modules with automatic cleanup."""

    def __init__(
        self,
        model: nn.Module,
        include_module_types: Optional[Tuple[Type[nn.Module], ...]] = None,
        exclude_module_types: Optional[Tuple[Type[nn.Module], ...]] = None,
        leaf_modules_only: bool = True,
        name_regex: Optional[str] = None,
    ):
        self.model = model
        self.include_module_types = include_module_types
        self.exclude_module_types = exclude_module_types or (nn.Sequential, nn.ModuleList, nn.ModuleDict)
        self.leaf_modules_only = leaf_modules_only
        self.name_regex = re.compile(name_regex) if name_regex else None

        self._forward_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._backward_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._target_modules: Dict[str, nn.Module] = {}
        self._discover_modules()

    def _discover_modules(self) -> None:
        """Find modules matching selection criteria."""
        self._target_modules.clear()
        for name, module in self.model.named_modules():
            # Skip root module itself if checking submodules
            if name == "":
                continue

            # Leaf module check
            if self.leaf_modules_only and len(list(module.children())) > 0:
                continue

            # Exclude types
            if self.exclude_module_types and isinstance(module, self.exclude_module_types):
                continue

            # Include types
            if self.include_module_types and not isinstance(module, self.include_module_types):
                continue

            # Regex filter
            if self.name_regex and not self.name_regex.search(name):
                continue

            self._target_modules[name] = module

    @property
    def target_modules(self) -> Dict[str, nn.Module]:
        """Dictionary of registered module names to modules."""
        return self._target_modules

    def register_forward_hook(
        self,
        hook_fn: Callable[[str, nn.Module, Any, Any], None],
    ) -> None:
        """Register a forward hook callable(module_name, module, inputs, outputs)."""
        for name, module in self._target_modules.items():
            def _make_hook(m_name: str):
                def _forward_hook(m: nn.Module, inp: Any, out: Any):
                    hook_fn(m_name, m, inp, out)
                return _forward_hook

            handle = module.register_forward_hook(_make_hook(name))
            self._forward_handles.append(handle)

    def register_backward_hook(
        self,
        hook_fn: Callable[[str, nn.Module, Any, Any], None],
    ) -> None:
        """Register a full backward hook callable(module_name, module, grad_inputs, grad_outputs)."""
        for name, module in self._target_modules.items():
            def _make_hook(m_name: str):
                def _backward_hook(m: nn.Module, grad_in: Any, grad_out: Any):
                    hook_fn(m_name, m, grad_in, grad_out)
                return _backward_hook

            # Prefer register_full_backward_hook
            if hasattr(module, "register_full_backward_hook"):
                handle = module.register_full_backward_hook(_make_hook(name))
            else:
                handle = module.register_backward_hook(_make_hook(name))
            self._backward_handles.append(handle)

    def remove_hooks(self) -> None:
        """Remove all attached hooks."""
        for handle in self._forward_handles:
            handle.remove()
        for handle in self._backward_handles:
            handle.remove()
        self._forward_handles.clear()
        self._backward_handles.clear()

    def __enter__(self) -> HookManager:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove_hooks()
