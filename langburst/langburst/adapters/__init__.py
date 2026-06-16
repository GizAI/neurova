from __future__ import annotations

from ..core.adapter import adapter_registry
from .hf_causal import Gemma4Adapter, HFAutoCausalAdapter
from .qwen36 import Qwen36A3BAdapter, Qwen36Adapter, choose_qwen_weight_device

_ENTRY_POINTS_LOADED = False


def ensure_adapters_loaded() -> None:
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    adapter_registry.load_entry_points()
    _ENTRY_POINTS_LOADED = True


adapter_registry.register(HFAutoCausalAdapter())
adapter_registry.register(Gemma4Adapter())


__all__ = [
    "Gemma4Adapter",
    "HFAutoCausalAdapter",
    "Qwen36A3BAdapter",
    "Qwen36Adapter",
    "choose_qwen_weight_device",
    "ensure_adapters_loaded",
]
