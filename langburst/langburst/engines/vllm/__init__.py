from __future__ import annotations

"""Optional vLLM engine package."""

from .bridge import (
    DEFAULT_MTP_KV_CACHE_MEMORY_BYTES,
    LOWBIT_MAMBA_MIN_BATCHED_TOKENS,
    LOWBIT_MTP_MIN_BATCHED_TOKENS,
    VLLMConversationStore,
    VLLMBridgeConfig,
    build_vllm_bridge_config,
    resolve_lowbit_enable_mtp,
    resolve_lowbit_max_num_batched_tokens,
    resolve_mtp_kv_cache_memory_bytes,
    vllm_engine_extra_kwargs,
)
from .provider import VLLMBackend, VLLMProvider

__all__ = [
    "DEFAULT_MTP_KV_CACHE_MEMORY_BYTES",
    "LOWBIT_MAMBA_MIN_BATCHED_TOKENS",
    "LOWBIT_MTP_MIN_BATCHED_TOKENS",
    "VLLMBackend",
    "VLLMBridgeConfig",
    "VLLMConversationStore",
    "VLLMProvider",
    "build_vllm_bridge_config",
    "resolve_lowbit_enable_mtp",
    "resolve_lowbit_max_num_batched_tokens",
    "resolve_mtp_kv_cache_memory_bytes",
    "vllm_engine_extra_kwargs",
]
