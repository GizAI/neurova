from __future__ import annotations

import math
import os

DEFAULT_SERVING_RECENT_WINDOW = 16384
DEFAULT_RESERVE_FREE_VRAM_MIB = 512
DEFAULT_MAX_STATE_POOL_SIZE = 1
DEFAULT_MAX_ACTIVE_REQUESTS = 1
DEFAULT_MAX_PROMPT_TOKENS = 16384
DEFAULT_MAX_GENERATION_TOKENS = 1024
DEFAULT_MAX_BATCHED_TOKENS = 256
DEFAULT_PREFILL_CHUNK_SIZE = 64
DEFAULT_KV_CACHE_DTYPE = "int4_bdr"
DEFAULT_KV_BLOCK_SIZE = 16
DEFAULT_KV_BLOCKS = 1024


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def serving_recent_window_default() -> int:
    return _env_int(
        "LANGBURST_CONTEXT_WINDOW",
        _env_int("LANGBURST_RECENT_WINDOW", DEFAULT_SERVING_RECENT_WINDOW),
    )


def max_prompt_tokens_default() -> int:
    return _env_int("LANGBURST_MAX_PROMPT_TOKENS", serving_recent_window_default())


def max_active_requests_default() -> int:
    return _env_int("LANGBURST_MAX_ACTIVE_REQUESTS", DEFAULT_MAX_ACTIVE_REQUESTS)


def max_state_pool_size_default() -> int:
    return _env_int("LANGBURST_MAX_STATE_POOL_SIZE", max_active_requests_default())


def kv_block_size_default() -> int:
    return _env_int("LANGBURST_KV_BLOCK_SIZE", DEFAULT_KV_BLOCK_SIZE)


def kv_cache_dtype_default() -> str:
    return os.environ.get("LANGBURST_KV_CACHE_DTYPE", DEFAULT_KV_CACHE_DTYPE)


def kv_blocks_default() -> int:
    explicit = os.environ.get("LANGBURST_KV_BLOCKS")
    if explicit is not None and explicit.strip() != "":
        return int(explicit)
    blocks_per_request = math.ceil(serving_recent_window_default() / kv_block_size_default())
    return max(1, blocks_per_request * max_active_requests_default())
