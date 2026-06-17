from __future__ import annotations

from langburst.engines.native.block_table import KVBlockTable
from langburst.engines.native.kv_policy import KVCachePolicy


def test_kv_cache_policy_caps_free_block_reserve_for_small_tables():
    table = KVBlockTable(num_blocks=8, block_size=2)
    policy = KVCachePolicy.from_env(
        enabled=True,
        block_table=table,
        env={
            "LANGBURST_PREFIX_CACHE_MIN_FREE_BLOCKS": "512",
            "LANGBURST_PREFIX_CACHE_MIN_FREE_MIB": "0",
        },
    )

    assert policy.min_free_blocks == 1
    assert policy.min_prefix_tokens == 2


def test_kv_cache_policy_rejects_prefix_when_blocks_are_reserved():
    table = KVBlockTable(num_blocks=16, block_size=2)
    table.ensure_tokens("active", 28)
    policy = KVCachePolicy.from_env(
        enabled=True,
        block_table=table,
        env={
            "LANGBURST_PREFIX_CACHE_MIN_FREE_BLOCKS": "512",
            "LANGBURST_PREFIX_CACHE_MIN_FREE_MIB": "0",
        },
    )

    decision = policy.admit_prefix_store(prefix_len=2, block_table=table, cuda_available=False)

    assert not decision.allowed
    assert decision.reason == "kv_block_pressure"
    assert decision.needed_blocks == 1
    assert decision.free_blocks == 2


def test_kv_cache_policy_rejects_prefix_under_gpu_memory_pressure():
    table = KVBlockTable(num_blocks=16, block_size=4)
    policy = KVCachePolicy.from_env(
        enabled=True,
        block_table=table,
        env={
            "LANGBURST_PREFIX_CACHE_MIN_FREE_BLOCKS": "0",
            "LANGBURST_PREFIX_CACHE_MIN_FREE_MIB": "384",
        },
    )

    decision = policy.admit_prefix_store(
        prefix_len=4,
        block_table=table,
        cuda_available=True,
        cuda_free_mib=128,
    )

    assert not decision.allowed
    assert decision.reason == "gpu_memory_pressure"
    assert decision.free_mib == 128


def test_kv_cache_policy_exposes_prefix_cache_kwargs():
    table = KVBlockTable(num_blocks=16, block_size=8)
    policy = KVCachePolicy.from_env(
        enabled=True,
        block_table=table,
        env={
            "LANGBURST_PREFIX_CACHE_MAX_ENTRIES": "3",
            "LANGBURST_PREFIX_CACHE_MAX_TOKENS": "1024",
            "LANGBURST_PREFIX_CACHE_MIN_FREE_BLOCKS": "0",
            "LANGBURST_PREFIX_CACHE_MIN_FREE_MIB": "0",
        },
    )

    kwargs = policy.prefix_cache_kwargs(block_table=table)

    assert kwargs["enabled"] is True
    assert kwargs["min_prefix_tokens"] == 8
    assert kwargs["max_entries"] == 3
    assert kwargs["max_cached_tokens"] == 1024
    assert callable(kwargs["release_blocks"])
