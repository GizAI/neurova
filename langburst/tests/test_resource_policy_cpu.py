from __future__ import annotations

from langburst.engines.native.resource_policy import EngineResourcePolicy
from langburst.tuning import marlin_direct_max_batch


def test_resource_policy_from_env_derives_kv_blocks_from_context_and_concurrency():
    policy = EngineResourcePolicy.from_env(
        {
            "LANGBURST_CONTEXT_WINDOW": "32768",
            "LANGBURST_KV_BLOCK_SIZE": "16",
            "LANGBURST_MAX_ACTIVE_REQUESTS": "2",
            "LANGBURST_MAX_QUEUED_REQUESTS": "8",
            "LANGBURST_PREFILL_CHUNK_SIZE": "64",
            "LANGBURST_DECODE_PREFILL_INTERLEAVE_STEPS": "3",
        }
    )

    assert policy.max_prompt_tokens == 32768
    assert policy.kv_block_size == 16
    assert policy.kv_blocks == 4096
    assert policy.max_active_requests == 2
    assert policy.max_queued_requests == 8
    assert policy.max_state_pool_size == 2
    assert policy.prefill_chunk_size == 64
    assert policy.decode_prefill_interleave_steps == 3


def test_resource_policy_from_env_accepts_explicit_deployment_budget():
    policy = EngineResourcePolicy.from_env(
        {
            "LANGBURST_CONTEXT_WINDOW": "8192",
            "LANGBURST_MAX_PROMPT_TOKENS": "4096",
            "LANGBURST_KV_BLOCK_SIZE": "32",
            "LANGBURST_KV_BLOCKS": "777",
            "LANGBURST_RESERVE_FREE_VRAM_MIB": "256",
            "LANGBURST_RUNTIME_OVERHEAD_MIB": "128",
            "LANGBURST_MAX_STATE_POOL_SIZE": "1",
        }
    )

    assert policy.max_prompt_tokens == 4096
    assert policy.kv_block_size == 32
    assert policy.kv_blocks == 777
    assert policy.reserve_free_vram_mib == 256
    assert policy.runtime_overhead_mib == 128
    assert policy.max_state_pool_size == 1


def test_resource_policy_caps_prefill_chunk_to_marlin_direct_batch(monkeypatch):
    monkeypatch.setenv("LANGBURST_MARLIN_DIRECT_MAX_BATCH", "32")

    policy = EngineResourcePolicy(prefill_chunk_size=256)

    assert policy.prefill_chunk_size == marlin_direct_max_batch()
