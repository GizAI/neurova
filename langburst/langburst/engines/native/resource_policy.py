from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping

from ...core.defaults import (
    DEFAULT_KV_BLOCK_SIZE,
    DEFAULT_KV_BLOCKS,
    DEFAULT_MAX_ACTIVE_REQUESTS,
    DEFAULT_MAX_BATCHED_TOKENS,
    DEFAULT_MAX_GENERATION_TOKENS,
    DEFAULT_MAX_PROMPT_TOKENS,
    DEFAULT_MAX_STATE_POOL_SIZE,
    DEFAULT_PREFILL_CHUNK_SIZE,
    DEFAULT_RESERVE_FREE_VRAM_MIB,
    DEFAULT_SERVING_RECENT_WINDOW,
)
from ...tuning import marlin_direct_max_batch

DEFAULT_MAX_QUEUED_REQUESTS = 8
DEFAULT_RUNTIME_OVERHEAD_MIB = 384
DEFAULT_MAX_PREFILL_ROWS_PER_BATCH = 0
DEFAULT_BATCH_WAIT_MS = 8.0


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return int(default)
    return int(raw)


def _env_optional_float(env: Mapping[str, str], name: str, default: float | None) -> float | None:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    text = raw.strip().lower()
    if text in {"0", "none", "off", "false"}:
        return None
    return float(raw)


def _env_optional_int(env: Mapping[str, str], name: str, default: int | None) -> int | None:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    text = raw.strip().lower()
    if text in {"0", "none", "off", "false"}:
        return None
    return int(raw)


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return bool(default)
    text = raw.strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of: 1/0, true/false, on/off")


def _env_int_tuple(env: Mapping[str, str], name: str) -> tuple[int, ...]:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return ()
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if any(value < 1 for value in values):
        raise ValueError(f"{name} values must be >= 1")
    return values


def _env_context_window(env: Mapping[str, str]) -> int:
    return _env_int(
        env,
        "LANGBURST_CONTEXT_WINDOW",
        _env_int(env, "LANGBURST_RECENT_WINDOW", DEFAULT_SERVING_RECENT_WINDOW),
    )


def _env_kv_block_size(env: Mapping[str, str]) -> int:
    return _env_int(env, "LANGBURST_KV_BLOCK_SIZE", DEFAULT_KV_BLOCK_SIZE)


def _env_max_active_requests(env: Mapping[str, str]) -> int:
    return _env_int(env, "LANGBURST_MAX_ACTIVE_REQUESTS", DEFAULT_MAX_ACTIVE_REQUESTS)


def _env_kv_blocks(env: Mapping[str, str]) -> int:
    explicit = env.get("LANGBURST_KV_BLOCKS")
    if explicit is not None and explicit.strip() != "":
        return int(explicit)
    block_size = _env_kv_block_size(env)
    context_tiers = _env_int_tuple(env, "LANGBURST_CONTEXT_TIERS")
    tier_slots = _env_int_tuple(env, "LANGBURST_CONTEXT_TIER_SLOTS")
    if context_tiers:
        if not tier_slots:
            tier_slots = tuple(1 for _ in context_tiers)
        if len(context_tiers) != len(tier_slots):
            raise ValueError("LANGBURST_CONTEXT_TIERS and LANGBURST_CONTEXT_TIER_SLOTS must have the same length")
        return max(1, sum(((tier + block_size - 1) // block_size) * slots for tier, slots in zip(context_tiers, tier_slots)))
    context_window = _env_context_window(env)
    max_active = _env_max_active_requests(env)
    return max(1, ((context_window + block_size - 1) // block_size) * max_active)


@dataclass(frozen=True)
class EngineResourcePolicy:
    """Single source of truth for native serving resource limits.

    Deployment scripts, CLI defaults, admission, state arenas, paged KV, and
    health reporting should consume this resolved policy instead of repeating
    environment-variable parsing or hard-coded VRAM assumptions.
    """

    max_loaded_models: int = 1
    max_active_requests: int = DEFAULT_MAX_ACTIVE_REQUESTS
    max_queued_requests: int = 0
    admission_timeout_s: float | None = None
    reserve_free_vram_mib: int = DEFAULT_RESERVE_FREE_VRAM_MIB
    max_state_pool_size: int = DEFAULT_MAX_STATE_POOL_SIZE
    max_prompt_tokens: int | None = DEFAULT_MAX_PROMPT_TOKENS
    max_generation_tokens: int | None = DEFAULT_MAX_GENERATION_TOKENS
    max_num_batched_tokens: int = DEFAULT_MAX_BATCHED_TOKENS
    prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE
    max_prefill_rows_per_batch: int = DEFAULT_MAX_PREFILL_ROWS_PER_BATCH
    decode_prefill_interleave_steps: int = 16
    exclusive_prefill_tokens: int | None = None
    batch_wait_ms: float = DEFAULT_BATCH_WAIT_MS
    kv_block_size: int = DEFAULT_KV_BLOCK_SIZE
    kv_blocks: int = DEFAULT_KV_BLOCKS
    runtime_overhead_mib: int = DEFAULT_RUNTIME_OVERHEAD_MIB
    context_tiers: tuple[int, ...] = ()
    context_tier_slots: tuple[int, ...] = ()
    allow_context_overflow: bool = True

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "EngineResourcePolicy":
        source = os.environ if env is None else env
        context_tiers = _env_int_tuple(source, "LANGBURST_CONTEXT_TIERS")
        context_tier_slots = _env_int_tuple(source, "LANGBURST_CONTEXT_TIER_SLOTS")
        exclusive_prefill_tokens = _env_optional_int(source, "LANGBURST_EXCLUSIVE_PREFILL_TOKENS", None)
        if "LANGBURST_EXCLUSIVE_PREFILL_TOKENS" not in source and len(context_tiers) > 1:
            exclusive_prefill_tokens = sorted(context_tiers)[0] + 1
        return cls(
            max_loaded_models=_env_int(source, "LANGBURST_MAX_LOADED_MODELS", 1),
            max_active_requests=_env_max_active_requests(source),
            max_queued_requests=_env_int(source, "LANGBURST_MAX_QUEUED_REQUESTS", DEFAULT_MAX_QUEUED_REQUESTS),
            admission_timeout_s=_env_optional_float(source, "LANGBURST_ADMISSION_TIMEOUT_S", None),
            reserve_free_vram_mib=_env_int(source, "LANGBURST_RESERVE_FREE_VRAM_MIB", DEFAULT_RESERVE_FREE_VRAM_MIB),
            max_state_pool_size=_env_int(source, "LANGBURST_MAX_STATE_POOL_SIZE", _env_max_active_requests(source)),
            max_prompt_tokens=_env_int(source, "LANGBURST_MAX_PROMPT_TOKENS", _env_context_window(source)),
            max_generation_tokens=_env_int(source, "LANGBURST_MAX_GENERATION_TOKENS", DEFAULT_MAX_GENERATION_TOKENS),
            max_num_batched_tokens=_env_int(source, "LANGBURST_MAX_NUM_BATCHED_TOKENS", DEFAULT_MAX_BATCHED_TOKENS),
            prefill_chunk_size=_env_int(source, "LANGBURST_PREFILL_CHUNK_SIZE", DEFAULT_PREFILL_CHUNK_SIZE),
            max_prefill_rows_per_batch=_env_int(
                source,
                "LANGBURST_MAX_PREFILL_ROWS_PER_BATCH",
                _env_max_active_requests(source),
            ),
            decode_prefill_interleave_steps=_env_int(source, "LANGBURST_DECODE_PREFILL_INTERLEAVE_STEPS", 16),
            exclusive_prefill_tokens=exclusive_prefill_tokens,
            batch_wait_ms=float(source.get("LANGBURST_BATCH_WAIT_MS", DEFAULT_BATCH_WAIT_MS)),
            kv_block_size=_env_kv_block_size(source),
            kv_blocks=_env_kv_blocks(source),
            runtime_overhead_mib=_env_int(source, "LANGBURST_RUNTIME_OVERHEAD_MIB", DEFAULT_RUNTIME_OVERHEAD_MIB),
            context_tiers=context_tiers,
            context_tier_slots=context_tier_slots,
            allow_context_overflow=_env_bool(source, "LANGBURST_ALLOW_CONTEXT_OVERFLOW", True),
        )

    def __post_init__(self) -> None:
        if self.max_loaded_models < 1:
            raise ValueError("max_loaded_models must be >= 1")
        if self.max_active_requests < 1:
            raise ValueError("max_active_requests must be >= 1")
        if self.max_queued_requests < 0:
            raise ValueError("max_queued_requests must be >= 0")
        if self.admission_timeout_s is not None and self.admission_timeout_s < 0:
            raise ValueError("admission_timeout_s must be >= 0")
        if self.reserve_free_vram_mib < 0:
            raise ValueError("reserve_free_vram_mib must be >= 0")
        if self.max_state_pool_size < 0:
            raise ValueError("max_state_pool_size must be >= 0")
        if self.max_prompt_tokens is not None and self.max_prompt_tokens < 1:
            raise ValueError("max_prompt_tokens must be >= 1")
        if self.max_generation_tokens is not None and self.max_generation_tokens < 1:
            raise ValueError("max_generation_tokens must be >= 1")
        if self.max_num_batched_tokens < 1:
            raise ValueError("max_num_batched_tokens must be >= 1")
        if self.prefill_chunk_size < 1:
            raise ValueError("prefill_chunk_size must be >= 1")
        if self.prefill_chunk_size > marlin_direct_max_batch():
            object.__setattr__(self, "prefill_chunk_size", marlin_direct_max_batch())
        if self.max_prefill_rows_per_batch < 0:
            raise ValueError("max_prefill_rows_per_batch must be >= 0")
        if self.decode_prefill_interleave_steps < 1:
            raise ValueError("decode_prefill_interleave_steps must be >= 1")
        if self.exclusive_prefill_tokens is not None and self.exclusive_prefill_tokens < 1:
            raise ValueError("exclusive_prefill_tokens must be >= 1")
        if self.batch_wait_ms < 0:
            raise ValueError("batch_wait_ms must be >= 0")
        if self.kv_block_size < 1:
            raise ValueError("kv_block_size must be >= 1")
        if self.kv_blocks < 1:
            raise ValueError("kv_blocks must be >= 1")
        if self.runtime_overhead_mib < 0:
            raise ValueError("runtime_overhead_mib must be >= 0")
        if bool(self.context_tiers) != bool(self.context_tier_slots):
            raise ValueError("context_tiers and context_tier_slots must be set together")
        if len(self.context_tiers) != len(self.context_tier_slots):
            raise ValueError("context_tiers and context_tier_slots must have the same length")
        if any(tier < 1 for tier in self.context_tiers):
            raise ValueError("context_tiers values must be >= 1")
        if any(slots < 1 for slots in self.context_tier_slots):
            raise ValueError("context_tier_slots values must be >= 1")
        if self.context_tiers:
            pairs = tuple(sorted(zip(self.context_tiers, self.context_tier_slots), key=lambda pair: pair[0]))
            object.__setattr__(self, "context_tiers", tuple(tier for tier, _slots in pairs))
            object.__setattr__(self, "context_tier_slots", tuple(slots for _tier, slots in pairs))
            object.__setattr__(self, "max_active_requests", sum(self.context_tier_slots))
            if self.max_state_pool_size < self.max_active_requests:
                object.__setattr__(self, "max_state_pool_size", self.max_active_requests)
            if self.max_prompt_tokens is None or self.max_prompt_tokens < self.context_tiers[-1]:
                object.__setattr__(self, "max_prompt_tokens", self.context_tiers[-1])

    def summary(self) -> dict[str, object]:
        return {
            "max_loaded_models": self.max_loaded_models,
            "max_active_requests": self.max_active_requests,
            "max_queued_requests": self.max_queued_requests,
            "admission_timeout_s": self.admission_timeout_s,
            "reserve_free_vram_mib": self.reserve_free_vram_mib,
            "max_state_pool_size": self.max_state_pool_size,
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_generation_tokens": self.max_generation_tokens,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "prefill_chunk_size": self.prefill_chunk_size,
            "max_prefill_rows_per_batch": self.max_prefill_rows_per_batch,
            "decode_prefill_interleave_steps": self.decode_prefill_interleave_steps,
            "exclusive_prefill_tokens": self.exclusive_prefill_tokens,
            "batch_wait_ms": self.batch_wait_ms,
            "kv_block_size": self.kv_block_size,
            "kv_blocks": self.kv_blocks,
            "runtime_overhead_mib": self.runtime_overhead_mib,
            "context_tiers": self.context_tiers,
            "context_tier_slots": self.context_tier_slots,
            "allow_context_overflow": self.allow_context_overflow,
        }
