from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping

import torch

from .block_table import KVBlockTable


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return int(default)
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


@dataclass(frozen=True)
class KVCacheBudgetDecision:
    allowed: bool
    reason: str = "ok"
    needed_blocks: int = 0
    free_blocks: int | None = None
    min_free_blocks: int = 0
    free_mib: int | None = None
    min_free_mib: int = 0

    def summary(self) -> dict[str, int | str | bool | None]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "needed_blocks": self.needed_blocks,
            "free_blocks": self.free_blocks,
            "min_free_blocks": self.min_free_blocks,
            "free_mib": self.free_mib,
            "min_free_mib": self.min_free_mib,
        }


@dataclass(frozen=True)
class KVCachePolicy:
    """Single serving-time policy for prefix reuse and KV cache pressure.

    The block table owns physical KV pages, RadixPrefixCache owns reusable
    prefix entries, and this policy owns all budget decisions between them.
    """

    prefix_cache_enabled: bool
    min_prefix_tokens: int
    max_prefix_entries: int
    max_prefix_tokens: int
    min_free_mib: int
    min_free_blocks: int

    @classmethod
    def from_env(
        cls,
        *,
        enabled: bool,
        block_table: KVBlockTable | None,
        env: Mapping[str, str] | None = None,
    ) -> "KVCachePolicy":
        source = os.environ if env is None else env
        block_size = int(block_table.block_size) if block_table is not None else 16
        configured_min_free_blocks = _env_int(source, "LANGBURST_PREFIX_CACHE_MIN_FREE_BLOCKS", 512)
        if block_table is not None:
            configured_min_free_blocks = min(configured_min_free_blocks, max(0, block_table.num_blocks // 8))
        return cls(
            prefix_cache_enabled=bool(enabled),
            min_prefix_tokens=max(1, block_size),
            max_prefix_entries=max(1, _env_int(source, "LANGBURST_PREFIX_CACHE_MAX_ENTRIES", 2)),
            max_prefix_tokens=max(block_size, _env_int(source, "LANGBURST_PREFIX_CACHE_MAX_TOKENS", 16_384)),
            min_free_mib=_env_int(source, "LANGBURST_PREFIX_CACHE_MIN_FREE_MIB", 384),
            min_free_blocks=configured_min_free_blocks,
        )

    def prefix_cache_kwargs(self, *, block_table: KVBlockTable | None) -> dict[str, object]:
        return {
            "enabled": self.prefix_cache_enabled,
            "min_prefix_tokens": self.min_prefix_tokens,
            "max_entries": self.max_prefix_entries,
            "max_cached_tokens": self.max_prefix_tokens,
            "release_blocks": block_table.release_pinned_blocks if block_table is not None else None,
        }

    def admit_prefix_store(
        self,
        *,
        prefix_len: int,
        block_table: KVBlockTable | None,
        cuda_available: bool | None = None,
        cuda_free_mib: int | None = None,
    ) -> KVCacheBudgetDecision:
        prefix_len = int(prefix_len)
        if not self.prefix_cache_enabled:
            return KVCacheBudgetDecision(False, reason="disabled")
        if prefix_len < self.min_prefix_tokens:
            return KVCacheBudgetDecision(False, reason="prefix_too_short")

        needed_blocks = 0
        free_blocks: int | None = None
        if block_table is not None:
            needed_blocks = (prefix_len + block_table.block_size - 1) // block_table.block_size
            free_blocks = int(block_table.free_block_count)
            if self.min_free_blocks > 0 and free_blocks - needed_blocks < self.min_free_blocks:
                return KVCacheBudgetDecision(
                    False,
                    reason="kv_block_pressure",
                    needed_blocks=needed_blocks,
                    free_blocks=free_blocks,
                    min_free_blocks=self.min_free_blocks,
                    min_free_mib=self.min_free_mib,
                )

        if self.min_free_mib > 0:
            if cuda_available is None:
                cuda_available = torch.cuda.is_available()
            if cuda_available:
                if cuda_free_mib is None:
                    free_bytes, _total_bytes = torch.cuda.mem_get_info()
                    cuda_free_mib = int(free_bytes // (1024 * 1024))
                if cuda_free_mib < self.min_free_mib:
                    return KVCacheBudgetDecision(
                        False,
                        reason="gpu_memory_pressure",
                        needed_blocks=needed_blocks,
                        free_blocks=free_blocks,
                        min_free_blocks=self.min_free_blocks,
                        free_mib=int(cuda_free_mib),
                        min_free_mib=self.min_free_mib,
                    )

        return KVCacheBudgetDecision(
            True,
            needed_blocks=needed_blocks,
            free_blocks=free_blocks,
            min_free_blocks=self.min_free_blocks,
            free_mib=cuda_free_mib,
            min_free_mib=self.min_free_mib,
        )

    def summary(self) -> dict[str, int | bool]:
        return {
            "prefix_cache_enabled": self.prefix_cache_enabled,
            "min_prefix_tokens": self.min_prefix_tokens,
            "max_prefix_entries": self.max_prefix_entries,
            "max_prefix_tokens": self.max_prefix_tokens,
            "min_free_mib": self.min_free_mib,
            "min_free_blocks": self.min_free_blocks,
        }
