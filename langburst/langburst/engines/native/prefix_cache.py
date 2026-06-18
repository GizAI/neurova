from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Sequence


@dataclass
class PrefixCacheEntry:
    namespace: str
    tokens: tuple[int, ...]
    state: Any
    block_ids: tuple[int, ...] = ()
    created_monotonic: float = field(default_factory=time.monotonic)
    last_used_monotonic: float = field(default_factory=time.monotonic)
    hits: int = 0

    @property
    def prefix_len(self) -> int:
        return len(self.tokens)


@dataclass(frozen=True)
class PrefixCacheHit:
    prefix_len: int
    state: Any
    block_ids: tuple[int, ...]


@dataclass(frozen=True)
class PrefixCacheStats:
    entries: int
    cached_tokens: int
    hits: int
    misses: int
    evictions: int

    def summary(self) -> dict[str, int]:
        return {
            "entries": self.entries,
            "cached_tokens": self.cached_tokens,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }


class _TrieNode:
    __slots__ = ("children", "entry")

    def __init__(self) -> None:
        self.children: dict[int, _TrieNode] = {}
        self.entry: PrefixCacheEntry | None = None


class RadixPrefixCache:
    """Token-prefix cache with adapter state snapshots and optional KV blocks.

    This is LangBurst's model-agnostic Automatic Prefix Caching boundary. It
    caches only immutable prefix boundaries selected by the caller. Paged KV
    ownership is explicit: cached block IDs are pinned by the KV block table,
    then attached to new request rows by the runner.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        min_prefix_tokens: int = 16,
        max_entries: int = 128,
        max_cached_tokens: int = 131_072,
        release_blocks: Callable[[Sequence[int]], None] | None = None,
    ) -> None:
        if min_prefix_tokens < 1:
            raise ValueError("min_prefix_tokens must be >= 1")
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if max_cached_tokens < min_prefix_tokens:
            raise ValueError("max_cached_tokens must be >= min_prefix_tokens")
        self.enabled = bool(enabled)
        self.min_prefix_tokens = int(min_prefix_tokens)
        self.max_entries = int(max_entries)
        self.max_cached_tokens = int(max_cached_tokens)
        self._release_blocks = release_blocks
        self._roots: dict[str, _TrieNode] = {}
        self._entries: dict[tuple[str, tuple[int, ...]], PrefixCacheEntry] = {}
        self._cached_tokens = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def lookup(self, token_ids: Sequence[int], *, max_prefix_len: int | None = None, namespace: str | None = None) -> PrefixCacheHit | None:
        if not self.enabled:
            return None
        limit = len(token_ids) if max_prefix_len is None else min(len(token_ids), max(0, int(max_prefix_len)))
        node = self._roots.get(str(namespace or ""))
        if node is None:
            self._misses += 1
            return None
        best: PrefixCacheEntry | None = None
        for idx, token in enumerate(token_ids[:limit], start=1):
            child = node.children.get(int(token))
            if child is None:
                break
            node = child
            if node.entry is not None and idx >= self.min_prefix_tokens:
                best = node.entry
        if best is None:
            self._misses += 1
            return None
        best.hits += 1
        best.last_used_monotonic = time.monotonic()
        self._hits += 1
        return PrefixCacheHit(prefix_len=best.prefix_len, state=best.state, block_ids=best.block_ids)

    def insert(self, token_ids: Sequence[int], state: Any, *, block_ids: Sequence[int] = (), namespace: str | None = None) -> bool:
        if not self.enabled:
            return False
        tokens = tuple(int(t) for t in token_ids)
        if len(tokens) < self.min_prefix_tokens:
            return False
        if len(tokens) > self.max_cached_tokens:
            if self._release_blocks is not None:
                new_block_ids = tuple(int(b) for b in block_ids)
                if new_block_ids:
                    self._release_blocks(new_block_ids)
            return False
        ns = str(namespace or "")
        key = (ns, tokens)
        new_block_ids = tuple(int(b) for b in block_ids)
        if key in self._entries:
            entry = self._entries[key]
            old_block_ids = entry.block_ids
            entry.state = state
            if old_block_ids == new_block_ids:
                if self._release_blocks is not None and new_block_ids:
                    self._release_blocks(new_block_ids)
            else:
                if self._release_blocks is not None and old_block_ids:
                    self._release_blocks(old_block_ids)
                entry.block_ids = new_block_ids
            entry.last_used_monotonic = time.monotonic()
            return True
        node = self._roots.setdefault(ns, _TrieNode())
        for token in tokens:
            node = node.children.setdefault(token, _TrieNode())
        entry = PrefixCacheEntry(namespace=ns, tokens=tokens, state=state, block_ids=new_block_ids)
        node.entry = entry
        self._entries[key] = entry
        self._cached_tokens += len(tokens)
        self._evict_if_needed()
        return True

    def clear(self) -> None:
        if self._release_blocks is not None:
            for entry in self._entries.values():
                if entry.block_ids:
                    self._release_blocks(entry.block_ids)
        self._roots.clear()
        self._entries.clear()
        self._cached_tokens = 0

    def stats(self) -> PrefixCacheStats:
        return PrefixCacheStats(
            entries=len(self._entries),
            cached_tokens=self._cached_tokens,
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
        )

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self.max_entries or self._cached_tokens > self.max_cached_tokens:
            victim_key, victim = min(self._entries.items(), key=lambda item: (item[1].last_used_monotonic, item[1].created_monotonic))
            self._delete(victim_key, victim)

    def _delete(self, key: tuple[str, tuple[int, ...]], entry: PrefixCacheEntry) -> None:
        namespace, tokens = key
        node = self._roots.get(namespace)
        if node is None:
            self._entries.pop(key, None)
            return
        stack: list[tuple[_TrieNode, int]] = []
        for token in tokens:
            child = node.children.get(token)
            if child is None:
                break
            stack.append((node, token))
            node = child
        if node.entry is entry:
            node.entry = None
        removed = self._entries.pop(key, None)
        if removed is not None:
            self._cached_tokens = max(0, self._cached_tokens - len(tokens))
        if self._release_blocks is not None and entry.block_ids:
            self._release_blocks(entry.block_ids)
        for parent, token in reversed(stack):
            child = parent.children[token]
            if child.entry is None and not child.children:
                del parent.children[token]
            else:
                break
        if not self._roots[namespace].children and self._roots[namespace].entry is None:
            self._roots.pop(namespace, None)
        self._evictions += 1
