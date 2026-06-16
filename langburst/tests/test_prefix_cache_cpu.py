from __future__ import annotations

from langburst.engines.native_impl.prefix_cache import RadixPrefixCache


def test_radix_prefix_cache_returns_longest_bounded_hit():
    cache = RadixPrefixCache(min_prefix_tokens=2)
    assert cache.insert([1, 2], {"state": "short"}, block_ids=(7,))
    assert cache.insert([1, 2, 3, 4], {"state": "long"}, block_ids=(8, 9))

    hit = cache.lookup([1, 2, 3, 4, 5], max_prefix_len=4)

    assert hit is not None
    assert hit.prefix_len == 4
    assert hit.state == {"state": "long"}
    assert hit.block_ids == (8, 9)
    assert cache.stats().hits == 1

    bounded = cache.lookup([1, 2, 3, 4, 5], max_prefix_len=2)
    assert bounded is not None
    assert bounded.prefix_len == 2
    assert bounded.state == {"state": "short"}


def test_radix_prefix_cache_eviction_releases_pinned_blocks():
    released: list[tuple[int, ...]] = []
    cache = RadixPrefixCache(
        min_prefix_tokens=1,
        max_entries=1,
        release_blocks=lambda blocks: released.append(tuple(int(b) for b in blocks)),
    )

    assert cache.insert([1], "a", block_ids=(1,))
    assert cache.insert([2], "b", block_ids=(2,))

    assert released == [(1,)]
    assert cache.stats().entries == 1
    assert cache.stats().evictions == 1
    cache.clear()
    assert released == [(1,), (2,)]


def test_radix_prefix_cache_duplicate_insert_releases_redundant_pin():
    released: list[tuple[int, ...]] = []
    cache = RadixPrefixCache(
        min_prefix_tokens=1,
        release_blocks=lambda blocks: released.append(tuple(int(b) for b in blocks)),
    )

    assert cache.insert([1], "a", block_ids=(3,))
    assert cache.insert([1], "b", block_ids=(3,))

    hit = cache.lookup([1], max_prefix_len=1)
    assert hit is not None
    assert hit.state == "b"
    assert hit.block_ids == (3,)
    assert released == [(3,)]
