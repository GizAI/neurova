from __future__ import annotations

from langburst.tuning import PrefillAttentionPolicy


def test_extend_prefill_policy_uses_effective_recent_window():
    policy = PrefillAttentionPolicy(
        fresh_sdpa_tokens=2048,
        extend_sdpa_tokens=2048,
        min_free_mib=384,
        recent_tokens=32,
    )

    assert policy.allows_extend_sdpa(live_tokens=65536)


def test_extend_prefill_policy_keeps_full_context_limit_without_recent_window():
    policy = PrefillAttentionPolicy(
        fresh_sdpa_tokens=2048,
        extend_sdpa_tokens=2048,
        min_free_mib=384,
        recent_tokens=0,
    )

    assert policy.allows_extend_sdpa(live_tokens=2048)
    assert not policy.allows_extend_sdpa(live_tokens=2049)
