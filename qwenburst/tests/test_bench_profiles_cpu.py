from __future__ import annotations

from qwenburst.bench_profiles import ProfileBenchResult


def test_profile_bench_result_tok_s():
    result = ProfileBenchResult(profile="stateful", features={}, generated=128, elapsed_s=4.0)
    assert result.tok_s == 32.0
