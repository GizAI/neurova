from __future__ import annotations

from qwenburst.speculative import SpeculativeProbeResult


def test_speculative_probe_result_requires_high_acceptance():
    assert not SpeculativeProbeResult("native_mtp1", total=16, accepted=0, min_accept_rate=0.55).viable
    assert not SpeculativeProbeResult("native_mtp1", total=16, accepted=8, min_accept_rate=0.55).viable
    assert SpeculativeProbeResult("native_mtp1", total=16, accepted=9, min_accept_rate=0.55).viable


def test_speculative_benchmark_keep_requires_identity_and_speedup():
    from qwenburst.speculative import SpeculativeBenchmarkResult

    assert SpeculativeBenchmarkResult("native_mtp1", 100, 10.0, 9.6, 10, 12, True).keep
    assert not SpeculativeBenchmarkResult("native_mtp1", 100, 10.0, 9.8, 10, 12, True).keep
    assert not SpeculativeBenchmarkResult("native_mtp1", 100, 10.0, 8.0, 10, 12, False).keep
