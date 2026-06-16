from __future__ import annotations

import json

import torch

from langburst.speculation import (
    DraftRequest,
    SpeculativeBenchmarkResult,
    SpeculativeDecodePolicy,
    SpeculativeProbeResult,
)
from langburst.speculative_verifier import NativeNextNVerifier, TargetVerification
from langburst.engines.native.policy import RuntimePolicyResolver
from langburst.tuning import verify_nextn_mode


class StaticProposer:
    method = "native_mtp1"

    def propose_tensors(self, request):
        return request.signals["first_token"].new_tensor([2, 3])


class Snapshot:
    def __init__(self, pos):
        self.pos = pos

    def restore_(self, state):
        state.pos = self.pos


class VerifierState:
    pos = 0

    def speculative_write_snapshot(self, _num_tokens):
        return Snapshot(self.pos)


def test_speculative_probe_result_requires_high_acceptance():
    assert not SpeculativeProbeResult("native_mtp1", total=16, accepted=0, min_accept_rate=0.55).viable
    assert not SpeculativeProbeResult("native_mtp1", total=16, accepted=8, min_accept_rate=0.55).viable
    assert SpeculativeProbeResult("native_mtp1", total=16, accepted=9, min_accept_rate=0.55).viable


def test_speculative_benchmark_keep_requires_identity_and_speedup():
    assert SpeculativeBenchmarkResult("native_mtp1", 100, 10.0, 9.6, 10, 12, True).keep
    assert not SpeculativeBenchmarkResult("native_mtp1", 100, 10.0, 9.0, 0, 12, True).keep
    assert not SpeculativeBenchmarkResult("native_mtp1", 100, 10.0, 9.8, 10, 12, True).keep
    assert not SpeculativeBenchmarkResult("native_mtp1", 100, 10.0, 8.0, 10, 12, False).keep
    assert not SpeculativeBenchmarkResult("native_mtp1", 100, 10.0, 9.0, 10, 12, True, fallback_reason="accept_rate").keep


def test_draft_request_accepts_optional_model_signals():
    request = DraftRequest(history=[1], max_draft=2, signals={"hidden": object()})
    assert request.max_draft == 2
    assert "hidden" in request.signals


def test_speculative_verifier_default_is_transaction_block(monkeypatch):
    monkeypatch.delenv("LANGBURST_SPECULATIVE_VERIFIER", raising=False)
    assert RuntimePolicyResolver().verifier_mode() == "transaction_block"


def test_speculative_verifier_env_override(monkeypatch):
    monkeypatch.setenv("LANGBURST_SPECULATIVE_VERIFIER", "sequential")
    assert RuntimePolicyResolver().verifier_mode() == "sequential"


def test_verify_nextn_mode_default_and_env(monkeypatch):
    monkeypatch.delenv("LANGBURST_VERIFY_NEXTN_MODE", raising=False)
    assert verify_nextn_mode() == "fused"
    monkeypatch.setenv("LANGBURST_VERIFY_NEXTN_MODE", "block")
    assert verify_nextn_mode() == "block"


def test_runtime_policy_resolver_reads_env(monkeypatch):
    monkeypatch.setenv("LANGBURST_MTP_MAX_DRAFT", "10")
    monkeypatch.setenv("LANGBURST_MTP_MIN_VERIFIED", "3")
    monkeypatch.setenv("LANGBURST_MTP_ACCEPT_THRESHOLD", "0.75")
    monkeypatch.setenv("LANGBURST_MTP_MAX_REJECTIONS", "2")
    policy = RuntimePolicyResolver().speculative_policy()

    assert policy == SpeculativeDecodePolicy(
        max_draft=10,
        verifier_mode="transaction_block",
        adaptive=True,
        min_verified=3,
        accept_threshold=0.75,
        max_rejections=2,
        min_speedup=1.03,
    )


def test_runtime_policy_resolver_reads_speed_positive_autotune_json(monkeypatch, tmp_path):
    path = tmp_path / "nextn_autotune.json"
    path.write_text(json.dumps({
        "keep": True,
        "policy": {
            "max_draft": 6,
            "verifier_mode": "transaction_block",
            "adaptive": True,
            "min_verified": 2,
            "accept_threshold": 0.8,
            "max_rejections": 1,
            "min_speedup": 1.05,
        },
    }))
    monkeypatch.setenv("LANGBURST_MTP_AUTOTUNE_JSON", str(path))
    for name in (
        "LANGBURST_MTP_MAX_DRAFT",
        "LANGBURST_MTP_MIN_VERIFIED",
        "LANGBURST_MTP_ACCEPT_THRESHOLD",
        "LANGBURST_MTP_MAX_REJECTIONS",
        "LANGBURST_MTP_MIN_SPEEDUP",
        "LANGBURST_SPECULATIVE_VERIFIER",
    ):
        monkeypatch.delenv(name, raising=False)

    policy = RuntimePolicyResolver().speculative_policy()

    assert policy == SpeculativeDecodePolicy(
        max_draft=6,
        verifier_mode="transaction_block",
        adaptive=True,
        min_verified=2,
        accept_threshold=0.8,
        max_rejections=1,
        min_speedup=1.05,
    )


def test_runtime_policy_resolver_ignores_non_kept_autotune_json(monkeypatch, tmp_path):
    path = tmp_path / "nextn_autotune.json"
    path.write_text(json.dumps({"keep": False, "policy": {"max_draft": 8}}))
    monkeypatch.setenv("LANGBURST_MTP_AUTOTUNE_JSON", str(path))
    monkeypatch.delenv("LANGBURST_MTP_MAX_DRAFT", raising=False)

    assert RuntimePolicyResolver().speculative_policy().max_draft == 1


def test_native_nextn_verifier_uses_batch_verify_callback():
    calls = []

    def verify_tokens(token_ids, state, num_candidates):
        calls.append((list(token_ids), num_candidates))
        state.pos += len(token_ids)
        logits = torch.full((8,), -1000.0)
        logits[4] = 1000.0
        return TargetVerification(
            target_ids=torch.tensor([2, 3], dtype=torch.long),
            logits=logits,
            raw_hidden=torch.ones((4,)),
        )

    logits = torch.full((8,), -1000.0)
    logits[1] = 1000.0
    verifier = NativeNextNVerifier(
        model=object(),
        proposer=StaticProposer(),
        sample_next=lambda row: torch.argmax(row).reshape(()).to(dtype=torch.long),
        max_draft=2,
        mode="transaction_block",
        verify_tokens=verify_tokens,
    )

    step = verifier.step(
        logits=logits,
        raw_hidden=torch.zeros((4,)),
        state=VerifierState(),
        remaining_tokens=4,
    )

    assert calls == [([1, 2, 3], 2)]
    assert [int(token.item()) for token in step.tokens] == [1, 2, 3, 4]
    assert step.accepted == 2
    assert step.verified == 2
    assert verifier.accept_rate == 1.0


def test_transaction_block_verifier_requires_runtime_verify_callback():
    try:
        NativeNextNVerifier(
            model=object(),
            proposer=StaticProposer(),
            sample_next=lambda row: torch.argmax(row).reshape(()).to(dtype=torch.long),
            max_draft=2,
            mode="transaction_block",
        )
    except ValueError as exc:
        assert "verify_tokens" in str(exc)
    else:
        raise AssertionError("transaction_block verifier accepted a missing runtime callback")
