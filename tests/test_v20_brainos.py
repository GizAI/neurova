from pathlib import Path

from neurova import (
    FinalCognitiveOS,
    DomainShardRouter,
    SemanticTestTimeAdapter,
    EpistemicImmuneSystem,
    SleepReplayConsolidator,
    MeaningAtomCalculus,
    TinySemanticEncoder,
    generate_nl_ir_examples,
)
from neurova.ir import ClaimIR, IRCandidate


def test_v20_tiny_encoder_is_structured_not_lm():
    rows = generate_nl_ir_examples(300, seed=3)
    enc = TinySemanticEncoder().fit(rows)
    assert not hasattr(enc, "generate")
    assert "autoregressive" in enc.objective
    preds = enc.predict("market shock leads to volatility", top_k=3)
    assert preds and preds[0].ir_type in {"CausalClaimIR", "ClaimIR", "ComparisonIR", "TemporalClaimIR", "NegatedClaimIR", "ExceptionIR"}


def test_v20_meaning_calculus_has_required_ops():
    calc = MeaningAtomCalculus()
    for op in ["compose", "negate", "time_scope", "cause_chain", "compare_chain", "exception_block", "derive"]:
        assert calc.has(op)
    assert calc.apply("cause_chain", {"a": "rain", "b": "wet ground"})["ok"]


def test_v20_domain_adapter_immune_sleep_grounding(tmp_path: Path):
    os = FinalCognitiveOS(root=tmp_path / "v20")
    assert os.domain_router.route("implement python function with tests").name == "code"
    assert os.domain_router.route("portfolio Sharpe CAGR drawdown").name == "quant"
    os.adapter.observe_alias("zephyr", "kibo")
    os.observe("kibo is robot")
    os.observe("robot is machine")
    assert "Yes" in os.observe("is zephyr machine?").response
    # create at least one failure trajectory for sleep replay
    os.observe("is unknown_entity machine?")
    report = os.sleep.run()
    assert report.trajectories >= 1
    assert os.memory.stats()["sleep_reports"] >= 1
    assert os.memory.stats()["learned_strategies"] >= 1
    assert os.memory.stats()["memory_actions"] > 0


def test_v20_epistemic_quarantine_api(tmp_path: Path):
    os = FinalCognitiveOS(root=tmp_path / "immune")
    os.observe("kibo is mineral")
    row = os.memory.find_claim("kibo", "is", "mineral", "positive")
    os.immune.quarantine_version(row["version_id"], "manual low-trust source test")
    counts = os.memory.claim_version_status_counts()
    assert counts.get("quarantined", 0) == 1
    assert os.memory.stats()["immune_events"] >= 1


def test_v20_smoke_is_extended(tmp_path: Path):
    os = FinalCognitiveOS(root=tmp_path / "smoke")
    report = os.run_smoke()
    assert report["percentage"] >= 95
    names = {r["name"]: r for r in report["rows"]}
    for key in [
        "meaning atom calculus",
        "semantic beam candidates",
        "tiny semantic encoder objective",
        "domain shard routing",
        "semantic test-time adapter",
        "sleep replay consolidation",
        "grounded verifier",
    ]:
        assert names[key]["passed"], names[key]
