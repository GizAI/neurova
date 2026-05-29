from pathlib import Path
from neurova import (
    FinalCognitiveOS,
    HybridSemanticCompiler,
    NegatedClaimIR,
    ComparisonIR,
    CausalClaimIR,
    CompositeIR,
    NeuralCognitiveCompiler,
    generate_nl_ir_examples,
)


def test_compiler_core_ir():
    c = HybridSemanticCompiler()
    assert isinstance(c.compile("teach: kibo is not mineral")[0].ir, NegatedClaimIR)
    assert isinstance(c.compile("alice is taller than bob")[0].ir, ComparisonIR)
    assert isinstance(c.compile("rain causes wet ground")[0].ir, CausalClaimIR)
    assert isinstance(c.compile("철수는 영희보다 크다")[0].ir, ComparisonIR)
    assert isinstance(c.compile("alice is taller than bob and bob is taller than charlie")[0].ir, CompositeIR)


def test_neural_cognitive_compiler_is_not_lm():
    m = NeuralCognitiveCompiler()
    assert not hasattr(m, "generate")
    assert "next-token" not in m.report_for(HybridSemanticCompiler(m).compile("is kibo machine?")[0]).objective


def test_claim_version_and_contradiction_order_independence(tmp_path: Path):
    a = FinalCognitiveOS(root=tmp_path / "a")
    a.observe("kibo is mineral")
    a.observe("kibo is not mineral")
    stats_a = a.memory.stats()
    b = FinalCognitiveOS(root=tmp_path / "b")
    b.observe("kibo is not mineral")
    b.observe("kibo is mineral")
    stats_b = b.memory.stats()
    assert stats_a["claim_versions"] == 4  # 2 seed + 2 observe
    assert stats_b["claim_versions"] == 4  # 2 seed + 2 observe
    assert stats_a["contradictions"] == 1
    assert stats_b["contradictions"] == 1
    assert "inconsistent" in a.observe("is kibo mineral?").response.lower()


def test_temporal_exception_comparison_causal_world_and_gate(tmp_path: Path):
    os = FinalCognitiveOS(root=tmp_path / "state")
    os.observe("on 2025 alice is ceo")
    os.observe("on 2026 bob is ceo")
    assert "bob" in os.observe("who is CEO in 2026?").response.lower()
    os.observe("all birds can fly")
    os.observe("penguin is bird")
    os.observe("penguin is exception to bird can fly")
    assert "blocked by exception" in os.observe("can penguin fly?").response.lower()
    os.observe("alice is taller than bob and bob is taller than charlie")
    assert "yes" in os.observe("is alice taller than charlie?").response.lower()
    os.observe("rain causes wet ground and wet ground causes slippery road")
    assert "yes" in os.observe("rain causes slippery road?").response.lower()
    os.observe('world: state={"light":"off"}; action=press_switch; next={"light":"on"}')
    assert os.world.predict("press_switch")
    cid = os.self_improver.propose_from_failure("x", "parser_error", "add a regression")
    assert os.self_improver.promote_if_passes(cid, [lambda: True]).promoted


def test_dataset_generator():
    rows = generate_nl_ir_examples(100, seed=1)
    assert len(rows) == 100
    assert {"text", "ir_type", "slots"}.issubset(rows[0])


def test_final_smoke(tmp_path: Path):
    os = FinalCognitiveOS(root=tmp_path / "state")
    report = os.run_smoke()
    assert report["percentage"] >= 95
    names = {r["name"]: r for r in report["rows"]}
    for key in [
        "temporal who",
        "comparison transitivity",
        "korean comparison transitivity",
        "causal chain",
        "exception blocks rule",
        "world transition memory",
        "self improvement gate",
    ]:
        assert names[key]["passed"], names[key]
    assert report["memory_stats"]["claim_versions"] > 0
    assert report["memory_stats"]["graph_nodes"] > 0
    assert report["memory_stats"]["graph_edges"] > 0
    assert report["memory_stats"]["memory_actions"] > 0
    assert report["memory_stats"]["trajectories"] > 0
    assert report["memory_stats"]["promotion_candidates"] > 0


def test_v17_learned_semantic_parser_generalizes_without_lm():
    from neurova import LearnedSemanticParser, MeaningAtomTable, evaluate_parser, generate_nl_ir_examples
    from neurova.ir import CausalClaimIR, ClaimIR, ComparisonIR
    rows = generate_nl_ir_examples(800, seed=23)
    parser = LearnedSemanticParser(rows)
    assert "next-token" not in parser.objective
    assert isinstance(parser.parse("supply shock leads to higher prices")[0].ir, CausalClaimIR)
    reverse = parser.parse("volatility happens because of market shock")[0].ir
    assert isinstance(reverse, CausalClaimIR)
    assert reverse.cause == "market shock" and reverse.effect == "volatility"
    assert isinstance(parser.parse("orion is classified as robot")[0].ir, ClaimIR)
    assert isinstance(parser.parse("junho is above alice")[0].ir, ComparisonIR)
    result = evaluate_parser(parser, rows[:200])
    assert result.type_accuracy >= 0.95
    assert result.slot_exact_accuracy >= 0.90
    atoms = MeaningAtomTable()
    assert atoms.validate_inventory()
    assert "Comparison" in atoms.atoms_for_ir("ComparisonIR")


def test_v17_compiler_uses_learned_parser_and_active_teacher(tmp_path: Path):
    from neurova import HybridSemanticCompiler
    from neurova.ir import CausalClaimIR, CompositeIR
    c = HybridSemanticCompiler()
    cand = c.compile("supply shock leads to higher prices")[0]
    assert isinstance(cand.ir, CausalClaimIR)
    assert "learned" in cand.parser
    combo = c.compile("supply shock leads to higher prices and higher prices causes volatility")[0]
    assert isinstance(combo.ir, CompositeIR)
    c.compile("@@@not parseable@@@")
    assert c.active_teacher.queue
