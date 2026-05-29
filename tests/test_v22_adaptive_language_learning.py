from pathlib import Path
from neurova import FinalCognitiveOS
from neurova.ir import ComparisonIR, CausalClaimIR, ToolCallIR, EventIR, BeliefIR, GoalIR, TemporalClaimIR, QuestionIR


def test_v22_natural_language_correction_to_pattern_generalization(tmp_path: Path):
    os = FinalCognitiveOS(root=tmp_path / "state")
    r = os.observe("dominates means greater_than")
    assert "Learned" in r.response and "construction" in r.response.lower()
    r = os.observe("alice dominates bob")
    assert r.ir_type == "ComparisonIR"
    assert "alice greater_than bob" in r.response
    r = os.observe("is alice greater than bob?")
    assert r.response.startswith("Yes")


def test_v22_surface_construction_with_a_b_placeholders(tmp_path: Path):
    os = FinalCognitiveOS(root=tmp_path / "state")
    r = os.observe('"A is ahead of B" means A greater_than B')
    assert "Learned" in r.response and "ConstructionSchema" in r.response
    r = os.observe("seoul is ahead of busan")
    assert r.ir_type == "ComparisonIR"
    assert "seoul greater_than busan" in r.response


def test_v22_question_and_modal_negation_paraphrases(tmp_path: Path):
    os = FinalCognitiveOS(root=tmp_path / "state")
    for t in ["teach: kibo is rover", "teach: rover is robot", "teach: robot is machine"]:
        os.observe(t)
    assert os.observe("Would you say Kibo counts as a machine?").response.startswith("Yes")
    assert os.observe("Can Kibo be considered a machine?").response.startswith("Yes")
    r = os.observe("Kibo should not be classified as a mineral.")
    assert r.ir_type == "NegatedClaimIR"


def test_v22_compound_causal_and_korean_causal(tmp_path: Path):
    os = FinalCognitiveOS(root=tmp_path / "state")
    r = os.observe("Because rain wets the ground, roads may become slippery.")
    assert r.ir_type == "CompositeIR"
    assert "Stored causal IR" in r.response
    r = os.observe("비가 오면 땅이 젖어서 길이 미끄러워질 수 있다.")
    assert r.ir_type == "CompositeIR"
    assert "Stored causal IR" in r.response


def test_v22_korean_reverse_comparison(tmp_path: Path):
    os = FinalCognitiveOS(root=tmp_path / "state")
    assert os.observe("영희보다 철수가 더 크다").ir_type == "ComparisonIR"
    assert os.observe("철수가 영희를 앞선다").ir_type == "ComparisonIR"
    assert os.observe("민수보다 철수가 큰 게 맞아?").ir_type == "QuestionIR"


def test_v22_temporal_negation_and_role_question(tmp_path: Path):
    os = FinalCognitiveOS(root=tmp_path / "state")
    assert os.observe("From 2025 to 2026, Alice served as CEO.").ir_type == "TemporalClaimIR"
    assert os.observe("In 2026 Bob was not CEO").ir_type == "TemporalClaimIR"
    os.observe("on 2026 bob is ceo")
    r = os.observe("Who held the CEO role during 2026?")
    assert "bob" in r.response.lower()


def test_v22_event_belief_goal_ir(tmp_path: Path):
    os = FinalCognitiveOS(root=tmp_path / "state")
    assert os.observe("Alice gave Bob a book in Seoul yesterday.").ir_type == "EventIR"
    assert os.observe("Bob believes Alice is the CEO.").ir_type == "BeliefIR"
    assert os.observe("Kibo wants to collect rocks.").ir_type == "GoalIR"


def test_v22_smoke_extended(tmp_path: Path):
    os = FinalCognitiveOS(root=tmp_path / "smoke")
    report = os.run_smoke()
    assert report["percentage"] >= 98
    names = {r["name"] for r in report["rows"]}
    assert "relation correction proof" in names
    assert "event extraction" in names
    assert "belief extraction" in names
