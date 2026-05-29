from pathlib import Path
from neurova import FinalCognitiveOS
from neurova.datasets import generate_v25_multitask_corpus
from neurova.semantic.neural_perception import NeuralSemanticPerception


def make_os(tmp_path):
    return FinalCognitiveOS(root=tmp_path / "os")


def test_v25_neural_semantic_perception_no_lm():
    rows = generate_v25_multitask_corpus(600, seed=1)
    model = NeuralSemanticPerception().fit(rows, epochs=2)
    assert not hasattr(model, "generate")
    assert "next-token" not in model.objective
    pred = model.predict("Alice gave Bob a book yesterday", 3)
    assert pred
    assert any(p.label == "EventIR" for p in pred)


def test_v25_natural_correction_and_fcg_variants(tmp_path):
    os = make_os(tmp_path)
    assert "Learned" in os.observe('When I say "A dominates B", it means A is greater than B.').response and "Schema" in os.observe('When I say "A dominates B", it means A is greater than B.').response
    assert "Stored comparison" in os.observe("alice dominates bob").response
    assert "Yes" in os.observe("is alice greater than bob?").response
    assert "Learned" in os.observe('"A is slightly ahead of B" means A greater_than B.').response and "Schema" in os.observe('"A is slightly ahead of B" means A greater_than B.').response
    assert "Stored comparison" in os.observe("orion is ahead of kibo").response
    assert "Yes" in os.observe("is orion ahead of kibo?").response


def test_v25_less_than_inverse_and_question(tmp_path):
    os = make_os(tmp_path)
    assert "Learned" in os.observe('"A lags behind B" means A less_than B.').response and "Schema" in os.observe('"A lags behind B" means A less_than B.').response
    assert "Stored comparison" in os.observe("alice lags behind bob").response
    assert "Yes" in os.observe("is bob greater than alice?").response


def test_v25_event_world_state_grounding(tmp_path):
    os = make_os(tmp_path)
    assert "Stored event" in os.observe("Alice gave a book to Bob in Seoul yesterday.").response
    assert "Yes" in os.observe("Does Bob have the book?").response
    os2 = make_os(tmp_path / "r2")
    assert "Stored event" in os2.observe("Bob received a package from Alice yesterday.").response
    assert "Yes" in os2.observe("Does Bob have package?").response


def test_v25_temporal_contradiction_and_stop_event(tmp_path):
    os = make_os(tmp_path)
    os.observe("From 2025 to 2026, Alice served as CEO.")
    os.observe("In 2026, Alice was not the CEO.")
    assert "Inconsistent evidence" in os.observe("Who held the CEO role during 2026?").response
    os2 = make_os(tmp_path / "r2")
    os2.observe("Alice became CEO in 2025.")
    os2.observe("Alice stopped being CEO in 2027.")
    assert "Yes" in os2.observe("Who was CEO in 2026?").response
    assert "cannot prove" in os2.observe("Who was CEO in 2027?").response


def test_v25_exception_belief_goal_korean(tmp_path):
    os = make_os(tmp_path)
    assert "CompositeIR stored" in os.observe("Penguins are birds; however, they usually do not fly.").response
    assert "Blocked by exception" in os.observe("Can a penguin fly even though it is a bird?").response
    assert "Stored belief" in os.observe("Bob thinks Alice is not the CEO.").response
    assert "Yes" in os.observe("Does Bob believe Alice is not CEO?").response
    assert "Stored goal" in os.observe("Kibo plans on collecting rocks.").response
    assert "Stored comparison" in os.observe("철수는 영희에 비해 우위에 있다.").response
    assert "Stored comparison" in os.observe("철수는 영희보다 크지 않다.").response


def test_v25_large_corpus_and_smoke(tmp_path):
    rows = generate_v25_multitask_corpus(2500, seed=5)
    labels = {r["ir_type"] for r in rows}
    assert len(rows) == 2500
    assert {"ToolCallIR", "EventIR", "TemporalClaimIR", "BeliefIR", "QuestionIR"}.issubset(labels)
    report = make_os(tmp_path).run_smoke()
    assert report["percentage"] == 100.0
    assert report["total"] >= 72
