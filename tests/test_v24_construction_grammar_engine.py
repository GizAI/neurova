from pathlib import Path

from neurova import FinalCognitiveOS
from neurova.ir import ComparisonIR, QuestionIR, TemporalClaimIR, EventIR, CompositeIR


def make_os(tmp_path):
    return FinalCognitiveOS(Path(tmp_path) / "state")


def test_natural_feedback_induces_feature_construction_and_questions(tmp_path):
    os = make_os(tmp_path)
    assert "Learned" and "Schema" in r.response in os.observe('When I say "A dominates B", it means A is greater than B.').response
    r = os.observe("alice dominates bob")
    assert r.ir_type == "ComparisonIR"
    assert "alice greater_than bob" in r.response
    q = os.observe("is alice greater than bob?")
    assert "Yes" in q.response
    assert len(os.compiler.construction_grammar) >= 1


def test_feature_construction_optional_modifier_and_question_variant(tmp_path):
    os = make_os(tmp_path)
    os.observe('"A is slightly ahead of B" means A greater_than B')
    assert os.observe("apollo is ahead of zephyr").ir_type == "ComparisonIR"
    q = os.observe("is apollo ahead of zephyr?")
    assert "Yes" in q.response


def test_question_paraphrases_not_static_regex_only(tmp_path):
    os = make_os(tmp_path)
    os.observe("teach: kibo is rover")
    os.observe("teach: rover is robot")
    os.observe("teach: robot is machine")
    for prompt in [
        "Is it fair to call Kibo a machine?",
        "Could Kibo be treated as a kind of machine?",
        "Would Kibo qualify as a machine?",
    ]:
        assert "Yes" in os.observe(prompt).response


def test_korean_particle_variants(tmp_path):
    os = make_os(tmp_path)
    assert os.observe("철수는 영희에 비해 우위에 있다").ir_type == "ComparisonIR"
    assert "Yes" in os.observe("철수가 영희보다 앞서 있니?").response
    assert os.observe("철수는 영희보다 크지 않다").ir_type == "ComparisonIR"


def test_temporal_interval_contradiction(tmp_path):
    os = make_os(tmp_path)
    assert os.observe("Alice was CEO from 2025 through 2026.").ir_type == "TemporalClaimIR"
    assert os.observe("In 2026, Alice was not the CEO.").ir_type == "TemporalClaimIR"
    r = os.observe("Who was CEO in 2026?")
    assert "Inconsistent evidence" in r.response


def test_exception_discourse_blocks_question(tmp_path):
    os = make_os(tmp_path)
    r = os.observe("Penguins are birds; however, they usually do not fly.")
    assert r.ir_type == "CompositeIR"
    q = os.observe("Can a penguin fly even though it is a bird?")
    assert "Blocked by exception" in q.response


def test_transfer_frame_and_derived_possession(tmp_path):
    os = make_os(tmp_path)
    assert os.observe("Alice gave a book to Bob in Seoul yesterday.").ir_type == "EventIR"
    assert "Yes" in os.observe("does bob have book?").response
    assert os.observe("Bob received a book from Alice yesterday.").ir_type == "EventIR"


def test_v24_smoke_extended(tmp_path):
    os = make_os(tmp_path)
    report = os.run_smoke()
    assert report["passed"] == report["total"]
    names = {r["name"] for r in report["rows"]}
    assert "v24 feature construction grammar" in names
    assert "v24 temporal inconsistent role" in names
