from pathlib import Path

from neurova import FinalCognitiveOS, ConstructionLearner
from neurova.ir import CausalClaimIR, ComparisonIR


def test_v21_one_shot_construction_learning(tmp_path: Path):
    os = FinalCognitiveOS(root=tmp_path / "v21")
    r = os.observe("learn construction: frost brings icy roads => causal(frost, icy roads)")
    assert "Learned" in r.response
    p = os.observe("heat brings expansion")
    assert p.ir_type == "CausalClaimIR"
    assert "Stored causal IR" in p.response
    q = os.observe("heat causes expansion?")
    assert "Yes" in q.response


def test_v21_construction_learner_is_general_not_generation():
    learner = ConstructionLearner()
    learner.learn("alpha outranks beta", ComparisonIR(left="alpha", comparator="greater_than", right="beta"))
    cands = learner.parse("seoul outranks busan")
    assert cands
    assert isinstance(cands[0].ir, ComparisonIR)
    assert cands[0].ir.left == "seoul"
    assert cands[0].ir.right == "busan"
    assert not hasattr(learner, "generate")
    assert "autoregressive" in learner.objective


def test_v21_smoke_includes_construction_learning(tmp_path: Path):
    os = FinalCognitiveOS(root=tmp_path / "smoke")
    report = os.run_smoke()
    assert report["percentage"] >= 95
    names = {r["name"]: r for r in report["rows"]}
    for key in ["one-shot construction parse", "one-shot construction proof", "construction learner objective"]:
        assert names[key]["passed"], names[key]
