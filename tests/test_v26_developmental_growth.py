from pathlib import Path
from neurova import FinalCognitiveOS, DevelopmentalGrowthLab, LLMSeedKnowledgeBank, IntrinsicMotivationEngine


def test_llm_seed_distillation_is_gated_and_not_autoregressive(tmp_path):
    os = FinalCognitiveOS(tmp_path / "seeded", auto_seed=False)
    report = LLMSeedKnowledgeBank().add_builtin_child_language_seed().install(os)
    assert report["construction"] >= 5
    assert report["event_frame"] >= 4
    assert "next-token" not in os.compiler.neural_perception.objective.lower()
    assert "autoregressive" in os.compiler.neural_perception.objective.lower()


def test_event_world_grounding_extended_frames(tmp_path):
    os = FinalCognitiveOS(tmp_path / "world", auto_seed=False)
    LLMSeedKnowledgeBank().add_builtin_child_language_seed().install(os)
    assert "Stored event IR" in os.observe("Sora bought a book from Dami yesterday.").response
    assert "Yes" in os.observe("Does Sora have book?").response
    assert "Stored event IR" in os.observe("Teacher moved the box from classroom to library.").response
    assert "library" in os.observe("Where is the box?").response.lower()


def test_developmental_correction_and_variant_learning(tmp_path):
    os = FinalCognitiveOS(tmp_path / "learn", auto_seed=False)
    assert "Learned construction" in os.observe('Actually, "A sparks B" means A causes B.').response
    assert "Stored causal IR" in os.observe("heat sparks expansion").response
    assert "Yes" in os.observe("heat causes expansion?").response
    assert "Learned construction" in os.observe('No, by "A outruns B" I mean A is greater than B.').response
    assert "Stored comparison IR" in os.observe("orion outruns zephyr").response
    assert "Yes" in os.observe("is orion greater than zephyr?").response


def test_elementary_workbook_growth_lab(tmp_path):
    report = DevelopmentalGrowthLab(tmp_path / "growth").run()
    assert report["after"]["score"] > report["before"]["score"]
    assert report["after"]["passed"] >= 16
    assert report["growth_delta"] >= 0.20
    assert "not proof of human-level intelligence" in report["claim"]


def test_intrinsic_motivation_and_sleep_consolidation(tmp_path):
    os = FinalCognitiveOS(tmp_path / "motivation", auto_seed=False)
    os.observe("Unparseable strange utterance zzz qqq")
    goals = IntrinsicMotivationEngine().propose(os)
    assert goals
    sleep_report = os.sleep.run()
    assert sleep_report.actions_logged >= 1
