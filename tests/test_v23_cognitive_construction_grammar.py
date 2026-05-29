from pathlib import Path
from neurova import FinalCognitiveOS
from neurova.ir import ToolCallIR, ComparisonIR, CausalClaimIR, EventIR, BeliefIR, SpeechActIR, CompositeIR


def make_os(tmp_path):
    return FinalCognitiveOS(Path(tmp_path) / "os")


def test_natural_language_correction_to_construction(tmp_path):
    os = make_os(tmp_path)
    r = os.observe('When I say "A outruns B", I mean A is faster than B.')
    assert "Learned" in r.response and "Schema" in r.response
    r2 = os.observe('alpha outruns beta')
    assert r2.ir_type == "ComparisonIR"
    assert "alpha greater_than beta" in r2.response
    r3 = os.observe('is alpha greater than beta?')
    assert "Yes" in r3.response


def test_causal_correction_to_construction(tmp_path):
    os = make_os(tmp_path)
    os.observe('By "A sparks B" I mean A causes B.')
    r = os.observe('heat sparks expansion')
    assert r.ir_type == "CausalClaimIR"
    assert "heat causes expansion" in r.response
    proof = os.observe('heat causes expansion?')
    assert "Yes" in proof.response


def test_discourse_event_belief_speech_frames(tmp_path):
    os = make_os(tmp_path)
    assert os.observe('Alice handed Bob a package in Seoul yesterday.').ir_type == "EventIR"
    assert os.observe('Bob thinks Alice is not the CEO.').ir_type == "BeliefIR"
    assert os.observe('Alice asked Bob to open the door.').ir_type == "SpeechActIR"
    assert os.observe('if rain falls, roads become slippery').ir_type == "CompositeIR"


def test_korean_particle_parser_variants(tmp_path):
    os = make_os(tmp_path)
    r = os.observe('철수가 영희를 능가한다')
    assert r.ir_type == "ComparisonIR"
    assert "철수 greater_than 영희" in r.response
    r2 = os.observe('영희보다 준호가 더 크다')
    assert r2.ir_type == "ComparisonIR"


def test_v23_smoke_includes_new_benchmark(tmp_path):
    os = make_os(tmp_path)
    report = os.run_smoke()
    assert report["passed"] == report["total"]
    names = {r["name"] for r in report["rows"]}
    assert "v23 natural correction parse" in names
    assert "speech act extraction" in names
