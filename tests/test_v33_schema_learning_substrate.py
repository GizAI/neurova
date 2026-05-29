from pathlib import Path

from neurova.agent import FinalCognitiveOS
from neurova.schema_learning import SchemaLearningSubstrate, HardcodeDetector
from neurova.v33_schema_audit import run_v33_audit


def test_schema_substrate_learns_construction_and_wrapper(tmp_path):
    os = FinalCognitiveOS(tmp_path / "os")
    os.observe('When I say "A glarns B", it means A is greater than B.')
    os.observe('"Would you say P?" means asking whether P.')
    res = os.observe('Would you say Mira glarns Taro?').response.lower()
    assert 'mira greater_than taro' in res


def test_schema_substrate_learns_event_frame_world_effect(tmp_path):
    os = FinalCognitiveOS(tmp_path / "os")
    os.observe('When A carries B from C to D, it means A moves B from C to D, and after that B is located at D.')
    os.observe('Eve carried the box from Berlin to Rome.')
    res = os.observe('Where is box?').response.lower()
    assert 'rome' in res


def test_schema_substrate_records_prediction_error_and_consolidates(tmp_path):
    sub = SchemaLearningSubstrate(tmp_path / "schema.sqlite3")
    eid = sub.observe('Would you classify Kibo as a machine?', predicted_ir={'bad': 'ClaimIR'}, outcome='failed')
    sub.record_error(eid, 'wrapper_operation_error', {'want': 'QuestionIR'}, {'got': 'ClaimIR'}, 0.9)
    report = sub.consolidate()
    assert report['error_families']['wrapper_operation_error'] == 1


def test_hardcode_guard_and_v33_audit(tmp_path):
    audit = run_v33_audit(tmp_path / "audit")
    assert audit.after_passed == audit.total
    assert audit.after_passed > audit.before_passed
    assert audit.hardcode_guard_passed
    guard = HardcodeDetector(Path(__file__).resolve().parents[1]).scan()
    assert guard['passed'], guard['hits']
