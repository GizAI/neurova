from neurova.v29_systemic_learning import V29SystemicLearningAudit, V29SystemicBenchmark, V29SystemicTutor
from neurova.agent import FinalCognitiveOS


def test_v29_systemic_audit_growth(tmp_path):
    report = V29SystemicLearningAudit(tmp_path/'audit').run()
    assert report['leakage_guard']['passed']
    assert report['before']['score'] < report['after']['score']
    assert report['after']['passed'] == report['after']['total']


def test_v29_operations_are_not_sentence_memorization(tmp_path):
    os = FinalCognitiveOS(tmp_path/'os')
    bench = V29SystemicBenchmark()
    V29SystemicTutor(os, bench.tutor_lessons()).run()
    # The system was never taught this exact sentence; it must reuse the learned grammar operation.
    assert 'Yes' in os.observe('did Mira glarn Taro?').response
    assert 'Yes' in os.observe('Would you say Mira glarns Taro?').response
    assert 'rome' in os.observe('Where is box?').response.lower()
