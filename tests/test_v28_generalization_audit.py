from neurova.agent import FinalCognitiveOS
from neurova.v28_generalization import V28GeneralizationAudit


def test_v28_generalization_audit_passes(tmp_path):
    report = V28GeneralizationAudit(tmp_path/'audit').run()
    assert report['leakage_guard']['passed']
    assert report['before']['score'] < report['after']['score']
    assert report['after']['passed'] == report['after']['total']
    assert report['ablation']['construction_only']['score'] < report['after']['score']


def test_v28_runtime_method(tmp_path):
    os = FinalCognitiveOS(tmp_path/'os')
    report = os.run_v28_generalization_audit()
    assert report['after']['passed'] == report['after']['total']
