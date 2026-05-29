from pathlib import Path
from neurova.v30_final_audit import V30FinalAnswerAudit
from neurova.agent import FinalCognitiveOS


def test_v30_final_audit_passes(tmp_path: Path):
    report = V30FinalAnswerAudit().run(tmp_path / 'audit')
    assert report.after_passed == report.total, [(r.name, r.after) for r in report.rows if not r.passed]
    assert report.before_passed < report.after_passed
    assert all(report.checklist.values())


def test_v30_no_autoregressive_generation(tmp_path: Path):
    os = FinalCognitiveOS(root=tmp_path / 'os')
    assert 'next-token' not in os.cognitive_model.report_for(os.compiler.compile('is kibo robot?')[0]).objective
    assert not hasattr(os.compiler.neural_perception, 'generate')
