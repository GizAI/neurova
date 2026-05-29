from neurova.v32_predictive_agi_audit import V32PredictiveAGIAudit
from neurova.predictive_developmental import PredictiveSocialCognitiveLoop
from neurova.official_benchmark_loaders import OfficialBenchmarkLoader
from pathlib import Path


def test_predictive_loop_consolidates_failure_clusters():
    loop = PredictiveSocialCognitiveLoop()
    loop.observe("Would you say nova glarns mira?", "cannot prove", "QuestionIR", False)
    loop.observe("I had a rough day", "fallback", "ResearchTaskIR", False)
    report = loop.consolidate()
    assert report["failure_clusters"]
    assert any(s["status"] in {"experimental", "stable"} for s in report["promoted_skills"])


def test_official_loader_reports_missing_files_without_claiming_score(tmp_path):
    loader = OfficialBenchmarkLoader()
    rep = loader.run_scan_file(str(tmp_path / "no_scan.txt"))
    assert not rep.loaded
    assert rep.total == 0
    assert "not evaluated" in rep.note


def test_v32_predictive_audit_growth(tmp_path):
    report = V32PredictiveAGIAudit(tmp_path / "audit").run()
    assert report.total >= 10
    assert report.passed == report.total
    assert report.external_compatible["accuracy"] == 1.0
    assert report.predictive_consolidation["failure_clusters"]
