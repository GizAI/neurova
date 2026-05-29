from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict
import json, tempfile
from .agent import FinalCognitiveOS
from .predictive_developmental import PredictiveSocialCognitiveLoop, save_predictive_report
from .external_benchmarks import ExternalBenchmarkSuite

@dataclass
class AuditCase:
    category: str
    text: str
    expected_contains: str
    before_ok: bool = False
    after_ok: bool = False
    before_response: str = ""
    after_response: str = ""

@dataclass
class V32AuditReport:
    passed: int
    total: int
    accuracy: float
    cases: List[dict]
    external_compatible: dict
    predictive_consolidation: dict
    official_note: str

class V32PredictiveAGIAudit:
    """A stricter audit for developmental prediction, not an AGI proof.

    It tests grammar-operation learning, dialogue/world prediction, temporal
    contradiction, and official-benchmark-compatible reasoning. It also records
    prediction errors and sleep-like consolidation.
    """
    def __init__(self, root: Path | None = None):
        self.root = Path(root or tempfile.mkdtemp(prefix="brainos_v32_audit_"))
        self.root.mkdir(parents=True, exist_ok=True)

    def _cases(self) -> List[AuditCase]:
        return [
            AuditCase("wrapper_operation", "Would you say Nova outclasses Mira?", "nova greater_than mira"),
            AuditCase("wrapper_operation", "Did Nova outclass Mira?", "nova greater_than mira"),
            AuditCase("passive_operation", "Mira was outclassed by Nova.", "nova greater_than mira"),
            AuditCase("negation_scope", "Nova does not outclass Mira.", "nova less_than mira"),
            AuditCase("causal_wrapper", "Does heat trigger expansion?", "heat causes expansion"),
            AuditCase("fluent_world", "Where is box?", "oslo"),
            AuditCase("temporal_contradiction", "Who served as principal during 2021?", "Inconsistent evidence"),
            AuditCase("belief_coreference", "Does he believe she is CEO?", "bob believes alice is ceo"),
            AuditCase("dialogue_support", "I had a rough day. Can you help me think this through?", "I can help"),
            AuditCase("dialogue_smalltalk", "This is hilarious.", "pretty"),
            AuditCase("meta_memory", "What did we just learn about Kibo?", "kibo is rover"),
            AuditCase("korean_comparison", "철수가 영희보다 우세하다고 봐도 되나?", "철수 greater_than 영희"),
        ]

    def _teach(self, os: FinalCognitiveOS):
        for t in [
            "Kibo is rover.",
            "When I say \"A outclasses B\", it means A is greater than B.",
            "Actually, \"A triggers B\" means A causes B.",
            "Heat triggers expansion.",
            "Nova outclasses Mira.",
            "Eve carried the box from Berlin to Rome.",
            "Mira moved the box from Rome to Oslo.",
            "Mina was principal from 2020 through 2022.",
            "Mina was not principal during 2021.",
            "Bob believes that Alice is CEO.",
            "영희에 비해 철수가 앞선다.",
        ]:
            os.observe(t)

    def run(self) -> V32AuditReport:
        cases=self._cases()
        loop=PredictiveSocialCognitiveLoop()
        before=FinalCognitiveOS(self.root/"before")
        for c in cases:
            r=before.observe(c.text).response
            c.before_response=r
            c.before_ok=c.expected_contains.lower() in r.lower()
            loop.observe(c.text, r, "unknown", c.before_ok)
        after=FinalCognitiveOS(self.root/"after")
        self._teach(after)
        for c in cases:
            r=after.observe(c.text).response
            c.after_response=r
            c.after_ok=c.expected_contains.lower() in r.lower()
            loop.observe(c.text, r, "unknown", c.after_ok)
        consolidation=loop.consolidate()
        save_predictive_report(self.root/"v32_predictive_records.json", loop)
        ext=ExternalBenchmarkSuite().run_all()
        passed=sum(c.after_ok for c in cases)
        total=len(cases)
        report=V32AuditReport(passed,total,0 if not total else passed/total,[asdict(c) for c in cases],ext,consolidation,"Official dataset loaders are provided in neurova/official_benchmark_loaders.py, but this audit uses compatible generated subsets unless official files are supplied.")
        (self.root/"v32_predictive_agi_audit_report.json").write_text(json.dumps(asdict(report),ensure_ascii=False,indent=2),encoding="utf-8")
        return report
