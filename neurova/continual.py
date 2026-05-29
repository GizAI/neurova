from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List
import json, time, uuid


@dataclass
class RegressionCase:
    name: str
    query: str
    predicate: Callable[[str], bool]
    expected: str = "predicate true"


@dataclass
class GateReport:
    candidate_id: str
    passed: int
    total: int
    promoted: bool
    failures: List[str] = field(default_factory=list)


class ContinualLearningGate:
    """Regression + benchmark gate for parser/memory/construction updates."""
    def __init__(self):
        self.cases: List[RegressionCase] = []

    def add_case(self, name: str, query: str, predicate: Callable[[str], bool], expected: str = "predicate true"):
        self.cases.append(RegressionCase(name, query, predicate, expected))

    def evaluate(self, runner: Callable[[str], str], candidate_id: str | None = None, promote_threshold: float = 1.0) -> GateReport:
        cid = candidate_id or "gate_" + uuid.uuid4().hex[:12]
        failures: List[str] = []
        passed = 0
        for c in self.cases:
            try:
                observed = runner(c.query)
                ok = bool(c.predicate(observed))
            except Exception as e:
                observed, ok = repr(e), False
            if ok:
                passed += 1
            else:
                failures.append(f"{c.name}: expected {c.expected}; observed={observed}")
        total = len(self.cases)
        ratio = passed / max(1, total)
        return GateReport(cid, passed, total, ratio >= promote_threshold, failures)

    def to_json(self, report: GateReport) -> str:
        return json.dumps(report.__dict__, ensure_ascii=False)
