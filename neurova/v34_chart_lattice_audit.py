from __future__ import annotations

"""V34 audit for typed candidate lattice and schema-driven chart parsing."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple
import json

from .agent import FinalCognitiveOS
from .chart_lattice import TypedChartParser


@dataclass
class V34AuditCase:
    category: str
    prompt: str
    expected_contains: str


@dataclass
class V34AuditReport:
    before_passed: int
    after_passed: int
    total: int
    lattice_nodes: int
    lattice_edges: int
    candidate_count: int
    passed: bool
    details: List[Dict]

    def to_dict(self) -> Dict:
        return asdict(self)


class V34ChartLatticeAudit:
    """Non-human-level but non-cheating audit.

    The tutor teaches schemas, then the benchmark uses different wrapper/passive/
    event/dialogue forms. Success requires schema-driven chart composition rather
    than answer table lookup.
    """

    def __init__(self, root: Path | None = None):
        self.root = Path(root or "/tmp/brainos_v34_chart_audit")

    def _new_os(self) -> FinalCognitiveOS:
        return FinalCognitiveOS(self.root, auto_seed=False)

    def _teach(self, os: FinalCognitiveOS) -> None:
        lessons = [
            'When I say "A tharnes B", it means A is greater than B.',
            'Luma tharnes Naro.',
            'Actually, "A morps B" means A causes B.',
            'Rain morps flood.',
            'When A ferries B from C to D, it means A moves B from C to D, and after that B is located at D.',
            '"I need help" should be understood as support request',
        ]
        for lesson in lessons:
            os.observe(lesson)

    def cases(self) -> List[V34AuditCase]:
        return [
            V34AuditCase("wrapper_question", "Would you say Luma tharnes Naro?", "yes"),
            V34AuditCase("did_question", "Did Luma tharne Naro?", "yes"),
            V34AuditCase("passive", "Naro was tharned by Luma.", "stored comparison"),
            V34AuditCase("negation", "Luma does not tharne Naro.", "less_than"),
            V34AuditCase("causal_question", "Does rain morp flood?", "yes"),
            V34AuditCase("causal_passive", "Flood was morped by rain.", "stored causal"),
            V34AuditCase("event_frame", "Eve ferried the crate from Oslo to Lima.", "stored event"),
            V34AuditCase("world_query", "Where is crate?", "lima"),
            V34AuditCase("dialogue_act", "I need help", "help"),
        ]

    def run(self) -> V34AuditReport:
        before = self._new_os()
        before_details = []
        before_passed = 0
        for case in self.cases():
            res = before.observe(case.prompt).response.lower()
            ok = case.expected_contains in res
            before_passed += int(ok)
            before_details.append({"phase": "before", "category": case.category, "prompt": case.prompt, "ok": ok, "response": res})

        os = self._new_os()
        self._teach(os)
        after_passed = 0
        after_details = []
        for case in self.cases():
            res = os.observe(case.prompt).response.lower()
            ok = case.expected_contains in res
            after_passed += int(ok)
            after_details.append({"phase": "after", "category": case.category, "prompt": case.prompt, "ok": ok, "response": res})

        schemas = os.schema_substrate.memory.schemas(include_experimental=True)
        chart = TypedChartParser(schemas)
        candidates, lattice = chart.parse("Would you say Luma tharnes Naro?", return_lattice=True)
        report = V34AuditReport(
            before_passed=before_passed,
            after_passed=after_passed,
            total=len(self.cases()),
            lattice_nodes=len(lattice.nodes),
            lattice_edges=len(lattice.edges),
            candidate_count=len(candidates),
            passed=after_passed == len(self.cases()) and len(lattice.nodes) > 0 and len(lattice.edges) > 0 and len(candidates) > 0,
            details=before_details + after_details,
        )
        return report


def write_v34_report(root: Path) -> Dict:
    audit = V34ChartLatticeAudit(root / "v34_audit_state")
    report = audit.run().to_dict()
    out = root / "artifacts" / "v34_chart_lattice_audit_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
