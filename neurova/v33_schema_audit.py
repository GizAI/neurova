from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
import tempfile

from .agent import FinalCognitiveOS
from .schema_learning import SchemaLearningSubstrate, HardcodeDetector


@dataclass
class V33Case:
    category: str
    prompt: str
    expected_substring: str


@dataclass
class V33AuditResult:
    before_passed: int
    after_passed: int
    total: int
    hardcode_guard_passed: bool
    details: List[Dict[str, str]] = field(default_factory=list)

    @property
    def before_accuracy(self) -> float:
        return self.before_passed / self.total if self.total else 0.0

    @property
    def after_accuracy(self) -> float:
        return self.after_passed / self.total if self.total else 0.0


def _check(os: FinalCognitiveOS, case: V33Case) -> bool:
    try:
        res = os.observe(case.prompt).response.lower()
    except Exception as e:
        res = f"exception: {e}".lower()
    return case.expected_substring.lower() in res


class V33SchemaLearningAudit:
    """Non-AGI, targeted audit proving schema learning rather than hardcoded parser growth."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else Path(tempfile.mkdtemp(prefix="brainos_v33_audit_"))
        self.cases = [
            V33Case("wrapper_schema", "Would you say Mira glarns Taro?", "mira greater_than taro"),
            V33Case("construction_schema", "Mira glarns Taro.", "mira greater_than taro"),
            V33Case("event_schema", "Eve carried the box from Berlin to Rome.", "world_effects"),
            V33Case("event_world_query", "Where is box?", "rome"),
            V33Case("dialogue_schema", "I'm stuck and I need help.", "state the goal"),
        ]

    def run(self) -> V33AuditResult:
        before_os = FinalCognitiveOS(self.root / "before")
        before_passed = sum(1 for c in self.cases if _check(before_os, c))

        after_os = FinalCognitiveOS(self.root / "after")
        # General schema lessons. These are not the exact test prompts; they teach operations.
        lessons = [
            'When I say "A glarns B", it means A is greater than B.',
            '"Would you say P?" means asking whether P.',
            'When A carries B from C to D, it means A moves B from C to D, and after that B is located at D.',
            '"I need help" should be understood as support request',
        ]
        for lesson in lessons:
            after_os.observe(lesson)
        after_passed = 0
        details = []
        for c in self.cases:
            res = after_os.observe(c.prompt).response
            ok = c.expected_substring.lower() in res.lower()
            after_passed += 1 if ok else 0
            details.append({"category": c.category, "prompt": c.prompt, "response": res, "ok": str(ok)})

        guard = HardcodeDetector(Path(__file__).resolve().parents[1]).scan()
        return V33AuditResult(before_passed=before_passed, after_passed=after_passed, total=len(self.cases), hardcode_guard_passed=guard["passed"], details=details)


def run_v33_audit(root: Path | None = None) -> V33AuditResult:
    return V33SchemaLearningAudit(root).run()
