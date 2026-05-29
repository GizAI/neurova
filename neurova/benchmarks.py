from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass
class BenchmarkRow:
    name: str
    passed: bool
    observed: str
    expected: str


class BrainOSBenchmark:
    def __init__(self):
        self.rows: List[BenchmarkRow] = []

    def check(self, name: str, fn: Callable[[], bool], observed: str = "", expected: str = "true") -> None:
        try:
            passed = bool(fn())
        except Exception as e:
            passed = False
            observed = repr(e)
        self.rows.append(BenchmarkRow(name, passed, observed, expected))

    def report(self) -> Dict:
        passed = sum(r.passed for r in self.rows)
        return {"passed": passed, "total": len(self.rows), "percentage": round(100 * passed / max(1, len(self.rows)), 1), "rows": [r.__dict__ for r in self.rows]}
