
from __future__ import annotations
from dataclasses import dataclass
from .external_benchmarks import ExternalBenchmarkSuite

@dataclass
class V31ExternalAuditResult:
    passed: int
    total: int
    accuracy: float
    detail: dict

class V31ExternalAudit:
    """Runs official-benchmark-compatible, evaluation-time generated checks.

    This is not a claim of full official benchmark saturation. It is a no-leakage
    compatibility harness for SCAN-style compositional command semantics,
    bAbI-style object-location QA, and CLUTRR-style two-hop kinship reasoning.
    """
    def run(self) -> V31ExternalAuditResult:
        detail = ExternalBenchmarkSuite().run_all()
        return V31ExternalAuditResult(detail['passed'], detail['total'], detail['accuracy'], detail)
