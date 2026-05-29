from __future__ import annotations
from dataclasses import dataclass
from typing import List
from ..ir import IRCandidate

@dataclass
class VerificationReport:
    ok: bool
    errors: List[str]

class SemanticVerifier:
    def verify(self, cand: IRCandidate) -> VerificationReport:
        errs = list(cand.validation_errors)
        errs += [f"missing:{x}" for x in cand.missing_fields]
        return VerificationReport(not errs, errs)
