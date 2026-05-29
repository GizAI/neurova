from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
from .ir import ClaimIR, IRCandidate, QuestionIR
from .memory import EvidenceGraphMemory


@dataclass
class ImmuneReport:
    status: str
    reasons: List[str]
    quarantined: bool = False


class EpistemicImmuneSystem:
    """Protects memory from parse errors, contradictions, and low-confidence claims."""
    def __init__(self, memory: EvidenceGraphMemory):
        self.memory = memory

    def inspect_candidate(self, cand: IRCandidate) -> ImmuneReport:
        reasons: List[str] = []
        if cand.confidence < 0.35 or cand.ambiguity > 0.65:
            reasons.append("low_confidence_or_high_ambiguity")
        if cand.validation_errors or cand.missing_fields:
            reasons.append("schema_validation_error")
        if isinstance(cand.ir, ClaimIR):
            opposite = "negative" if cand.ir.polarity == "positive" else "positive"
            if self.memory.find_claim(cand.ir.subject, cand.ir.relation, cand.ir.object, opposite, cand.ir.valid_from):
                reasons.append("opposite_claim_exists")
        status = "quarantine" if reasons else "clear"
        self.memory.log_action("EPISTEMIC_INSPECT", type(cand.ir).__name__, {"status": status, "reasons": reasons, "parser": cand.parser}, -0.2 if reasons else 0.2)
        return ImmuneReport(status=status, reasons=reasons, quarantined=bool(reasons))

    def quarantine_version(self, version_id: str, reason: str) -> None:
        self.memory.set_claim_version_status(version_id, "quarantined", reason)

    def audit(self) -> Dict[str, int]:
        stats = self.memory.stats()
        return {"contradictions": stats.get("contradictions", 0), "memory_actions": stats.get("memory_actions", 0)}
