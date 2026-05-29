from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
from .ir import ClaimIR, NegatedClaimIR, RuleIR
from .memory import EvidenceGraphMemory


@dataclass
class SkillResult:
    action: str
    target: str
    changed: bool
    detail: Dict


class MemorySkillEngine:
    """Explicit memory operations that can later become learned policy actions."""
    def __init__(self, memory: EvidenceGraphMemory):
        self.memory = memory

    def add_claim(self, claim: ClaimIR, evidence_id: str | None = None) -> SkillResult:
        vid = self.memory.upsert_claim(claim, evidence_id)
        return SkillResult("ADD_CLAIM", claim.normalized_key(), True, {"version_id": vid})

    def mark_contradiction_if_needed(self, claim: ClaimIR) -> SkillResult:
        cons = self.memory.contradictions_for_key(claim.normalized_key())
        return SkillResult("MARK_CONTRADICTION", claim.normalized_key(), bool(cons), {"count": len(cons)})

    def promote_rule(self, rule: RuleIR) -> SkillResult:
        rid = self.memory.add_rule(rule)
        return SkillResult("PROMOTE_RULE", rule.signature(), True, {"rule_id": rid})

    def create_regression_test(self, name: str, prompt: str, expected: str) -> SkillResult:
        pid = self.memory.add_promotion_candidate("regression_test", {"name": name, "prompt": prompt, "expected": expected}, status="candidate")
        self.memory.log_action("CREATE_REGRESSION_TEST", name, {"prompt": prompt, "expected": expected, "candidate_id": pid}, 0.5)
        return SkillResult("CREATE_REGRESSION_TEST", name, True, {"candidate_id": pid})
