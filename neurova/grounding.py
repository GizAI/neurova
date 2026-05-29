from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
from .ir import ProofIR
from .world import StateTransitionWorldModel


@dataclass
class GroundingReport:
    ok: bool
    channel: str
    detail: Dict


class GroundedVerifier:
    """Links symbolic answers to proof, world, or execution evidence."""
    def verify_proof(self, proof: ProofIR) -> GroundingReport:
        ok = proof.success and bool(proof.steps)
        if proof.status in {"refuted", "blocked_by_exception", "inconsistent"}:
            ok = bool(proof.steps) or bool(proof.active_memory_trace)
        return GroundingReport(ok, "proof", {"status": proof.status, "steps": len(proof.steps), "trace": proof.active_memory_trace})

    def verify_world_action(self, world: StateTransitionWorldModel, action: str) -> GroundingReport:
        preds = world.predict(action)
        return GroundingReport(bool(preds), "world", {"action": action, "predictions": preds[:3]})

    def verify_execution(self, success: bool, attempts: int) -> GroundingReport:
        return GroundingReport(bool(success), "execution", {"success": success, "attempts": attempts})
