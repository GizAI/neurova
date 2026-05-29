from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import re
import uuid

from ..ir import (
    ClaimIR,
    NegatedClaimIR,
    TemporalClaimIR,
    CausalClaimIR,
    ComparisonIR,
    CognitiveIR,
    IRCandidate,
)


def _clean(x: str) -> str:
    return re.sub(r"\s+", " ", str(x).strip().lower())


def _slot_values(ir: CognitiveIR) -> Dict[str, str]:
    if isinstance(ir, TemporalClaimIR):
        return {"subject": ir.subject, "object": ir.object, "time_expr": ir.time_expr or ir.valid_from or ""}
    if isinstance(ir, NegatedClaimIR):
        return {"subject": ir.subject, "object": ir.object}
    if isinstance(ir, ClaimIR):
        return {"subject": ir.subject, "object": ir.object}
    if isinstance(ir, CausalClaimIR):
        return {"cause": ir.cause, "effect": ir.effect}
    if isinstance(ir, ComparisonIR):
        return {"left": ir.left, "right": ir.right}
    return {}


def _ir_type(ir: CognitiveIR) -> str:
    return type(ir).__name__


@dataclass
class SemanticConstruction:
    """A reusable surface construction mapped to one semantic operation.

    This is the compact alternative to adding endless regexes: one corrected
    example is abstracted into a typed construction, and then reused on new
    entities. It is not a language model and has no text-generation objective.
    """
    construction_id: str
    ir_type: str
    regex: str
    slots: List[str]
    source_text: str
    source_ir: Dict[str, Any]
    support_count: int = 1
    confidence: float = 0.82

    def pattern(self) -> re.Pattern:
        return re.compile(self.regex, re.I)


class ConstructionLearner:
    """One-shot semantic construction learner.

    Learns reusable NL -> IR mappings from a single supervised pair.
    Example:
        frost brings icy roads => CausalClaimIR(frost, icy roads)
    produces a construction that parses:
        heat brings expansion => CausalClaimIR(heat, expansion)

    This is the simple high-leverage core for broadening language ability
    without next-token imitation.
    """
    name = "construction_learner_no_lm"
    objective = "surface construction -> typed IR; one-shot abstraction; no autoregressive generation"

    def __init__(self):
        self.constructions: List[SemanticConstruction] = []

    def __len__(self) -> int:
        return len(self.constructions)

    def export(self) -> List[Dict[str, Any]]:
        return [asdict(c) for c in self.constructions]

    def learn(self, text: str, ir: CognitiveIR, confidence: float = 0.86) -> Optional[SemanticConstruction]:
        slots = _slot_values(ir)
        slots = {k: _clean(v) for k, v in slots.items() if _clean(v)}
        if not slots:
            return None
        low = _clean(text)
        temp = low
        used: List[str] = []
        # Replace longer slot values first so nested values do not collide.
        for slot, value in sorted(slots.items(), key=lambda kv: len(kv[1]), reverse=True):
            if value not in temp:
                continue
            marker = f"§{slot}§"
            temp = temp.replace(value, marker, 1)
            used.append(slot)
        # Need at least one variable slot; two slots make robust relational constructions.
        if not used:
            return None
        regex = re.escape(temp)
        for slot in used:
            regex = regex.replace(re.escape(f"§{slot}§"), f"(?P<{slot}>.+?)")
        regex = r"^" + regex.replace(r"\ ", r"\s+") + r"[?.!]*$"
        # Merge identical constructions by regex/type.
        for c in self.constructions:
            if c.ir_type == _ir_type(ir) and c.regex == regex:
                c.support_count += 1
                c.confidence = min(0.98, max(c.confidence, confidence) + 0.01)
                return c
        c = SemanticConstruction(
            construction_id="constr_" + uuid.uuid4().hex[:12],
            ir_type=_ir_type(ir),
            regex=regex,
            slots=used,
            source_text=text,
            source_ir=getattr(ir, "__dict__", {}).copy(),
            support_count=1,
            confidence=confidence,
        )
        self.constructions.append(c)
        return c

    def parse(self, text: str) -> List[IRCandidate]:
        low = _clean(text)
        out: List[IRCandidate] = []
        for c in self.constructions:
            m = c.pattern().match(low)
            if not m:
                continue
            slots = {k: v.strip(" ?.!,'\"") for k, v in m.groupdict().items() if v is not None}
            ir = self._instantiate(c.ir_type, slots, c.source_ir)
            if ir is None:
                continue
            out.append(IRCandidate(
                ir=ir,
                confidence=min(0.98, c.confidence + 0.01 * min(c.support_count, 5)),
                parser="construction_learner",
                notes=["one-shot construction abstraction", c.construction_id],
                model_score=0.06 + 0.01 * min(c.support_count, 5),
            ))
        return sorted(out, key=lambda x: x.total_score, reverse=True)

    def _instantiate(self, ir_type: str, slots: Dict[str, str], defaults: Dict[str, Any]) -> Optional[CognitiveIR]:
        blocked = {"almost", "wants", "want", "wanted", "failed", "tries", "tried", "expected", "said", "claims", "claim"}
        for v in slots.values():
            if set(str(v).lower().split()) & blocked:
                return None
        if ir_type == "CausalClaimIR":
            return CausalClaimIR(cause=slots.get("cause", defaults.get("cause", "")), effect=slots.get("effect", defaults.get("effect", "")), confidence=0.86)
        if ir_type == "ComparisonIR":
            return ComparisonIR(left=slots.get("left", defaults.get("left", "")), comparator=defaults.get("comparator", "greater_than"), right=slots.get("right", defaults.get("right", "")), confidence=0.86)
        if ir_type == "NegatedClaimIR":
            return NegatedClaimIR(subject=slots.get("subject", defaults.get("subject", "")), relation=defaults.get("relation", "is"), object=slots.get("object", defaults.get("object", "")), confidence=0.86)
        if ir_type == "TemporalClaimIR":
            t = slots.get("time_expr", defaults.get("time_expr") or defaults.get("valid_from", ""))
            return TemporalClaimIR(subject=slots.get("subject", defaults.get("subject", "")), relation=defaults.get("relation", "is"), object=slots.get("object", defaults.get("object", "")), time_expr=t, valid_from=t, confidence=0.86)
        if ir_type == "ClaimIR":
            return ClaimIR(subject=slots.get("subject", defaults.get("subject", "")), relation=defaults.get("relation", "is"), object=slots.get("object", defaults.get("object", "")), confidence=0.86)
        return None


__all__ = ["SemanticConstruction", "ConstructionLearner"]
