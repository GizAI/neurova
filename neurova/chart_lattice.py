from __future__ import annotations

"""V34 typed chart parser and candidate lattice.

This module is intentionally schema-driven.  It does not add new one-off
surface patterns to the legacy parser cascade; instead it composes learned
schemas into a typed candidate lattice and lets constraints/verification rank
or reject candidates.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import json
import re

from .ir import (
    ClaimIR,
    NegatedClaimIR,
    CausalClaimIR,
    ComparisonIR,
    EventIR,
    QuestionIR,
    ToolCallIR,
    SpeechActIR,
    SupportRequestIR,
    MetaMemoryQuestionIR,
    IRCandidate,
)


def _norm(x: str) -> str:
    x = x.strip().lower()
    x = re.sub(r"[?.!]+$", "", x)
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"^(a|an|the)\s+", "", x)
    return x.strip()


def _singular(x: str) -> str:
    x = _norm(x)
    if x.endswith("ies") and len(x) > 4:
        return x[:-3] + "y"
    if x.endswith(("ches", "shes", "xes", "ses", "zes")) and len(x) > 4:
        return x[:-2]
    if x.endswith("s") and len(x) > 3 and not x.endswith("ss"):
        return x[:-1]
    return x


@dataclass
class TokenSpan:
    start: int
    end: int
    text: str
    role: str = "token"
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LatticeNode:
    node_id: str
    span: TokenSpan
    kind: str
    payload: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


@dataclass
class LatticeEdge:
    source: str
    target: str
    label: str
    score: float = 0.0
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TypedCandidate:
    ir: Any
    node_ids: List[str]
    type_signature: str
    score: float
    parser: str
    constraints: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class TypedCandidateLattice:
    text: str
    tokens: List[str] = field(default_factory=list)
    nodes: List[LatticeNode] = field(default_factory=list)
    edges: List[LatticeEdge] = field(default_factory=list)
    candidates: List[TypedCandidate] = field(default_factory=list)
    diagnostics: List[str] = field(default_factory=list)

    def add_node(self, kind: str, text: str, start: int = 0, end: Optional[int] = None, score: float = 0.0, **payload: Any) -> str:
        nid = f"n{len(self.nodes)}"
        self.nodes.append(LatticeNode(nid, TokenSpan(start, end if end is not None else start + 1, text), kind, payload, score))
        return nid

    def add_edge(self, source: str, target: str, label: str, score: float = 0.0, **payload: Any) -> None:
        self.edges.append(LatticeEdge(source, target, label, score, payload))

    def add_candidate(self, cand: TypedCandidate) -> None:
        self.candidates.append(cand)

    def best(self) -> List[TypedCandidate]:
        return sorted([c for c in self.candidates if c.ok], key=lambda c: c.score, reverse=True)

    def to_report(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "tokens": self.tokens,
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
            "candidates": [
                {
                    "type_signature": c.type_signature,
                    "score": c.score,
                    "parser": c.parser,
                    "constraints": c.constraints,
                    "errors": c.errors,
                    "notes": c.notes,
                    "ir_type": type(c.ir).__name__,
                    "ir_text": getattr(c.ir, "text", lambda: str(c.ir))(),
                }
                for c in self.candidates
            ],
            "diagnostics": self.diagnostics,
        }


class TypedConstraintVerifier:
    """Lightweight semantic safety checks for lattice candidates."""

    NONASSERTIVE_CUES = re.compile(r"\b(almost|nearly|wants?\s+to|wanted\s+to|failed\s+to|tried\s+to|is\s+said\s+to|was\s+expected\s+to|would\s+.*\s+if)\b", re.I)

    def verify(self, text: str, cand: TypedCandidate) -> TypedCandidate:
        raw = text.lower().strip()
        ir = cand.ir
        if isinstance(ir, (ComparisonIR, CausalClaimIR, ClaimIR)) and self.NONASSERTIVE_CUES.search(raw):
            cand.errors.append("nonassertive_or_intensional_scope")
            cand.constraints.append("blocked_false_friend_scope")
        # Prevent classic wrapper-as-subject failures.
        if isinstance(ir, (ClaimIR, ComparisonIR)):
            subj = getattr(ir, "subject", None) or getattr(ir, "left", "")
            if isinstance(subj, str) and re.match(r"^(would you|is it true|do you think|does |did )", subj.strip(), re.I):
                cand.errors.append("wrapper_leaked_into_subject")
        return cand


class TypedChartParser:
    """Schema-driven chart parser.

    Inputs are learned schemas from SchemaLearningSubstrate.  The chart parser
    decomposes wrapper operations before matching inner clauses, builds typed
    candidate nodes, and returns ranked IRCandidates.  It is deliberately not a
    full natural-language parser, but it is the first cohesive typed candidate
    lattice in this project.
    """

    def __init__(self, schemas: List[Tuple[str, str, Dict[str, Any], float, str]]):
        self.schemas = schemas
        self.verifier = TypedConstraintVerifier()

    def parse(self, text: str, return_lattice: bool = False) -> List[IRCandidate] | Tuple[List[IRCandidate], TypedCandidateLattice]:
        lattice = self.build_lattice(text)
        out: List[IRCandidate] = []
        for tc in lattice.best():
            cand = IRCandidate(tc.ir, min(0.995, tc.score), tc.parser, notes=tc.notes + ["typed_lattice"])
            cand.model_score += 0.72
            out.append(cand)
        if return_lattice:
            return out, lattice
        return out

    def build_lattice(self, text: str) -> TypedCandidateLattice:
        raw = text.strip()
        low = _norm(raw)
        lattice = TypedCandidateLattice(text=raw, tokens=re.findall(r"[\w가-힣']+|[?.!,]", raw))
        root = lattice.add_node("utterance", raw, 0, len(lattice.tokens), score=1.0)
        # Decompose wrappers first; each inner parse becomes a child candidate.
        for inner, op, label in self._wrapper_inner(low):
            wn = lattice.add_node("wrapper", label, 0, len(lattice.tokens), score=0.95, operation=op, inner=inner)
            lattice.add_edge(root, wn, "decompose_wrapper", 0.95)
            inner_lattice = self.build_lattice(inner)
            for child in inner_lattice.best():
                ir = child.ir
                if op == "question":
                    final = ir if isinstance(ir, QuestionIR) else QuestionIR(target=ir, requested_mode="proof")
                    sig = f"Question[{child.type_signature}]"
                elif op == "negate":
                    final = self._negate_ir(ir)
                    sig = f"Negate[{child.type_signature}]"
                else:
                    final = ir
                    sig = child.type_signature
                node = lattice.add_node("candidate", getattr(final, "type_name", lambda: type(final).__name__)(), 0, len(lattice.tokens), score=child.score, inner=inner)
                lattice.add_edge(wn, node, "apply_wrapper_operation", child.score)
                tc = TypedCandidate(final, [wn, node], sig, min(0.99, child.score + 0.03), "v34_typed_chart", ["wrapper_first", "inner_ir_composition"], notes=child.notes)
                lattice.add_candidate(self.verifier.verify(raw, tc))
        # Direct schema matches.
        for sid, typ, schema, conf, status in self.schemas:
            if typ == "ConstructionSchema":
                for tc in self._match_construction(raw, schema, conf, sid):
                    node = lattice.add_node("construction", schema.get("name", sid), 0, len(lattice.tokens), score=tc.score, schema_id=sid)
                    lattice.add_edge(root, node, "match_construction", tc.score)
                    tc.node_ids.append(node)
                    lattice.add_candidate(self.verifier.verify(raw, tc))
            elif typ == "EventFrameSchema":
                for tc in self._match_event_frame(raw, schema, conf, sid):
                    node = lattice.add_node("event_frame", schema.get("name", sid), 0, len(lattice.tokens), score=tc.score, schema_id=sid)
                    lattice.add_edge(root, node, "match_event_frame", tc.score)
                    tc.node_ids.append(node)
                    lattice.add_candidate(self.verifier.verify(raw, tc))
            elif typ == "DialogueActSchema":
                tc = self._match_dialogue_act(raw, schema, conf, sid)
                if tc:
                    node = lattice.add_node("dialogue_act", schema.get("act_type", "dialogue"), 0, len(lattice.tokens), score=tc.score, schema_id=sid)
                    lattice.add_edge(root, node, "match_dialogue_act", tc.score)
                    tc.node_ids.append(node)
                    lattice.add_candidate(self.verifier.verify(raw, tc))
        if not lattice.candidates:
            lattice.diagnostics.append("no_schema_candidate")
        return lattice

    def _wrapper_inner(self, low: str) -> List[Tuple[str, str, str]]:
        out: List[Tuple[str, str, str]] = []
        wrappers = [
            (r"^would\s+you\s+(?:say|think)\s+(.+?)\??$", "question", "would_you_say"),
            (r"^is\s+it\s+true\s+that\s+(.+?)\??$", "question", "is_it_true_that"),
            (r"^do\s+you\s+think\s+(.+?)\??$", "question", "do_you_think"),
            (r"^can\s+we\s+say\s+(.+?)\??$", "question", "can_we_say"),
        ]
        for pat, op, label in wrappers:
            m = re.match(pat, low, re.I)
            if m:
                out.append((m.group(1).strip(), op, label))
        m = re.match(r"^(?:did|does|do)\s+(.+?)\s+([a-z][a-z\-]+)\s+(.+?)\??$", low, re.I)
        if m:
            subj, verb, obj = m.groups()
            out.append((f"{subj} {verb} {obj}", "question", "do_support_question"))
            if not verb.endswith("s"):
                out.append((f"{subj} {verb}s {obj}", "question", "do_support_question_s_variant"))
        m = re.match(r"^(.+?)\s+(?:does|did|do)\s+(?:not|n't)\s+([a-z][a-z\-]+)\s+(.+?)$", low, re.I)
        if m:
            subj, verb, obj = m.groups()
            out.append((f"{subj} {verb} {obj}", "negate", "do_support_negation"))
            if not verb.endswith("s"):
                out.append((f"{subj} {verb}s {obj}", "negate", "do_support_negation_s_variant"))
        m = re.match(r"^(.+?)\s+(?:is|was|were)\s+([a-z][a-z\-]+?)(?:ed)?\s+by\s+(.+?)\??$", low, re.I)
        if m:
            obj, verb, subj = m.groups()
            out.append((f"{subj} {verb} {obj}", "assert", "passive_relation"))
            if not verb.endswith("s"):
                out.append((f"{subj} {verb}s {obj}", "assert", "passive_relation_s_variant"))
        return out

    def _negate_ir(self, ir: Any) -> Any:
        if isinstance(ir, ComparisonIR):
            inv = "less_than" if ir.comparator == "greater_than" else "greater_than" if ir.comparator == "less_than" else ir.comparator
            return ComparisonIR(left=ir.left, comparator=inv, right=ir.right)
        if isinstance(ir, CausalClaimIR):
            return CausalClaimIR(cause=ir.cause, effect=ir.effect, polarity="negative", relation=ir.relation)
        if isinstance(ir, ClaimIR):
            return NegatedClaimIR(subject=ir.subject, relation=ir.relation, object=ir.object)
        return ir

    def _match_construction(self, raw: str, schema: Dict[str, Any], conf: float, sid: str) -> List[TypedCandidate]:
        patterns = [schema.get("form", "")] + list(schema.get("variants", []))
        out: List[TypedCandidate] = []
        for pat in patterns:
            slots = self._match_vars(pat, raw)
            if not slots:
                continue
            ir = self._instantiate_meaning(schema.get("meaning_schema", ""), slots)
            if not ir:
                continue
            out.append(TypedCandidate(ir, [], self._signature(ir), min(0.98, conf + 0.16), "v34_chart_construction", ["schema_slot_unification"], notes=[f"schema={sid}", f"pattern={pat}"]))
        return out

    def _match_event_frame(self, raw: str, schema: Dict[str, Any], conf: float, sid: str) -> List[TypedCandidate]:
        patterns = [schema.get("form", "")] + list(schema.get("variants", []))
        out: List[TypedCandidate] = []
        for pat in patterns:
            slots = self._match_vars(pat, raw)
            if not slots:
                continue
            roles = schema.get("roles", {})
            actor = slots.get(roles.get("actor") or roles.get("giver") or "A", "")
            patient = slots.get(roles.get("patient") or roles.get("object") or "B", "")
            recipient = slots.get(roles.get("recipient") or roles.get("borrower") or "B")
            dst = slots.get(roles.get("destination") or "D")
            action = "move" if any(e.get("relation") == "located_at" for e in schema.get("effects", [])) else "give"
            ev = EventIR(actor=_norm(actor), action=action, patient=_norm(patient), recipient=_norm(recipient) if recipient and action != "move" else None, location=_norm(dst) if dst else None)
            out.append(TypedCandidate(ev, [], "EventFrame[roles,effects]", min(0.98, conf + 0.14), "v34_chart_event_frame", ["role_unification", "world_effect_schema"], notes=[f"schema={sid}"]))
        return out

    def _match_dialogue_act(self, raw: str, schema: Dict[str, Any], conf: float, sid: str) -> Optional[TypedCandidate]:
        act = schema.get("act_type", "support_request")
        low = raw.lower()
        if act == "support_request" and any(w in low for w in ["stuck", "help", "confused", "worried", "rough day", "not sure"]):
            return TypedCandidate(SupportRequestIR(state="confused", request="help_think_through"), [], "DialogueAct[SupportRequest]", min(0.95, conf + 0.1), "v34_chart_dialogue_act", ["social_state_transition"], notes=[f"schema={sid}"])
        if schema.get("act_type") == "meta_memory_query" and "learn" in low and "about" in low:
            m = re.search(r"about\s+(.+?)\??$", raw, re.I)
            target = _norm(m.group(1)) if m else "recent"
            return TypedCandidate(MetaMemoryQuestionIR(target=target), [], "DialogueAct[MetaMemoryQuestion]", min(0.95, conf + 0.1), "v34_chart_dialogue_act", ["metacognitive_query"], notes=[f"schema={sid}"])
        return None

    def _instantiate_meaning(self, meaning: str, slots: Dict[str, str]) -> Optional[Any]:
        get = lambda k: _norm(slots.get(k, k))
        if meaning.startswith("ComparisonIR"):
            rel = "greater_than" if "greater_than" in meaning else "less_than" if "less_than" in meaning else "equal_to"
            return ComparisonIR(left=get("A"), comparator=rel, right=get("B"))
        if meaning.startswith("CausalClaimIR"):
            return CausalClaimIR(cause=get("A"), effect=get("B"))
        if meaning.startswith("NegatedClaimIR"):
            return NegatedClaimIR(subject=get("A"), relation="is", object=get("B"))
        if meaning.startswith("ClaimIR"):
            return ClaimIR(subject=get("A"), relation="is", object=get("B"))
        return None

    def _match_vars(self, pattern: str, raw: str) -> Optional[Dict[str, str]]:
        if not pattern:
            return None
        pat = pattern.strip().strip("?.!")
        raw_clean = raw.strip().strip("?.!")
        tokens = pat.split()
        regex_parts: List[str] = []
        vars_seen: List[str] = []
        for tok in tokens:
            if tok in {"A", "B", "C", "D", "P"}:
                regex_parts.append(r"(.+?)")
                vars_seen.append(tok)
            else:
                regex_parts.append(self._lexeme_regex(tok))
        rx = r"^" + r"\s+".join(regex_parts) + r"$"
        m = re.match(rx, raw_clean, re.I)
        if not m:
            return None
        return {var: _singular(val) for var, val in zip(vars_seen, m.groups())}

    def _lexeme_regex(self, tok: str) -> str:
        low = tok.lower()
        alts = {tok}
        if low.endswith("ies") and len(tok) > 4:
            base = tok[:-3] + "y"
            alts.update({base, base + "s", base[:-1] + "ies", base[:-1] + "ied", base + "ed", base + "d"})
        elif low.endswith("es") and len(tok) > 4:
            base = tok[:-2]
            alts.update({base, base + "s", base + "es", base + "ed", base + "d"})
        elif low.endswith("s") and len(tok) > 3 and not low.endswith("ss"):
            base = tok[:-1]
            alts.update({base, base + "s", base + "ed", base + "d"})
        else:
            alts.update({tok + "s", tok + "ed", tok + "d"})
            if low.endswith("y") and len(tok) > 2:
                alts.update({tok[:-1] + "ies", tok[:-1] + "ied"})
        # Also permit doubled consonant-ish no-op simplification for nonce verbs.
        return r"(?:" + "|".join(sorted((re.escape(a) for a in alts), key=len, reverse=True)) + r")"

    def _signature(self, ir: Any) -> str:
        if isinstance(ir, ComparisonIR):
            return f"Comparison[{ir.comparator}]"
        if isinstance(ir, CausalClaimIR):
            return "CausalClaim[causes]"
        if isinstance(ir, NegatedClaimIR):
            return "NegatedClaim"
        if isinstance(ir, ClaimIR):
            return f"Claim[{ir.relation}]"
        return type(ir).__name__


def ircandidate_lattice_report(candidates: List[IRCandidate], lattice: TypedCandidateLattice) -> str:
    return json.dumps({"candidate_count": len(candidates), "lattice": lattice.to_report()}, ensure_ascii=False, indent=2)
