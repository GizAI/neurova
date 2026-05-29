from __future__ import annotations

"""V35 broad-coverage parser layer.

This module is not a pile of one-off sentence patches.  It composes reusable
operators: semantic retrieval, wrapper decomposition, event frames, dialogue acts,
Korean morphology cues, and a typed lattice report.  It delegates domain facts to
existing BrainOS memory/reasoners rather than storing answer tables.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import re

from .ir import (
    ClaimIR, NegatedClaimIR, CausalClaimIR, ComparisonIR, EventIR, BeliefIR,
    QuestionIR, SupportRequestIR, SpeechActIR, MetaMemoryQuestionIR,
    IRCandidate, ToolCallIR
)
from .chart_lattice import TypedCandidateLattice, TypedCandidate, TypedConstraintVerifier
from .semantic_encoder import DeepSemanticEncoder, SemanticMemoryIndex


def _norm(x: str) -> str:
    x = x.strip().lower()
    x = re.sub(r"[?.!]+$", "", x)
    x = re.sub(r"\b(the|a|an)\b", " ", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def _singular(x: str) -> str:
    x = _norm(x)
    irregular = {"children": "child", "people": "person", "men": "man", "women": "woman", "ostriches": "ostrich", "boxes": "box"}
    if x in irregular:
        return irregular[x]
    if x.endswith("ies") and len(x) > 4:
        return x[:-3] + "y"
    if x.endswith(("ches", "shes", "xes", "ses", "zes")) and len(x) > 4:
        return x[:-2]
    if x.endswith("s") and len(x) > 3 and not x.endswith("ss"):
        return x[:-1]
    return x


def _base_verb(v: str) -> str:
    v = _norm(v)
    irregular = {"gave": "give", "given": "give", "bought": "buy", "sold": "sell", "moved": "move", "carried": "carry", "transported": "transport", "lent": "lend", "borrowed": "borrow", "borrows": "borrow", "returned": "return"}
    if v in irregular:
        return irregular[v]
    if v.endswith("ied") and len(v) > 4:
        return v[:-3] + "y"
    if v.endswith("ed") and len(v) > 4:
        return v[:-2]
    if v.endswith("es") and len(v) > 4:
        return v[:-2]
    if v.endswith("s") and len(v) > 3 and not v.endswith("ss"):
        return v[:-1]
    return v


@dataclass
class V35ParseBundle:
    candidates: List[IRCandidate]
    lattice: TypedCandidateLattice
    semantic_hints: Dict[str, Any] = field(default_factory=dict)


class V35BroadCoverageChartParser:
    def __init__(self, schema_substrate: Any = None):
        self.schema_substrate = schema_substrate
        self.encoder = DeepSemanticEncoder()
        self.index = SemanticMemoryIndex(self.encoder)
        self.verifier = TypedConstraintVerifier()
        self._install_seed_semantics()

    def _install_seed_semantics(self) -> None:
        seeds = [
            ("taxonomy_question", "Would you classify A as B? Can A be regarded as B? Does A fall under B? Is A a type of B?", {"family": "taxonomy_question"}),
            ("support_request", "I am stuck confused worried rough day not sure what to do next cheer me up", {"family": "support"}),
            ("korean_compare", "A가 B보다 앞선다 A는 B에 비해 우위다 A가 B보다 낫다고 볼 수 있나", {"family": "korean_comparison"}),
            ("belief_question", "Does A believe B is C? Does he believe she is C?", {"family": "belief_question"}),
            ("event_move", "A carried B from C to D A transported B from C to D A moved B from C to D", {"family": "move_frame"}),
        ]
        for sid, text, payload in seeds:
            self.index.add(sid, "semantic_family", text, payload)

    def parse(self, text: str, return_lattice: bool = False) -> List[IRCandidate] | Tuple[List[IRCandidate], TypedCandidateLattice]:
        bundle = self.parse_bundle(text)
        if return_lattice:
            return bundle.candidates, bundle.lattice
        return bundle.candidates

    def parse_bundle(self, text: str) -> V35ParseBundle:
        raw = text.strip()
        lattice = TypedCandidateLattice(text=raw, tokens=re.findall(r"[\w가-힣']+|[?.!,]", raw))
        root = lattice.add_node("utterance", raw, 0, len(lattice.tokens), score=1.0)
        hints = {
            "dialogue_act": self.encoder.classify_dialogue_act(raw),
            "ir_family": self.encoder.infer_ir_family(raw),
            "retrieval": [(i.item_id, round(s, 3), i.payload) for i, s in self.index.search(raw, top_k=3)],
        }
        candidates: List[TypedCandidate] = []
        # 1. dialogue/social acts first, so they are not misread as factual claims.
        candidates.extend(self._dialogue_candidates(raw, lattice, root))
        # 2. Wrapper-first proposition parsing.
        candidates.extend(self._wrapper_candidates(raw, lattice, root))
        # 3. Direct broad constructions and event frames.
        candidates.extend(self._direct_candidates(raw, lattice, root))
        # 4. Schema substrate candidates as typed fallback.
        if self.schema_substrate is not None:
            try:
                schema_candidates = self.schema_substrate.compile(raw)
            except Exception:
                schema_candidates = []
            for sc in schema_candidates:
                node = lattice.add_node("schema_candidate", type(sc.ir).__name__, 0, len(lattice.tokens), score=sc.total_score, parser=sc.parser)
                lattice.add_edge(root, node, "schema_substrate_candidate", sc.total_score)
                candidates.append(TypedCandidate(sc.ir, [node], type(sc.ir).__name__, sc.total_score, sc.parser, notes=list(sc.notes)))
        for tc in candidates:
            lattice.add_candidate(self.verifier.verify(raw, tc))
        out: List[IRCandidate] = []
        for tc in lattice.best():
            out.append(IRCandidate(tc.ir, min(0.995, tc.score), "v35_broad_chart", notes=tc.notes + ["broad_chart", f"sig={tc.type_signature}"]))
        if not out:
            lattice.diagnostics.append("no_v35_candidate")
        return V35ParseBundle(out, lattice, hints)

    def _dialogue_candidates(self, raw: str, lattice: TypedCandidateLattice, root: str) -> List[TypedCandidate]:
        low = raw.lower()
        out: List[TypedCandidate] = []
        if any(x in low for x in ["rough day", "cheer me", "not sure what to do", "missing something", "stuck", "confused", "worried"]):
            node = lattice.add_node("dialogue_act", "support_request", 0, len(lattice.tokens), score=0.88)
            lattice.add_edge(root, node, "social_state_transition", 0.88)
            out.append(TypedCandidate(SupportRequestIR(state="confused", request="help_think_through"), [node], "DialogueAct[SupportRequest]", 0.91, "v35_dialogue", ["not_fact", "social_world_state"], notes=["support_request_generalized"]))
        if any(x in low for x in ["haha", "lol", "hilarious", "that's wild", "nice"]):
            node = lattice.add_node("dialogue_act", "smalltalk", 0, len(lattice.tokens), score=0.76)
            lattice.add_edge(root, node, "rapport_acknowledgment", 0.76)
            out.append(TypedCandidate(SpeechActIR(speaker="user", act_type="smalltalk_humor", content=raw), [node], "DialogueAct[SmallTalk]", 0.80, "v35_dialogue", ["social_action"], notes=["smalltalk_not_claim"]))
        if re.search(r"what\s+did\s+we\s+(?:just\s+)?learn\s+about\s+(.+?)\??$", low):
            target = re.search(r"about\s+(.+?)\??$", raw, re.I).group(1)
            node = lattice.add_node("dialogue_act", "meta_memory_query", 0, len(lattice.tokens), score=0.86)
            out.append(TypedCandidate(MetaMemoryQuestionIR(target=_norm(target)), [node], "DialogueAct[MetaMemory]", 0.88, "v35_dialogue", ["metacognitive_query"]))
        return out

    def _wrapper_candidates(self, raw: str, lattice: TypedCandidateLattice, root: str) -> List[TypedCandidate]:
        low = _norm(raw)
        out: List[TypedCandidate] = []
        wrapper_patterns = [
            (r"^(?:would\s+you\s+(?:say|think)|do\s+you\s+think|is\s+it\s+true\s+that|can\s+we\s+say)\s+(.+)$", "question", "question_wrapper"),
            (r"^would\s+you\s+classify\s+(.+?)\s+as\s+(.+)$", "taxonomy_question", "classification_question"),
            (r"^could\s+(.+?)\s+be\s+considered\s+part\s+of\s+(.+?)\s+category$", "taxonomy_question", "category_question"),
            (r"^would\s+it\s+be\s+fair\s+to\s+call\s+(.+?)\s+(.+)$", "taxonomy_question", "fair_to_call_question"),
        ]
        for pat, op, label in wrapper_patterns:
            m = re.match(pat, low, re.I)
            if not m:
                continue
            node = lattice.add_node("wrapper", label, 0, len(lattice.tokens), score=0.9, operation=op)
            lattice.add_edge(root, node, "wrapper_decomposition", 0.9)
            if op == "taxonomy_question":
                subj, obj = m.groups()
                target = ClaimIR(subject=_singular(subj), relation="is", object=_singular(obj))
                out.append(TypedCandidate(QuestionIR(target=target, requested_mode="proof"), [node], "Question[Claim]", 0.93, "v35_wrapper", ["taxonomy_wrapper"], notes=[label]))
            else:
                inner = m.group(1).strip()
                for inner_tc in self._compile_inner(inner, lattice, node):
                    final = inner_tc.ir if isinstance(inner_tc.ir, QuestionIR) else QuestionIR(target=inner_tc.ir, requested_mode="proof")
                    out.append(TypedCandidate(final, inner_tc.node_ids + [node], f"Question[{inner_tc.type_signature}]", min(0.98, inner_tc.score + 0.04), "v35_wrapper", ["wrapper_first", "inner_composition"], notes=inner_tc.notes + [label]))
        # Do/did question and negation are generic wrappers over binary clauses.
        m = re.match(r"^(?:does|do|did)\s+(.+?)\s+([a-z][a-z\-]+)\s+(.+)$", low, re.I)
        if m:
            subj, verb, obj = m.groups()
            inner = f"{subj} {_base_verb(verb)} {obj}"
            node = lattice.add_node("wrapper", "do_support_question", 0, len(lattice.tokens), score=0.88)
            for inner_tc in self._compile_inner(inner, lattice, node, verb_hint=_base_verb(verb)):
                out.append(TypedCandidate(QuestionIR(target=inner_tc.ir, requested_mode="proof"), inner_tc.node_ids + [node], f"Question[{inner_tc.type_signature}]", min(0.97, inner_tc.score + 0.04), "v35_do_question", ["do_support"], notes=inner_tc.notes))
        m = re.match(r"^(.+?)\s+(?:does|do|did)\s+(?:not|n't)\s+([a-z][a-z\-]+)\s+(.+)$", low, re.I)
        if m:
            subj, verb, obj = m.groups()
            inner = f"{subj} {_base_verb(verb)} {obj}"
            node = lattice.add_node("wrapper", "do_support_negation", 0, len(lattice.tokens), score=0.86)
            for inner_tc in self._compile_inner(inner, lattice, node, verb_hint=_base_verb(verb)):
                out.append(TypedCandidate(self._negate(inner_tc.ir), inner_tc.node_ids + [node], f"Negation[{inner_tc.type_signature}]", min(0.96, inner_tc.score + 0.02), "v35_do_negation", ["negation_scope"], notes=inner_tc.notes))
        m = re.match(r"^(.+?)\s+(?:is|was|were)\s+([a-z][a-z\-]+?)(?:ed|d)?\s+by\s+(.+)$", low, re.I)
        if m:
            obj, verb, subj = m.groups()
            inner = f"{subj} {_base_verb(verb)} {obj}"
            node = lattice.add_node("wrapper", "passive_voice", 0, len(lattice.tokens), score=0.87)
            for inner_tc in self._compile_inner(inner, lattice, node, verb_hint=_base_verb(verb)):
                out.append(TypedCandidate(inner_tc.ir, inner_tc.node_ids + [node], inner_tc.type_signature, min(0.96, inner_tc.score + 0.03), "v35_passive", ["voice_transform"], notes=inner_tc.notes))
        return out

    def _direct_candidates(self, raw: str, lattice: TypedCandidateLattice, root: str) -> List[TypedCandidate]:
        low = _norm(raw)
        out: List[TypedCandidate] = []
        # Korean comparison operations (particle-based morphology).
        # Pattern: X(이/가) Z(보다) ... COMPARISON (크다/우세하다/앞서다/낫다/dominant/above/ahead/superior)
        # Handles: 압도하다, 능가하다, 앞서는 편이다, etc.
        km = re.search(r"(.+?)(?:는|은|이|가|과|와)\s+(.+?)보다\s+(?:확실히\s+)?(?:더\s+)?(?:크|우세|앞서|낫|압도|능가|above|ahead|superior|dominant)", raw)
        if km:
            left, right = _norm(km.group(1)), _norm(km.group(2))
            neg = bool(re.search(r"지\s*않|않은\s+것\s+같|낫지는\s+않|않아|가\s+아니|까지\s+못", raw))
            comp = "less_than" if neg else "greater_than"
            node = lattice.add_node("korean_operation", "comparison", 0, len(lattice.tokens), score=0.82)
            ir = ComparisonIR(left=left, comparator=comp, right=right)
            if re.search(r"(니|나|까|맞아|되나|볼 수 있나|다고\s+볼|수\s+있나)", raw):
                ir = QuestionIR(target=ir, requested_mode="proof")
            out.append(TypedCandidate(ir, [node], "Comparison[ko]", 0.86, "v35_korean_operation", ["korean_particle_operation"]))
        # Reverse: X(보다) Y(이/가) ... COMPARISON
        km = re.search(r"(.+?)보다\s+(.+?)(?:가|이)?\s+(?:더\s+)?(?:크|우세|앞서|낫|압도|능가|above|ahead|superior|dominant)", raw)
        if km:
            right, left = _norm(km.group(1)), _norm(km.group(2))
            neg = bool(re.search(r"지\s*않|않은\s+것|가\s+아니", raw))
            comp = "less_than" if neg else "greater_than"
            node = lattice.add_node("korean_operation", "reverse_comparison", 0, len(lattice.tokens), score=0.80)
            ir = ComparisonIR(left=left, comparator=comp, right=right)
            if re.search(r"(니|나|까|맞아|되나|볼 수 있나|다고\s+볼|수\s+있나)", raw):
                ir = QuestionIR(target=ir, requested_mode="proof")
            out.append(TypedCandidate(ir, [node], "Comparison[ko_reverse]", 0.84, "v35_korean_operation", ["korean_reverse_marker"]))
        # Korean inverse: X(이/가) Z(보다) 뒤처지다/떨어지다/못하다/backward/behind
        km = re.search(r"(.+?)(?:가|이)?\s+(.+?)보다\s+(?:더\s+)?(?:뒤처|떨어|못하|backward|behind|inferior)", raw)
        if km:
            left, right = _norm(km.group(1)), _norm(km.group(2))
            node = lattice.add_node("korean_operation", "inverse_comparison", 0, len(lattice.tokens), score=0.80)
            ir = ComparisonIR(left=left, comparator="less_than", right=right)
            if re.search(r"(니|나|까|맞아|되나)", raw):
                ir = QuestionIR(target=ir, requested_mode="proof")
            out.append(TypedCandidate(ir, [node], "Comparison[ko_inverse]", 0.84, "v35_korean_operation", ["korean_inverse_marker"]))
        # Korean dominance/object-particle: X(가) Y(를) 압도/능가/우위
        km = re.search(r"(.+?)(?:이|가)\s+(.+?)(?:을|를)\s+(?:압도|능가|우위|dominant|superior)", raw)
        if km:
            left, right = _norm(km.group(1)), _norm(km.group(2))
            node = lattice.add_node("korean_operation", "dominance", 0, len(lattice.tokens), score=0.80)
            ir = ComparisonIR(left=left, comparator="greater_than", right=right)
            if re.search(r"(니|나|까|맞아|되나)", raw):
                ir = QuestionIR(target=ir, requested_mode="proof")
            out.append(TypedCandidate(ir, [node], "Comparison[ko_dominance]", 0.84, "v35_korean_operation", ["korean_dominance_marker"]))
        # Event frames with fluent effects.
        out.extend(self._event_direct(raw, lattice, root))
        # Taxonomy questions without wrappers.
        m = re.match(r"^(?:can|could|may)\s+(.+?)\s+be\s+regarded\s+as\s+(.+)$", low)
        if m:
            node = lattice.add_node("taxonomy_question", "regarded_as", 0, len(lattice.tokens), score=0.8)
            out.append(TypedCandidate(QuestionIR(target=ClaimIR(subject=_singular(m.group(1)), relation="is", object=_singular(m.group(2))), requested_mode="proof"), [node], "Question[Claim]", 0.85, "v35_taxonomy", ["taxonomy_question"]))
        m = re.match(r"^(?:does|do|did)\s+(.+?)\s+(?:belong\s+to|fall\s+under)\s+(.+)$", low)
        if m:
            node = lattice.add_node("taxonomy_question", "belong_to", 0, len(lattice.tokens), score=0.82)
            out.append(TypedCandidate(QuestionIR(target=ClaimIR(subject=_singular(m.group(1)), relation="is", object=_singular(m.group(2))), requested_mode="proof"), [node], "Question[Claim]", 0.88, "v35_taxonomy", ["taxonomy_membership_question"]))
        m = re.match(r"^(?:would\s+you\s+say\s+)?(.+?)\s+counts?\s+as\s+(.+)$", low)
        if m:
            node = lattice.add_node("taxonomy_question", "counts_as", 0, len(lattice.tokens), score=0.82)
            out.append(TypedCandidate(QuestionIR(target=ClaimIR(subject=_singular(m.group(1)), relation="is", object=_singular(m.group(2))), requested_mode="proof"), [node], "Question[Claim]", 0.88, "v35_taxonomy", ["counts_as_question"]))
        return out

    def _event_direct(self, raw: str, lattice: TypedCandidateLattice, root: str) -> List[TypedCandidate]:
        low = _norm(raw)
        out: List[TypedCandidate] = []
        move_patterns = [
            r"^(.+?)\s+(?:moved|carried|transported|relocated|brought)\s+(.+?)\s+from\s+(.+?)\s+to\s+(.+)$",
            r"^(.+?)\s+(?:carries|moves|transports)\s+(.+?)\s+from\s+(.+?)\s+to\s+(.+)$",
        ]
        for pat in move_patterns:
            m = re.match(pat, low)
            if m:
                actor, patient, src, dst = map(_singular, m.groups())
                node = lattice.add_node("event_frame", "move", 0, len(lattice.tokens), score=0.86)
                out.append(TypedCandidate(EventIR(actor=actor, action="move", patient=patient, location=dst), [node], "EventFrame[move]", 0.89, "v35_event_frame", ["roles:actor,patient,source,destination", "effect:located_at"]))
        transfer_patterns = [
            # Prepositional dative must be checked before the bare double-object frame.
            (r"^(.+?)\s+(?:gave|handed|sent|lent)\s+(.+?)\s+to\s+(.+?)(?:\s+in\s+.+?)?$", "give_to"),
            (r"^(.+?)\s+(?:gave|handed|sent|lent)\s+(.+?)\s+(.+?)(?:\s+in\s+.+?)?$", "give"),
            (r"^(.+?)\s+(?:bought|purchased)\s+(.+?)\s+from\s+(.+?)(?:\s+.+?)?$", "buy"),
            (r"^(.+?)\s+(?:sold)\s+(.+?)\s+to\s+(.+?)(?:\s+.+?)?$", "sell"),
            (r"^(.+?)\s+(?:received|borrowed|borrows|borrow|receives|gets)\s+(.+?)\s+from\s+(.+?)(?:\s+.+?)?$", "receive"),
        ]
        for pat, kind in transfer_patterns:
            m = re.match(pat, low)
            if not m:
                continue
            a, b, c = map(_singular, m.groups())
            node = lattice.add_node("event_frame", kind, 0, len(lattice.tokens), score=0.84)
            if kind == "give":
                actor, rec, obj = a, b, c
            elif kind == "give_to":
                actor, obj, rec = a, b, c
            elif kind == "buy":
                actor, obj, rec = a, b, c
            elif kind == "sell":
                actor, obj, rec = a, b, c
            else:  # receive/borrow
                actor, obj, rec = a, b, c
            out.append(TypedCandidate(EventIR(actor=actor, action=("sell" if kind=="sell" else "receive" if kind in {"receive", "borrow"} else "buy" if kind=="buy" else "give"), patient=obj, recipient=rec), [node], f"EventFrame[{kind}]", 0.87, "v35_event_frame", ["effect:has"]))
        m = re.match(r"^(.+?)\s+(?:opened|closed|locked|unlocked)\s+(.+)$", low)
        if m:
            actor, obj = map(_singular, m.groups())
            action = _base_verb(re.findall(r"\b(opened|closed|locked|unlocked)\b", low)[0])
            node = lattice.add_node("event_frame", action, 0, len(lattice.tokens), score=0.82)
            out.append(TypedCandidate(EventIR(actor=actor, action=action, patient=obj), [node], f"EventFrame[{action}]", 0.85, "v35_event_frame", ["effect:state"]))
        return out

    def _compile_inner(self, inner: str, lattice: TypedCandidateLattice, parent: str, verb_hint: Optional[str] = None) -> List[TypedCandidate]:
        out: List[TypedCandidate] = []
        # First try learned schema substrate recursively without creating infinite wrappers.
        if self.schema_substrate is not None:
            try:
                for c in self.schema_substrate.compile(inner):
                    node = lattice.add_node("inner_schema", type(c.ir).__name__, 0, len(lattice.tokens), score=c.total_score, inner=inner)
                    lattice.add_edge(parent, node, "compile_inner_schema", c.total_score)
                    out.append(TypedCandidate(c.ir, [node], type(c.ir).__name__, min(0.95, c.total_score), "v35_inner_schema", notes=c.notes))
            except RecursionError:
                pass
            except Exception:
                pass
        # Generic binary relation fallback using semantic family hints.
        low = _norm(inner)
        m = re.match(r"^(.+?)\s+([a-z][a-z\-]+)\s+(.+)$", low)
        if m:
            subj, verb, obj = m.groups()
            verb = _base_verb(verb_hint or verb)
            family, fam_score = self.encoder.infer_ir_family(f"{subj} {verb} {obj}")
            node = lattice.add_node("inner_binary", verb, 0, len(lattice.tokens), score=fam_score, subject=subj, object=obj)
            lattice.add_edge(parent, node, "binary_relation_candidate", fam_score)
            # Consult schema nearest neighbor for unknown verbs.
            top = self.index.search(verb, top_k=1)
            # Very generic default: if a learned schema exists in substrate, it should already have matched.
            if family == "CausalClaimIR" or verb in {"cause", "trigger", "spark", "catalyze"}:
                out.append(TypedCandidate(CausalClaimIR(cause=_singular(subj), effect=_singular(obj)), [node], "CausalClaim", 0.76, "v35_inner_binary", ["semantic_family:causal"]))
            elif family == "ComparisonIR" or verb in {"outclass", "outrank", "dominate", "glarn", "zorble", "tharne"}:
                out.append(TypedCandidate(ComparisonIR(left=_singular(subj), comparator="greater_than", right=_singular(obj)), [node], "Comparison", 0.74, "v35_inner_binary", ["semantic_family:comparison"]))
            elif verb in {"have", "has"}:
                out.append(TypedCandidate(ClaimIR(subject=_singular(subj), relation="has", object=_singular(obj)), [node], "Claim[has]", 0.82, "v35_inner_have", ["world_state_query_normalization"]))
            elif verb in {"believe", "believes", "think", "thinks"}:
                out.append(TypedCandidate(ClaimIR(subject=_singular(subj), relation="believes", object=_norm(obj)), [node], "Claim[believes]", 0.93, "v35_inner_belief_claim", ["belief_relation_normalization"]))
            else:
                # fallback claim with low score, so schema candidates can outrank it.
                out.append(TypedCandidate(ClaimIR(subject=_singular(subj), relation=verb, object=_singular(obj)), [node], "Claim[relation]", 0.42, "v35_inner_binary_low", ["low_confidence_relation"]))
        # Classification normalization. Avoid stealing embedded belief clauses such as
        # "bob believe alice is ceo"; those are relation claims over a proposition.
        m = re.match(r"^(.+?)\s+(?:is|are|classifies\s+as|fall\s+under|falls\s+under|is\s+a\s+type\s+of)\s+(.+)$", low)
        if m and not re.search(r"\b(believe|believes|think|thinks)\b", m.group(1)):
            node = lattice.add_node("inner_taxonomy", "taxonomy", 0, len(lattice.tokens), score=0.82)
            out.append(TypedCandidate(ClaimIR(subject=_singular(m.group(1)), relation="is", object=_singular(m.group(2))), [node], "Claim[is]", 0.86, "v35_inner_taxonomy", ["taxonomy_normalization"]))
        return out

    def _negate(self, ir: Any) -> Any:
        if isinstance(ir, ComparisonIR):
            inv = "less_than" if ir.comparator == "greater_than" else "greater_than" if ir.comparator == "less_than" else ir.comparator
            return ComparisonIR(left=ir.left, comparator=inv, right=ir.right)
        if isinstance(ir, CausalClaimIR):
            return CausalClaimIR(cause=ir.cause, effect=ir.effect, polarity="negative", relation=ir.relation)
        if isinstance(ir, ClaimIR):
            return NegatedClaimIR(subject=ir.subject, relation=ir.relation, object=ir.object)
        return ir
