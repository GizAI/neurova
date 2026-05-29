from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple
import re
import uuid

from ..ir import (
    ClaimIR, NegatedClaimIR, TemporalClaimIR, CausalClaimIR, ComparisonIR,
    QuestionIR, CognitiveIR, IRCandidate
)


def _clean(x: str) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip().lower().strip(" .?!,'\""))


def _strip_articles(x: str) -> str:
    x = _clean(x)
    x = re.sub(r"^(a|an|the)\s+", "", x)
    if x.endswith("s") and len(x) > 3 and not x.endswith("ss"):
        x = x[:-1]
    return x


def _slot_values(ir: CognitiveIR) -> Dict[str, str]:
    if isinstance(ir, TemporalClaimIR):
        return {"subject": ir.subject, "object": ir.object, "time_expr": ir.time_expr or ir.valid_from or ir.valid_during or ""}
    if isinstance(ir, NegatedClaimIR):
        return {"subject": ir.subject, "object": ir.object}
    if isinstance(ir, ClaimIR):
        return {"subject": ir.subject, "object": ir.object}
    if isinstance(ir, CausalClaimIR):
        return {"cause": ir.cause, "effect": ir.effect}
    if isinstance(ir, ComparisonIR):
        return {"left": ir.left, "right": ir.right}
    return {}


@dataclass
class FeatureConstraint:
    slot: str
    kind: str = "entity_or_phrase"
    required: bool = True


@dataclass
class ConstructionVariant:
    name: str
    regex: str
    polarity: str = "positive"  # positive | negative | question | inverse_question
    transform: str = "identity"  # identity | question | negation | inverse


@dataclass
class FeatureConstruction:
    """Feature-structure construction, not a one-off regex patch.

    A construction stores a form schema, semantic schema, slots, constraints,
    variants, and evidence.  It approximates FCG-style form-meaning pairings in
    a compact, dependency-free runtime: parse = unify(surface features, slots),
    then instantiate semantic operation.
    """
    construction_id: str
    name: str
    form_schema: str
    semantic_schema: str
    ir_type: str
    slots: List[str]
    constraints: List[FeatureConstraint] = field(default_factory=list)
    variants: List[ConstructionVariant] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    counterexamples: List[str] = field(default_factory=list)
    support_count: int = 1
    confidence: float = 0.88

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CognitiveConstructionGrammar:
    """Generalization engine for no-LLM construction grammar learning.

    This replaces "regex patch accumulation" with form-meaning constructions:
    - slot variables are inferred from the target IR;
    - variants are generated automatically (assertion, question, negation,
      inverse/reverse where semantically valid);
    - parsing performs lightweight feature unification and semantic instantiation.
    """
    name = "v24_cognitive_construction_grammar"
    objective = "feature-structure construction grammar; form-meaning constructions; non-autoregressive objective; no generation"

    def __init__(self):
        self.constructions: List[FeatureConstruction] = []

    def __len__(self) -> int:
        return len(self.constructions)

    def export(self) -> List[Dict[str, Any]]:
        return [c.to_dict() for c in self.constructions]

    def learn(self, surface_text: str, target_ir: CognitiveIR, confidence: float = 0.9) -> Optional[FeatureConstruction]:
        sem = self._semantic_schema(target_ir)
        if not sem:
            return None
        slots = {k: _clean(v) for k, v in _slot_values(target_ir).items() if _clean(v)}
        if not slots:
            return None
        form_schema, used = self._abstract_form(surface_text, slots)
        if not used:
            return None
        ir_type = type(target_ir).__name__
        variants = self._variants(form_schema, sem, ir_type)
        constraints = [FeatureConstraint(slot=s) for s in used]
        for c in self.constructions:
            if c.form_schema == form_schema and c.semantic_schema == sem:
                c.support_count += 1
                c.confidence = min(0.99, c.confidence + 0.02)
                if surface_text not in c.examples:
                    c.examples.append(surface_text)
                return c
        c = FeatureConstruction(
            construction_id="fcg_" + uuid.uuid4().hex[:12],
            name=f"{ir_type}:{form_schema[:48]}",
            form_schema=form_schema,
            semantic_schema=sem,
            ir_type=ir_type,
            slots=used,
            constraints=constraints,
            variants=variants,
            examples=[surface_text],
            confidence=confidence,
        )
        self.constructions.append(c)
        return c

    def parse(self, text: str) -> List[IRCandidate]:
        low = _clean(text)
        out: List[IRCandidate] = []
        for c in self.constructions:
            for v in c.variants:
                m = re.match(v.regex, low, re.I)
                if not m:
                    continue
                slots = {k: _strip_articles(val) for k, val in m.groupdict().items() if val is not None}
                if not self._unify(c, slots):
                    continue
                ir = self._instantiate(c.semantic_schema, slots, variant=v)
                if ir is None:
                    continue
                notes = ["feature-structure construction", c.construction_id, f"variant={v.name}", f"schema={c.form_schema}"]
                out.append(IRCandidate(ir=ir, confidence=min(0.99, c.confidence + 0.01 * min(c.support_count, 5)), parser=self.name, notes=notes, model_score=0.42))
        return sorted(out, key=lambda x: x.total_score, reverse=True)

    def _unify(self, c: FeatureConstruction, slots: Dict[str, str]) -> bool:
        for cons in c.constraints:
            if cons.required and not slots.get(cons.slot):
                return False
            if slots.get(cons.slot) in {"", "?"}:
                return False
        # Avoid degenerate same-slot comparisons/causes unless explicitly learned.
        if "left" in slots and "right" in slots and slots["left"] == slots["right"]:
            return False
        if "cause" in slots and "effect" in slots and slots["cause"] == slots["effect"]:
            return False
        # Scope/modality guards: do not let near-surface traps become positive facts.
        # Example: "Zed almost glarns Rex" must not instantiate {left}="zed almost".
        blocked = {"almost", "wants", "want", "wanted", "failed", "tries", "tried", "expected", "said", "claims", "claim"}
        for val in slots.values():
            toks = set(str(val).lower().split())
            if toks & blocked:
                return False
        return True

    def _semantic_schema(self, ir: CognitiveIR) -> Optional[str]:
        if isinstance(ir, CausalClaimIR):
            return "causal({cause},{effect})"
        if isinstance(ir, ComparisonIR):
            return f"compare({{left}},{ir.comparator},{{right}})"
        if isinstance(ir, TemporalClaimIR):
            return f"temporal({{subject}},{ir.relation},{{object}},{{time_expr}})"
        if isinstance(ir, NegatedClaimIR):
            return f"not_claim({{subject}},{ir.relation},{{object}})"
        if isinstance(ir, ClaimIR):
            return f"claim({{subject}},{ir.relation},{{object}})"
        return None

    def _abstract_form(self, surface_text: str, slots: Dict[str, str]) -> Tuple[str, List[str]]:
        # If the teacher gives A/B placeholders, keep those as slots directly.
        s = _clean(surface_text)
        used: List[str] = []
        placeholder_map = [("left", "a"), ("right", "b"), ("cause", "a"), ("effect", "b"), ("subject", "a"), ("object", "b"), ("time_expr", "t")]
        # Prefer actual target role names by IR schema.
        for slot, ph in placeholder_map:
            if slot in slots and re.search(rf"\b{ph}\b", s):
                s = re.sub(rf"\b{ph}\b", "{" + slot + "}", s, count=1)
                if slot not in used:
                    used.append(slot)
        # Then replace actual slot surface values, longest first.
        for slot, value in sorted(slots.items(), key=lambda kv: len(kv[1]), reverse=True):
            if "{" + slot + "}" in s:
                continue
            if value and value in s:
                s = s.replace(value, "{" + slot + "}", 1)
                if slot not in used:
                    used.append(slot)
        # For binary relation teachings like "dominates" supplied with target compare(A,B), create canonical slot schema.
        if not used and len(slots) >= 2:
            pass
        return re.sub(r"\s+", " ", s).strip(), used

    def _schema_to_regex(self, schema: str) -> str:
        # Relax modifiers/adverbs and articles between tokens.
        parts: List[str] = []
        i = 0
        for m in re.finditer(r"\{(\w+)\}", schema):
            lit = schema[i:m.start()]
            parts.append(self._literal_to_regex(lit))
            slot = m.group(1)
            parts.append(rf"(?P<{slot}>.+?)")
            i = m.end()
        parts.append(self._literal_to_regex(schema[i:]))
        return r"^\s*" + "".join(parts).strip() + r"\s*[?.!]*\s*$"

    def _literal_to_regex(self, lit: str) -> str:
        lit = _clean(lit)
        if not lit:
            return r"\s*"
        raw_tokens = lit.split()
        tokens = []
        # Allow construction-specific modifiers to become optional features rather than frozen literals.
        # This is the key step that turns "A is slightly ahead of B" into a reusable
        # construction that also accepts "A is ahead of B" and question variants.
        optional_mods = {"slightly", "really", "clearly", "usually", "normally", "generally", "typically"}
        for t in raw_tokens:
            # Modifiers are constructional features, not obligatory lexical anchors.
            # They are accepted by the inter-token gap below, so we omit them from
            # the literal token sequence to generalize: "slightly ahead" -> "ahead".
            if t in optional_mods:
                continue
            tokens.append(re.escape(t))
        gap = r"\s+(?:slightly\s+|really\s+|clearly\s+|usually\s+|normally\s+|generally\s+|typically\s+)?"
        return r"\s*" + gap.join(tokens) + r"\s*"

    def _variants(self, form_schema: str, semantic_schema: str, ir_type: str) -> List[ConstructionVariant]:
        variants: List[ConstructionVariant] = []
        base_re = self._schema_to_regex(form_schema)
        variants.append(ConstructionVariant("declarative", base_re, "positive", "identity"))
        # Question variants: convert statement construction into interrogative forms.
        q_schema = form_schema
        if q_schema.startswith("{left} is "):
            q_schema = "is {left} " + q_schema[len("{left} is "):]
        elif q_schema.startswith("{subject} is "):
            q_schema = "is {subject} " + q_schema[len("{subject} is "):]
        else:
            q_schema = self._do_support_question_schema(form_schema)
        q_re = self._schema_to_regex(q_schema)
        variants.append(ConstructionVariant("question", q_re, "question", "question"))
        did_schema = self._did_support_question_schema(form_schema)
        if did_schema != q_schema and did_schema != form_schema:
            variants.append(ConstructionVariant("did_question", self._schema_to_regex(did_schema), "question", "question"))
        variants.append(ConstructionVariant("would_you_say_question", r"^\s*(?:would\s+you\s+say|do\s+you\s+think|is\s+it\s+true\s+that|is\s+it\s+fair\s+to\s+(?:say|call)|could|can|may)\s+" + self._schema_to_regex(form_schema).lstrip("^\\s*").rstrip("\\s*[?.!]*\\s*$") + r"\s*[?.!]*\s*$", "question", "question"))
        # Negation variants for be- and verb-based constructions.
        if " is " in form_schema:
            neg_schema = form_schema.replace(" is ", " is not ", 1)
            variants.append(ConstructionVariant("negated", self._schema_to_regex(neg_schema), "negative", "negation"))
        else:
            neg_schema = self._do_support_negation_schema(form_schema)
            if neg_schema != form_schema:
                variants.append(ConstructionVariant("do_support_negated", self._schema_to_regex(neg_schema), "negative", "negation"))
            did_neg_schema = self._did_support_negation_schema(form_schema)
            if did_neg_schema != form_schema:
                variants.append(ConstructionVariant("did_support_negated", self._schema_to_regex(did_neg_schema), "negative", "negation"))
        # Passive/reverse variants for transitive verb constructions.
        passive = self._passive_schema(form_schema)
        if passive:
            variants.append(ConstructionVariant("passive", self._schema_to_regex(passive), "positive", "identity"))
            variants.append(ConstructionVariant("past_passive", self._schema_to_regex(passive.replace(" is ", " was ", 1)), "positive", "identity"))
        # Comparison-specific reverse lexical variants.
        if semantic_schema.startswith("compare"):
            if "ahead of" in form_schema:
                rev = form_schema.replace("{left} is ahead of {right}", "{right} is behind {left}").replace("{left} ahead of {right}", "{right} behind {left}")
                if rev != form_schema:
                    variants.append(ConstructionVariant("inverse_behind", self._schema_to_regex(rev), "positive", "identity"))
            if "above" in form_schema:
                rev = form_schema.replace("{left} is above {right}", "{right} is below {left}").replace("{left} above {right}", "{right} below {left}")
                if rev != form_schema:
                    variants.append(ConstructionVariant("inverse_below", self._schema_to_regex(rev), "positive", "identity"))
            if "lags behind" in form_schema:
                rev = form_schema.replace("{left} lags behind {right}", "{right} is ahead of {left}")
                if rev != form_schema:
                    variants.append(ConstructionVariant("inverse_ahead_from_lag", self._schema_to_regex(rev), "positive", "identity"))
        return variants

    def _verb_base(self, phrase: str) -> str:
        # conservative English lemmatizer for learned construction variants.
        toks = phrase.split()
        if not toks:
            return phrase
        v = toks[0]
        if v.endswith("ies") and len(v) > 4:
            v = v[:-3] + "y"
        elif v.endswith("es") and (v.endswith("ches") or v.endswith("shes") or v.endswith("sses") or v.endswith("xes") or v.endswith("zes")):
            v = v[:-2]
        elif v.endswith("s") and len(v) > 3 and not v.endswith("ss"):
            v = v[:-1]
        toks[0] = v
        return " ".join(toks)

    def _regular_participle(self, base_phrase: str) -> str:
        toks = base_phrase.split()
        if not toks:
            return base_phrase
        v = toks[0]
        if v.endswith("e"):
            pp = v + "d"
        elif len(v) >= 3 and re.search(r"[aeiou][bcdfghjklmnpqrstvwxyz]$", v):
            pp = v + v[-1] + "ed"
        else:
            pp = v + "ed"
        irregular = {"beat":"beaten", "buy":"bought", "sell":"sold", "give":"given", "take":"taken", "send":"sent", "outdo":"outdone"}
        toks[0] = irregular.get(v, pp)
        return " ".join(toks)

    def _do_support_question_schema(self, form_schema: str) -> str:
        m = re.match(r"^(\{(?:left|subject|cause)\})\s+(.+?)\s+(\{(?:right|object|effect)\})$", form_schema)
        if not m:
            return "does " + form_schema
        subj, verb_phrase, obj = m.groups()
        return f"does {subj} {self._verb_base(verb_phrase)} {obj}"

    def _do_support_negation_schema(self, form_schema: str) -> str:
        m = re.match(r"^(\{(?:left|subject|cause)\})\s+(.+?)\s+(\{(?:right|object|effect)\})$", form_schema)
        if not m:
            return form_schema
        subj, verb_phrase, obj = m.groups()
        return f"{subj} does not {self._verb_base(verb_phrase)} {obj}"

    def _did_support_question_schema(self, form_schema: str) -> str:
        m = re.match(r"^(\{(?:left|subject|cause)\})\s+(.+?)\s+(\{(?:right|object|effect)\})$", form_schema)
        if not m:
            return form_schema
        subj, verb_phrase, obj = m.groups()
        return f"did {subj} {self._verb_base(verb_phrase)} {obj}"

    def _did_support_negation_schema(self, form_schema: str) -> str:
        m = re.match(r"^(\{(?:left|subject|cause)\})\s+(.+?)\s+(\{(?:right|object|effect)\})$", form_schema)
        if not m:
            return form_schema
        subj, verb_phrase, obj = m.groups()
        return f"{subj} did not {self._verb_base(verb_phrase)} {obj}"

    def _passive_schema(self, form_schema: str) -> Optional[str]:
        m = re.match(r"^(\{left\})\s+(.+?)\s+(\{right\})$", form_schema)
        if not m:
            return None
        _left, verb_phrase, _right = m.groups()
        # only simple transitive verb phrases, avoid particles like 'lags behind'.
        if any(p in verb_phrase for p in [" behind", " ahead", " above", " below", " to "]):
            return None
        return f"{{right}} is {self._regular_participle(self._verb_base(verb_phrase))} by {{left}}"

    def _instantiate(self, semantic_schema: str, slots: Dict[str, str], variant: ConstructionVariant) -> Optional[CognitiveIR]:
        def fill(name: str) -> str:
            return slots.get(name, "").strip()
        m = re.match(r"causal\(\{cause\},\{effect\}\)", semantic_schema)
        if m:
            ir = CausalClaimIR(cause=fill("cause"), effect=fill("effect"), confidence=0.9)
            if variant.transform == "question":
                return QuestionIR(target=ir, requested_mode="proof")
            return ir
        m = re.match(r"compare\(\{left\},(greater_than|less_than|equal_to),\{right\}\)", semantic_schema)
        if m:
            comp = m.group(1)
            ir = ComparisonIR(left=fill("left"), comparator=comp, right=fill("right"), confidence=0.9)
            if variant.transform == "question":
                return QuestionIR(target=ir, requested_mode="proof")
            if variant.transform == "negation":
                inv = "less_than" if comp == "greater_than" else "greater_than" if comp == "less_than" else "not_equal_to"
                return ComparisonIR(left=fill("left"), comparator=inv, right=fill("right"), confidence=0.74)
            return ir
        m = re.match(r"claim\(\{subject\},(.+?),\{object\}\)", semantic_schema)
        if m:
            rel = m.group(1)
            ir = ClaimIR(subject=fill("subject"), relation=rel, object=fill("object"), confidence=0.9)
            if variant.transform == "question":
                return QuestionIR(target=ir, requested_mode="proof")
            if variant.transform == "negation":
                return NegatedClaimIR(subject=fill("subject"), relation=rel, object=fill("object"), confidence=0.86)
            return ir
        m = re.match(r"not_claim\(\{subject\},(.+?),\{object\}\)", semantic_schema)
        if m:
            rel = m.group(1)
            ir = NegatedClaimIR(subject=fill("subject"), relation=rel, object=fill("object"), confidence=0.9)
            if variant.transform == "question":
                return QuestionIR(target=ir, requested_mode="proof")
            return ir
        m = re.match(r"temporal\(\{subject\},(.+?),\{object\},\{time_expr\}\)", semantic_schema)
        if m:
            rel = m.group(1)
            ir = TemporalClaimIR(subject=fill("subject"), relation=rel, object=fill("object"), time_expr=fill("time_expr"), valid_from=fill("time_expr"), valid_during=fill("time_expr"), confidence=0.9)
            if variant.transform == "question":
                return QuestionIR(target=ir, requested_mode="proof")
            return ir
        return None


__all__ = ["FeatureConstraint", "ConstructionVariant", "FeatureConstruction", "CognitiveConstructionGrammar"]
