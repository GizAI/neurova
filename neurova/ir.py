from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union
from time import time
import json, uuid


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class BaseIR:
    id: str = field(default_factory=lambda: new_id("ir"))
    confidence: float = 0.7
    scope: str = "global"
    evidence_ids: List[str] = field(default_factory=list)

    def type_name(self) -> str:
        return type(self).__name__

    def to_json(self) -> str:
        d = asdict(self)
        d["_type"] = self.type_name()
        return json.dumps(d, ensure_ascii=False)


@dataclass
class EvidenceRefIR(BaseIR):
    source_id: str = "user"
    quote: str = ""
    reliability: float = 0.6
    created_at: float = field(default_factory=time)


@dataclass
class ClaimIR(BaseIR):
    subject: str = ""
    relation: str = "is"
    object: str = ""
    polarity: str = "positive"  # positive | negative | unknown
    modality: str = "asserted"
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    source_id: Optional[str] = None
    reliability: float = 0.6
    support_count: int = 0
    refute_count: int = 0
    last_verified_at: Optional[float] = None
    exceptions: List[str] = field(default_factory=list)

    def normalized_key(self) -> str:
        return f"{self.subject.lower()}|{self.relation.lower()}|{self.object.lower()}"

    def text(self) -> str:
        neg = "not " if self.polarity == "negative" else ""
        return f"{self.subject} {self.relation} {neg}{self.object}".strip()


@dataclass
class NegatedClaimIR(ClaimIR):
    polarity: str = "negative"


@dataclass
class TemporalClaimIR(ClaimIR):
    time_expr: str = ""
    valid_during: Optional[str] = None


@dataclass
class CausalClaimIR(BaseIR):
    cause: str = ""
    effect: str = ""
    polarity: str = "positive"  # positive | negative
    relation: str = "causes"

    def text(self) -> str:
        neg = "does not cause " if self.polarity == "negative" else "causes "
        return f"{self.cause} {neg}{self.effect}"

    def normalized_key(self) -> str:
        return f"{self.cause.lower()}|causes|{self.effect.lower()}"


@dataclass
class ComparisonIR(BaseIR):
    left: str = ""
    comparator: str = "greater_than"
    right: str = ""

    def text(self) -> str:
        return f"{self.left} {self.comparator} {self.right}"

    def normalized_key(self) -> str:
        return f"{self.left.lower()}|{self.comparator}|{self.right.lower()}"


@dataclass
class RuleIR(BaseIR):
    condition_relation: str = "is"
    condition_object: str = ""
    conclusion_relation: str = "is"
    conclusion_object: str = ""
    exceptions: List[str] = field(default_factory=list)

    def signature(self) -> str:
        return f"{self.condition_relation.lower()}|{self.condition_object.lower()}=>{self.conclusion_relation.lower()}|{self.conclusion_object.lower()}"

    def text(self) -> str:
        return f"if X {self.condition_relation} {self.condition_object} then X {self.conclusion_relation} {self.conclusion_object}"


@dataclass
class QuantifiedRuleIR(RuleIR):
    quantifier: str = "all"


@dataclass
class ExceptionIR(BaseIR):
    rule_id: str = ""
    exception_subject: str = ""
    exception_text: str = ""
    condition_object: str = ""
    conclusion_relation: str = ""
    conclusion_object: str = ""


@dataclass
class ContradictionIR(BaseIR):
    claim_a: str = ""
    claim_b: str = ""
    reason: str = ""


@dataclass
class QuestionIR(BaseIR):
    target: Union[ClaimIR, NegatedClaimIR, TemporalClaimIR, CausalClaimIR, ComparisonIR, None] = None
    requested_mode: str = "answer"


@dataclass
class ProgramSpecIR(BaseIR):
    name: str = ""
    function_name: str = ""
    inputs: List[str] = field(default_factory=list)
    outputs: str = ""
    invariants: List[str] = field(default_factory=list)
    tests: List[str] = field(default_factory=list)


@dataclass
class WritingTaskIR(BaseIR):
    topic: str = ""
    target_words: int = 500
    style: str = "structured"


@dataclass
class ResearchTaskIR(BaseIR):
    question: str = ""
    requires_sources: bool = True


@dataclass
class ExperimentIR(BaseIR):
    hypothesis: str = ""
    intervention: str = ""
    metric: str = ""


@dataclass
class PlanIR(BaseIR):
    goal: str = ""
    steps: List[str] = field(default_factory=list)


@dataclass
class EventIR(BaseIR):
    actor: str = ""
    action: str = ""
    patient: Optional[str] = None
    recipient: Optional[str] = None
    instrument: Optional[str] = None
    location: Optional[str] = None
    time_expr: Optional[str] = None
    manner: Optional[str] = None
    cause: Optional[str] = None
    result: Optional[str] = None
    polarity: str = "positive"

    def text(self) -> str:
        bits = [self.actor, self.action]
        if self.patient:
            bits.append(self.patient)
        if self.recipient:
            bits.append("to " + self.recipient)
        if self.location:
            bits.append("in " + self.location)
        if self.time_expr:
            bits.append("at " + self.time_expr)
        return " ".join([b for b in bits if b]).strip()


@dataclass
class BeliefIR(BaseIR):
    holder: str = ""
    proposition: Any = None
    confidence_label: str = "unknown"
    source: Optional[str] = None


@dataclass
class GoalIR(BaseIR):
    agent: str = ""
    desired_state: str = ""


@dataclass
class SpeechActIR(BaseIR):
    speaker: str = ""
    act_type: str = "statement"
    content: Any = None


@dataclass
class ToolCallIR(BaseIR):
    tool_name: str = ""
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompositeIR(BaseIR):
    items: List[Any] = field(default_factory=list)
    source_text: str = ""


@dataclass
class WrapperConstructionIR(BaseIR):
    """A learned grammar operation over an inner proposition, not a sentence template.

    Example: "would you say [P]?" -> QuestionIR(P).
    This is a first-class representation for learning grammar operations instead of
    memorizing individual question sentences.
    """
    wrapper_pattern: str = ""
    operation: str = "question"
    inner_slot: str = "P"
    source_example: str = ""


@dataclass
class EventFrameIR(BaseIR):
    """A learnable event frame schema with roles and world-state effects."""
    frame_name: str = ""
    surface_schema: str = ""
    roles: Dict[str, str] = field(default_factory=dict)
    effects: List[Dict[str, str]] = field(default_factory=list)
    variants: List[str] = field(default_factory=list)


@dataclass
class TemporalQuerySchemaIR(BaseIR):
    """A learnable temporal query schema, e.g. "Who served as ROLE during T?"."""
    surface_schema: str = ""
    role_slot: str = "ROLE"
    time_slot: str = "T"
    relation: str = "is"


@dataclass
class MetaMemoryQuestionIR(BaseIR):
    target: str = ""
    scope: str = "recent"
    request: str = "learned_facts_summary"


@dataclass
class SupportRequestIR(BaseIR):
    state: str = "confused"
    request: str = "help_think_through"


@dataclass
class ProofStepIR(BaseIR):
    conclusion: str = ""
    premises: List[str] = field(default_factory=list)
    rule_applied: str = ""
    verifier_status: str = "unchecked"
    alternative_paths: List[str] = field(default_factory=list)
    failed_paths: List[str] = field(default_factory=list)


@dataclass
class ProofIR(BaseIR):
    query: str = ""
    success: bool = False
    status: str = "unknown"  # proved | refuted | unknown | ambiguous | inconsistent | blocked_by_exception | time_dependent
    steps: List[ProofStepIR] = field(default_factory=list)
    active_memory_trace: List[str] = field(default_factory=list)
    verifier_status: str = "unchecked"

    def render(self) -> str:
        # Natural language rendering for successful proofs
        if self.success and self.steps:
            ans = []
            for s in self.steps:
                c = s.conclusion
                # Simple pronoun mapping
                c = " " + c + " "
                c = c.replace(" i am ", " __YOU_ARE__ ").replace(" i is ", " __YOU_ARE__ ")
                c = c.replace(" you are ", " I am ").replace(" you is ", " I am ")
                c = c.replace(" __YOU_ARE__ ", " you are ")
                c = c.replace(" my ", " __YOUR__ ").replace(" your ", " my ")
                c = c.replace(" __YOUR__ ", " your ")
                c = c.strip()
                if c not in ans:
                    ans.append(c)
            
            q = self.query.lower()
            if q.startswith("what ") or q.startswith("who ") or q.startswith("where ") or q.startswith("when "):
                res = " and ".join(ans)
                return res[0].upper() + res[1:] + "."
            elif q.startswith("is ") or q.startswith("does ") or q.startswith("do ") or q.startswith("can "):
                return "Yes. " + ans[0][0].upper() + ans[0][1:] + "."

        if self.success:
            lines = [f"Yes. I can prove: {self.query}"]
        elif self.status == "refuted":
            lines = [f"No. I can refute: {self.query}"]
        elif self.status == "inconsistent":
            lines = [f"Inconsistent evidence for: {self.query}"]
        elif self.status == "blocked_by_exception":
            lines = [f"Blocked by exception: {self.query}"]
        elif self.status == "time_dependent":
            lines = [f"Time-dependent answer for: {self.query}"]
        else:
            if self.query.lower() in ["what is you", "who are you"]:
                return "I don't have enough self-knowledge to answer that yet."
            lines = [f"I cannot prove: {self.query}"]
        for i, s in enumerate(self.steps, 1):
            prem = ", ".join(s.premises) if s.premises else "known"
            lines.append(f"{i}. {s.conclusion} because {prem}; rule={s.rule_applied}; verifier={s.verifier_status}")
        if self.active_memory_trace:
            lines.append("[active_memory] " + " -> ".join(self.active_memory_trace))
        return "\n".join(lines)


CognitiveIR = Union[
    ClaimIR, NegatedClaimIR, TemporalClaimIR, CausalClaimIR, ComparisonIR, RuleIR,
    QuantifiedRuleIR, ExceptionIR, ContradictionIR, QuestionIR, ProgramSpecIR,
    WritingTaskIR, ResearchTaskIR, ExperimentIR, PlanIR, EventIR, BeliefIR, GoalIR, SpeechActIR, WrapperConstructionIR, EventFrameIR, TemporalQuerySchemaIR, MetaMemoryQuestionIR, SupportRequestIR, ToolCallIR, CompositeIR, ProofIR
]


@dataclass
class IRCandidate:
    ir: CognitiveIR
    confidence: float
    parser: str
    ambiguity: float = 0.0
    missing_fields: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    model_score: float = 0.0
    memory_score: float = 0.0

    @property
    def total_score(self) -> float:
        return self.confidence + self.model_score + self.memory_score - self.ambiguity - 0.2 * len(self.validation_errors)
