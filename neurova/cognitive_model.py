from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
import re
from typing import Dict, List, Sequence
from .ir import ClaimIR, CausalClaimIR, ComparisonIR, IRCandidate, QuestionIR

def toks(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_가-힣]+", s.lower())

@dataclass
class CognitiveModelReport:
    backend: str
    objective: str
    ranked: bool
    selected_type: str
    score: float

class NeuralCognitiveCompiler:
    """Not an LLM: no next-token loss, no text generation.
    Learns/ranks: IR candidates, evidence relevance, proof operator, world transition, memory utility.
    """
    name = "neural_cognitive_compiler_no_lm"
    def __init__(self):
        self.type_weights: Dict[str, float] = defaultdict(float)
        self.token_type_weights: Dict[str, Counter[str]] = defaultdict(Counter)
        self.operator_counts: Counter[str] = Counter()
        self.transition_counts: Counter[tuple[str, str]] = Counter()
        self.memory_utility: Counter[str] = Counter()
        self.updates = 0

    def update_from_selection(self, text: str, selected_type: str, reward: float = 1.0):
        self.type_weights[selected_type] += 0.05 * reward
        for t in toks(text):
            self.token_type_weights[t][selected_type] += reward
        self.updates += 1

    def rank_ir_candidates(self, text: str, candidates: List[IRCandidate], memory=None) -> List[IRCandidate]:
        tokens = toks(text)
        for c in candidates:
            typ = type(c.ir).__name__
            token_score = sum(self.token_type_weights[t][typ] for t in tokens) / max(1, len(tokens))
            priority = 1.0 if any(str(n).startswith(("v22_adaptive_language_priority", "adaptive_construction_priority")) for n in getattr(c, "notes", [])) else 0.0
            pname = str(getattr(c, "parser", ""))
            if pname.startswith("v30_"):
                priority += 4.0
            elif pname.startswith("v29_"):
                priority += 2.25
            elif pname.startswith("v28_"):
                priority += 0.65
            elif pname.startswith("v27_"):
                priority += 0.35
            if typ in {"EventFrameIR", "WrapperConstructionIR", "TemporalQuerySchemaIR", "MetaMemoryQuestionIR", "SupportRequestIR", "SpeechActIR"}:
                priority += 0.85
            c.model_score = self.type_weights[typ] + 0.03 * token_score + priority
            c.memory_score = self._memory_score(c, memory)
        return sorted(candidates, key=lambda x: x.total_score, reverse=True)

    def _memory_score(self, c: IRCandidate, memory) -> float:
        if not memory: return 0.0
        ir = c.ir
        try:
            if isinstance(ir, QuestionIR) and isinstance(ir.target, ClaimIR):
                pos = memory.find_claim(ir.target.subject, ir.target.relation, ir.target.object, "positive")
                neg = memory.find_claim(ir.target.subject, ir.target.relation, ir.target.object, "negative")
                return 0.08 * bool(pos) + 0.04 * bool(neg)
            if isinstance(ir, ClaimIR):
                opposite = "negative" if ir.polarity == "positive" else "positive"
                return -0.1 if memory.find_claim(ir.subject, ir.relation, ir.object, opposite) else 0.02
        except Exception:
            return 0.0
        return 0.0

    def score_evidence(self, query: str, evidence: str, reliability: float = 0.6) -> float:
        q, e = set(toks(query)), set(toks(evidence))
        return 0.65 * (len(q & e) / max(1, len(q))) + 0.35 * reliability

    def choose_proof_operator(self, query_type: str, state: dict) -> str:
        if query_type == "comparison": op = "comparison_graph_search"
        elif query_type == "causal": op = "causal_graph_search"
        elif state.get("negated"): op = "refutation_search"
        elif state.get("taxonomy"): op = "taxonomy_forward"
        else: op = "active_retrieve_then_unknown"
        self.operator_counts[op] += 1
        return op

    def observe_transition(self, action: str, effect: str, reward: float = 1.0):
        self.transition_counts[(action.lower(), effect.lower())] += reward

    def predict_world_transition(self, action: str, candidate_effects: Sequence[str]):
        total = sum(v for (a, _), v in self.transition_counts.items() if a == action.lower())
        rows = []
        for e in candidate_effects:
            rows.append((e, (self.transition_counts[(action.lower(), e.lower())] + 1) / (total + len(candidate_effects) + 1)))
        return sorted(rows, key=lambda x: x[1], reverse=True)

    def score_memory_utility(self, key: str, success: bool) -> float:
        self.memory_utility[key] += 1.0 if success else -0.5
        return self.memory_utility[key]

    def report_for(self, selected: IRCandidate) -> CognitiveModelReport:
        return CognitiveModelReport(self.name, "IR/evidence/proof/world/memory scoring; no autoregressive text-generation objective", True, type(selected.ir).__name__, selected.total_score)
