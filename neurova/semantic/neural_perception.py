from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
import math, re
from typing import Dict, Iterable, List, Tuple


TOKEN_RE = re.compile(r"[A-Za-z0-9_가-힣]+")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def features(text: str) -> List[str]:
    low = normalize(text)
    toks = TOKEN_RE.findall(low)
    out = ["bias", f"len={min(12, len(toks))}"]
    out.extend("tok=" + t for t in toks)
    out.extend("bi=" + a + "_" + b for a, b in zip(toks, toks[1:]))
    out.extend("tri=" + a + "_" + b + "_" + c for a, b, c in zip(toks, toks[1:], toks[2:]))
    compact = re.sub(r"\s+", " ", low)
    for n in (2, 3, 4, 5):
        for i in range(max(0, len(compact) - n + 1)):
            out.append(f"ch{n}=" + compact[i:i+n])
    # Linguistic feature hints, still non-autoregressive.
    hints = {
        "has_question": "?" in text or (toks[:1] and toks[0] in {"is", "can", "could", "would", "does", "do", "who", "what"}),
        "has_negation": any(t in {"not", "no", "cannot", "can't", "않다", "아니다"} for t in toks),
        "has_temporal": any(re.fullmatch(r"\d{4}", t) for t in toks) or any(t in {"during", "from", "until", "through"} for t in toks),
        "has_causal": any(t in {"because", "causes", "cause", "caused", "leads", "sparks", "brings", "때문에", "오면"} for t in toks),
        "has_event": any(t in {"gave", "give", "handed", "sent", "received", "opened", "closed", "moved", "collected", "asked", "ordered"} for t in toks),
        "has_belief": any(t in {"believes", "believe", "thinks", "think", "knows", "know"} for t in toks),
        "has_goal": any(t in {"wants", "want", "intends", "plans"} for t in toks),
    }
    out.extend(k for k, v in hints.items() if v)
    return out


@dataclass
class SemanticScore:
    label: str
    score: float


@dataclass
class NeuralSemanticPerception:
    """Tiny structured neural-style semantic perception.

    This is deliberately not an LLM: it has no decoder, no next-token loss, and no
    autoregressive generation.  It is an online linear structured predictor used to
    bias IR candidate ranking, event-frame role decisions, and correction routing.
    """
    weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    labels: set[str] = field(default_factory=set)
    updates: int = 0
    objective: str = "structured semantic perception only; no autoregressive language-model objective"

    def fit(self, rows: Iterable[dict], epochs: int = 2) -> "NeuralSemanticPerception":
        data = [(r.get("text", ""), r.get("ir_type") or r.get("label", "")) for r in rows if r.get("text") and (r.get("ir_type") or r.get("label"))]
        for _, y in data:
            self.labels.add(y)
        for _ in range(max(1, epochs)):
            for text, gold in data:
                pred = self.predict(text, 1)[0].label if self.labels else gold
                if pred != gold:
                    fs = features(text)
                    for f in fs:
                        self.weights[gold][f] += 1.0
                        self.weights[pred][f] -= 1.0
                    self.updates += 1
        return self

    def score(self, text: str, label: str) -> float:
        fs = features(text)
        w = self.weights.get(label, {})
        return sum(w.get(f, 0.0) for f in fs) / math.sqrt(max(1, len(fs)))

    def predict(self, text: str, top_k: int = 5) -> List[SemanticScore]:
        if not self.labels:
            return []
        rows = [SemanticScore(label, self.score(text, label)) for label in self.labels]
        return sorted(rows, key=lambda r: r.score, reverse=True)[:top_k]

    def candidate_bias(self, text: str, ir_type: str) -> float:
        if ir_type not in self.labels:
            return 0.0
        s = self.score(text, ir_type)
        # bound contribution so symbolic/verifier layers remain in control
        return max(-0.25, min(0.45, s / 6.0))

    def explain(self, text: str, label: str, top_n: int = 8) -> List[Tuple[str, float]]:
        fs = features(text)
        w = self.weights.get(label, {})
        rows = [(f, w.get(f, 0.0)) for f in fs if abs(w.get(f, 0.0)) > 0]
        return sorted(rows, key=lambda x: abs(x[1]), reverse=True)[:top_n]
