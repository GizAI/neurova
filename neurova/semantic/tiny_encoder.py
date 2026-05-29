from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import hashlib, math, re
from typing import Dict, Iterable, List, Tuple


def _features(text: str) -> List[str]:
    low = text.lower().strip()
    toks = re.findall(r"[A-Za-z0-9_가-힣]+", low)
    feats = ["bias"]
    feats += ["tok=" + t for t in toks]
    feats += ["bi=" + a + "_" + b for a, b in zip(toks, toks[1:])]
    # Character ngrams make the model useful for tiny corpora and Korean particles.
    compact = re.sub(r"\s+", " ", low)
    for n in (2, 3, 4):
        for i in range(max(0, len(compact) - n + 1)):
            feats.append(f"ch{n}=" + compact[i:i+n])
    return feats


@dataclass
class TypeScore:
    ir_type: str
    score: float


class TinySemanticEncoder:
    """Tiny no-LLM structured encoder for semantic parsing.

    It is deliberately not an autoregressive LM: no decoder, no autoregressive loss,
    and no text generation. It learns a linear classifier over token/character
    features so small corpora can bias IR candidate ranking.
    """
    objective = "hashed feature IR-type classification only; no autoregressive text generation"

    def __init__(self):
        self.weights: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.labels: set[str] = set()
        self.updates = 0

    def fit(self, rows: Iterable[dict], epochs: int = 3) -> "TinySemanticEncoder":
        data = [(r.get("text", ""), r.get("ir_type", "")) for r in rows if r.get("text") and r.get("ir_type")]
        for _, y in data:
            self.labels.add(y)
        for _ in range(max(1, epochs)):
            for text, gold in data:
                pred = self.predict(text, top_k=1)[0].ir_type if self.labels else gold
                if pred != gold:
                    feats = _features(text)
                    for f in feats:
                        self.weights[gold][f] += 1.0
                        self.weights[pred][f] -= 1.0
                    self.updates += 1
        return self

    def score(self, text: str, label: str) -> float:
        feats = _features(text)
        w = self.weights.get(label, {})
        return sum(w.get(f, 0.0) for f in feats) / math.sqrt(max(1, len(feats)))

    def predict(self, text: str, top_k: int = 5) -> List[TypeScore]:
        if not self.labels:
            return []
        rows = [TypeScore(label, self.score(text, label)) for label in self.labels]
        return sorted(rows, key=lambda r: r.score, reverse=True)[:top_k]
