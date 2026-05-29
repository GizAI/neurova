from __future__ import annotations

"""V35 semantic perception layer.

This is deliberately not an autoregressive language model.  It is a compact
semantic encoder/retriever used to propose likely schemas, dialogue acts, and
IR families.  If torch is available it can host a small encoder later; the
portable default is deterministic hashed n-gram features with cosine search.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple
import hashlib
import math
import re


def _norm(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[?.!,]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9가-힣']+", _norm(text))


def _hash(s: str, dims: int) -> int:
    return int(hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest(), 16) % dims


@dataclass
class SemanticVector:
    dims: int
    weights: Dict[int, float] = field(default_factory=dict)

    def add(self, idx: int, value: float) -> None:
        self.weights[idx] = self.weights.get(idx, 0.0) + value

    def norm(self) -> float:
        return math.sqrt(sum(v * v for v in self.weights.values())) or 1.0

    def cosine(self, other: "SemanticVector") -> float:
        if self.dims != other.dims:
            return 0.0
        if len(self.weights) > len(other.weights):
            return other.cosine(self)
        dot = sum(v * other.weights.get(i, 0.0) for i, v in self.weights.items())
        return dot / (self.norm() * other.norm())


@dataclass
class SemanticItem:
    item_id: str
    kind: str
    text: str
    payload: Dict[str, Any]
    vector: SemanticVector


class DeepSemanticEncoder:
    """Small semantic perception interface.

    The class is intentionally generation-free.  It supports semantic retrieval,
    failure clustering, and coarse dialogue/IR intent hints.  A future GPU model
    can replace `encode()` without changing callers.
    """

    def __init__(self, dims: int = 256):
        self.dims = dims
        self.backend = "hashed_ngram_semantic_encoder"
        try:
            import torch  # noqa: F401
            self.torch_available = True
        except Exception:
            self.torch_available = False

    def encode(self, text: str) -> SemanticVector:
        toks = _tokens(text)
        v = SemanticVector(self.dims)
        for t in toks:
            v.add(_hash("tok:" + t, self.dims), 1.0)
            if len(t) > 3:
                v.add(_hash("stem:" + self._stem(t), self.dims), 0.6)
        for n in (2, 3):
            for i in range(max(0, len(toks) - n + 1)):
                gram = " ".join(toks[i:i+n])
                v.add(_hash(f"{n}gram:" + gram, self.dims), 1.2 if n == 2 else 1.4)
        # Semantic cue buckets.  These are not answer rules; they help retrieval.
        cue_map = {
            "question": ["would", "could", "can", "does", "did", "who", "where", "what", "is", "?"],
            "support": ["stuck", "confused", "worried", "rough", "help", "lost"],
            "taxonomy": ["classify", "regarded", "considered", "type", "kind", "category", "under", "member"],
            "event_move": ["move", "moved", "carry", "carried", "transport", "transported", "from", "to", "where"],
            "belief": ["believe", "believes", "think", "thinks", "that"],
            "negation": ["not", "n't", "cannot", "unlikely", "hardly", "failed"],
            "korean_compare": ["보다", "앞", "우세", "크", "낫", "뒤처"],
        }
        low = text.lower()
        for bucket, cues in cue_map.items():
            if any(c in low for c in cues):
                v.add(_hash("cue:" + bucket, self.dims), 2.0)
        return v

    def _stem(self, t: str) -> str:
        for suf in ("ing", "ed", "es", "s"):
            if len(t) > len(suf) + 2 and t.endswith(suf):
                return t[: -len(suf)]
        return t

    def classify_dialogue_act(self, text: str) -> Tuple[str, float]:
        low = text.lower()
        if any(x in low for x in ["stuck", "confused", "worried", "rough day", "cheer me", "not sure what"]):
            return "support_request", 0.86
        if any(x in low for x in ["haha", "lol", "hilarious", "wild", "nice"]):
            return "smalltalk", 0.78
        if any(x in low for x in ["no,", "actually", "i mean", "correction", "should be understood"]):
            return "correction", 0.82
        if re.match(r"^(who|what|where|when|why|how|does|did|is|can|could|would)\b", low):
            return "question", 0.76
        return "assertion", 0.55

    def infer_ir_family(self, text: str) -> Tuple[str, float]:
        low = text.lower()
        if any(x in low for x in ["greater", "ahead", "outclass", "outrank", "behind", "보다", "우세"]):
            return "ComparisonIR", 0.74
        if any(x in low for x in ["cause", "causes", "trigger", "leads to", "because"]):
            return "CausalClaimIR", 0.72
        if any(x in low for x in ["believe", "think", "thinks"]):
            return "BeliefIR", 0.70
        if any(x in low for x in ["from", "to", "moved", "carried", "bought", "sold", "gave", "received"]):
            return "EventIR", 0.67
        if any(x in low for x in ["in 20", "during", "from 20", "through", "before"]):
            return "TemporalClaimIR", 0.66
        return "ClaimIR", 0.50


class SemanticMemoryIndex:
    def __init__(self, encoder: Optional[DeepSemanticEncoder] = None):
        self.encoder = encoder or DeepSemanticEncoder()
        self.items: List[SemanticItem] = []

    def add(self, item_id: str, kind: str, text: str, payload: Optional[Dict[str, Any]] = None) -> None:
        self.items.append(SemanticItem(item_id, kind, text, payload or {}, self.encoder.encode(text)))

    def search(self, text: str, kind: Optional[str] = None, top_k: int = 5) -> List[Tuple[SemanticItem, float]]:
        q = self.encoder.encode(text)
        scored = []
        for item in self.items:
            if kind and item.kind != kind:
                continue
            scored.append((item, q.cosine(item.vector)))
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

    def cluster_failures(self, failures: Iterable[str], threshold: float = 0.35) -> List[List[str]]:
        clusters: List[List[str]] = []
        vectors: List[SemanticVector] = []
        for f in failures:
            fv = self.encoder.encode(f)
            placed = False
            for i, cv in enumerate(vectors):
                if fv.cosine(cv) >= threshold:
                    clusters[i].append(f)
                    # cheap centroid update
                    for idx, val in fv.weights.items():
                        cv.add(idx, val / max(1, len(clusters[i])))
                    placed = True
                    break
            if not placed:
                clusters.append([f])
                vectors.append(fv)
        return clusters


# ===========================================================================
# V36 Evolution: Schema Retrieval, Text<->IR Contrastive, Role Tagging
# ===========================================================================

class SchemaRetrievalIndex(SemanticMemoryIndex):
    """Specialized index for schema retrieval by embedding similarity.

    Stores schemas with their forms, meaning, and type for fast lookup.
    When a new utterance comes in, retrieve the most similar schemas
    to guide parsing before falling back to regex patterns.
    """

    def add_schema(self, schema_id: str, forms: list, meaning: str, schema_type: str, confidence: float = 0.5) -> None:
        text = " ".join(forms)
        self.add(schema_id, "schema", text, {
            "meaning": meaning,
            "schema_type": schema_type,
            "forms": forms,
            "confidence": confidence,
        })

    def retrieve_schemas(self, text: str, top_k: int = 5, min_score: float = 0.15) -> list:
        results = self.search(text, kind="schema", top_k=top_k)
        return [
            {
                "schema_id": item.item_id,
                "score": round(score, 4),
                "schema_type": item.payload.get("schema_type", ""),
                "meaning": item.payload.get("meaning", ""),
                "forms": item.payload.get("forms", []),
                "confidence": item.payload.get("confidence", 0.0),
            }
            for item, score in results
            if score >= min_score
        ]


class ContrastiveLearningBuffer:
    """Accumulates text<->IR pairs for contrastive learning.

    Positive pairs: (text, correct IR type/meaning).
    Hard negatives: (similar text, wrong IR type/meaning).
    When enough pairs accumulate, they can train the semantic encoder.
    """

    def __init__(self, max_size: int = 10000):
        self.positives: list = []  # [(text, ir_type, meaning)]
        self.negatives: list = []  # [(text, wrong_ir_type)]
        self.max_size = max_size

    def add_positive(self, text: str, ir_type: str, meaning: str = "") -> None:
        self.positives.append((text, ir_type, meaning))
        if len(self.positives) > self.max_size:
            self.positives = self.positives[-self.max_size:]

    def add_negative(self, text: str, wrong_ir_type: str) -> None:
        self.negatives.append((text, wrong_ir_type))
        if len(self.negatives) > self.max_size:
            self.negatives = self.negatives[-self.max_size:]

    @property
    def size(self) -> int:
        return len(self.positives) + len(self.negatives)

    def export_pairs(self) -> dict:
        return {
            "positives": self.positives[-100:],
            "negatives": self.negatives[-100:],
            "total_positives": len(self.positives),
            "total_negatives": len(self.negatives),
        }


class SlotRoleTagger:
    """Lightweight slot/role tagger for event frames and construction schemas.

    Uses keyword and position heuristics. A future GPU model can replace
    the tag() method without changing callers.
    """

    ROLE_CUES = {
        "agent": ["gave", "sent", "moved", "carried", "bought", "asked", "told", "opened"],
        "patient": ["a", "the", "it"],
        "recipient": ["to", "from"],
        "location": ["in", "at", "from", "to"],
        "time": ["yesterday", "today", "tomorrow", "ago", "in 20"],
        "instrument": ["with", "using", "by means of"],
    }

    def tag(self, text: str) -> list:
        """Return a list of (token, role) pairs."""
        import re as _re
        tokens = _re.findall(r"[a-zA-Z가-힣0-9']+|[.,!?]", text)
        result = []
        for i, tok in enumerate(tokens):
            low = tok.lower()
            role = "O"
            for r, cues in self.ROLE_CUES.items():
                if low in cues:
                    role = f"B-{r}_cue"
                    break
            result.append((tok, role))
        return result
