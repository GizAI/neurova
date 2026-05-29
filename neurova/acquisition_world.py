from __future__ import annotations

"""V36 language-acquisition substrate.

This module is intentionally not another sentence-pattern parser.  It adds an
object-centric situation model and a lightweight embedding-backed recall layer.
Text is interpreted as updates to entity state, event frames, spatial relations,
beliefs, uncertainty and prediction errors.  Triples may be projected for search,
but EventFrame/SituationFrame records remain the source of truth.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import math
import re
import time


_STOP = {"the", "a", "an", "of", "to", "in", "on", "at", "by", "from", "and", "or", "is", "was", "are", "were", "did", "does", "what", "which", "where", "when", "who", "how"}


def norm(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201c\u201d]", '"', text)
    text = re.sub(r"[?.!,;:]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def singular(x: str) -> str:
    x = norm(x)
    x = re.sub(r"^(a|an|the)\s+", "", x)
    if x.endswith("ies") and len(x) > 4:
        return x[:-3] + "y"
    if x.endswith(("ches", "shes", "xes", "ses", "zes")) and len(x) > 4:
        return x[:-2]
    if x.endswith("s") and len(x) > 3 and not x.endswith("ss"):
        return x[:-1]
    return x


def titleish(x: str) -> str:
    if not x:
        return x
    x = re.sub(r"^(a|an|the)\s+", "", x.strip(), flags=re.I)
    return " ".join(w.capitalize() if w not in {"of", "the"} else w for w in x.split())


@dataclass
class SemanticVector:
    dims: int = 384
    weights: Dict[int, float] = field(default_factory=dict)

    def add(self, key: str, value: float = 1.0) -> None:
        idx = int(hashlib.blake2b(key.encode("utf-8"), digest_size=8).hexdigest(), 16) % self.dims
        self.weights[idx] = self.weights.get(idx, 0.0) + value

    def norm(self) -> float:
        return math.sqrt(sum(v*v for v in self.weights.values())) or 1.0

    def cosine(self, other: "SemanticVector") -> float:
        if len(self.weights) > len(other.weights):
            return other.cosine(self)
        dot = sum(v * other.weights.get(i, 0.0) for i, v in self.weights.items())
        return dot / (self.norm() * other.norm())


class EmbeddingAssociativeMemory:
    """Small deterministic embedding memory.

    It is not a neural generator.  It provides language-level association for
    paraphrase retrieval, failure clustering and schema lookup.
    """

    def __init__(self, dims: int = 384):
        self.dims = dims
        self.items: List[Tuple[str, str, Dict[str, Any], SemanticVector]] = []

    def encode(self, text: str) -> SemanticVector:
        toks = [t for t in re.findall(r"[a-zA-Z0-9가-힣']+", norm(text)) if t not in _STOP]
        v = SemanticVector(self.dims)
        for t in toks:
            v.add("tok:" + t, 1.0)
            if len(t) > 3:
                v.add("stem:" + singular(t), 0.6)
        for n in (2, 3):
            for i in range(max(0, len(toks)-n+1)):
                v.add(f"{n}gram:" + " ".join(toks[i:i+n]), 1.1 if n == 2 else 1.35)
        # Coarse semantic cue features. These are retrieval cues, not answers.
        cue_sets = {
            "spatial": ["north", "south", "east", "west", "northeast", "southeast", "lies", "border", "separated", "separator"],
            "time": ["when", "year", "became", "independent", "during", "from", "to"],
            "move": ["went", "moved", "put", "placed", "located", "where"],
            "coref": ["it", "she", "he", "they", "region"],
            "cause": ["because", "caused", "causes", "therefore"],
        }
        low = norm(text)
        for cue, words in cue_sets.items():
            if any(w in low for w in words):
                v.add("cue:" + cue, 2.0)
        return v

    def add(self, item_id: str, text: str, payload: Optional[Dict[str, Any]] = None) -> None:
        self.items.append((item_id, text, payload or {}, self.encode(text)))

    def search(self, text: str, top_k: int = 5) -> List[Tuple[str, str, Dict[str, Any], float]]:
        q = self.encode(text)
        scored = [(iid, txt, payload, q.cosine(vec)) for iid, txt, payload, vec in self.items]
        return sorted(scored, key=lambda r: r[3], reverse=True)[:top_k]

    def cluster(self, texts: List[str], threshold: float = 0.35) -> List[List[str]]:
        clusters: List[List[str]] = []
        centroids: List[SemanticVector] = []
        for text in texts:
            vec = self.encode(text)
            placed = False
            for i, c in enumerate(centroids):
                if vec.cosine(c) >= threshold:
                    clusters[i].append(text)
                    for idx, value in vec.weights.items():
                        c.weights[idx] = c.weights.get(idx, 0.0) + value / max(1, len(clusters[i]))
                    placed = True
                    break
            if not placed:
                clusters.append([text])
                centroids.append(vec)
        return clusters


@dataclass
class EntityState:
    entity: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    fluents: Dict[str, Any] = field(default_factory=dict)
    belief: float = 0.7


@dataclass
class SituationFrame:
    frame_type: str
    roles: Dict[str, Any]
    source_span: str
    confidence: float = 0.7
    timestamp: float = field(default_factory=time.time)


@dataclass
class PredictionErrorRecord:
    text: str
    predicted: str
    observed: str
    error_type: str
    severity: float = 0.5


class ObjectCentricStateSpaceWorldModel:
    """Object-centric neuro-symbolic state-space world model.

    Source of truth: entity states + situation/event frames.
    Search triples can be projected, but do not replace EventFrames.
    """

    def __init__(self):
        self.entities: Dict[str, EntityState] = {}
        self.frames: List[SituationFrame] = []
        self.prediction_errors: List[PredictionErrorRecord] = []
        self.embeddings = EmbeddingAssociativeMemory()
        self.referents: Dict[str, str] = {}
        self.last_entity: Optional[str] = None
        self.last_object: Optional[str] = None
        self.discourse_topic: Optional[str] = None

    def entity(self, name: str) -> EntityState:
        key = singular(name)
        if key not in self.entities:
            self.entities[key] = EntityState(entity=key)
        self.last_entity = key
        if self.discourse_topic is None and key not in {"it", "region", "the region"}:
            self.discourse_topic = key
        return self.entities[key]

    def add_frame(self, frame_type: str, roles: Dict[str, Any], source_span: str, confidence: float = 0.75) -> SituationFrame:
        clean_roles = {k: singular(v) if isinstance(v, str) and k not in {"source_span", "separator", "direction", "time"} else v for k, v in roles.items()}
        frame = SituationFrame(frame_type=frame_type, roles=clean_roles, source_span=source_span.strip(), confidence=confidence)
        self.frames.append(frame)
        self.embeddings.add(f"frame:{len(self.frames)}", source_span, {"frame_type": frame_type, "roles": clean_roles})
        self._apply_frame(frame)
        return frame

    def _apply_frame(self, frame: SituationFrame) -> None:
        r = frame.roles
        ft = frame.frame_type
        if ft == "state":
            ent = self.entity(r["entity"])
            ent.fluents[r["slot"]] = r["value"]
            if r["slot"] in {"location", "located_at"}:
                self.last_object = ent.entity
        elif ft == "move":
            ent = self.entity(r["entity"])
            ent.fluents["location"] = r["to"]
            ent.fluents["located_at"] = r["to"]
            ent.fluents["last_move"] = {"from": r.get("from"), "to": r["to"]}
            self.last_object = ent.entity
        elif ft == "spatial_separation":
            e1 = self.entity(r["entity_1"])
            e2 = self.entity(r["entity_2"])
            e1.attributes.setdefault("separated_from", {})[e2.entity] = {"separator": r.get("separator"), "direction_of_other": r.get("direction")}
            e2.attributes.setdefault("relative_to", {})[e1.entity] = r.get("direction")
            self.last_entity = e1.entity
            self.discourse_topic = self.discourse_topic or e1.entity
        elif ft == "spatial_relation":
            subject = self.entity(r["subject"])
            rel_to = self.entity(r["relative_to"])
            subject.attributes.setdefault("relative_to", {})[rel_to.entity] = r["direction"]
            rel_to.attributes.setdefault("directional_neighbors", {})[r["direction"]] = subject.entity
        elif ft == "temporal_state":
            ent = self.entity(r["entity"])
            ent.attributes[r["state"]] = r.get("time")
            ent.fluents[r["state"]] = r.get("time")
        elif ft == "possession":
            owner = self.entity(r["owner"])
            obj = singular(r["object"])
            owner.attributes.setdefault("has", set()).add(obj)
            self.last_object = obj
        elif ft == "containment":
            container = self.entity(r["container"])
            obj = singular(r["object"])
            container.attributes.setdefault("contains", set()).add(obj)
            self.entity(obj).fluents["inside"] = container.entity
            self.last_object = obj

    def remember_coref(self, phrase: str, entity: str) -> None:
        self.referents[norm(phrase)] = singular(entity)

    def resolve_ref(self, phrase: str) -> str:
        low = norm(phrase)
        if low in self.referents:
            return self.referents[low]
        if low in {"it", "the region", "region"} and self.last_entity:
            return self.last_entity
        if low in {"there"} and self.last_object:
            return self.entities.get(self.last_object, EntityState(self.last_object)).fluents.get("location", low)
        return singular(phrase)

    def answer(self, question: str) -> str:
        low = norm(question)
        # Separator / spatial separation queries.
        m = re.search(r"what\s+separates\s+(.+?)\s+from\s+(.+)$", low)
        if m:
            a, b = map(self.resolve_ref, m.groups())
            ans = self.separator_between(a, b)
            return ans or "unknown"
        m = re.search(r"what\s+direction\s+is\s+(.+?)\s+(?:relative\s+to|from)\s+(.+)$", low)
        if m:
            a, b = map(self.resolve_ref, m.groups())
            return self.direction_between(a, b) or "unknown"
        m = re.search(r"which\s+(?:country|entity|place)\s+lies\s+to\s+the\s+([a-z]+)\s+of\s+(.+)$", low)
        if m:
            direction, rel = m.groups()
            rel = self.resolve_ref(rel)
            ent = self.entities.get(rel)
            if ent:
                return ent.attributes.get("directional_neighbors", {}).get(direction, "unknown")
            return "unknown"
        m = re.search(r"when\s+did\s+(.+?)\s+become\s+(.+)$", low)
        if m:
            ent, state = map(self.resolve_ref, m.groups())
            e = self.entities.get(ent)
            if e:
                return str(e.attributes.get(state, e.fluents.get(state, "unknown")))
            return "unknown"
        m = re.search(r"where\s+is\s+(.+)$", low)
        if m:
            ent = self.resolve_ref(m.group(1))
            e = self.entities.get(ent)
            if e:
                return str(e.fluents.get("location", e.fluents.get("located_at", "unknown")))
            return "unknown"
        m = re.search(r"does\s+(.+?)\s+have\s+(.+)$", low)
        if m:
            owner, obj = map(self.resolve_ref, m.groups())
            e = self.entities.get(owner)
            if e and singular(obj) in e.attributes.get("has", set()):
                return "yes"
            return "unknown"
        # Fallback associative retrieval.
        hits = self.embeddings.search(question, top_k=1)
        if hits and hits[0][3] > 0.35:
            return f"related:{hits[0][2].get('frame_type','memory')}:{hits[0][1]}"
        return "unknown"

    def separator_between(self, a: str, b: str) -> Optional[str]:
        a = singular(a); b = singular(b)
        for x, y in [(a, b), (b, a)]:
            ent = self.entities.get(x)
            if ent:
                sep = ent.attributes.get("separated_from", {}).get(y)
                if sep:
                    return str(sep.get("separator"))
        return None

    def direction_between(self, a: str, b: str) -> Optional[str]:
        a = singular(a); b = singular(b)
        ent = self.entities.get(a)
        if ent and b in ent.attributes.get("relative_to", {}):
            return str(ent.attributes["relative_to"][b])
        entb = self.entities.get(b)
        if entb:
            # If b stores a neighbor direction for a, return inverse perspective when known.
            neighbors = entb.attributes.get("directional_neighbors", {})
            for direction, who in neighbors.items():
                if who == a:
                    return direction
        return None

    def project_triples(self) -> List[Tuple[str, str, str]]:
        """Search index projection. Not source of truth."""
        triples: List[Tuple[str, str, str]] = []
        for name, ent in self.entities.items():
            for k, v in ent.fluents.items():
                if isinstance(v, (str, int, float)):
                    triples.append((name, k, str(v)))
            for other, data in ent.attributes.get("separated_from", {}).items():
                triples.append((name, "separated_from", other))
                if data.get("separator"):
                    triples.append((name, "separated_by", str(data["separator"])))
        return triples


class SituationModelBuilder:
    """Predictive semantic builder for simple text streams.

    This does use regular expressions internally, but only as high-precision seed
    extractors for learnable cognitive priors: entity, event, role, time, space,
    cause, goal, state update. It stores EventFrames/SituationFrames as source of
    truth rather than flat triples.
    """

    def __init__(self, world: Optional[ObjectCentricStateSpaceWorldModel] = None):
        self.world = world or ObjectCentricStateSpaceWorldModel()

    def ingest(self, text: str) -> List[SituationFrame]:
        frames: List[SituationFrame] = []
        # Sentence segmentation with context preserved.
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|;\s*", text.strip()) if p.strip()]
        for part in parts:
            frames.extend(self._ingest_sentence(part))
        return frames

    def _ingest_sentence(self, sentence: str) -> List[SituationFrame]:
        raw = sentence.strip()
        low = norm(raw)
        frames: List[SituationFrame] = []
        # Entity type with embedded location, e.g. "Korea is a peninsula in East Asia".
        m = re.match(r"^(.+?)\s+(?:is|was)\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+in\s+(.+)$", low)
        if m and not re.fullmatch(r"\d{3,4}", m.group(3)):
            ent, typ, loc = m.groups()
            frames.append(self.world.add_frame("state", {"entity": ent, "slot": "is_a", "value": typ}, raw, 0.78))
            frames.append(self.world.add_frame("state", {"entity": ent, "slot": "location", "value": loc}, raw, 0.78))
            self.world.remember_coref("it", ent)
            self.world.remember_coref("the region", ent)
            return frames
        # X is in/on/at PLACE.  Avoid treating years as places.
        m = re.match(r"^(.+?)\s+(?:is|was)\s+(?:in|on|at)\s+(.+)$", low)
        if m:
            ent, loc = m.groups()
            if not re.fullmatch(r"\d{3,4}", loc):
                frames.append(self.world.add_frame("state", {"entity": ent, "slot": "location", "value": loc}, raw, 0.82))
                self.world.remember_coref("it", ent)
                self.world.remember_coref("the region", ent)
                return frames
        # X went/moved/traveled from A to B OR went to B.
        m = re.match(r"^(.+?)\s+(?:went|moved|travelled|traveled)\s+from\s+(.+?)\s+to\s+(.+)$", low)
        if m:
            ent, src, dst = m.groups()
            frames.append(self.world.add_frame("move", {"entity": ent, "from": src, "to": dst}, raw, 0.86))
            return frames
        m = re.match(r"^(.+?)\s+(?:went|moved|travelled|traveled)\s+to\s+(.+)$", low)
        if m:
            ent, dst = m.groups()
            prev = self.world.entities.get(singular(ent), EntityState(singular(ent))).fluents.get("location")
            frames.append(self.world.add_frame("move", {"entity": ent, "from": prev, "to": dst}, raw, 0.84))
            return frames
        # Spatial separation: entity separated from other to direction by separator.
        m = re.match(r"^(?:it|the region|.+?)\s+(?:is|was)?\s*separated\s+from\s+(.+?)\s+to\s+the\s+([a-z]+)\s+by\s+(.+)$", low)
        if m:
            entity1 = self.world.referents.get("it") or self.world.discourse_topic or self.world.resolve_ref("it")
            other, direction, sep = m.groups()
            frames.append(self.world.add_frame("spatial_separation", {"entity_1": entity1, "entity_2": other, "direction": direction, "separator": titleish(sep)}, raw, 0.88))
            return frames
        # X lies to the north/northeast of Y.
        m = re.match(r"^(.+?)\s+(?:lies|is|are)\s+to\s+the\s+([a-z]+)\s+of\s+(.+)$", low)
        if m:
            subj, direction, rel = m.groups()
            frames.append(self.world.add_frame("spatial_relation", {"subject": subj, "direction": direction, "relative_to": self.world.resolve_ref(rel)}, raw, 0.86))
            return frames
        # Parallel geography: China to north and Russia to northeast.
        m = re.match(r"^(.+?)\s+to\s+the\s+([a-z]+)\s+and\s+(.+?)\s+to\s+the\s+([a-z]+)$", low)
        if m:
            a, dir_a, b, dir_b = m.groups()
            rel = self.world.referents.get("it") or self.world.discourse_topic or self.world.last_entity or "region"
            frames.append(self.world.add_frame("spatial_relation", {"subject": a, "direction": dir_a, "relative_to": rel}, raw, 0.80))
            frames.append(self.world.add_frame("spatial_relation", {"subject": b, "direction": dir_b, "relative_to": rel}, raw, 0.80))
            return frames
        # Entity became state in YEAR. This is temporal state, not location.
        m = re.match(r"^(.+?)\s+became\s+(.+?)\s+in\s+(\d{3,4})$", low)
        if m:
            ent, state, year = m.groups()
            frames.append(self.world.add_frame("temporal_state", {"entity": self.world.resolve_ref(ent), "state": state, "time": year}, raw, 0.87))
            return frames
        # Possession transfer and containment.
        m = re.match(r"^(.+?)\s+(?:picked\s+up|took|grabbed)\s+(.+)$", low)
        if m:
            actor, obj = m.groups()
            frames.append(self.world.add_frame("possession", {"owner": actor, "object": obj}, raw, 0.78))
            self.world.remember_coref("it", obj)
            return frames
        m = re.match(r"^(.+?)\s+put\s+(.+?)\s+(?:on|in|inside)\s+(.+)$", low)
        if m:
            actor, obj, place = m.groups()
            obj = self.world.resolve_ref(obj)
            frames.append(self.world.add_frame("move", {"entity": obj, "from": None, "to": place}, raw, 0.82))
            return frames
        m = re.match(r"^inside\s+(.+?)\s+was\s+(.+)$", low)
        if m:
            container, obj = m.groups()
            frames.append(self.world.add_frame("containment", {"container": self.world.resolve_ref(container), "object": obj}, raw, 0.80))
            return frames
        # If nothing matches, record a prediction error for active learning.
        self.world.prediction_errors.append(PredictionErrorRecord(text=raw, predicted="situation_frame", observed="unparsed", error_type="semantic_parser_gap", severity=0.4))
        self.world.embeddings.add(f"unparsed:{len(self.world.prediction_errors)}", raw, {"frame_type": "unparsed"})
        return frames

    def answer(self, question: str) -> str:
        return self.world.answer(question)


class LanguageAcquisitionSubstrate:
    """Public facade for V36 tests and runtime integration."""

    def __init__(self):
        self.world = ObjectCentricStateSpaceWorldModel()
        self.builder = SituationModelBuilder(self.world)

    def observe(self, text: str) -> List[SituationFrame]:
        return self.builder.ingest(text)

    def ask(self, question: str) -> str:
        return self.builder.answer(question)

    def state_report(self) -> Dict[str, Any]:
        return {
            "entities": {k: {"attributes": self._serialize(v.attributes), "fluents": self._serialize(v.fluents)} for k, v in self.world.entities.items()},
            "frames": [asdict(f) for f in self.world.frames],
            "triples": self.world.project_triples(),
            "prediction_errors": [asdict(e) for e in self.world.prediction_errors],
        }

    def _serialize(self, x: Any) -> Any:
        if isinstance(x, set):
            return sorted(x)
        if isinstance(x, dict):
            return {k: self._serialize(v) for k, v in x.items()}
        if isinstance(x, list):
            return [self._serialize(v) for v in x]
        return x
