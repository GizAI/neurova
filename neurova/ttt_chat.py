"""
Neurova TTT Chat Prototype

A practical conversational engine built around a frozen remote Qwen embedding
endpoint plus test-time-updated episodic / construction / correction memory.

Run:
  EMBEDDING_URL=http://ml-dmc8:8081/v1/embeddings \
  EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B \
  PYTHONPATH=. python -m neurova.ttt_chat_cli

This is not a full LLM. It is a TTT memory-and-worldmodel chat substrate:
- remote vLLM/OpenAI-compatible Qwen embeddings
- fast episodic memory updated during dialogue
- correction memory: `correct: <question> => <answer>`
- source/evidence memory
- light structured world model for common factual dialogue
- fallback semantic recall over evidence
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterable, List, Optional, Tuple
import hashlib
import json
import math
import os
import re
import time
import urllib.request
import urllib.error

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _key(s: str) -> str:
    s = _norm_text(s).lower()
    s = re.sub(r"^[\"'`]+|[\"'`.,;:!?]+$", "", s)
    for art in ("a ", "an ", "the ", "this ", "that "):
        while s.startswith(art):
            s = s[len(art):]
    return s.strip()


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    if np is not None:
        av = np.asarray(a, dtype=np.float32)
        bv = np.asarray(b, dtype=np.float32)
        denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
        return float(np.dot(av, bv) / denom) if denom else 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class RemoteQwenEmbedder:
    """OpenAI-compatible vLLM embedding client with deterministic fallback.

    The fallback is intentionally weak but makes tests/dev work when the remote
    endpoint is unavailable. Production should set `require_remote=True`.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 20.0,
        dim: int = 2560,
        require_remote: bool = False,
    ):
        self.url = url or os.environ.get("EMBEDDING_URL", "http://ml-dmc8:8081/v1/embeddings")
        self.model = model or os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")
        self.timeout = timeout
        self.dim = int(os.environ.get("EMBEDDING_DIM", dim))
        self.require_remote = require_remote or os.environ.get("EMBEDDING_REQUIRE_REMOTE") == "1"
        self._cache: Dict[str, List[float]] = {}

    def embed(self, text: str) -> List[float]:
        text = _norm_text(text)
        if not text:
            return [0.0] * self.dim
        if text in self._cache:
            return self._cache[text]
        try:
            payload = json.dumps({"model": self.model, "input": text}).encode("utf-8")
            req = urllib.request.Request(
                self.url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            vec = data["data"][0]["embedding"]
            vec = [float(x) for x in vec]
        except Exception as exc:
            if self.require_remote:
                raise RuntimeError(f"Embedding endpoint failed: {self.url}: {exc}") from exc
            vec = self._fallback_embed(text)
        vec = self._normalize(vec)
        self._cache[text] = vec
        return vec

    def _normalize(self, vec: List[float]) -> List[float]:
        norm = math.sqrt(sum(x * x for x in vec))
        if not norm:
            return vec
        return [x / norm for x in vec]

    def _fallback_embed(self, text: str) -> List[float]:
        # Feature hashing over words + character trigrams. Not semantic; dev only.
        dim = min(self.dim, 384)
        arr = [0.0] * dim
        words = re.findall(r"[\w.\-]+", text.lower())
        feats = list(words)
        compact = "_".join(words)
        feats.extend(compact[i : i + 3] for i in range(max(0, len(compact) - 2)))
        for feat in feats:
            h = hashlib.blake2b(feat.encode(), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % dim
            sign = 1.0 if (h[4] & 1) else -1.0
            arr[idx] += sign
        if self.dim > dim:
            arr.extend([0.0] * (self.dim - dim))
        return arr


@dataclass
class MemoryItem:
    kind: str
    text: str
    payload: Dict[str, Any]
    vector: List[float]
    created_at: float = field(default_factory=time.time)
    strength: float = 1.0


class TTTAssociativeMemory:
    """Test-time-updated vector memory.

    Every observation/correction inserts or strengthens memory immediately.
    This is the fast memory layer. It is not a parametric LLM; it is the
    substrate that lets the engine adapt inside a conversation.
    """

    def __init__(self, embedder: RemoteQwenEmbedder):
        self.embedder = embedder
        self.items: List[MemoryItem] = []

    def add(self, kind: str, text: str, payload: Optional[Dict[str, Any]] = None, strength: float = 1.0) -> MemoryItem:
        item = MemoryItem(kind=kind, text=_norm_text(text), payload=payload or {}, vector=self.embedder.embed(text), strength=strength)
        self.items.append(item)
        return item

    def search(self, query: str, kind: Optional[str] = None, k: int = 5) -> List[Tuple[float, MemoryItem]]:
        qv = self.embedder.embed(query)
        scored: List[Tuple[float, MemoryItem]] = []
        for it in self.items:
            if kind and it.kind != kind:
                continue
            score = _cosine(qv, it.vector) * it.strength
            scored.append((score, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]

    def to_jsonable(self) -> List[Dict[str, Any]]:
        # Do not persist huge vectors by default; keep text/payload/strength.
        return [
            {"kind": it.kind, "text": it.text, "payload": it.payload, "created_at": it.created_at, "strength": it.strength}
            for it in self.items
        ]

    def load_jsonable(self, rows: Iterable[Dict[str, Any]]) -> None:
        self.items.clear()
        for row in rows:
            self.add(row.get("kind", "episode"), row.get("text", ""), row.get("payload", {}), row.get("strength", 1.0))


@dataclass
class EntityState:
    name: str
    aliases: List[str] = field(default_factory=list)
    attributes: Dict[str, str] = field(default_factory=dict)
    relations: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)

    def add_relation(self, rel: str, target: str, **props: str) -> None:
        self.relations.setdefault(rel, [])
        row = {"target": target, **{k: v for k, v in props.items() if v}}
        if row not in self.relations[rel]:
            self.relations[rel].append(row)


class TinyWorldModel:
    """Small explicit state model, deliberately limited but debuggable."""

    def __init__(self):
        self.entities: Dict[str, EntityState] = {}
        self.user_id = "user"
        self.assistant_id = "neurova"
        self.get(self.user_id)
        self.get(self.assistant_id).attributes["name"] = "Neurova"

    def get(self, name: str) -> EntityState:
        n = _key(name)
        if not n:
            n = "unknown"
        if n not in self.entities:
            self.entities[n] = EntityState(name=n)
        return self.entities[n]

    def alias(self, entity: str, alias: str) -> None:
        e = self.get(entity)
        a = _key(alias)
        if a and a not in e.aliases:
            e.aliases.append(a)
        if a:
            self.entities[a] = e

    def resolve(self, name: str) -> str:
        n = _key(name)
        if n in ("i", "me", "my", "myself"):
            return self.user_id
        if n in ("you", "your", "yourself"):
            return self.assistant_id
        if n in self.entities:
            return self.entities[n].name
        # alias lookup
        for k, e in self.entities.items():
            if n == k or n in e.aliases:
                return e.name
        return n

    def observe_statement(self, text: str) -> List[str]:
        """Extract a few robust high-value facts. Everything else remains evidence."""
        t = _norm_text(text)
        low = t.lower().strip(" .")
        updates: List[str] = []

        # Dialogue grounding: I am X / my name is X.
        m = re.match(r"(?:i am|i'm|my name is|call me)\s+(.+)$", low, re.I)
        if m:
            name = _norm_text(m.group(1)).strip(" .")
            user = self.get(self.user_id)
            user.attributes["name"] = name
            self.alias(self.user_id, name)
            user.evidence.append(t)
            updates.append(f"user.name={name}")
            return updates

        # X is Y / X is in Y / X was founded by Y in T
        m = re.match(r"(.+?)\s+(?:is|are|was|were)\s+(?:a\s+|an\s+|the\s+)?(.+)$", low, re.I)
        if m:
            subj = self.resolve(m.group(1))
            rest = _norm_text(m.group(2))
            ent = self.get(subj)
            ent.evidence.append(t)
            # location phrase
            loc_m = re.search(r"\bin\s+([a-z0-9 .\-]+)$", rest)
            if loc_m and not re.match(r"\d{3,4}$", loc_m.group(1).strip()):
                ent.attributes["location"] = _key(loc_m.group(1))
                updates.append(f"{subj}.location={ent.attributes['location']}")
            # founded by
            f_m = re.search(r"founded\s+by\s+(.+?)(?:\s+in\s+(\d{4}))?$", rest)
            if f_m:
                founder = _key(f_m.group(1))
                year = f_m.group(2) or ""
                ent.add_relation("founded_by", founder, time=year)
                updates.append(f"{subj}.founded_by={founder}")
            else:
                # classification, keep before long prepositional modifier
                cls = re.split(r"\s+(?:in|on|at|near|with|from|by|to|between)\s+", rest)[0]
                cls = _key(cls)
                if cls:
                    ent.attributes.setdefault("is_a", cls)
                    updates.append(f"{subj}.is_a={cls}")

        # X founded Y in YEAR
        m = re.match(r"(.+?)\s+founded\s+(.+?)(?:\s+in\s+(\d{4}))?$", low, re.I)
        if m:
            founder = self.resolve(m.group(1))
            org = self.resolve(m.group(2))
            year = m.group(3) or ""
            self.get(org).add_relation("founded_by", founder, time=year)
            self.get(org).evidence.append(t)
            updates.append(f"{org}.founded_by={founder}")

        # X gave Y to Z / X gave Z Y (very small transfer dynamics)
        m = re.match(r"(.+?)\s+gave\s+(.+?)\s+to\s+(.+)$", low, re.I)
        if m:
            giver, obj, rec = self.resolve(m.group(1)), self.resolve(m.group(2)), self.resolve(m.group(3))
            self.get(obj).attributes["holder"] = rec
            self.get(obj).evidence.append(t)
            updates.append(f"holder({obj})={rec}")
        m = re.match(r"(.+?)\s+gave\s+(.+?)\s+(.+)$", low, re.I)
        if m and not updates:
            giver, rec, obj = self.resolve(m.group(1)), self.resolve(m.group(2)), self.resolve(m.group(3))
            self.get(obj).attributes["holder"] = rec
            self.get(obj).evidence.append(t)
            updates.append(f"holder({obj})={rec}")

        return updates

    def answer_structured(self, question: str) -> Optional[str]:
        q = _key(question.rstrip("?"))
        # Dialogue identity.
        if q in ("who am i", "what am i"):
            name = self.get(self.user_id).attributes.get("name")
            return f"You are {name}." if name else "I don't know who you are yet."
        if q in ("who are you", "what are you"):
            name = self.get(self.assistant_id).attributes.get("name", "Neurova")
            return f"I am {name}."

        m = re.match(r"who founded (.+)$", q)
        if m:
            ent = self.get(self.resolve(m.group(1)))
            rels = ent.relations.get("founded_by", [])
            if rels:
                r = rels[-1]
                year = f" in {r.get('time')}" if r.get("time") else ""
                return f"{ent.name} was founded by {r['target']}{year}."

        m = re.match(r"when was (.+?) founded$", q)
        if m:
            ent = self.get(self.resolve(m.group(1)))
            rels = ent.relations.get("founded_by", [])
            for r in reversed(rels):
                if r.get("time"):
                    return f"It was founded in {r['time']}."

        m = re.match(r"where is (.+)$", q)
        if m:
            ent = self.get(self.resolve(m.group(1)))
            loc = ent.attributes.get("location")
            return f"It is in {loc}." if loc else None

        m = re.match(r"who has (.+)$", q)
        if m:
            obj = self.get(self.resolve(m.group(1)))
            holder = obj.attributes.get("holder")
            return f"{holder} has {obj.name}." if holder else None

        m = re.match(r"what is (.+)$", q)
        if m:
            ent = self.get(self.resolve(m.group(1)))
            parts = []
            if ent.attributes.get("name"):
                parts.append(f"name is {ent.attributes['name']}")
            if ent.attributes.get("is_a"):
                parts.append(f"is {ent.attributes['is_a']}")
            if ent.attributes.get("location"):
                parts.append(f"is in {ent.attributes['location']}")
            for rel, rows in ent.relations.items():
                if rows:
                    vals = ", ".join(r["target"] for r in rows[-3:])
                    parts.append(f"{rel}: {vals}")
            if parts:
                return f"{ent.name}: " + "; ".join(parts) + "."
        return None

    def to_jsonable(self) -> Dict[str, Any]:
        out = {}
        seen = set()
        for name, e in self.entities.items():
            if id(e) in seen:
                continue
            seen.add(id(e))
            out[name] = asdict(e)
        return out

    def load_jsonable(self, data: Dict[str, Any]) -> None:
        self.entities.clear()
        for name, row in data.items():
            e = EntityState(**row)
            self.entities[name] = e
            for a in e.aliases:
                self.entities[a] = e
        self.get(self.user_id)
        self.get(self.assistant_id)


class TTTChatEngine:
    """Conversation engine with remote Qwen embeddings and test-time memory."""

    def __init__(self, persist_path: Optional[str] = None, embedder: Optional[RemoteQwenEmbedder] = None):
        self.embedder = embedder or RemoteQwenEmbedder()
        self.memory = TTTAssociativeMemory(self.embedder)
        self.world = TinyWorldModel()
        self.persist_path = persist_path or os.environ.get("NEUROVA_TTT_MEMORY", "")
        self.last_question: Optional[str] = None
        self.last_answer: Optional[str] = None
        if self.persist_path and os.path.exists(self.persist_path):
            self.load(self.persist_path)

    def hear(self, text: str) -> str:
        text = _norm_text(text)
        if not text:
            return "Yes?"
        low = text.lower()

        if low in ("status", ":status"):
            return self.status()
        if low in ("model", ":model"):
            return self.model_dump()
        if low in ("sleep", ":sleep"):
            return self.sleep()
        if low.startswith("correct:"):
            return self.correct(text[len("correct:") :].strip())
        if low.startswith("learn:"):
            return self.learn(text[len("learn:") :].strip())

        if self._is_question(text):
            ans = self.answer(text)
            self.last_question, self.last_answer = text, ans
            self.memory.add("question", text, {"answer": ans}, strength=0.7)
            self._autosave()
            return ans

        updates = self.world.observe_statement(text)
        self.memory.add("evidence", text, {"updates": updates}, strength=1.0)
        self._autosave()
        return "Got it. I've stored that information." if updates else "I heard you. I stored it as evidence."

    def _is_question(self, text: str) -> bool:
        if text.endswith("?"):
            return True
        return bool(re.match(r"^(who|what|when|where|why|how|which|is|are|do|does|did|can|could|has|have)\b", text.lower()))

    def learn(self, statement: str) -> str:
        updates = self.world.observe_statement(statement)
        self.memory.add("teaching", statement, {"updates": updates}, strength=1.4)
        self._autosave()
        return "Learned." if updates else "Stored as teaching evidence."

    def correct(self, payload: str) -> str:
        # Format: correct: question => answer. If no question, use last question.
        if "=>" in payload:
            q, a = [p.strip() for p in payload.split("=>", 1)]
        else:
            q, a = self.last_question or "", payload.strip()
        if not q or not a:
            return "Use: correct: <question> => <answer>"
        self.memory.add("correction", q, {"answer": a}, strength=2.5)
        # Also try to learn factual content from the answer itself.
        self.world.observe_statement(a)
        self._autosave()
        return "Correction learned for similar future questions."

    def answer(self, question: str) -> str:
        # 1) Exact/semantic correction memory is strongest TTT behavior.
        hits = self.memory.search(question, kind="correction", k=3)
        if hits and hits[0][0] >= float(os.environ.get("NEUROVA_CORRECTION_THRESHOLD", "0.72")):
            return hits[0][1].payload.get("answer", "I learned an answer, but it is empty.")

        # 2) Structured world model.
        structured = self.world.answer_structured(question)
        if structured:
            return structured

        # 3) Evidence recall with Qwen embeddings.
        hits = self.memory.search(question, kind="evidence", k=3)
        if hits and hits[0][0] >= float(os.environ.get("NEUROVA_EVIDENCE_THRESHOLD", "0.62")):
            snippets = [h[1].text for h in hits[:2]]
            return "I don't have a clean structured answer yet, but the closest evidence I remember is: " + " / ".join(snippets)

        return "I don't know yet. Teach me with: correct: <question> => <answer>"

    def sleep(self) -> str:
        # Lightweight consolidation: strengthen corrections that have nearby questions/evidence.
        strengthened = 0
        for score, corr in self.memory.search(" ".join([m.text for m in self.memory.items[-20:]]), kind="correction", k=20):
            if score > 0.55:
                corr.strength = min(3.5, corr.strength + 0.05)
                strengthened += 1
        self._autosave()
        return f"Consolidated. Strengthened {strengthened} correction memories."

    def status(self) -> str:
        return f"{len(self.world.to_jsonable())} entities, {len(self.memory.items)} memory items, embedding_model={self.embedder.model}, endpoint={self.embedder.url}"

    def model_dump(self) -> str:
        rows = []
        for name, row in self.world.to_jsonable().items():
            attrs = row.get("attributes", {})
            rels = row.get("relations", {})
            if not attrs and not rels:
                continue
            rows.append(f"- {name}: attrs={attrs}, rels={rels}")
        return "\n".join(rows) if rows else "<empty>"

    def save(self, path: str) -> None:
        data = {"world": self.world.to_jsonable(), "memory": self.memory.to_jsonable()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.world.load_jsonable(data.get("world", {}))
        self.memory.load_jsonable(data.get("memory", []))

    def _autosave(self) -> None:
        if self.persist_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.persist_path)), exist_ok=True)
            self.save(self.persist_path)


__all__ = ["RemoteQwenEmbedder", "TTTAssociativeMemory", "TTTChatEngine", "TinyWorldModel"]
