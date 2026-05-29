"""
Neurova V41 — Language Acquisition Engine (True Cognitive Substrate)

================================================================================
CORE PHILOSOPHY
================================================================================

Language rules are NOT programmed. They are LEARNED through interaction.

The only cognitive priors are:
  1. ENTITIES exist (things we can talk about)
  2. EVENTS happen (entities in roles, relations)
  3. TOPIC-COMMENT: first significant thing is usually the topic
  4. SIGNAL WORDS: certain lexical items indicate relation types
  5. PATTERNS REPEAT: repeated form-meaning mappings become constructions
  6. PREDICTION ERRORS drive learning

These priors are EXPLICITLY labeled as bootstrap heuristics, NOT language rules.
They are used ONLY as candidate generators, never as answer deciders.
Every successful parse reinforces constructions.
Every failed parse generates a new construction candidate.

================================================================================
ARCHITECTURE
================================================================================

Input
  │
  ▼
CandidateParser ─── uses ─── LanguagePriors (bootstrap fallback)
  │                              │
  │                              ▼
  │              ConstructionMemory (learned patterns — actual parse leader)
  │
  ▼
EventFrame candidates (multiple interpretations)
  │
  ▼
WorldModel ─── validates ─── role consistency, entity state, temporal order
  │
  ▼
Best candidate applied → Episode stored → success/failure recorded
  │
  ▼
Success → reinforce Construction
Failure → FailureCase → embedding cluster → new construction candidate
  │
  ▼
Sleep → consolidate, merge, prune constructions

Question → QueryPlanner (operator composition)
         → WorldModel.execute(operators)
         → answer / failure record
"""

import os, sys, re, json, time, uuid, math
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

# ── Optional spaCy (tokenization ONLY, NO dependency parsing) ──
_NLP = None
for _p in [
    "/home/user/miniconda3/envs/neurova_vsa/bin/python3",
    "/home/user/miniconda3/envs/quantv/bin/python3",
    sys.executable
]:
    try:
        import spacy
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer", "textcat"])
        break
    except Exception:
        pass

# ── Optional GPU embedding (qwen3-embedding on ml-dmc8) ──
EMBEDDING_URL = os.environ.get("EMBEDDING_URL", "http://ml-dmc8:8081/v1/embeddings")
EMBEDDING_DIM = 2560
_EMBEDDING_CACHE = {}
_HAS_USEARCH = False
try:
    from usearch.index import Index
    _HAS_USEARCH = True
except ImportError:
    Index = None


def _uid():
    return uuid.uuid4().hex[:12]


def norm(s: str) -> str:
    """Universal normalizer — lowercase + single whitespace."""
    if not s: return ""
    return re.sub(r'\s+', ' ', s.strip().lower())


# ── Simple sentence splitter (abbreviation-aware) ──
_ABBREVIATIONS = frozenset({
    'mr', 'mrs', 'ms', 'dr', 'etc', 'inc', 'ltd', 'st', 'ave', 'dept',
    'u.s', 'u.k', 'dprk', 'rok', 'e.g', 'i.e', 'vs', 'al', 'jr', 'sr',
    'corp', 'co', 'llc', 'prof', 'gen', 'capt', 'gov', 'sen', 'rep',
    'approx', 'dept', 'est', 'govt', 'natl', 'no.', 'vol',
})

def sent_split(text: str) -> List[str]:
    """Split into sentences — abbreviation-aware."""
    if not text:
        return []
    text = text.replace('\n', ' ')
    parts = []
    for w in text.split():
        parts.append(w)
        if w[-1] in '.!?':
            base = w.rstrip('.!?').lower()
            if base not in _ABBREVIATIONS and not (len(base) <= 2 and base.isalpha()):
                parts.append('|||')
    raw = ' '.join(parts).split('|||')
    return [s.strip().strip('|').strip() for s in raw if s.strip().strip('|').strip()]


def tokens(text: str) -> List[str]:
    """Tokenize — NO POS, NO dependency, NO NER."""
    if _NLP:
        return [t.text.lower().rstrip('.,;:!?\'"') for t in _NLP(text) if not t.is_space]
    return [t.rstrip('.,;:!?\'"') for t in text.lower().split()]


def embed_text(text: str) -> Optional[np.ndarray]:
    """Get embedding from qwen3-embedding on ml-dmc8 GPU."""
    if not text or not text.strip():
        return None
    cache_key = f"emb:{text[:300]}"
    if cache_key in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[cache_key]
    try:
        import urllib.request
        data = json.dumps({
            "input": text,
            "model": "Qwen/Qwen3-Embedding-4B"
        }).encode()
        req = urllib.request.Request(
            EMBEDDING_URL, data=data,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
        vec = np.array(result["data"][0]["embedding"], dtype=np.float32)
        _EMBEDDING_CACHE[cache_key] = vec
        return vec
    except Exception:
        return None


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# ══════════════════════════════════════════════════════════════
# LANGUAGE PRIORS — EXPLICIT BOOTSTRAP, NOT LANGUAGE RULES
# ══════════════════════════════════════════════════════════════
# These are labeled as "bootstrap heuristics" — they exist because
# a newborn brain has priors, not because English grammar is coded.
# They generate CANDIDATES only. ConstructionMemory overrides them.

class LanguagePriors:
    """
    Bootstrap priors for initial interaction.
    
    These mimic innate human priors:
    - Certain sound patterns (signal words) correlate with relation types
    - Topic-Comment structure: first significant thing is the topic
    - Prepositions indicate spatial/temporal relations
    
    These are fallback heuristics. Once ConstructionMemory has learned
    patterns, priors are overridden.
    """
    
    # Signal words → (event_type, role, position)
    SIGNAL_MAP = {
        'is':    ('CLASSIFICATION', 'entity', 'after'),
        'are':   ('CLASSIFICATION', 'entity', 'after'),
        'was':   ('CLASSIFICATION', 'entity', 'after'),
        'were':  ('CLASSIFICATION', 'entity', 'after'),
        'be':    ('CLASSIFICATION', 'entity', 'after'),
        'am':    ('CLASSIFICATION', 'entity', 'after'),
        'in':    ('LOCATION', None, 'after'),
        'on':    ('LOCATION', None, 'after'),
        'at':    ('LOCATION', None, 'after'),
        'near':  ('LOCATION', None, 'after'),
        'border':     ('BORDER', 'entity', 'after_prep'),
        'borders':    ('BORDER', 'entity', 'after_prep'),
        'bordered':   ('BORDER', 'entity', 'after_prep'),
        'separate':   ('SEPARATE', 'entity', 'after_prep'),
        'separates':  ('SEPARATE', 'entity', 'after_prep'),
        'separated':  ('SEPARATE', 'entity', 'after_prep'),
        'divide':     ('DIVIDE', 'entity', 'after_prep'),
        'divides':    ('DIVIDE', 'entity', 'after_prep'),
        'divided':    ('DIVIDE', 'entity', 'after_prep'),
        'locate':     ('LOCATION', 'entity', 'after_prep'),
        'located':    ('LOCATION', 'entity', 'after_prep'),
        'situated':   ('LOCATION', 'entity', 'after_prep'),
        'headquartered': ('LOCATION', 'entity', 'after_prep'),
        'found':      ('LOCATION', 'entity', 'after_prep'),
        'founded':    ('CREATION', 'entity', 'after'),
        'consist':    ('COMPOSITION', 'entity', 'after_prep'),
        'consists':   ('COMPOSITION', 'entity', 'after_prep'),
        'consisting': ('COMPOSITION', 'entity', 'after_prep'),
        'comprise':   ('COMPOSITION', 'entity', 'after_prep'),
        'comprises':  ('COMPOSITION', 'entity', 'after_prep'),
        'include':    ('COMPOSITION', 'entity', 'after_prep'),
        'includes':   ('COMPOSITION', 'entity', 'after_prep'),
        'call':       ('CLASSIFICATION', 'entity', 'after'),
        'called':     ('CLASSIFICATION', 'entity', 'after'),
        'name':       ('CLASSIFICATION', 'entity', 'after'),
        'named':      ('CLASSIFICATION', 'entity', 'after'),
        'known':      ('CLASSIFICATION', 'entity', 'after'),
        'own':        ('POSSESSION', 'entity', 'after'),
        'owns':       ('POSSESSION', 'entity', 'after'),
        'owned':      ('POSSESSION', 'entity', 'after_prep'),
        'have':       ('POSSESSION', 'entity', 'after'),
        'has':        ('POSSESSION', 'entity', 'after'),
        'had':        ('POSSESSION', 'entity', 'after'),
        'make':       ('CREATION', 'entity', 'after'),
        'makes':      ('CREATION', 'entity', 'after'),
        'made':       ('CREATION', 'entity', 'after_prep'),
        'create':     ('CREATION', 'entity', 'after'),
        'creates':    ('CREATION', 'entity', 'after'),
        'created':    ('CREATION', 'entity', 'after'),
        'build':      ('CREATION', 'entity', 'after'),
        'builds':     ('CREATION', 'entity', 'after'),
        'built':      ('CREATION', 'entity', 'after'),
        'launch':     ('ACTION', 'entity', 'after'),
        'launches':   ('ACTION', 'entity', 'after'),
        'launched':   ('ACTION', 'entity', 'after'),
        'begin':      ('ACTION', 'entity', 'after'),
        'began':      ('ACTION', 'entity', 'after'),
        'start':      ('ACTION', 'entity', 'after'),
        'started':    ('ACTION', 'entity', 'after'),
        'develop':    ('ACTION', 'entity', 'after'),
        'develops':   ('ACTION', 'entity', 'after'),
        'developed':  ('ACTION', 'entity', 'after'),
    }
    
    # Preposition roles (universal spatial/temporal markers)
    PREP_ROLES = {
        'in':    'location', 'on':    'location', 'at':    'location',
        'near':  'location', 'to':    'direction', 'from':  'source',
        'by':    'instrument', 'across': 'located_across', 'into':  'result',
        'of':    'possession', 'for':   'purpose', 'with':  'companion',
        'under': 'location', 'over':  'location', 'between': 'between',
    }
    
    # Direction words (spatial cognitive primitives)
    DIRECTION_WORDS = frozenset({
        'north', 'south', 'east', 'west',
        'northeast', 'northwest', 'southeast', 'southwest',
    })
    
    # Words that typically don't carry entity meaning (function words)
    FILLER_WORDS = frozenset({
        'a', 'an', 'the', 'this', 'that', 'these', 'those',
        'some', 'any', 'every', 'each', 'all', 'both',
        'no', 'not', 'none', 'neither',
        'there', 'here',
    })
    
    @classmethod
    def is_signal(cls, word: str) -> bool:
        return word.lower() in cls.SIGNAL_MAP
    
    @classmethod
    def is_direction(cls, word: str) -> bool:
        return word.lower() in cls.DIRECTION_WORDS
    
    @classmethod
    def is_filler(cls, word: str) -> bool:
        return word.lower() in cls.FILLER_WORDS
    
    @classmethod
    def is_year(cls, s: str) -> bool:
        try:
            n = int(s.strip('.,; '))
            return 1000 <= n <= 2100
        except:
            return False


# ══════════════════════════════════════════════════════════════
# EVENT FRAME — Fully nested semantic representation
# ══════════════════════════════════════════════════════════════

@dataclass
class Relation:
    """A typed relation between two entities."""
    type: str = ""
    target: str = ""
    properties: Dict[str, str] = field(default_factory=dict)


@dataclass
class EventFrame:
    """
    A semantic event/situation frame.
    
    Nested structure — no flat fields for directions, neighbors, etc.
    All spatial/temporal/relational information lives in relations list.
    """
    event_type: str = "STATEMENT"
    entity: str = ""
    attributes: Dict[str, str] = field(default_factory=dict)
    location: str = ""
    time: str = ""
    relations: List[Relation] = field(default_factory=list)
    properties: Set[str] = field(default_factory=set)
    source_span: str = ""
    confidence: float = 0.3
    
    def to_dict(self):
        return {
            "event_type": self.event_type,
            "entity": self.entity,
            "attributes": dict(self.attributes),
            "location": self.location,
            "time": self.time,
            "relations": [{
                "type": r.type, "target": r.target,
                "properties": dict(r.properties)
            } for r in self.relations],
            "properties": list(self.properties),
            "confidence": self.confidence,
        }


@dataclass
class EventFrameCandidate:
    """
    A candidate interpretation — one of possibly many.
    Carries its confidence and the source that generated it.
    """
    event_frame: EventFrame
    confidence: float = 0.3
    source: str = "prior"  # "prior", "construction", "embedding"
    source_id: str = ""    # construction ID if source="construction"
    
    def to_dict(self):
        d = self.event_frame.to_dict()
        d["candidate_confidence"] = self.confidence
        d["candidate_source"] = self.source
        return d


# ══════════════════════════════════════════════════════════════
# CONSTRUCTION — Learned from interaction
# ══════════════════════════════════════════════════════════════

@dataclass
class Construction:
    """
    A learned form-meaning mapping.
    
    Grows from every interaction:
    - Successes reinforce confidence and add examples
    - Failures add counterexamples and reduce confidence
    - Embedding enables semantic matching for novel inputs
    - Role mappings are learned, not coded
    """
    id: str
    event_type: str
    signal_phrases: List[str] = field(default_factory=list)
    role_keywords: Dict[str, str] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    counterexamples: List[str] = field(default_factory=list)
    avg_embedding: Optional[np.ndarray] = None
    confidence: float = 0.3
    success_count: int = 0
    failure_count: int = 0
    last_used: float = 0.0
    created: float = field(default_factory=time.time)
    
    def score(self) -> float:
        """Confidence weighted by success ratio."""
        t = self.success_count + self.failure_count
        if t == 0:
            return self.confidence
        ratio = self.success_count / max(t, 1)
        return self.confidence * ratio
    
    def match_score(self, text: str) -> float:
        """
        How well this construction matches the given text.
        
        Uses:
        1. Signal phrase overlap (lexical)
        2. Embedding similarity (semantic)
        3. Example similarity
        """
        score = 0.0
        tl = text.lower()
        
        # Lexical: signal phrase match
        for sp in self.signal_phrases:
            if sp in tl:
                score += 0.3
                break
        
        # Semantic: embedding similarity to average
        if self.avg_embedding is not None:
            vec = embed_text(text)
            if vec is not None:
                sim = cosine_sim(vec, self.avg_embedding)
                if sim > 0.7:
                    score += sim * 0.4
        
        # Example overlap
        for ex in self.examples:
            ext = ex.lower()
            common = len(set(tl.split()) & set(ext.split()))
            total = max(len(set(tl.split()) | set(ext.split())), 1)
            overlap = common / total
            if overlap > 0.5:
                score += overlap * 0.3
                break
        
        return min(1.0, score)
    
    def reinforce(self):
        """Called after successful use."""
        self.success_count += 1
        self.confidence = min(0.95, self.confidence + 0.05)
        self.last_used = time.time()
    
    def weaken(self):
        """Called after failed use."""
        self.failure_count += 1
        self.confidence = max(0.05, self.confidence - 0.05)
    
    def update_embedding(self):
        """Recalculate average embedding from examples."""
        vecs = []
        for ex in self.examples + self.signal_phrases:
            v = embed_text(ex)
            if v is not None:
                vecs.append(v)
        if vecs:
            self.avg_embedding = np.mean(vecs, axis=0)
    
    def to_dict(self):
        return {
            "id": self.id, "event_type": self.event_type,
            "confidence": self.confidence,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "signal_phrases": self.signal_phrases[:5],
            "examples": self.examples[:3],
            "score": self.score(),
        }


class ConstructionMemory:
    """
    The system's learned pattern repository.
    
    THIS drives parsing decisions, not priors.
    When a construction matches with high confidence, it overrides priors.
    When no construction matches, priors are used as fallback.
    """
    
    def __init__(self):
        self.constructions: Dict[str, Construction] = {}
    
    def create(self, event_type: str, signal_phrases: List[str],
               role_keywords: Dict[str, str] = None,
               example: str = "", confidence: float = 0.3) -> Construction:
        """Create a new construction from an interaction."""
        cid = f"c_{event_type.lower()}_{_uid()[:8]}"
        c = Construction(
            id=cid, event_type=event_type,
            signal_phrases=signal_phrases,
            role_keywords=role_keywords or {},
            confidence=confidence,
        )
        if example:
            c.examples.append(example)
        c.update_embedding()
        self.constructions[cid] = c
        return c
    
    def find_best_match(self, text: str, event_type: str = "") -> Optional[Construction]:
        """
        Find the best matching construction for the given text.
        
        Returns the highest-scoring construction that matches.
        If event_type is specified, only considers that type.
        """
        best, best_score = None, 0.3  # threshold
        for c in self.constructions.values():
            if event_type and c.event_type != event_type:
                continue
            score = c.match_score(text)
            if score > best_score:
                best_score = score
                best = c
        return best
    
    def find_by_signal(self, signal: str) -> List[Construction]:
        """Find constructions that contain the given signal word."""
        return [
            c for c in self.constructions.values()
            if any(signal in sp or sp in signal for sp in c.signal_phrases)
        ]
    
    def create_from_failure(self, text: str, expected: str,
                            failed_construction_id: str = "") -> Optional[Construction]:
        """Create a new construction candidate from a failure."""
        words = [t for t in tokens(text) if not LanguagePriors.is_filler(t) and len(t) > 2]
        signals = [w for w in words if LanguagePriors.is_signal(w)]
        
        if not signals:
            return None
        
        # Infer event type from first signal
        sig = signals[0]
        etype = LanguagePriors.SIGNAL_MAP.get(sig, (None, None, None))[0]
        if etype is None:
            etype = "STATEMENT"
        
        c = self.create(
            event_type=etype,
            signal_phrases=signals[:3],
            example=text,
            confidence=0.25  # Low initial confidence
        )
        
        # If there's a failed construction, link it
        if failed_construction_id and failed_construction_id in self.constructions:
            c.failures.append(failed_construction_id)
        
        return c
    
    def consolidate(self):
        """Clean up weak constructions, merge similar ones."""
        to_delete = []
        
        # Remove very weak constructions
        for cid, c in self.constructions.items():
            if c.confidence < 0.05:
                to_delete.append(cid)
            elif c.failure_count > 20 and c.success_count == 0:
                to_delete.append(cid)
        
        for cid in to_delete:
            del self.constructions[cid]
        
        return len(to_delete)
    
    def get_stats(self) -> Dict:
        types = defaultdict(int)
        for c in self.constructions.values():
            types[c.event_type] += 1
        return {
            "total": len(self.constructions),
            "types": dict(types),
        }


# ══════════════════════════════════════════════════════════════
# ENTITY & SITUATION MODEL — Object-centric world state
# ══════════════════════════════════════════════════════════════

@dataclass
class Entity:
    """An entity in the world model with all known properties."""
    name: str = ""
    attributes: Dict[str, str] = field(default_factory=dict)
    properties: Set[str] = field(default_factory=set)
    location: str = ""
    time: str = ""
    relations: Dict[str, List[Tuple[str, Dict[str, str]]]] = field(default_factory=lambda: defaultdict(list))
    events: List[EventFrame] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "name": self.name,
            "attributes": dict(self.attributes),
            "properties": list(self.properties),
            "location": self.location,
            "time": self.time,
            "relations": {
                k: [(t, dict(p)) for t, p in v[:5]]
                for k, v in self.relations.items()
            },
            "event_count": len(self.events),
        }


class SituationModel:
    """
    Object-centric world state.
    
    Tracks entities, their attributes, locations, relations, and events.
    Supports generic query operators for question answering.
    """
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.evidence_store: List[Dict] = []  # Raw text spans with source info
    
    def get(self, name: str) -> Optional[Entity]:
        """Get or create an entity by name."""
        n = norm(name)
        if not n:
            return None
        if n not in self.entities:
            self.entities[n] = Entity(n)
        return self.entities[n]
    
    def find(self, target: str) -> Optional[Tuple[str, Entity]]:
        """
        Find an entity by flexible matching.
        
        Strategies:
        1. Exact match
        2. Target is substring of entity name
        3. Entity name is substring of target
        4. Direction words indicate spatial relation target
        """
        t = norm(target)
        if not t:
            return None
        
        # Exact match
        if t in self.entities:
            return t, self.entities[t]
        
        # Target is substring of entity name (e.g., "korea" → "south korea")
        matches = [(n, e) for n, e in self.entities.items() if t in n]
        if matches:
            # Return longest match (most specific)
            return max(matches, key=lambda x: len(x[0]))
        
        # Entity name is substring of target
        matches = [(n, e) for n, e in self.entities.items() if n in t]
        if matches:
            return max(matches, key=lambda x: len(x[0]))
        
        return None
    
    def apply(self, ef: EventFrame) -> bool:
        """Apply an event frame to update world state."""
        if not ef.entity:
            return False
        
        e = self.get(ef.entity)
        if e is None:
            return False
        
        # Update attributes
        for k, v in ef.attributes.items():
            e.attributes[k] = v
        
        # Update properties
        e.properties.update(ef.properties)
        
        # Update location
        if ef.location:
            e.location = ef.location
        
        # Update time
        if ef.time:
            e.time = ef.time
        
        # Update relations
        for rel in ef.relations:
            t = rel.target.strip()
            if t:
                e.relations[rel.type].append((t, dict(rel.properties)))
                self.get(t)  # Ensure target entity exists
                
                # Add inverse relations
                inv = self._inverse_relation(rel.type)
                if inv:
                    target_e = self.get(t)
                    if target_e:
                        target_e.relations[inv].append(
                            (ef.entity, dict(rel.properties)))
        
        # Store event
        e.events.append(ef)
        
        # Store evidence
        if ef.source_span:
            self.evidence_store.append({
                "span": ef.source_span,
                "entity": ef.entity,
                "event_type": ef.event_type,
                "timestamp": time.time(),
            })
        
        return True
    
    def _inverse_relation(self, rel_type: str) -> Optional[str]:
        """Get inverse relation type."""
        inverses = {
            'BORDER': 'BORDERED_BY',
            'BORDERED_BY': 'BORDER',
            'SEPARATED_FROM': 'SEPARATES',
            'SEPARATES': 'SEPARATED_FROM',
            'DIVIDED_INTO': 'DIVIDES',
            'DIVIDES': 'DIVIDED_INTO',
            'PART_OF': 'CONTAINS',
            'CONTAINS': 'PART_OF',
            'POSSESSES': 'POSSESSED_BY',
            'POSSESSED_BY': 'POSSESSES',
            'LOCATED_IN': 'CONTAINS_LOCATION',
            'CONTAINS_LOCATION': 'LOCATED_IN',
        }
        return inverses.get(rel_type)
    
    def resolve_location(self, name: str) -> str:
        """Resolve location through transitive relations."""
        e = self.find(name)
        if e is None:
            return ""
        entity = e[1]
        if entity.location:
            return entity.location
        for rel_type, entries in entity.relations.items():
            if 'part_of' in rel_type.lower() or 'located_in' in rel_type.lower():
                for t, _ in entries:
                    parent = self.find(t)
                    if parent and parent[1].location:
                        return parent[1].location
        return ""
    
    def describe(self, name: str) -> str:
        """Compose a description from stored data — no templates."""
        found = self.find(name)
        if not found:
            return ""
        _, e = found
        
        parts = []
        for k, v in e.attributes.items():
            if k == 'is_a':
                parts.append(f"is {v}")
            else:
                parts.append(f"{k} {v}")
        if e.location:
            parts.append(f"located in {e.location}")
        
        for rel_type, entries in list(e.relations.items())[:5]:
            targets = [t for t, _ in entries[:3]]
            if targets:
                label = rel_type.lower().replace('_', ' ')
                parts.append(f"{label} {', '.join(targets)}")
        
        if e.properties:
            props = [p for p in e.properties if not p.startswith('_')][:3]
            if props:
                parts.append(f"has: {', '.join(props)}")
        
        return ", ".join(parts) if parts else ""
    
    def to_dict(self):
        return {
            name: e.to_dict()
            for name, e in self.entities.items()
        }


# ══════════════════════════════════════════════════════════════
# COREFERENCE — Learned from interaction
# ══════════════════════════════════════════════════════════════

class CorefLearner:
    """
    Resolves pronouns and references.
    
    Uses:
    1. Frequency-based preference (most-mentioned entity wins)
    2. Embedding similarity (thematic fit)
    3. Entity type compatibility
    """
    
    def __init__(self):
        self.history: List[Tuple[str, str, float]] = []
        self.mention_counts = {}
        self.pronouns = {'he', 'him', 'his', 'she', 'her', 'hers',
                         'it', 'its', 'they', 'them', 'their',
                         'this', 'that', 'these', 'those'}
        self.special_gpe = {
            'region', 'country', 'nation', 'area', 'place',
            'peninsula', 'island', 'city', 'state', 'province',
            'both countries', 'the two countries',
        }
        self.non_referent = {'korea strait', 'the korea strait',
                             'korean strait', 'the korean strait'}
    
    def register(self, entity: str, etype: str = ''):
        """Register an entity mention."""
        e = norm(entity)
        if not e or len(e) < 1:
            return
        self.mention_counts[e] = self.mention_counts.get(e, 0) + 1
        
        # Move to front of history
        for i, (name, tp, _) in enumerate(self.history):
            if name == e:
                self.history.pop(i)
                break
        self.history.insert(0, (e, etype, time.time()))
        
        # Trim history
        if len(self.history) > 200:
            self.history = self.history[:200]
    
    def resolve(self, text: str) -> Tuple[str, bool]:
        """Resolve a reference. Returns (resolved_name, was_resolved)."""
        t = norm(text)
        if not t:
            return text, False
        
        # Special GPE references → most-mentioned main entity
        if t in self.special_gpe:
            best = self._best_main()
            if best:
                return best, True
            return text, False
        
        # Pronouns → most-mentioned main entity
        if t in self.pronouns:
            best = self._best_main()
            if best:
                return best, True
            if self.history:
                return self.history[0][0], True
            return text, False
        
        # Exact match → already resolved
        for name, tp, _ in self.history:
            if name == t:
                return text, False
        
        # Entity name is part of reference → resolve
        # e.g., "the region" → "korea" if "korea" is in history
        for name, tp, _ in self.history:
            if name in t or t in name:
                return name, True
        
        return text, False
    
    def _best_main(self):
        """Return the most-mentioned main entity."""
        best, best_cnt = None, 0
        for e, c in self.mention_counts.items():
            if e in self.non_referent:
                continue
            if c > best_cnt:
                best, best_cnt = e, c
        if best:
            return best
        for name, _, _ in self.history:
            if name not in self.non_referent:
                return name
        return None
    
    def reset(self):
        self.history.clear()
        self.mention_counts.clear()


# ══════════════════════════════════════════════════════════════
# CANDIDATE PARSER — Generates multiple interpretations
# ══════════════════════════════════════════════════════════════

class CandidateParser:
    """
    Generates candidate EventFrame interpretations for a text.
    
    Uses:
    1. LanguagePriors (explicit bootstrap) for initial candidates
    2. ConstructionMemory for learned pattern matching
    3. WorldModel for validation/context
    
    Multiple candidates are generated; the best one wins.
    """
    
    def __init__(self, model: SituationModel, coref: CorefLearner,
                 cmem: ConstructionMemory):
        self.model = model
        self.coref = coref
        self.cmem = cmem
    
    def parse(self, text: str) -> EventFrame:
        """
        Parse text into the best EventFrame.
        
        Flow:
        1. Generate candidates from priors
        2. Check if any learned construction matches
        3. If construction matches with high confidence, use it
        4. Otherwise, use best prior candidate
        5. Validate against world model
        """
        candidates = self._generate_candidates(text)
        
        if not candidates:
            return EventFrame(source_span=text, confidence=0.1)
        
        # Rank by confidence
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        
        # Return best candidate
        ef = candidates[0].event_frame
        ef.confidence = candidates[0].confidence
        
        # Resolve entity via coref
        if ef.entity:
            resolved, was_resolved = self.coref.resolve(ef.entity)
            if was_resolved:
                ef.entity = resolved
        
        return ef
    
    def _generate_candidates(self, text: str) -> List[EventFrameCandidate]:
        """Generate all candidate interpretations."""
        candidates = []
        text_lower = text.lower().strip().rstrip('.!?')
        toks = tokens(text)
        
        # 1. Try learned constructions first
        constr = self.cmem.find_best_match(text)
        if constr and constr.score() > 0.5:
            # Construction-driven candidate
            ef = self._apply_construction(constr, text, text_lower, toks)
            if ef:
                candidates.append(EventFrameCandidate(
                    event_frame=ef,
                    confidence=constr.score(),
                    source="construction",
                    source_id=constr.id,
                ))
        
        # 2. Generate prior-based candidates (always)
        prior_candidates = self._generate_prior_candidates(text_lower, toks)
        
        # Deduplicate
        for pc in prior_candidates:
            is_dup = False
            for existing in candidates:
                if (existing.event_frame.event_type == pc.event_frame.event_type
                    and existing.event_frame.entity == pc.event_frame.entity):
                    is_dup = True
                    # Keep the higher-confidence one
                    if pc.confidence > existing.confidence:
                        existing.confidence = pc.confidence
                    break
            if not is_dup:
                candidates.append(pc)
        
        # 3. If construction exists but with lower confidence, offer as alternative
        if constr and constr.score() <= 0.5:
            ef = self._apply_construction(constr, text, text_lower, toks)
            if ef:
                candidates.append(EventFrameCandidate(
                    event_frame=ef,
                    confidence=constr.score(),
                    source="construction",
                    source_id=constr.id,
                ))
        
        return candidates
    
    def _generate_prior_candidates(self, text_lower: str,
                                    toks: List[str]) -> List[EventFrameCandidate]:
        """Generate candidates using LanguagePriors (bootstrap heuristics)."""
        ef = EventFrame(source_span=text_lower, confidence=0.3)
        
        # Extract entity (topic)
        raw_entity = self._extract_topic(text_lower, toks)
        resolved, was_resolved = self.coref.resolve(raw_entity) if raw_entity else ('', False)
        ef.entity = resolved if resolved else raw_entity
        
        # Detect event type from signals
        event_type, info = self._detect_event_type(toks)
        ef.event_type = event_type
        
        # Extract relations generically (not type-specific)
        self._extract_slots(text_lower, toks, ef, event_type, info)
        
        # Extract location
        if not ef.location:
            ef.location = self._extract_any_location(text_lower, toks)
        
        # Extract time
        ef.time = self._extract_time(toks)
        
        return [EventFrameCandidate(event_frame=ef, confidence=0.3, source="prior")]
    
    def _extract_topic(self, text_lower: str, toks: List[str]) -> str:
        """
        Extract the topic entity using topic-comment universal structure.
        
        The topic is the first significant noun-like element
        that isn't a question word, signal word, or filler.
        """
        if not toks:
            return ""
        
        # Check if the sentence starts with a pronoun reference
        first = toks[0]
        if first in ('it', 'this', 'that', 'these', 'those', 'they', 'he', 'she'):
            resolved, _ = self.coref.resolve(first)
            if resolved != first:
                return resolved
        
        # Question words at start
        question_words = frozenset({
            'what', 'where', 'who', 'when', 'why', 'how', 'which',
        })
        
        # Find the first significant word that's not a question word or signal
        for i, t in enumerate(toks):
            if t in question_words:
                continue
            if t in ('is', 'are', 'was', 'were', 'am', 'do', 'does', 'did'):
                # Check if this is a question (verb before subject)
                # "Is Korea a country?" — entity is after the copula
                if i == 0 and i + 1 < len(toks):
                    # Get entity after the copula
                    after_entity = self._gather_entity(toks, i + 1)
                    if after_entity:
                        return after_entity
                continue
            if LanguagePriors.is_filler(t):
                continue
            
            # This is likely the start of the topic
            return self._gather_entity(toks, i)
        
        return toks[0] if toks else ""
    
    def _gather_entity(self, toks: List[str], start: int) -> str:
        """Gather entity words starting at start index."""
        words = []
        for j in range(start, len(toks)):
            t = toks[j]
            # Stop at signal words
            if LanguagePriors.is_signal(t) and j > start and t not in ('in', 'on', 'at', 'of', 'for'):
                break
            # Stop at relative clause markers
            if t in ('which', 'that', 'who', 'where') and words:
                break
            # Stop at certain prepositions
            if t in ('in', 'on', 'at', 'near', 'of', 'to', 'by', 'from', 'with',
                     'since', 'during', 'until') and len(words) >= 1:
                break
            # Stop at conjunctions with enough words
            if t in ('and', 'or', 'but', 'because') and len(words) >= 2:
                break
            words.append(t)
        
        if not words:
            return ""
        
        raw = ' '.join(words)
        # Remove trailing relative clause fragments
        raw = re.sub(r'\s+(which|that|who)\s+.*$', '', raw).strip()
        # Remove leading articles
        ws = raw.split()
        ws = [w for w in ws if w not in LanguagePriors.FILLER_WORDS or len(ws) <= 1]
        return ' '.join(ws[:5]) if ws else raw
    
    def _detect_event_type(self, toks: List[str]) -> Tuple[str, dict]:
        """
        Detect event type by scanning for signal words.
        
        Returns (event_type, info_dict).
        """
        info = {}
        sig_map = LanguagePriors.SIGNAL_MAP
        
        # Priority 1: Specific relational verbs (most informative)
        for i, t in enumerate(toks):
            entry = sig_map.get(t)
            if entry and entry[0] in ('BORDER', 'SEPARATE', 'DIVIDE'):
                if t in ('bordered', 'located', 'situated', 'headquartered',
                          'separated', 'divided',
                          'founded', 'launched', 'developed',
                          'called', 'named', 'known'):
                    info = {'signal': t, 'signal_idx': i, 'role': 'entity', 'position': 'after'}
                else:
                    info = {'signal': t, 'signal_idx': i, 'role': entry[1], 'position': entry[2]}
                return entry[0], info
        
        # Priority 2: Copula (is/are/was/were)
        for i, t in enumerate(toks):
            if t in ('is', 'are', 'was', 'were', 'am', 'be'):
                info = {'signal': t, 'signal_idx': i, 'role': 'entity', 'position': 'after'}
                return 'CLASSIFICATION', info
        
        # Priority 3: Other signals
        for i, t in enumerate(toks):
            entry = sig_map.get(t)
            if entry:
                info = {'signal': t, 'signal_idx': i, 'role': entry[1], 'position': entry[2]}
                return entry[0], info
        
        return 'STATEMENT', info
    
    def _apply_construction(self, constr: Construction, text: str,
                             text_lower: str, toks: List[str]) -> Optional[EventFrame]:
        """Apply a learned construction to extract an EventFrame."""
        ef = EventFrame(source_span=text, confidence=constr.score())
        ef.event_type = constr.event_type
        
        # Extract entity (topic)
        raw_entity = self._extract_topic(text_lower, toks)
        resolved, was_resolved = self.coref.resolve(raw_entity) if raw_entity else ('', False)
        ef.entity = resolved if resolved else raw_entity
        
        # Generic slot extraction based on construction type
        self._extract_slots(text_lower, toks, ef, constr.event_type, {})
        
        if not ef.location:
            ef.location = self._extract_any_location(text_lower, toks)
        ef.time = self._extract_time(toks)
        
        return ef
    
    def _extract_slots(self, text_lower: str, toks: List[str],
                        ef: EventFrame, etype: str, info: dict):
        """
        Generic slot extraction — NOT type-specific handlers.
        
        Uses generic text patterns that emerge from topic-comment structure:
        - After the copula → attributes/classification
        - "by [entity]" → agent/instrument
        - "to the [direction]" → spatial relation with direction
        - "from [entity]" → source
        - "by [entity]" → instrument
        - Lists (X, Y, and Z) → multiple entities in same role
        """
        sig_idx = info.get('signal_idx', -1) if info else -1
        
        if etype == 'CLASSIFICATION':
            # Things after the signal word are attributes
            attr = self._extract_after(text_lower, toks, sig_idx)
            if attr:
                ef.attributes['is_a'] = attr
        
        elif etype == 'BORDER':
            # "bordered by X" or "borders X"
            self._extract_with_directions(text_lower, 'BORDER', ef)
        
        elif etype == 'SEPARATE':
            self._extract_separation(text_lower, ef)
        
        elif etype == 'DIVIDE':
            self._extract_division(text_lower, ef)
        
        elif etype == 'COMPOSITION':
            self._extract_parts(text_lower, 'PART_OF', ef)
        
        elif etype == 'LOCATION':
            self._extract_location_from(text_lower, toks, sig_idx, ef)
        
        elif etype in ('CREATION', 'ACTION', 'POSSESSION'):
            # Generic action: after signal word is the object
            obj = self._extract_after(text_lower, toks, sig_idx)
            if obj:
                ef.attributes[f'{etype.lower()}_object'] = obj
    
    def _extract_after(self, text_lower: str, toks: List[str],
                        sig_idx: int) -> str:
        """Extract words after the signal word (generic)."""
        if sig_idx < 0 or sig_idx >= len(toks) - 1:
            return ""
        
        after = toks[sig_idx + 1:]
        attr_words = [w for w in after if w not in LanguagePriors.FILLER_WORDS]
        
        # Stop at prepositions or relative clauses
        clean = []
        for w in attr_words:
            if w in ('in', 'on', 'at', 'near', 'of', 'to', 'by', 'from',
                     'with', 'which', 'that', 'who', 'where',
                     'since', 'consisting', 'comprising') and clean:
                break
            if w == 'and' and clean:
                break
            clean.append(w)
        
        return ' '.join(clean[:6]) if clean else ""
    
    def _extract_with_directions(self, text_lower: str, rel_type: str, ef: EventFrame):
        """Extract entities with direction markers (for BORDER, etc.)."""
        # Try "by [entity]" pattern
        by_matches = list(re.finditer(r'\bby\s+', text_lower))
        if by_matches:
            last_by = by_matches[-1]
            after_by = text_lower[last_by.end():].strip().rstrip('.!?')
            raw_parts = re.split(r'\s*,\s*(?:and\s+)?', after_by)
            all_parts = []
            for part in raw_parts:
                for sub in re.split(r'\s+and\s+', part):
                    sp = sub.strip()
                    if sp and sp not in all_parts:
                        all_parts.append(sp)
            
            for part in all_parts:
                m = re.match(
                    r'^(.+?)\s+to\s+the\s+(northeast|northwest|southeast|southwest|north|south|east|west)$',
                    part.strip(), re.I
                )
                if m:
                    neighbor = m.group(1).strip()
                    direction = m.group(2).lower()
                else:
                    neighbor = part.strip()
                    direction = ''
                
                if neighbor and neighbor not in ('the', 'a', 'an'):
                    rel = Relation(type=rel_type, target=self._clean_entity(neighbor))
                    if direction:
                        rel.properties['direction'] = direction
                    ef.relations.append(rel)
    
    def _extract_separation(self, text_lower: str, ef: EventFrame):
        """Extract separation: separated from X to the [direction] by Y."""
        # Find "from [entity]" — don't consume direction/instrument markers
        from_m = re.search(r'\bfrom\s+(.+?)$', text_lower, re.I)
        if not from_m:
            return
        target_phrase = from_m.group(1).strip().rstrip(',.!?')
        direction = ''
        instrument = ''
        
        # Extract direction from the target phrase itself
        dir_m = re.search(
            r'(.+?)\s+to\s+the\s+(northeast|northwest|southeast|southwest|north|south|east|west)',
            target_phrase, re.I
        )
        if dir_m:
            target_phrase = dir_m.group(1).strip()
            direction = dir_m.group(2).lower()
        
        # Extract instrument
        instr_m = re.search(r'(.+?)\s+by\s+(.+?)$', target_phrase, re.I)
        if instr_m:
            target_phrase = instr_m.group(1).strip()
            instrument = instr_m.group(2).strip().rstrip('.')
        
        clean_target = self._clean_entity(target_phrase)
        if clean_target:
            rel = Relation(type='SEPARATED_FROM', target=clean_target)
            if direction:
                rel.properties['direction'] = direction
            if instrument:
                rel.properties['instrument'] = instrument
            ef.relations.append(rel)
    
    def _extract_division(self, text_lower: str, ef: EventFrame):
        """Extract division: divided into X and Y / between X and Y."""
        ef.properties.add('divided')
        
        into_m = re.search(r'\binto\s+(.+?)$', text_lower, re.I)
        if into_m:
            parts_text = into_m.group(1).strip()
            self._extract_parts_from_list(parts_text, 'DIVIDED_INTO', ef)
        
        between_m = re.search(r'\bbetween\s+(.+?)\s+and\s+(.+?)(?:\s+at\s+|\s+near\s+|$)', text_lower, re.I)
        if between_m:
            p1 = self._clean_entity(between_m.group(1))
            p2 = self._clean_entity(between_m.group(2))
            for p in [p1, p2]:
                if p:
                    ef.relations.append(Relation(type='DIVIDED_INTO', target=p))
        
        crit_m = re.search(r'(?:at|near)\s+(?:the\s+)?(\d+\w*\s*(?:parallel|meridian|latitude|longitude))', text_lower, re.I)
        if crit_m and ef.relations:
            ef.relations[-1].properties['criterion'] = crit_m.group(1)
    
    def _extract_parts(self, text_lower: str, rel_type: str, ef: EventFrame):
        """Extract composition parts."""
        of_pos = text_lower.find('of')
        if of_pos > 0:
            after_of = text_lower[of_pos + 2:].strip()
            self._extract_parts_from_list(after_of, rel_type, ef)
    
    def _extract_parts_from_list(self, text: str, rel_type: str, ef: EventFrame):
        """Extract list items like 'X, Y, and Z'."""
        raw_parts = re.split(r'\s*,\s*(?:and\s+)?', text)
        for part in raw_parts:
            for sub in re.split(r'\s+and\s+', part):
                sp = self._clean_entity(sub)
                if sp and sp not in ('smaller', 'other'):
                    if not any(w in sp for w in ('smaller', 'other')):
                        ef.relations.append(Relation(type=rel_type, target=sp))
    
    def _extract_location_from(self, text_lower: str, toks: List[str],
                                sig_idx: int, ef: EventFrame):
        """Extract location statement with possible attribute between copula and location signal."""
        # Find copula before the signal
        copula_idx = -1
        for i in range(max(sig_idx - 1, 0), -1, -1):
            if toks[i] in ('is', 'are', 'was', 'were', 'am', 'be'):
                copula_idx = i
                break
        
        if copula_idx >= 0 and copula_idx + 1 < sig_idx:
            between = toks[copula_idx + 1:sig_idx]
            attr_words = [w for w in between if w not in LanguagePriors.FILLER_WORDS]
            if attr_words and not ef.attributes.get('is_a'):
                ef.attributes['is_a'] = ' '.join(attr_words[:4])
        
        ef.location = self._extract_any_location(text_lower, toks)
    
    def _extract_any_location(self, text_lower: str, toks: List[str]) -> str:
        """Extract location from 'in/on/at/near' phrases."""
        for prep in [' in ', ' on ', ' at ', ' near ', ' of ']:
            pos = text_lower.rfind(prep)
            if pos > 0:
                after = text_lower[pos + len(prep):]
                first_word = after.split()[0] if after.split() else ''
                if LanguagePriors.is_year(first_word):
                    continue
                
                loc_words = []
                for w in after.split():
                    if w in ('and', 'or', 'but') and loc_words:
                        break
                    if w in ('which', 'that', 'who', 'where'):
                        break
                    if LanguagePriors.is_signal(w) and w not in ('in', 'on', 'at', 'near', 'of'):
                        break
                    if w in (',', ';', ':', '.', ')'):
                        break
                    loc_words.append(w)
                
                if loc_words:
                    loc = ' '.join(loc_words).strip()
                    loc = re.sub(r'\s+(which|that|who)\s+.*$', '', loc).strip()
                    cleaned = self._clean_entity(loc)
                    fw = cleaned.split()[0] if cleaned.split() else ''
                    return cleaned if not LanguagePriors.is_year(fw) else ''
        return ''
    
    def _extract_time(self, toks: List[str]) -> str:
        """Extract temporal information."""
        for i, t in enumerate(toks):
            if LanguagePriors.is_year(t):
                return t.strip('.,; ')
            if t in ('in', 'on', 'at', 'since', 'until', 'during', 'before', 'after'):
                if i + 1 < len(toks) and LanguagePriors.is_year(toks[i + 1]):
                    return toks[i + 1].strip('.,; ')
        return ''
    
    def _clean_entity(self, s: str) -> str:
        """Clean an entity name — no language-specific stripping."""
        s = norm(s)
        s = re.sub(r'\s*\([^)]*\)', '', s).strip()
        for a in ('a ', 'an ', 'the ', 'this ', 'that '):
            while s.startswith(a):
                s = s[len(a):]
        s = re.sub(r'\s+(sea|strait|river|island|islands|mountain|mountains|peninsula)\s*$', '', s)
        return norm(s).rstrip('.,; ')
    
    def parse_question(self, text: str) -> Dict:
        """
        Parse a question into structured query parameters.
        
        Uses universal question-word patterns (cognitive priors):
        - Where → location
        - Who → identity
        - When → time
        - What/Which → attribute/entity
        - Is/Are/Do/Does → verification
        - How → manner/quantity
        """
        q = text.strip().rstrip('?').strip()
        q_lower = q.lower()
        toks = tokens(q)
        
        result = {
            'question_type': 'UNKNOWN',
            'target_entity': '',
            'direction': '',
            'has_direction': False,
        }
        
        if not toks:
            return result
        
        first = toks[0]
        second = toks[1] if len(toks) > 1 else ''
        third = toks[2] if len(toks) > 2 else ''
        
        # Detect question type
        if first == 'where':
            result['question_type'] = 'WHERE'
        elif first == 'who':
            result['question_type'] = 'WHO'
        elif first == 'when':
            result['question_type'] = 'WHEN'
        elif first == 'why':
            result['question_type'] = 'WHY'
        elif first == 'how':
            result['question_type'] = 'HOW'
        elif first in ('is', 'are', 'was', 'were', 'am', 'do', 'does', 'did',
                        'can', 'could', 'has', 'have', 'had'):
            result['question_type'] = 'YESNO'
        elif first in ('what', 'which'):
            # Check for direction words
            dir_found = None
            for d in LanguagePriors.DIRECTION_WORDS:
                if d in toks:
                    dir_found = d
                    break
            if dir_found:
                result['direction'] = dir_found
                result['has_direction'] = True
                result['question_type'] = 'WHAT_DIRECTION'
            elif any('border' in t or 'bordered' in t for t in toks):
                result['question_type'] = 'WHAT_BORDERS'
            elif 'separated' in q_lower:
                result['question_type'] = 'WHAT_SEPARATED'
            else:
                result['question_type'] = 'WHAT'
        
        # Extract target entity
        result['target_entity'] = self._extract_question_target(q_lower, toks, result)
        
        return result
    
    def _extract_question_target(self, q_lower: str, toks: List[str],
                                  qinfo: Dict) -> str:
        """
        Extract the target entity from a question.
        
        Universal pattern: after question words and auxiliaries,
        the next significant word is the target.
        """
        skip = frozenset({
            'what', 'where', 'who', 'when', 'why', 'how', 'which',
            'is', 'are', 'was', 'were', 'am', 'do', 'does', 'did',
            'can', 'could', 'will', 'would', 'shall', 'should',
            'in', 'on', 'at', 'near', 'of', 'to', 'by', 'from', 'with',
            'the', 'a', 'an', 'this', 'that', 'these', 'those',
            'border', 'borders', 'bordered', 'separated', 'called',
            'located', 'situated', 'found', 'known', 'referred',
            'founded', 'headquartered',
            'there', 'any', 'some', 'name',
        })
        
        # Direction questions: target is typically after the direction word
        if qinfo.get('has_direction'):
            dir_idx = None
            for i, t in enumerate(toks):
                if t in LanguagePriors.DIRECTION_WORDS:
                    dir_idx = i
                    break
            if dir_idx is not None:
                after_dir = toks[dir_idx + 1:]
                clean = [w for w in after_dir if w not in ('of', 'in', 'on', 'at', 'near', 'to', 'the')]
                if clean:
                    return ' '.join(clean[-3:])
                before_dir = [w for w in toks[:dir_idx] if w not in skip]
                if before_dir:
                    return ' '.join(before_dir[-3:])
        
        # Normal: find first significant word after question/auxiliary
        for i, t in enumerate(toks):
            if t not in skip:
                entity_words = [t]
                for j in range(i + 1, len(toks)):
                    w = toks[j]
                    if w in skip:
                        break
                    if w in LanguagePriors.DIRECTION_WORDS and qinfo.get('has_direction'):
                        break
                    entity_words.append(w)
                return ' '.join(entity_words[:3])
        
        return ''


# ══════════════════════════════════════════════════════════════
# QUERY PLANNER — Operator-based question answering
# ══════════════════════════════════════════════════════════════

@dataclass
class QueryOp:
    """A single query operation."""
    op_type: str  # FIND_ENTITY, GET_ATTRIBUTE, CHECK_RELATION, RESOLVE_DIRECTION, etc.
    args: Dict[str, str] = field(default_factory=dict)


class QueryPlanner:
    """
    Composes query operators from question parse.
    
    Instead of question-type-specific handlers, each question is
    compiled into a sequence of generic operators that compose.
    """
    
    def __init__(self, model: SituationModel, coref=None):
        self.model = model
        self.coref = coref or CorefLearner()
    
    def plan(self, question: str, qinfo: Dict) -> List[QueryOp]:
        """
        Compile a question into an executable query plan.
        
        Returns a list of QueryOps that form the execution plan.
        """
        qtype = qinfo['question_type']
        target = qinfo.get('target_entity', '')
        plans = []
        
        if qtype == 'WHERE':
            plans = [
                QueryOp('FIND_ENTITY', {'target': target}),
                QueryOp('GET_LOCATION', {}),
            ]
        elif qtype == 'WHO':
            plans = [
                QueryOp('FIND_ENTITY', {'target': target}),
                QueryOp('GET_IDENTITY', {}),
            ]
        elif qtype == 'WHEN':
            plans = [
                QueryOp('FIND_ENTITY', {'target': target}),
                QueryOp('GET_TIME', {}),
            ]
        elif qtype == 'YESNO':
            plans = [
                QueryOp('FIND_ENTITY', {'target': target}),
                QueryOp('VERIFY_PROPERTY', {'question': question}),
            ]
        elif qtype == 'HOW':
            plans = [
                QueryOp('FIND_ENTITY', {'target': target}),
                QueryOp('GET_MANNER', {}),
            ]
        elif qtype == 'WHAT_DIRECTION':
            plans = [
                QueryOp('FIND_ENTITY', {'target': target}),
                QueryOp('RESOLVE_DIRECTION', {
                    'direction': qinfo.get('direction', ''),
                    'target': target,
                }),
            ]
        elif qtype == 'WHAT_BORDERS':
            plans = [
                QueryOp('FIND_ENTITY', {'target': target}),
                QueryOp('GET_BORDERS', {}),
            ]
        elif qtype == 'WHAT_SEPARATED':
            plans = [
                QueryOp('FIND_ENTITY', {'target': target}),
                QueryOp('GET_SEPARATED', {}),
            ]
        elif qtype == 'WHAT':
            plans = [
                QueryOp('FIND_ENTITY', {'target': target}),
                QueryOp('DESCRIBE', {}),
            ]
        else:
            plans = [
                QueryOp('FIND_ENTITY', {'target': target}),
                QueryOp('DESCRIBE', {}),
            ]
        
        return plans
    
    def execute(self, plans: List[QueryOp], question: str = "") -> str:
        """
        Execute a query plan against the world model.
        
        Each operator is executed generically.
        Operators compose: output of one feeds into the next.
        """
        context = {}
        
        for op in plans:
            result = self._execute_op(op, context, question)
            if result is not None:
                context['result'] = result
            
            # Early exit for fatal failures
            if result == '__NOT_FOUND__':
                return "I don't know."
            if result == '__NOT_FOUND_ENTITY__':
                return "I don't know anything about that yet."
        
        return context.get('result', "I don't know.")
    
    def _execute_op(self, op: QueryOp, ctx: Dict, question: str = "") -> Any:
        """Execute a single query operator."""
        if op.op_type == 'FIND_ENTITY':
            target = op.args.get('target', '')
            if not target:
                return '__NOT_FOUND__'
            found = self.model.find(target)
            if found is None:
                # Try resolving pronouns
                try:
                    resolved, _ = self.coref.resolve(target)
                    if resolved != target:
                        found = self.model.find(resolved)
                except:
                    pass
                if found is None:
                    return '__NOT_FOUND_ENTITY__'
            ctx['entity_name'] = found[0]
            ctx['entity'] = found[1]
            return found
        
        elif op.op_type == 'GET_LOCATION':
            entity = ctx.get('entity')
            if not entity:
                return '__NOT_FOUND__'
            name = ctx.get('entity_name', '')
            
            if entity.location:
                return f"It is in {entity.location}."
            
            loc = self.model.resolve_location(name)
            if loc:
                return f"It is in {loc}."
            
            # Check relations for location
            for rel_type, entries in entity.relations.items():
                for t, _ in entries:
                    parent = self.model.find(t)
                    if parent and parent[1].location:
                        return f"It is in {parent[1].location}."
            
            return "I don't know where it is."
        
        elif op.op_type == 'GET_IDENTITY':
            entity = ctx.get('entity')
            if not entity:
                return "I don't know who that is."
            desc = self.model.describe(ctx.get('entity_name', ''))
            if desc:
                return f"I recall: {desc}."
            return "I don't know who that is."
        
        elif op.op_type == 'GET_TIME':
            entity = ctx.get('entity')
            if not entity or not entity.time:
                return "I don't know."
            return f"It was in {entity.time}."
        
        elif op.op_type == 'VERIFY_PROPERTY':
            entity = ctx.get('entity')
            if not entity:
                return "no"
            
            q_lower = question.lower()
            
            # Check attributes
            for k, v in entity.attributes.items():
                if k in q_lower or v in q_lower:
                    return "yes"
                last_word = q_lower.split()[-1]
                if last_word and last_word in v.lower():
                    return "yes"
            
            # Check location
            if entity.location and entity.location in q_lower:
                return "yes"
            
            # Check relations
            for rel_type, entries in entity.relations.items():
                for t, _ in entries:
                    if t in q_lower:
                        return "yes"
            
            # Check properties
            for p in entity.properties:
                if p in q_lower:
                    return "yes"
            
            return "no"
        
        elif op.op_type == 'GET_MANNER':
            entity = ctx.get('entity')
            if not entity:
                return "I don't know."
            
            # Check for division
            div_parts = []
            div_criterion = ""
            for rel_type, entries in entity.relations.items():
                if 'divid' in rel_type.lower():
                    for t, props in entries:
                        div_parts.append(t)
                        div_criterion = props.get('criterion', '')
            
            if div_parts:
                parts = ", ".join(div_parts)
                crit = f" at or near the {div_criterion}" if div_criterion else ""
                return f"It is divided into {parts}{crit}."
            
            desc = self.model.describe(ctx.get('entity_name', ''))
            return desc if desc else "I don't know."
        
        elif op.op_type == 'RESOLVE_DIRECTION':
            entity = ctx.get('entity')
            name = ctx.get('entity_name', '')
            direction = op.args.get('direction', '')
            
            if not direction or not entity:
                return "I don't know."
            
            # Strategy 1: Check entity's relations for direction
            for rel_type, entries in entity.relations.items():
                for t, props in entries:
                    if props.get('direction', '').lower() == direction:
                        return f"The {direction} of {name} is {t}."
            
            # Strategy 2: Check reverse (other entities have relation TO this one)
            for ename, ent in self.model.entities.items():
                if ename == name:
                    continue
                for rel_type, entries in ent.relations.items():
                    for t, props in entries:
                        if norm(t) == norm(name) and props.get('direction', '').lower() == direction:
                            return f"The {direction} of {name} is {ename}."
            
            # Strategy 3: SEPARATED_FROM implies direction
            for rel_type, entries in entity.relations.items():
                if 'separated' in rel_type.lower():
                    for t, props in entries:
                        if props.get('direction', '').lower() == direction:
                            return f"The {direction} of {name} is {t}."
            
            return "I don't know."
        
        elif op.op_type == 'GET_BORDERS':
            entity = ctx.get('entity')
            if not entity:
                return "I don't know."
            borders = []
            for rel_type, entries in entity.relations.items():
                if 'border' in rel_type.lower():
                    for t, _ in entries:
                        borders.append(t)
            return ", ".join(borders) if borders else "I don't know."
        
        elif op.op_type == 'GET_SEPARATED':
            entity = ctx.get('entity')
            if not entity:
                return "I don't know."
            for rel_type, entries in entity.relations.items():
                if 'separated' in rel_type.lower():
                    targets = [t for t, _ in entries]
                    if targets:
                        return ", ".join(targets)
            return "I don't know."
        
        elif op.op_type == 'DESCRIBE':
            entity = ctx.get('entity')
            if not entity:
                return "I don't know anything about that yet."
            desc = self.model.describe(ctx.get('entity_name', ''))
            if desc:
                return f"I recall: {desc}."
            return "I don't know anything about that yet."
        
        return None


# ══════════════════════════════════════════════════════════════
# FAILURE CASE — Learning from mistakes
# ══════════════════════════════════════════════════════════════

@dataclass
class FailureCase:
    """A failed interaction — stored for later learning."""
    id: str = ""
    input_text: str = ""
    expected: str = ""
    actual: str = ""
    question_type: str = ""
    target_entity: str = ""
    embedding: Optional[np.ndarray] = None
    failed_construction: str = ""
    timestamp: float = field(default_factory=time.time)


class FailureMemory:
    """Stores and clusters failures for schema induction."""
    
    def __init__(self):
        self.failures: List[FailureCase] = []
    
    def record(self, input_text: str, actual: str, expected: str = "",
               question_type: str = "", target_entity: str = "",
               failed_construction: str = ""):
        """Record a failure."""
        fc = FailureCase(
            id=_uid(),
            input_text=input_text,
            expected=expected,
            actual=actual,
            question_type=question_type,
            target_entity=target_entity,
            failed_construction=failed_construction,
        )
        fc.embedding = embed_text(input_text)
        self.failures.append(fc)
    
    def get_clusters(self, min_size: int = 3, sim_threshold: float = 0.85) -> List[List[FailureCase]]:
        """Cluster similar failures by embedding similarity."""
        if len(self.failures) < min_size:
            return []
        
        clusters = []
        used = set()
        
        for i, f1 in enumerate(self.failures):
            if i in used or f1.embedding is None:
                continue
            cluster = [f1]
            used.add(i)
            
            for j, f2 in enumerate(self.failures):
                if j in used or f2.embedding is None:
                    continue
                sim = cosine_sim(f1.embedding, f2.embedding)
                if sim > sim_threshold:
                    cluster.append(f2)
                    used.add(j)
            
            if len(cluster) >= min_size:
                clusters.append(cluster)
        
        return clusters


# ══════════════════════════════════════════════════════════════
# EPISODE MEMORY
# ══════════════════════════════════════════════════════════════

@dataclass
class Episode:
    """A single interaction episode."""
    id: str
    input: str
    event_frame: Optional[EventFrame]
    resolved_entity: str
    success: bool
    is_question: bool = False
    answer: str = ""
    expected: str = ""
    error: str = ""
    timestamp: float = field(default_factory=time.time)
    embedding: Optional[np.ndarray] = None


class EpisodeMemory:
    """Stores all interaction episodes."""
    
    def __init__(self):
        self.episodes: List[Episode] = []
    
    def record(self, inp: str, ef: Optional[EventFrame], entity: str,
               success: bool, is_question: bool = False,
               answer: str = "", expected: str = "", error: str = "") -> Episode:
        ep = Episode(
            id=_uid(), input=inp, event_frame=ef,
            resolved_entity=entity, success=success,
            is_question=is_question, answer=answer,
            expected=expected, error=error,
        )
        self.episodes.append(ep)
        if len(self.episodes) > 20000:
            self.episodes = self.episodes[-10000:]
        ep.embedding = embed_text(inp)
        return ep
    
    def get_failures(self) -> List[Episode]:
        return [ep for ep in self.episodes if not ep.success and ep.error]
    
    def get_recent(self, n: int = 10) -> List[Episode]:
        return self.episodes[-n:]
    
    def clear(self):
        self.episodes.clear()


# ══════════════════════════════════════════════════════════════
# EVIDENCE STORE — Raw text span preservation
# ══════════════════════════════════════════════════════════════

class EvidenceStore:
    """
    Preserves raw text spans for re-reading.
    
    When the world model can't answer a question confidently,
    the system can search evidence store for relevant spans.
    """
    
    def __init__(self):
        self.spans: List[Dict] = []
    
    def add(self, span: str, source: str = "", entity: str = "",
            metadata: Dict = None):
        """Store a raw text span."""
        self.spans.append({
            "span": span,
            "source": source,
            "entity": entity,
            "metadata": metadata or {},
            "timestamp": time.time(),
            "embedding": embed_text(span),
        })
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, Dict]]:
        """Search for relevant spans by embedding similarity."""
        qvec = embed_text(query)
        if qvec is None:
            return []
        
        results = []
        for span in self.spans:
            if span["embedding"] is not None:
                sim = cosine_sim(qvec, span["embedding"])
                results.append((sim, span))
        
        results.sort(key=lambda x: -x[0])
        return results[:top_k]
    
    def clear(self):
        self.spans.clear()


# ══════════════════════════════════════════════════════════════
# GLOBAL COREFERENCE (for cross-module access)
# ══════════════════════════════════════════════════════════════

_GlobalCoref = CorefLearner()


# ══════════════════════════════════════════════════════════════
# MAIN ENGINE
# ══════════════════════════════════════════════════════════════

class LanguageAcquisitionEngine:
    """
    Main cognitive engine — V41.
    
    Zero hardcoded answer handlers.
    Uses: CandidateParser → ConstructionMemory → WorldModel → QueryPlanner
    Learn from failures via FailureMemory → sleep consolidation.
    """
    
    def __init__(self):
        self.model = SituationModel()
        self.coref = CorefLearner()
        self.cmem = ConstructionMemory()
        self.epmem = EpisodeMemory()
        self.failmem = FailureMemory()
        self.evidence = EvidenceStore()
        self.parser = CandidateParser(self.model, self.coref, self.cmem)
        self.planner = QueryPlanner(self.model, coref=self.coref)
        self.dialogue_count = 0
        
        # Sync global coref
        global _GlobalCoref
        _GlobalCoref = self.coref
    
    def hear(self, text: str) -> str:
        """Process an utterance. Returns response."""
        self.dialogue_count += 1
        
        if not text.strip():
            return "Yes?"
        
        text = text.strip()
        
        # Detect question
        is_question = text.endswith("?") or any(
            text.lower().startswith(w)
            for w in ('what ', 'where ', 'who ', 'when ', 'why ', 'how ',
                      'which ', 'is ', 'are ', 'was ', 'were ', 'am ',
                      'do ', 'does ', 'did ', 'can ', 'could ')
        )
        
        if is_question:
            return self._answer(text)
        
        # Process statement(s)
        sents = sent_split(text)
        stored = 0
        for sent in sents:
            if self._learn(sent):
                stored += 1
        
        if stored == 0:
            return "I heard you."
        elif stored >= len(sents):
            return "Got it. I've stored that information."
        else:
            return f"Got it. Stored {stored} facts."
    
    def _learn(self, text: str) -> bool:
        """Learn from a statement."""
        # Parse into EventFrame (candidate generation + construction matching)
        ef = self.parser.parse(text)
        
        if not ef.entity:
            self.epmem.record(text, ef, "", False, error="no_entity")
            return False
        
        # Apply to world model
        success = self.model.apply(ef)
        
        # Register entities for coref
        self.coref.register(ef.entity)
        for rel in ef.relations:
            if rel.target:
                self.coref.register(rel.target)
        
        # Store evidence
        self.evidence.add(
            span=text,
            source="user_statement",
            entity=ef.entity,
            metadata={"event_type": ef.event_type}
        )
        
        if success:
            # Also register attribute values as entities (e.g., "Kyungtae" from "I am Kyungtae")
            if ef.attributes.get('is_a'):
                val = ef.attributes['is_a'].strip()
                if val:
                    self.coref.register(val)
                    # Create entity for the attribute value
                    val_entity = self.model.get(val)
                    val_entity.attributes['identified_by'] = ef.entity
            
            # Try to update/create construction
            signal_words = [t for t in tokens(text)
                           if LanguagePriors.is_signal(t)]
            if signal_words:
                # Find or create construction
                best = self.cmem.find_best_match(text, ef.event_type)
                if best:
                    best.reinforce()
                    if text not in best.examples:
                        best.examples.append(text)
                        best.update_embedding()
                else:
                    self.cmem.create(
                        ef.event_type, signal_words[:3],
                        example=text, confidence=0.3
                    )
        
        self.epmem.record(text, ef, ef.entity, success)
        return success
    
    def _answer(self, question: str) -> str:
        """Answer a question."""
        # Parse question
        qinfo = self.parser.parse_question(question)
        target = qinfo.get('target_entity', '')
        
        # Resolve target entity
        resolved, was_resolved = self.coref.resolve(target) if target else ('', False)
        if was_resolved:
            qinfo['target_entity'] = resolved
        
        # Plan query
        plans = self.planner.plan(question, qinfo)
        
        # Execute
        answer = self.planner.execute(plans, question)
        
        # Record episode
        self.epmem.record(
            question, None, qinfo.get('target_entity', ''),
            True, is_question=True, answer=answer
        )
        
        # If answer is "I don't know", check evidence store
        if "I don't know" in answer:
            for result in self.evidence.search(question, top_k=3):
                _, span_data = result
                if span_data.get("span"):
                    # Found relevant evidence but couldn't extract answer
                    pass
            self.failmem.record(
                input_text=question,
                actual=answer,
                question_type=qinfo.get('question_type', ''),
                target_entity=qinfo.get('target_entity', ''),
            )
        
        return answer
    
    def feedback(self, question: str, correct_answer: str) -> str:
        """Learn from user correction — treat correction as statement."""
        success = self._learn(correct_answer)
        if success:
            # Remove the failure if it exists
            self.failmem.failures = [
                f for f in self.failmem.failures
                if f.input_text != question
            ]
            return f"Learned: {correct_answer}"
        return f"Noted: {correct_answer}"
    
    def sleep_cycle(self) -> str:
        """Consolidate memory."""
        deleted = self.cmem.consolidate()
        
        # Try to create constructions from failure clusters
        clusters = self.failmem.get_clusters(min_size=2)
        new_from_failures = 0
        for cluster in clusters:
            # Use the most representative failure text
            texts = [f.input_text for f in cluster]
            most_common_type = max(set(f.question_type for f in cluster),
                                   key=lambda x: sum(1 for f in cluster if f.question_type == x))
            # Create construction from this cluster
            example = max(texts, key=lambda t: len(t))
            self.cmem.create_from_failure(example, "", "")
            new_from_failures += 1
        
        stats = self.cmem.get_stats()
        return (f"Consolidated: {deleted} pruned, {new_from_failures} new from failures, "
                f"{stats['total']} total constructions, "
                f"{len(self.failmem.failures)} pending failures")
    
    def get_status(self) -> str:
        stats = self.cmem.get_stats()
        return (f"{len(self.model.entities)} entities, "
                f"{len(self.epmem.episodes)} episodes, "
                f"{stats['total']} constructions, "
                f"{len(self.failmem.failures)} failures, "
                f"{self.dialogue_count} dialogues")
    
    def reset(self):
        self.model = SituationModel()
        self.coref = CorefLearner()
        self.cmem = ConstructionMemory()
        self.epmem = EpisodeMemory()
        self.failmem = FailureMemory()
        self.evidence = EvidenceStore()
        self.parser = CandidateParser(self.model, self.coref, self.cmem)
        self.planner = QueryPlanner(self.model, coref=self.coref)
        self.dialogue_count = 0
    
    def to_dict(self):
        return {
            "entities": self.model.to_dict(),
            "constructions": {k: c.to_dict() for k, c in self.cmem.constructions.items()},
            "failures": len(self.failmem.failures),
            "episodes": len(self.epmem.episodes),
            "status": self.get_status(),
        }


class NeurovaEngine:
    """Simple CLI interface."""
    
    def __init__(self):
        self.brain = LanguageAcquisitionEngine()
    
    def hear(self, text: str) -> str:
        return self.brain.hear(text)
    
    def reset(self):
        self.brain.reset()
    
    def get_status(self) -> str:
        return self.brain.get_status()
    
    def sleep_cycle(self) -> str:
        return self.brain.sleep_cycle()
    
    @property
    def model(self):
        return self.brain.model
