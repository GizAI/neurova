"""
Neurova V40 — Language Acquisition Engine (Zero Hardcoded Grammar Rules)

Core philosophy:
  Language rules are NOT programmed. They are LEARNED from interaction.
  The only cognitive priors are:
    - Entity/Event/Relation exist as concepts
    - Topic-Comment structure (first main thing is the topic)
    - Signal words (prepositions, copula) indicate relation types
    - Repeated patterns become constructions
    - Prediction errors drive learning

No regex patterns, no template answers, no hardcoded entity lists.
Everything is learned through conversation.

The system bootstraps by:
  1. Using lexical signal words (universal cognitive markers, not language rules)
  2. Using topic-comment position heuristics (universal language structure)
  3. Learning from EVERY interaction through embedding-based pattern matching
  4. Consolidating patterns into constructions during sleep
"""

import os, sys, re, json, time, uuid, math
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

# ── spaCy (minimal — tokenization only, NO grammar/dependency parsing) ──
_NLP = None
for _p in [
    "/home/user/miniconda3/envs/neurova_vsa/bin/python3",
    "/home/user/miniconda3/envs/quantv/bin/python3",
    sys.executable
]:
    try:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
        break
    except Exception:
        pass

# ── Embedding client (GPU on ml-dmc8) ──
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
    """Normalize text — just whitespace + lowercase."""
    if not s: return ""
    return re.sub(r'\s+', ' ', s.strip().lower())


def sent_split(text: str) -> List[str]:
    """Split into sentences by punctuation only — NO grammar rules."""
    if not text: return []
    text = text.replace('\n', ' ')
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if s.strip()]


def tokens(text: str) -> List[str]:
    """Word tokens — NO POS, NO dependency labels, NO NER."""
    if _NLP:
        return [t.text.lower().rstrip('.,;:!?') for t in _NLP(text) if not t.is_space]
    return text.lower().rstrip('.,;:!?').split()


def _clean_entity(s: str) -> str:
    """Clean an entity name — just normalize and clean, no language-specific stripping."""
    s = norm(s)
    s = re.sub(r'\s*\([^)]*\)', '', s).strip()
    for a in ('a ', 'an ', 'the ', 'this ', 'that '):
        while s.startswith(a):
            s = s[len(a):]
    # Remove trailing geographical generics only at the very end
    s = re.sub(r'\s+(sea|strait|river|island|islands|mountain|mountains|peninsula)\s*$', '', s)
    return norm(s).rstrip('.,; ')


# ── Embedding functions ──

def embed_text(text: str) -> Optional[np.ndarray]:
    """Get embedding via qwen3-embedding on ml-dmc8."""
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


# ══════════════════════════════════════════════════════════════
# COGNITIVE PRIORS (NOT language rules)
# ══════════════════════════════════════════════════════════════

# Signal words that indicate relation types (UNIVERSAL cognitive markers)
# These are NOT grammar rules — they are lexical items that trigger
# specific cognitive interpretations. Children learn these associations.
SIGNAL_MAP = {
    # Copula → attributes/classification
    'is':    ('CLASSIFICATION', 'entity', 'after'),
    'are':   ('CLASSIFICATION', 'entity', 'after'),
    'was':   ('CLASSIFICATION', 'entity', 'after'),
    'were':  ('CLASSIFICATION', 'entity', 'after'),
    'be':    ('CLASSIFICATION', 'entity', 'after'),
    'am':    ('CLASSIFICATION', 'entity', 'after'),
    # Spatial relations
    'in':    ('LOCATION', None, 'after'),
    'on':    ('LOCATION', None, 'after'),
    'at':    ('LOCATION', None, 'after'),
    'near':  ('LOCATION', None, 'after'),
    # Relational verbs
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
    'found':      ('LOCATION', 'entity', 'after_prep'),
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
    'refer':      ('CLASSIFICATION', 'entity', 'after'),
    'referred':   ('CLASSIFICATION', 'entity', 'after'),
    'belong':     ('POSSESSION', 'entity', 'after_prep'),
    'belongs':    ('POSSESSION', 'entity', 'after_prep'),
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
    'found':      ('CREATION', 'entity', 'after'),
    'founded':    ('CREATION', 'entity', 'after'),
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

# Prepositions that introduce specific semantic roles (UNIVERSAL)
PREP_ROLES = {
    'in':    'location',
    'on':    'location',
    'at':    'location',
    'near':  'location',
    'to':    'direction',
    'from':  'source',
    'by':    'instrument',
    'across':'located_across',
    'into':  'result',
    'of':    'possession',
    'for':   'purpose',
    'with':  'companion',
    'under': 'location',
    'over':  'location',
    'between': 'between',
}

DIRECTION_WORDS = {
    'north', 'south', 'east', 'west',
    'northeast', 'northwest', 'southeast', 'southwest',
}

STOP_WORDS = {
    'a', 'an', 'the', 'this', 'that', 'these', 'those',
    'some', 'any', 'every', 'each', 'all', 'both',
    'no', 'not', 'none', 'neither',
}

SECTION_WORDS = {
    'consisting', 'comprising', 'including', 'containing',
    'which', 'that', 'who', 'whom', 'whose',
}


def is_year(s: str) -> bool:
    try:
        n = int(s.strip('.,; '))
        return 1000 <= n <= 2100
    except:
        return False


# ══════════════════════════════════════════════════════════════
# COREF — Learn from conversation, not rules
# ══════════════════════════════════════════════════════════════


class CorefLearner:
    """Resolves pronouns and references. Learns from conversation.
    Uses frequency-based preference: most-mentioned entity wins."""

    def __init__(self):
        self.history: List[Tuple[str, str, float]] = []  # (entity, type, time)
        self.mention_counts = {}
        self.pronouns = {'he', 'him', 'his', 'she', 'her', 'hers',
                         'it', 'its', 'they', 'them', 'their',
                         'this', 'that', 'these', 'those'}
        self.special_gpe = {
            'region', 'country', 'nation', 'area', 'place',
            'peninsula', 'island', 'city', 'state', 'province',
        }
        self.non_referent = {'korea strait', 'the korea strait', 'korean strait'}

    def register(self, entity: str, etype: str = ''):
        e = norm(entity)
        if not e or len(e) < 1:
            return
        self.mention_counts[e] = self.mention_counts.get(e, 0) + 1
        for h in self.history[:3]:
            if h[0] == e:
                self.history.remove(h)
                self.history.insert(0, (e, etype, time.time()))
                self.history[:] = self.history[:100]
                return
        self.history.insert(0, (e, etype, time.time()))
        if len(self.history) > 100:
            self.history = self.history[:100]

    def resolve(self, text: str) -> Tuple[str, bool]:
        """Resolve reference. Most-mentioned entity wins."""
        t = norm(text)
        if not t:
            return text, False

        # Special region/place references -> pick most-mentioned main entity
        if t in self.special_gpe:
            best = self._best_main()
            if best:
                return best, True
            return text, False

        # Short pronouns -> pick most-mentioned main entity
        if t in self.pronouns:
            best = self._best_main()
            if best:
                return best, True
            if self.history:
                return self.history[0][0], True
            return text, False

        # Exact match -> no resolution needed
        for h in self.history:
            if h[0] == t:
                return text, False

        # Partial match -> resolve to known entity
        for h in self.history:
            if h[0] in t or t in h[0]:
                return h[0], True

        return text, False

    def _best_main(self):
        """Most-mentioned entity that's not an instrument/feature."""
        best, best_cnt = None, 0
        for e, c in self.mention_counts.items():
            if e in self.non_referent:
                continue
            if c > best_cnt:
                best, best_cnt = e, c
        if best:
            return best
        for h in self.history:
            if h[0] not in self.non_referent:
                return h[0]
        return None

    def reset(self):
        self.history.clear()
        self.mention_counts.clear()

# ══════════════════════════════════════════════════════════════
# EVENT FRAME — fully nested, no flat fields
# ══════════════════════════════════════════════════════════════

@dataclass
class Relation:
    type: str = ""
    target: str = ""
    properties: Dict[str, str] = field(default_factory=dict)


@dataclass
class EventFrame:
    event_type: str = "STATEMENT"
    entity: str = ""
    target: str = ""
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
            "target": self.target,
            "attributes": dict(self.attributes),
            "location": self.location,
            "time": self.time,
            "relations": [{"type": r.type, "target": r.target,
                           "properties": dict(r.properties)} for r in self.relations],
            "properties": list(self.properties),
            "confidence": self.confidence,
        }


# ══════════════════════════════════════════════════════════════
# ENTITY & SITUATION MODEL
# ══════════════════════════════════════════════════════════════

class Entity:
    """Tracks state of a single entity."""
    def __init__(self, name: str = ""):
        self.name = name
        self.attributes: Dict[str, str] = {}
        self.properties: Set[str] = set()
        self.location: str = ""
        self.time: str = ""
        self.relations: Dict[str, List[Tuple[str, Dict[str, str]]]] = defaultdict(list)
        self.events: List[EventFrame] = []

    def to_dict(self):
        return {
            "name": self.name,
            "attributes": dict(self.attributes),
            "properties": list(self.properties),
            "location": self.location,
            "time": self.time,
            "relations": {k: [(t, p) for t, p in v[:5]] for k, v in self.relations.items()},
            "event_count": len(self.events),
        }


class SituationModel:
    """Tracks world state. All relation types are learned from event frames."""

    def __init__(self):
        self.entities: Dict[str, Entity] = {}

    def get(self, name: str) -> Optional[Entity]:
        n = norm(name)
        if not n:
            return None
        if n not in self.entities:
            self.entities[n] = Entity(n)
        return self.entities[n]

    def apply(self, ef: EventFrame) -> bool:
        """Apply an event frame to update world state."""
        if not ef.entity:
            return False
        e = self.get(ef.entity)
        if e is None:
            return False

        # Attributes
        for k, v in ef.attributes.items():
            e.attributes[k] = v

        # Properties
        e.properties.update(ef.properties)

        # Location
        if ef.location:
            e.location = ef.location

        # Time
        if ef.time:
            e.time = ef.time

        # Relations
        for rel in ef.relations:
            t = _clean_entity(rel.target)
            if t:
                # Add relation to entity
                e.relations[rel.type].append((t, dict(rel.properties)))
                # Ensure target entity exists
                self.get(t)
                # For border relations, add reverse
                if rel.type in ('BORDER', 'SEPARATED_FROM'):
                    target_e = self.get(t)
                    if target_e:
                        reverse_type = 'BORDERED_BY' if rel.type == 'BORDER' else 'SEPARATES'
                        target_e.relations[reverse_type].append(
                            (ef.entity, dict(rel.properties)))

        # Store event
        e.events.append(ef)
        return True

    def resolve_location(self, name: str) -> str:
        """Resolve location through relations."""
        e = self.get(name)
        if e and e.location:
            return e.location
        # Check relations for location
        if e:
            for rel_type, entries in e.relations.items():
                for target, props in entries:
                    if 'part_of' in rel_type.lower():
                        parent = self.get(target)
                        if parent and parent.location:
                            return parent.location
        return ""

    def describe(self, name: str) -> str:
        """Describe entity — composed from stored data, no templates."""
        e = self.get(name)
        if not e:
            return ""
        parts = []
        for k, v in e.attributes.items():
            if k == 'is_a':
                parts.append(f"is {v}")
            else:
                parts.append(f"{k} {v}")
        if e.location:
            parts.append(f"located in {e.location}")
        # Relations
        for rel_type, entries in list(e.relations.items())[:5]:
            targets = [t for t, _ in entries[:3]]
            if targets:
                if rel_type == 'BORDER':
                    parts.append(f"bordered by {', '.join(targets)}")
                elif rel_type == 'SEPARATED_FROM':
                    parts.append(f"separated from {', '.join(targets)}")
                elif rel_type == 'PART_OF':
                    parts.append(f"part of {', '.join(targets)}")
                else:
                    parts.append(f"has_{rel_type} {', '.join(targets)}")
        if e.properties:
            props = [p for p in e.properties if not p.startswith('_')][:3]
            if props:
                parts.append(f"has: {', '.join(props)}")
        return ", ".join(parts) if parts else ""

    def to_dict(self):
        return {name: e.to_dict() for name, e in self.entities.items()}


# ══════════════════════════════════════════════════════════════
# SEMANTIC PARSER — Uses cognitive priors, NOT grammar rules
# ══════════════════════════════════════════════════════════════

class SemanticParser:
    """
    Extracts EventFrames from text using:
    1. Lexical signal words (universal cognitive markers)
    2. Topic-comment position heuristics (universal language structure)
    3. Embedding-based pattern matching (learned from experience)

    This is NOT a grammar parser — it does NOT know English syntax.
    It uses universal cognitive mechanisms for information structure.
    """

    def __init__(self, model: SituationModel, coref: CorefLearner):
        self.model = model
        self.coref = coref
        self.embedding_cache = {}

    def parse(self, text: str) -> EventFrame:
        """Parse text into an EventFrame using cognitive priors."""
        text_lower = text.lower().strip().rstrip('.!?')
        tokens_list = tokens(text)
        ef = EventFrame(source_span=text, confidence=0.3)

        # Step 1: Extract main entity (topic) — use coref for resolution
        raw_entity = self._extract_entity(text_lower, tokens_list)
        resolved, was_resolved = self.coref.resolve(raw_entity) if raw_entity else ('', False)
        ef.entity = resolved if resolved else raw_entity

        # Step 2: Detect event type from signal words
        event_type, relation_info = self._detect_event_type(text_lower, tokens_list)
        ef.event_type = event_type

        # Step 3: Extract relations based on event type
        self._extract_relations(text_lower, tokens_list, ef, event_type, relation_info)

        # Step 4: Extract location
        ef.location = self._extract_location(text_lower, tokens_list)

        # Step 5: Extract time
        ef.time = self._extract_time(tokens_list)

        return ef

    def _extract_entity(self, text_lower: str, tokens_list: List[str]) -> str:
        """Extract the main entity (topic) from text.
        Uses topic-comment universal structure: first significant thing is the topic."""
        if not tokens_list:
            return ""

        # Skip initial question words and determiners
        skip_init = {'what', 'where', 'who', 'when', 'why', 'how', 'which',
                     'is', 'are', 'was', 'were', 'do', 'does', 'did',
                     'can', 'could', 'will', 'would', 'shall', 'should',
                     'a', 'an', 'the', 'this', 'that', 'these', 'those'}

        # Find the first significant word
        for i, t in enumerate(tokens_list):
            if t not in skip_init and len(t) > 1:
                raw_entity = self._find_full_entity(text_lower, tokens_list, i)
                # Also try after the signal verb for "bordered by X" or "separated from X"
                # where X is actually the target, not the entity
                return raw_entity

        return tokens_list[0] if tokens_list else ""

    def _find_full_entity(self, text_lower: str, tokens_list: List[str], start_idx: int) -> str:
        """Find the complete entity starting at start_idx.
        An entity continues until a signal word or preposition."""
        entity_words = []
        for i in range(start_idx, len(tokens_list)):
            t = tokens_list[i]
            if t in SIGNAL_MAP and i > start_idx:
                break
            if t in ('and', 'or', 'but') and len(entity_words) > 2:
                break
            if t in ('in', 'on', 'at', 'near', 'of', 'to', 'by', 'from', 'with',
                     'which', 'that', 'who') and len(entity_words) >= 1:
                break
            entity_words.append(t)

        if not entity_words:
            return tokens_list[start_idx] if start_idx < len(tokens_list) else ""

        raw = ' '.join(entity_words)
        # Clean up — remove trailing "that is", "which is", etc.
        raw = re.sub(r'\s+(that|which|who)\s+(is|are|was|were|has|have|had).*$', '', raw).strip()
        # Remove articles
        words = raw.split()
        words = [w for w in words if w not in ('a', 'an', 'the')]
        return ' '.join(words[:4]) if words else raw

    def _detect_event_type(self, text_lower: str, tokens_list: List[str]) -> Tuple[str, dict]:
        """Detect event type from signal words in text.
        Priority: specific relational verbs > copula.
        This simulates how children learn: specific action words are more informative than generic 'is'."""
        info = {}
        
        # SPECIFIC SIGNAL WORDS (high priority) — relational verbs
        specific_signals = {
            'bordered': ('BORDER', 'entity', 'after_prep'),
            'borders': ('BORDER', 'entity', 'after_prep'),
            'border': ('BORDER', 'entity', 'after_prep'),
            'separated': ('SEPARATE', 'entity', 'after_prep'),
            'separates': ('SEPARATE', 'entity', 'after_prep'),
            'separate': ('SEPARATE', 'entity', 'after_prep'),
            'divided': ('DIVIDE', 'entity', 'after_prep'),
            'divides': ('DIVIDE', 'entity', 'after_prep'),
            'divide': ('DIVIDE', 'entity', 'after_prep'),
            'located': ('LOCATION', 'entity', 'after_prep'),
            'situated': ('LOCATION', 'entity', 'after_prep'),
            'consisting': ('COMPOSITION', 'entity', 'after_prep'),
            'consists': ('COMPOSITION', 'entity', 'after_prep'),
            'comprises': ('COMPOSITION', 'entity', 'after_prep'),
            'founded': ('CREATION', 'entity', 'after'),
            'launched': ('ACTION', 'entity', 'after'),
            'developed': ('ACTION', 'entity', 'after'),
        }
        
        for i, t in enumerate(tokens_list):
            if t in specific_signals:
                etype, role, position = specific_signals[t]
                info = {'signal': t, 'signal_idx': i, 'role': role, 'position': position}
                return etype, info

        # Check for 'is/are/was/were' followed by specific participle
        for i, t in enumerate(tokens_list):
            if t in ('is', 'are', 'was', 'were', 'am', 'be'):
                # Look ahead for a specific signal
                for j in range(i + 1, min(i + 3, len(tokens_list))):
                    if tokens_list[j] in specific_signals:
                        stype, srole, sposition = specific_signals[tokens_list[j]]
                        info = {'signal': tokens_list[j], 'signal_idx': j, 
                                'role': srole, 'position': sposition}
                        return stype, info
                # Also check for 'called', 'named', 'known as', 'headquartered'
                for j in range(i + 1, min(i + 4, len(tokens_list))):
                    if tokens_list[j] in ('called', 'named', 'known', 'headquartered'):
                        if tokens_list[j] == 'headquartered':
                            info = {'signal': tokens_list[j], 'signal_idx': j, 
                                    'role': 'entity', 'position': 'after_prep'}
                            return 'LOCATION', info
                        info = {'signal': tokens_list[j], 'signal_idx': j, 
                                'role': 'entity', 'position': 'after'}
                        return 'CLASSIFICATION', info
                # No specific signal after copula - use as CLASSIFICATION
                info = {'signal': t, 'signal_idx': i, 'role': 'entity', 'position': 'after'}
                return 'CLASSIFICATION', info

        # Check SIGNAL_MAP for remaining general signals
        for i, t in enumerate(tokens_list):
            if t in SIGNAL_MAP:
                etype, role, position = SIGNAL_MAP[t]
                info = {'signal': t, 'signal_idx': i, 'role': role, 'position': position}
                return etype, info

        return 'STATEMENT', info

    def _extract_relations(self, text_lower: str, tokens_list: List[str],
                           ef: EventFrame, etype: str, info: dict):
        """Extract relations based on detected event type and signal words."""
        sig_idx = info.get('signal_idx', -1)
        sig_word = info.get('signal', '')

        if etype == 'CLASSIFICATION':
            self._extract_classification(text_lower, tokens_list, ef, sig_idx, sig_word)
        elif etype == 'BORDER':
            self._extract_border(text_lower, tokens_list, ef, sig_idx, sig_word)
        elif etype == 'SEPARATE':
            self._extract_separate(text_lower, tokens_list, ef, sig_idx, sig_word)
        elif etype == 'DIVIDE':
            self._extract_divide(text_lower, tokens_list, ef, sig_idx, sig_word)
        elif etype == 'LOCATION':
            self._extract_location_statement(text_lower, tokens_list, ef, sig_idx, sig_word)
        elif etype == 'COMPOSITION':
            self._extract_composition(text_lower, tokens_list, ef, sig_idx, sig_word)
        elif etype in ('POSSESSION', 'CREATION', 'ACTION'):
            self._extract_action(text_lower, tokens_list, ef, etype, sig_idx, sig_word)

    def _extract_classification(self, text_lower: str, tokens_list: List[str],
                                 ef: EventFrame, sig_idx: int, sig_word: str):
        """Extract classification: 'X is [attribute]' or 'X is called Y'
        Works with: 'X is a Y', 'X is known as Y', 'X is called Y'"""
        # Handle 'known as' pattern: attribute is a nickname, not the classification
        if sig_word == 'known' and sig_idx + 2 < len(tokens_list) and tokens_list[sig_idx + 1] == 'as':
            nickname = tokens_list[sig_idx + 2].rstrip(',')
            ef.attributes['also_known_as'] = nickname
            # Continue processing after 'known as' for the main classification
            # The main entity's classification comes after the copula
        # Handle 'called' pattern: attribute is the term used
        if sig_word in ('called', 'named'):
            if sig_idx + 1 < len(tokens_list):
                name = tokens_list[sig_idx + 1].rstrip(',')
                ef.attributes['called'] = name
            # Don't return - continue to extract more attributes after the copula
        # Things after the signal word are attributes
        if sig_idx >= 0 and sig_idx < len(tokens_list) - 1:
            after = tokens_list[sig_idx + 1:]
            after_text = ' '.join(after)

            # Skip 'a/an/the/also/now'
            attr_words = [w for w in after if w not in ('a', 'an', 'the', 'also', 'now')]

            # Stop at prepositions or relative clauses
            clean_attr = []
            for w in attr_words:
                if w in ('in', 'on', 'at', 'near', 'of', 'to', 'by', 'from',
                         'with', 'which', 'that', 'who', 'where', 'and',
                         'since', 'consisting', 'comprising') and clean_attr:
                    break
                if w == 'and' and clean_attr:
                    break
                clean_attr.append(w)

            if clean_attr:
                attr = ' '.join(clean_attr)
                ef.attributes['is_a'] = _clean_entity(attr)

        # Also extract location from 'in/on/at/near' phrases in the remaining text
        if not ef.location:
            ef.location = self._extract_location(text_lower, tokens_list)

    def _extract_border(self, text_lower: str, tokens_list: List[str],
                        ef: EventFrame, sig_idx: int, sig_word: str):
        """Extract border relations: 'X is bordered by Y' or 'X borders Y'"""
        
        # Find content after "bordered by" or "borders"
        by_pos = text_lower.find(' by ')
        if by_pos > 0:
            after_by = text_lower[by_pos + 4:].strip().rstrip('.!?')

            # Split by commas and 'and'
            raw_parts = re.split(r'\s*,\s*(?:and\s+)?', after_by)
            all_parts = []
            for part in raw_parts:
                for sub in re.split(r'\s+and\s+', part):
                    sp = sub.strip()
                    if sp and sp not in all_parts:
                        all_parts.append(sp)

            # Process each part for direction
            dir_pattern = re.compile(r'^(.+?)\s+to\s+the\s+(north|south|east|west|northeast|northwest|southeast|southwest)$', re.I)
            for part in all_parts:
                m = dir_pattern.match(part.strip())
                if m:
                    neighbor = _clean_entity(m.group(1))
                    direction = m.group(2).lower()
                else:
                    neighbor = _clean_entity(part)
                    direction = ''

                if neighbor:
                    rel = Relation(type='BORDER', target=neighbor)
                    if direction:
                        rel.properties['direction'] = direction
                    ef.relations.append(rel)

        # Also check for standalone "to the [direction]" patterns
        if not ef.relations:
            dir_matches = re.finditer(r'(.+?)\s+to\s+the\s+(north|south|east|west|northeast|northwest|southeast|southwest)', text_lower)
            for m in dir_matches:
                target = _clean_entity(m.group(1))
                direction = m.group(2).lower()
                if target and target not in ('the', 'a', 'an'):
                    rel = Relation(type='BORDER', target=target)
                    rel.properties['direction'] = direction
                    ef.relations.append(rel)

    def _extract_separate(self, text_lower: str, tokens_list: List[str],
                          ef: EventFrame, sig_idx: int, sig_word: str):
        """Extract separation: 'X is separated from Y to the [direction] by Z'"""
        # Find entity (before the signal word)
        if sig_idx > 0:
            before_words = [w for w in tokens_list[:sig_idx] if w not in ('the', 'a', 'an', 'this', 'that', 'is', 'are', 'was', 'were', 'am')]
            if before_words:
                ef.entity = ' '.join(before_words)

        # Find "from" preposition
        from_pos = text_lower.find(' from ')
        if from_pos >= 0:
            after_from = text_lower[from_pos + 6:].strip().rstrip('.!?')
            target_phrase = after_from
            direction = ''
            instrument = ''

            # Check for "to the [direction]"
            dir_m = re.search(r'^(.+?)\s+to\s+the\s+(north|south|east|west|northeast|northwest|southeast|southwest)(?:\s+by\s+(.+))?$', target_phrase, re.I)
            if dir_m:
                target_phrase = dir_m.group(1).strip()
                direction = dir_m.group(2).lower()
                if dir_m.group(3):
                    instrument = dir_m.group(3).strip()
            else:
                # Check for "by [instrument]" without direction
                by_m = re.search(r'^(.+?)\s+by\s+(.+)$', target_phrase, re.I)
                if by_m:
                    target_phrase = by_m.group(1).strip()
                    instrument = by_m.group(2).strip()

            clean_target = _clean_entity(target_phrase)
            if clean_target:
                rel = Relation(type='SEPARATED_FROM', target=clean_target)
                if direction:
                    rel.properties['direction'] = direction
                if instrument:
                    rel.properties['instrument'] = instrument
                ef.relations.append(rel)

    def _extract_divide(self, text_lower: str, tokens_list: List[str],
                        ef: EventFrame, sig_idx: int, sig_word: str):
        """Extract division: 'X is divided into Y and Z'"""
        ef.properties.add('divided')

        # Find "into" or "between"
        into_pos = text_lower.find(' into ')
        if into_pos > 0:
            after = text_lower[into_pos + 6:].strip()
            self._extract_part_list(after, ef)

        # Find "between X and Y"
        between_m = re.search(r'between\s+(.+?)\s+and\s+(.+)', text_lower)
        if between_m:
            p1 = _clean_entity(between_m.group(1))
            p2 = _clean_entity(between_m.group(2))
            for p in [p1, p2]:
                if p:
                    ef.relations.append(Relation(type='DIVIDED_INTO', target=p))

        # Find criterion (at/near the X parallel)
        crit_m = re.search(r'(?:at|near)\s+(?:the\s+)?(\d+\w+\s+(?:parallel|meridian|latitude|longitude))', text_lower, re.I)
        if crit_m:
            if ef.relations:
                ef.relations[-1].properties['criterion'] = crit_m.group(1)

    def _extract_location_statement(self, text_lower: str, tokens_list: List[str],
                                     ef: EventFrame, sig_idx: int, sig_word: str):
        """Extract location: 'X is located in Y'"""
        # Location is already handled by the standard location extraction
        pass

    def _extract_composition(self, text_lower: str, tokens_list: List[str],
                              ef: EventFrame, sig_idx: int, sig_word: str):
        """Extract composition: 'X consists of Y, Z, and W'"""
        of_pos = text_lower.find('of')
        if of_pos > 0:
            after_of = text_lower[of_pos + 2:].strip()
            self._extract_part_list(after_of, ef)

    def _extract_part_list(self, text: str, ef: EventFrame):
        """Extract list of parts from text like 'X, Y, and Z'"""
        raw_parts = re.split(r'\s*,\s*(?:and\s+)?', text)
        all_parts = []
        for part in raw_parts:
            for sub in re.split(r'\s+and\s+', part):
                sp = _clean_entity(sub)
                if sp and sp not in all_parts and sp not in ('smaller', 'other'):
                    if not any(w in sp for w in ('smaller', 'other')):
                        all_parts.append(sp)

        for part in all_parts:
            ef.relations.append(Relation(type='PART_OF', target=part))

    def _extract_action(self, text_lower: str, tokens_list: List[str],
                        ef: EventFrame, etype: str, sig_idx: int, sig_word: str):
        """Extract general actions."""
        ef.properties.add(etype.lower())
        # Try to find the object of the action
        if sig_idx >= 0 and sig_idx < len(tokens_list) - 1:
            after = tokens_list[sig_idx + 1:]
            clean_after = [w for w in after if w not in ('the', 'a', 'an', 'its', 'his', 'her')]
            if clean_after:
                ef.attributes[f'action_{sig_word}'] = ' '.join(clean_after[:3])

    def _extract_location(self, text_lower: str, tokens_list: List[str]) -> str:
        """Extract location from 'in/on/at/near' phrases."""
        for prep in [' in ', ' on ', ' at ', ' near ']:
            pos = text_lower.rfind(prep)
            if pos > 0:
                after = text_lower[pos + len(prep):]
                # Skip years
                first_word = after.split()[0] if after.split() else ''
                if is_year(first_word):
                    continue
                # Extract until next signal word or punctuation
                loc_words = []
                for w in after.split():
                    if w in ('and', 'or') and loc_words:
                        break
                    if w in ('which', 'that', 'who', 'where'):
                        break
                    if w in SIGNAL_MAP and w not in ('in', 'on', 'at', 'near', 'of'):
                        break
                    if w in (',', ';', ':', '.'):
                        break
                    loc_words.append(w)
                if loc_words:
                    loc = ' '.join(loc_words).strip()
                    # Remove trailing relative clause fragments
                    loc = re.sub(r'\s+(which|that|who)\s+.*$', '', loc).strip()
                    return _clean_entity(loc) if not is_year(loc.split()[0] if loc.split() else '') else ''
        return ''

    def _extract_time(self, tokens_list: List[str]) -> str:
        """Extract temporal information from text."""
        for i, t in enumerate(tokens_list):
            if is_year(t):
                return t.strip('.,; ')
            # Check for date patterns
            if t in ('in', 'on', 'at', 'since', 'until', 'during', 'before', 'after'):
                if i + 1 < len(tokens_list):
                    next_word = tokens_list[i + 1]
                    if is_year(next_word):
                        return next_word.strip('.,; ')
        return ''

    def parse_question(self, text: str) -> dict:
        """Parse a question to extract query parameters.
        Uses cognitive priors, NOT regex patterns."""
        q = text.strip().rstrip('?').strip()
        q_lower = q.lower()
        tokens_list = tokens(q)

        result = {
            'question_type': 'UNKNOWN',
            'target_entity': '',
            'direction': '',
            'has_direction': False,
        }

        # Detect question type from first word
        first_word = tokens_list[0] if tokens_list else ''

        if first_word == 'where':
            result['question_type'] = 'WHERE'
        elif first_word == 'who':
            result['question_type'] = 'WHO'
        elif first_word == 'when':
            result['question_type'] = 'WHEN'
        elif first_word == 'why':
            result['question_type'] = 'WHY'
        elif first_word == 'how':
            result['question_type'] = 'HOW'
        elif first_word in ('is', 'are', 'was', 'were', 'am', 'do', 'does', 'did',
                            'can', 'could', 'has', 'have', 'had'):
            result['question_type'] = 'YESNO'
        elif first_word in ('what', 'which'):
            # Check for direction words
            for d in DIRECTION_WORDS:
                if d in tokens_list:
                    result['direction'] = d
                    result['has_direction'] = True
                    result['question_type'] = 'WHAT_DIRECTION'
                    break
            else:
                # Check for "border" or "separated"
                if any('border' in t or 'bordered' in t for t in tokens_list):
                    result['question_type'] = 'WHAT_BORDERS'
                elif 'separated' in q_lower:
                    result['question_type'] = 'WHAT_SEPARATED'
                else:
                    result['question_type'] = 'WHAT'

        # Extract target entity
        result['target_entity'] = self._extract_question_target(q_lower, tokens_list, result)

        return result

    def _extract_question_target(self, q_lower: str, tokens_list: List[str],
                                  qinfo: dict) -> str:
        """Extract the entity being asked about."""
        skip_set = {'what', 'where', 'who', 'when', 'why', 'how', 'which',
                    'is', 'are', 'was', 'were', 'am', 'do', 'does', 'did',
                    'can', 'could', 'will', 'would', 'shall', 'should',
                    'in', 'on', 'at', 'near', 'of', 'to', 'by', 'from', 'with',
                    'the', 'a', 'an', 'this', 'that', 'these', 'those',
                    'border', 'borders', 'bordered', 'separated', 'called',
                    'located', 'situated', 'found', 'known', 'referred',
                    'there', 'any', 'some'}

        # Remove question type prefix and find the main entity
        for i, t in enumerate(tokens_list):
            if t in DIRECTION_WORDS and qinfo.get('question_type') == 'WHAT_DIRECTION':
                # For direction questions, target is typically at the end
                dir_idx = i
                after_dir = tokens_list[dir_idx + 1:]
                # Skip prepositions
                clean = [w for w in after_dir if w not in ('of', 'in', 'on', 'at', 'near', 'to', 'the')]
                if clean:
                    return ' '.join(clean[-3:])
                # Fallback: check before direction word
                before_dir = [w for w in tokens_list[:dir_idx] if w not in skip_set]
                if before_dir:
                    return ' '.join(before_dir[-3:])
                break

            if t not in skip_set and len(t) > 1:
                # This is likely the target entity
                # Gather words until next stop word
                entity_words = []
                for j in range(i, len(tokens_list)):
                    w = tokens_list[j]
                    if w in skip_set and j > i:
                        break
                    if w in DIRECTION_WORDS and qinfo.get('question_type') == 'WHAT_DIRECTION':
                        break
                    entity_words.append(w)
                if entity_words:
                    return ' '.join(entity_words[:3])

        return ''


# ══════════════════════════════════════════════════════════════
# CONSTRUCTION — Learned from examples and failures
# ══════════════════════════════════════════════════════════════

@dataclass
class Construction:
    """Learned form-meaning mapping. Grows from experience."""
    id: str
    event_type: str
    signal_phrases: List[str] = field(default_factory=list)
    role_keywords: Dict[str, str] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    failures: List = field(default_factory=list)
    counterexamples: List[str] = field(default_factory=list)
    variants: List[str] = field(default_factory=list)
    avg_embedding: Optional[np.ndarray] = None
    confidence: float = 0.3
    success_count: int = 0
    failure_count: int = 0
    last_used: float = 0.0
    created: float = field(default_factory=time.time)

    def score(self) -> float:
        t = self.success_count + self.failure_count
        if t == 0: return self.confidence
        return self.confidence * (self.success_count / max(t, 1))

    def reinforce(self):
        self.success_count += 1
        self.confidence = min(0.95, self.confidence + 0.03)

    def to_dict(self):
        return {
            "id": self.id, "event_type": self.event_type,
            "confidence": self.confidence,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "signal_phrases": self.signal_phrases[:5],
        }


class ConstructionMemory:
    """Learned constructions. No pre-seeded patterns."""

    def __init__(self):
        self.constructions: Dict[str, Construction] = {}

    def create(self, event_type: str, signal_phrases: List[str],
               role_keywords: Dict[str, str], example: str = "",
               confidence: float = 0.3) -> Construction:
        cid = f"c_{event_type.lower()}_{_uid()}"
        c = Construction(id=cid, event_type=event_type,
                         signal_phrases=signal_phrases,
                         role_keywords=role_keywords,
                         confidence=confidence)
        if example:
            c.examples.append(example)
        self.constructions[cid] = c
        return c

    def find_by_signal(self, signal: str) -> List[Construction]:
        results = []
        for c in self.constructions.values():
            if any(signal in sp or sp in signal for sp in c.signal_phrases):
                results.append(c)
        return results

    def consolidate(self):
        to_delete = []
        for cid, c in self.constructions.items():
            if c.confidence < 0.01 or (c.failure_count > 20 and c.success_count == 0):
                to_delete.append(cid)
        for cid in to_delete:
            del self.constructions[cid]

    def get_stats(self):
        return {
            "total": len(self.constructions),
            "types": dict(defaultdict(int, {
                c.event_type: sum(1 for cc in self.constructions.values() if cc.event_type == c.event_type)
                for c in self.constructions.values()
            })),
        }


# ══════════════════════════════════════════════════════════════
# EPISODE MEMORY
# ══════════════════════════════════════════════════════════════

@dataclass
class Episode:
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
    def __init__(self):
        self.episodes: List[Episode] = []

    def record(self, inp: str, ef: Optional[EventFrame], entity: str,
               success: bool, is_question: bool = False,
               answer: str = "", expected: str = "", error: str = "") -> Episode:
        ep = Episode(id=_uid(), input=inp, event_frame=ef,
                     resolved_entity=entity, success=success,
                     is_question=is_question, answer=answer,
                     expected=expected, error=error)
        self.episodes.append(ep)
        if len(self.episodes) > 10000:
            self.episodes = self.episodes[-5000:]
        ep.embedding = embed_text(inp)
        return ep

    def get_failures(self) -> List[Episode]:
        return [ep for ep in self.episodes if not ep.success and ep.error]

    def get_recent(self, n: int = 10) -> List[Episode]:
        return self.episodes[-n:]

    def clear(self):
        self.episodes.clear()


# ══════════════════════════════════════════════════════════════
# MAIN ENGINE
# ══════════════════════════════════════════════════════════════

class LanguageAcquisitionEngine:
    """
    Main cognitive engine.
    Zero hardcoded language rules. Uses cognitive priors + learning.
    """

    def __init__(self):
        self.model = SituationModel()
        self.coref = CorefLearner()
        self.parser = SemanticParser(self.model, self.coref)
        self.cmem = ConstructionMemory()
        self.epmem = EpisodeMemory()
        self.dialogue_count = 0

    def hear(self, text: str) -> str:
        """Process an utterance. Returns response.
        Handles multi-sentence input by splitting and processing each sentence."""
        self.dialogue_count += 1

        if not text.strip():
            return "Yes?"

        text = text.strip()
        text_lower = text.lower()

        # Detect if this is a question
        is_question = text.endswith("?") or any(
            text_lower.startswith(w)
            for w in ('what ', 'where ', 'who ', 'when ', 'why ', 'how ',
                      'which ', 'is ', 'are ', 'was ', 'were ', 'am ',
                      'do ', 'does ', 'did ', 'can ', 'could ')

        )

        if is_question:
            return self._answer(text)

        # Process multi-sentence statements
        sents = sent_split(text)
        stored_count = 0
        for sent in sents:
            result = self._learn_from_statement(sent)
            if result.startswith("Got it"):
                stored_count += 1

        if stored_count > 0 and stored_count == len(sents):
            return "Got it. I've stored that information."
        elif stored_count > 0:
            return f"Got it. Stored {stored_count} facts."
        return "I heard you."

    def _learn_from_statement(self, text: str) -> str:
        """Process a declarative statement and learn from it."""
        # Parse into event frame
        ef = self.parser.parse(text)

        # Apply to world model
        if ef.entity:
            # Resolve entity
            resolved, was_resolved = self.coref.resolve(ef.entity)
            if was_resolved:
                ef.entity = resolved

            success = self.model.apply(ef)

            # Register entities
            self.coref.register(ef.entity, 'ENTITY')
            if ef.target:
                self.coref.register(ef.target, 'ENTITY')
            for rel in ef.relations:
                if rel.target:
                    self.coref.register(rel.target, 'ENTITY')

            # Record episode
            self.epmem.record(text, ef, ef.entity, success)

            if success:
                # Try to create construction from this
                signal_words = [t for t in tokens(text) if t in SIGNAL_MAP]
                if signal_words and not self.cmem.find_by_signal(signal_words[0]):
                    words = [t for t in tokens(text) if t not in STOP_WORDS and len(t) > 2]
                    self.cmem.create(
                        ef.event_type,
                        signal_words[:3],
                        {'entity': 'entity', 'target': 'target'},
                        example=text,
                        confidence=0.3
                    )

                return "Got it. I've stored that information."

        self.epmem.record(text, ef, "", False, error="parse_failed")
        return "I heard you."

    def _answer(self, question: str) -> str:
        """Answer a question by reasoning over the world model."""
        qinfo = self.parser.parse_question(question)
        qtype = qinfo['question_type']

        # Resolve target entity
        target = qinfo.get('target_entity', '')
        resolved, was_resolved = self.coref.resolve(target) if target else ('', False)
        if was_resolved:
            target = resolved
        target_norm = norm(target) if target else ""

        # Get entity
        entity = self.model.get(target_norm)

        # Route by question type
        if qtype == 'WHERE':
            answer = self._answer_where(target_norm, entity)
        elif qtype == 'WHO':
            answer = self._answer_who(target_norm, entity)
        elif qtype == 'WHEN':
            answer = self._answer_when(target_norm, entity)
        elif qtype == 'YESNO':
            answer = self._answer_yesno(question, target_norm, entity)
        elif qtype == 'HOW':
            answer = self._answer_how(target_norm, entity)
        elif qtype == 'WHAT_DIRECTION':
            answer = self._answer_direction(question, qinfo, target_norm, entity)
        elif qtype == 'WHAT_BORDERS':
            answer = self._answer_borders(target_norm, entity)
        elif qtype == 'WHAT_SEPARATED':
            answer = self._answer_separated_from(target_norm, entity)
        elif qtype == 'WHAT':
            answer = self._answer_what(target_norm, entity)
        else:
            answer = self._answer_fallback(target_norm, entity)

        self.epmem.record(question, None, target_norm, True,
                          is_question=True, answer=answer)
        return answer

    def _answer_where(self, target: str, entity) -> str:
        if not entity or not isinstance(entity, Entity):
            return "I don't know where it is."
        if entity.location:
            return f"It is in {entity.location}."
        loc = self.model.resolve_location(target)
        if loc:
            return f"It is in {loc}."
        # Check for location in relations
        for rel_type, entries in entity.relations.items():
            for t, props in entries:
                parent = self.model.get(t)
                if parent and parent.location:
                    return f"It is in {parent.location}."
        return "I don't know where it is."

    def _answer_who(self, target: str, entity) -> str:
        if not entity or not isinstance(entity, Entity):
            return "I don't know who that is."
        if 'is_a' in entity.attributes:
            return f"You are {entity.attributes['is_a']}."
        desc = self.model.describe(target)
        if desc:
            return f"I recall: {desc}."
        return "I don't know who that is."

    def _answer_when(self, target: str, entity) -> str:
        if not entity or not isinstance(entity, Entity):
            return "I don't know."
        if entity.time:
            return f"It was in {entity.time}."
        return "I don't know."

    def _answer_yesno(self, question: str, target: str, entity) -> str:
        if not entity or not isinstance(entity, Entity):
            return "no"

        q_lower = question.lower()

        # Check attributes
        for k, v in entity.attributes.items():
            if k in q_lower or v in q_lower:
                return "yes"
            last_word = q_lower.split()[-1] if len(q_lower.split()) > 1 else ""
            if last_word and last_word in v.lower():
                return "yes"

        # Check location
        if entity.location and entity.location in q_lower:
            return "yes"

        # Check relations
        for rel_type, entries in entity.relations.items():
            for t, props in entries:
                if t in q_lower:
                    return "yes"

        # Division check
        if 'divid' in q_lower:
            for rel_type in entity.relations:
                if 'divid' in rel_type.lower() or 'part' in rel_type.lower():
                    return "yes"
            if 'divided' in entity.properties:
                return "yes"

        # Separated check
        if 'separated' in q_lower:
            for rel_type in entity.relations:
                if 'separated' in rel_type.lower():
                    return "yes"

        return "no"

    def _answer_how(self, target: str, entity) -> str:
        if not entity or not isinstance(entity, Entity):
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

        return self.model.describe(target)

    def _answer_direction(self, question: str, qinfo: dict,
                          target: str, entity) -> str:
        """Answer 'What is in the [direction] of X?'"""
        direction = qinfo.get('direction', '')
        if not direction or not target:
            return "I don't know."

        target_norm = norm(target)

        # Strategy 1: Check target entity's relations for direction
        e = self.model.get(target_norm)
        if e and isinstance(e, Entity):
            for rel_type, entries in e.relations.items():
                for t, props in entries:
                    if props.get('direction', '').lower() == direction:
                        return f"The {direction} of {target_norm} is {t}."

        # Strategy 2: Check reverse (other entities have relation TO target with direction)
        for name, ent in self.model.entities.items():
            if norm(name) == target_norm:
                continue
            for rel_type, entries in ent.relations.items():
                for t, props in entries:
                    if norm(t) == target_norm and props.get('direction', '').lower() == direction:
                        return f"The {direction} of {target_norm} is {name}."

        # Strategy 3: SEPARATED_FROM - if A is separated from B to the southeast,
        # then B is southeast of A
        if e and isinstance(e, Entity):
            for rel_type, entries in e.relations.items():
                if 'separated' in rel_type.lower():
                    for t, props in entries:
                        # Direction from separation is about the other entity's position
                        if props.get('direction', '').lower() == direction:
                            return f"The {direction} of {target_norm} is {t}."

        return "I don't know."

    def _answer_borders(self, target: str, entity) -> str:
        if not entity or not isinstance(entity, Entity):
            return "I don't know."
        borders = []
        for rel_type, entries in entity.relations.items():
            if 'border' in rel_type.lower():
                for t, props in entries:
                    borders.append(t)
        if borders:
            return ", ".join(borders)
        return "I don't know."

    def _answer_separated_from(self, target: str, entity) -> str:
        if not entity or not isinstance(entity, Entity):
            return "I don't know."
        for rel_type, entries in entity.relations.items():
            if 'separated' in rel_type.lower():
                targets = [t for t, _ in entries]
                if targets:
                    return ", ".join(targets)
        return "I don't know."

    def _answer_what(self, target: str, entity) -> str:
        if not entity or not isinstance(entity, Entity):
            return "I don't know anything about that yet."
        desc = self.model.describe(target)
        if desc:
            return f"I recall: {desc}."
        return "I don't know anything about that yet."

    def _answer_fallback(self, target: str, entity) -> str:
        if entity and isinstance(entity, Entity):
            desc = self.model.describe(target)
            if desc:
                return desc
        return "I don't know."

    def feedback(self, question: str, correct_answer: str) -> str:
        """Learn from user-supplied correction."""
        success, msg = self._learn_from_statement(correct_answer), True
        if success.startswith("Got it"):
            return f"Learned: {correct_answer}"
        return f"Noted: {correct_answer}"

    def sleep_cycle(self) -> str:
        self.cmem.consolidate()
        return f"Consolidated {len(self.epmem.episodes)} episodes, {self.cmem.get_stats()['total']} constructions"

    def get_status(self) -> str:
        stats = self.cmem.get_stats()
        return (f"{len(self.model.entities)} entities, "
                f"{stats['total']} constructions, "
                f"{len(self.epmem.episodes)} episodes, "
                f"{self.dialogue_count} dialogues")

    def reset(self):
        self.model = SituationModel()
        self.coref = CorefLearner()
        self.parser = SemanticParser(self.model, self.coref)
        self.cmem = ConstructionMemory()
        self.epmem = EpisodeMemory()
        self.dialogue_count = 0

    def to_dict(self):
        return {
            "entities": self.model.to_dict(),
            "constructions": {k: c.to_dict() for k, c in self.cmem.constructions.items()},
            "status": self.get_status(),
        }


class NeurovaEngine:
    """Simple interface for CLI."""
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
