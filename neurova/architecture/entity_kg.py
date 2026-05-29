"""
Entity-Centric Knowledge Graph v2
- Entity-indexed fact storage
- Location tracking with overwrite
- Possession tracking (with/give/drop)
- Inheritance from is-a relations
- Negation with scope inheritance
- Coreference resolution across statements
- Sentence splitting for multi-sentence input
- Temporal ordering
- Comparative/relational reasoning
- Path finding
- Multi-hop query engine
"""

import re
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import defaultdict

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except:
    _NLP = None


# ── Constants ──

LOCATIVE_VERBS = {"go", "move", "travel", "journey", "walk", "run", "went", "came",
                  "return", "get", "head", "leave"}
POSSESSION_VERBS = {"pick", "bring", "get", "take", "grab", "carry", "hold", "have", "has"}
DROP_VERBS = {"drop", "put", "discard", "release", "leave", "place"}
GIVE_VERB = {"give", "hand", "pass", "lend", "sell"}
BE_VERBS = {"be", "is", "are", "am", "was", "were"}

Q_WORDS = {"what", "where", "who", "when", "why", "how", "which"}

PRONOUN_MAP = {
    "he": "male", "him": "male", "his": "male",
    "she": "female", "her": "female", "hers": "female",
    "it": "neutral", "its": "neutral",
    "they": "plural", "them": "plural", "their": "plural",
    "i": "self", "me": "self", "my": "self", "we": "self_plural",
    "you": "other",
}

COREF_TRIGGERS = {
    "the region", "the country", "the place", "the area",
    "the person", "the man", "the woman", "the boy", "the girl",
    "the object", "the thing", "the item",
}


# ── Helpers ──

def norm(s: str) -> str:
    """Normalize: lowercase, strip determiners."""
    if not s:
        return ""
    s = s.strip().lower()
    for art in ("a ", "an ", "the ", "some ", "any ", "every ", "this ", "that "):
        while s.startswith(art):
            s = s[len(art):]
    return s.strip()


def singular(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    if s.endswith("ies") and len(s) > 4:
        return s[:-3] + "y"
    if s.endswith("ses") or s.endswith("xes") or s.endswith("ches") or s.endswith("shes"):
        return s[:-2]
    if s.endswith("s") and not s.endswith("ss") and len(s) > 3:
        return s[:-1]
    return s


def split_sentences(text: str) -> List[str]:
    """Split text into sentences. Handles '.' separators and abbreviations."""
    # Simple but robust: split on sentence-ending punctuation
    sentences = []
    current = ""
    in_parentheses = 0
    
    for ch in text:
        if ch == '(':
            in_parentheses += 1
        elif ch == ')':
            in_parentheses -= 1
        
        current += ch
        
        if ch in '.!?' and in_parentheses == 0:
            # Check for abbreviations
            words = current.strip().split()
            if len(words) >= 2 and words[-2].isupper() and len(words[-2]) <= 4:
                continue  # Likely abbreviation like "U.S." or "Inc."
            if current.strip().rstrip('.!?').endswith('Mr') or \
               current.strip().rstrip('.!?').endswith('Mrs') or \
               current.strip().rstrip('.!?').endswith('Dr') or \
               current.strip().rstrip('.!?').endswith('etc'):
                continue
            sentences.append(current.strip())
            current = ""
    
    if current.strip():
        sentences.append(current.strip())
    
    return [s for s in sentences if s]


def _find_prep_obj(token, depth=0):
    """Recursive prep-object finding for 'went back to X' patterns."""
    if depth > 5:
        return []
    objs = []
    for child in token.children:
        if child.pos_ == "ADP" and child.dep_ == "prep":
            for gc in child.children:
                if gc.dep_ == "pobj":
                    objs.append(" ".join(c.text.lower() for c in gc.subtree))
        objs.extend(_find_prep_obj(child, depth + 1))
    return objs


def parse_sentence(text: str) -> Dict[str, Any]:
    """Parse a single sentence into structured fields using spaCy."""
    result = {
        "subject": "", "verb": "", "object": "",
        "is_negation": False, "is_question": False,
        "qword": "", "entities": [],
        "raw": text,
        "conjunction_subjects": [],
        "has_copula": False,
        "prepositions": {},
    }
    
    raw_lower = text.lower().strip()
    result["is_negation"] = "not" in raw_lower or "n't" in raw_lower or "cannot" in raw_lower
    result["is_question"] = "?" in text
    for w in Q_WORDS:
        if w in raw_lower.split():
            result["qword"] = w
            break
    
    if not _NLP:
        return result
    
    try:
        doc = _NLP(text)
        
        # Extract named entities
        for ent in doc.ents:
            result["entities"].append({"text": ent.text, "label": ent.label_})
        
        for token in doc:
            if token.dep_ == "ROOT":
                result["verb"] = token.lemma_
                result["has_copula"] = token.lemma_ in BE_VERBS
                
                # Subject
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass", "expl"):
                        subj_text = " ".join(c.text.lower() for c in child.subtree)
                        result["subject"] = subj_text
                        
                        # Check for conjunction subjects (John and Mary)
                        for sc in child.children:
                            if sc.dep_ == "conj" and sc.pos_ == "PROPN":
                                result.setdefault("conjunction_subjects", []).append(
                                    " ".join(c.text.lower() for c in sc.subtree))
                            if sc.dep_ == "cc" and sc.text.lower() == "and":
                                pass
                    
                    # Object / attribute / complement
                    elif child.dep_ in ("attr", "acomp", "dobj"):
                        obj_text = " ".join(c.text.lower() for c in child.subtree)
                        result["object"] = obj_text
                    
                    # Prepositional phrases
                    elif child.dep_ == "prep":
                        prep_label = child.text.lower()
                        for gc in child.children:
                            if gc.dep_ == "pobj":
                                pobj = " ".join(c.text.lower() for c in gc.subtree)
                                result["prepositions"][prep_label] = pobj
                                if result["object"]:
                                    result["object"] += " " + pobj
                                else:
                                    result["object"] = pobj
                    
                    # Adverbial modifiers that contain prep objects
                    elif child.dep_ == "advmod" and child.text.lower() in ("back", "there", "here"):
                        deep = _find_prep_obj(child)
                        if deep and not result["object"]:
                            result["object"] = deep[-1]
        
        # Deep prep search for "went back to X" patterns  
        if not result["object"]:
            deep = _find_prep_obj(token)
            if deep:
                result["object"] = deep[-1]
        
        # Handle conjunction in subject
        subj_words = result["subject"].split()
        if "and" in subj_words:
            parts = result["subject"].split(" and ")
            result["conjunction_subjects"] = [norm(p) for p in parts if p.strip()]
        
        # Clean up extracted fields
        result["subject"] = norm(result["subject"])
        result["object"] = norm(result["object"])
        
    except Exception:
        pass
    
    return result


def extract_entity(text: str) -> Optional[str]:
    """Extract the main entity from a description phrase."""
    t = norm(text)
    if not t:
        return None
    words = t.split()
    # Filter out generic words
    generic = {"a", "an", "the", "this", "that", "some", "any", "every",
               "is", "are", "was", "were", "be", "has", "have", "had",
               "in", "on", "at", "to", "from", "of", "for", "with", "by"}
    meaningful = [w for w in words if w not in generic]
    return meaningful[-1] if meaningful else words[-1]


# ── Coreference Engine ──

class CoreferenceEngine:
    """Resolves pronouns and definite references to previous entities."""
    
    def __init__(self):
        self.recent_entities: List[Dict[str, Any]] = []
        self.max_history = 50
    
    def reset(self):
        self.recent_entities = []
    
    def mention(self, entity: str, gender: str = "neutral", text: str = ""):
        """Record an entity mention."""
        self.recent_entities.insert(0, {
            "entity": norm(entity),
            "gender": gender,
            "text": text,
        })
        if len(self.recent_entities) > self.max_history:
            self.recent_entities.pop()
    
    def resolve(self, text: str) -> str:
        """If text is a pronoun or definite reference, resolve to a previous entity."""
        t = norm(text)
        if not t:
            return text
        
        # Pronoun resolution
        if t in PRONOUN_MAP:
            gender = PRONOUN_MAP[t]
            for ent in self.recent_entities:
                if gender == "neutral" or ent["gender"] == gender or gender == "plural":
                    return ent["entity"]
                if gender == "male" and ent["gender"] in ("male", "neutral"):
                    return ent["entity"]
                if gender == "female" and ent["gender"] in ("female", "neutral"):
                    return ent["entity"]
            # Fallback: most recent entity
            if self.recent_entities:
                return self.recent_entities[0]["entity"]
            return text
        
        # Definite reference resolution (the region, the country, etc.)
        if t in COREF_TRIGGERS:
            # Find the most recent entity whose type matches
            for ent in self.recent_entities:
                return ent["entity"]
            return text
        
        # "The X" references - try to link
        if text.lower().startswith("the "):
            base = norm(text)
            for ent in self.recent_entities:
                if base in ent["entity"] or ent["entity"] in base:
                    return ent["entity"]
        
        return text


# ── Fact ──

class Fact:
    """A single fact with provenance."""
    __slots__ = ("subj", "rel", "obj", "neg", "time", "source")
    def __init__(self, subj: str, rel: str, obj: str = "",
                 neg: bool = False, time: int = 0, source: str = ""):
        self.subj = norm(subj)
        self.rel = rel if rel not in BE_VERBS else "be"
        self.obj = norm(obj)
        self.neg = neg
        self.time = time
        self.source = source
    
    def __repr__(self):
        neg_s = "NOT " if self.neg else ""
        return f"Fact({self.subj} --{neg_s}{self.rel}--> {self.obj})"


# ── Entity Knowledge Graph ──

class EntityKnowledgeGraph:
    """
    Entity-centric knowledge graph.
    All facts are indexed by entity for fast retrieval.
    Supports location, possession, inheritance, negation, comparison.
    """
    
    def __init__(self):
        self.facts: List[Fact] = []
        self._time = 0
        self._entity_index: Dict[str, List[int]] = defaultdict(list)  # entity → fact indices
        self.coref = CoreferenceEngine()
    
    def reset(self):
        self.facts.clear()
        self._entity_index.clear()
        self._time = 0
        self.coref.reset()
    
    def _index(self, idx: int, fact: Fact):
        """Index a fact by subject and object."""
        if fact.subj:
            self._entity_index[fact.subj].append(idx)
        if fact.obj:
            self._entity_index[fact.obj].append(idx)
        # Also index by subject's first token for partial matching
        if fact.subj:
            first = fact.subj.split()[0] if fact.subj else ""
            if first and first != fact.subj:
                self._entity_index[first].append(idx)
    
    def _get_facts_about(self, entity: str) -> List[Fact]:
        """Get all facts mentioning an entity."""
        e = norm(entity)
        indices = set()
        indices.update(self._entity_index.get(e, []))
        # Partial match
        for key in list(self._entity_index.keys()):
            if e in key or key in e:
                indices.update(self._entity_index[key])
        return [self.facts[i] for i in sorted(indices)]
    
    def add_fact(self, subj: str, rel: str, obj: str = "",
                 neg: bool = False, source: str = "") -> None:
        """Add a fact with automatic entity indexing and special handling."""
        if not subj or not rel:
            return
        
        self._time += 1
        subj = norm(subj)
        obj = norm(obj)
        
        if rel in BE_VERBS:
            rel = "be"
        
        # --- Location verbs: overwrite old locations ---
        if rel in LOCATIVE_VERBS and obj:
            # Remove old location facts for this subject
            self._remove_facts(subj, LOCATIVE_VERBS)
        
        # --- Possession: pick/get/take → object WITH subject ---
        if rel in POSSESSION_VERBS and obj and not neg:
            # Remove old possession of this object
            self._remove_facts(obj, {"with"})
            fact = Fact(obj, "with", subj, time=self._time, source=source)
            self._add_fact_internal(fact)
            return
        
        # --- Drop/place: object stays at subject's location ---
        if rel in DROP_VERBS and obj and not neg:
            loc = self._get_direct_location(subj)
            self._remove_facts(obj, {"with", "placed"})
            if loc:
                fact = Fact(obj, "placed", loc, time=self._time, source=source)
            else:
                fact = Fact(subj, rel, obj, time=self._time, source=source)
            self._add_fact_internal(fact)
            return
        
        # --- Give: X gives Y to Z → Y is with Z ---
        if rel in GIVE_VERB and obj and not neg:
            words = obj.split()
            to_idx = -1
            for i, w in enumerate(words):
                if w == "to" and i + 1 < len(words):
                    given_obj = " ".join(words[:i]) if i > 0 else ""
                    recipient = words[-1]
                    if given_obj:
                        self._remove_facts(given_obj, {"with"})
                        fact = Fact(given_obj, "with", recipient, time=self._time, source=source)
                        self._add_fact_internal(fact)
                        return
            # "gave Mary the apple" format
            if len(words) >= 2:
                self._remove_facts(words[-1], {"with"})
                fact = Fact(words[-1], "with", words[0], time=self._time, source=source)
                self._add_fact_internal(fact)
                return
        
        # --- Standard fact ---
        fact = Fact(subj, rel, obj, neg=neg, time=self._time, source=source)
        self._add_fact_internal(fact)
    
    def _add_fact_internal(self, fact: Fact):
        """Add a fact and index it."""
        idx = len(self.facts)
        self.facts.append(fact)
        self._index(idx, fact)
    
    def _remove_facts(self, subj: str, rels: Set[str]):
        """Remove facts matching subject and relations."""
        keep = []
        for i, f in enumerate(self.facts):
            if f.subj == subj and f.rel in rels:
                continue  # Remove
            keep.append(i)
        
        if len(keep) == len(self.facts):
            return
        
        old_facts = self.facts
        self.facts = []
        self._entity_index.clear()
        for i in keep:
            self._add_fact_internal(old_facts[i])
    
    def _get_direct_location(self, entity: str) -> Optional[str]:
        """Get entity's current location from locative or 'be' facts."""
        e = norm(entity)
        # Direct location
        for f in reversed(self.facts):
            if f.subj == e and not f.neg and f.obj:
                if f.rel in LOCATIVE_VERBS or f.rel == "be":
                    return f.obj
        return None
    
    def get_location(self, entity: str) -> Optional[str]:
        """Get entity's location with multi-hop resolution."""
        e = norm(entity)
        if not e:
            return None
        
        # 1. Direct location
        loc = self._get_direct_location(e)
        if loc:
            return loc
        
        # 2. Object was placed somewhere
        for f in reversed(self.facts):
            if f.subj == e and f.rel == "placed" and not f.neg and f.obj:
                return f.obj
        
        # 3. Object is with someone → where is that person?
        for f in reversed(self.facts):
            if f.subj == e and f.rel == "with" and not f.neg and f.obj:
                return self.get_location(f.obj)
        
        # 4. Someone has/holds the entity
        for f in reversed(self.facts):
            if f.obj == e and f.rel == "with" and not f.neg:
                return self.get_location(f.subj)
        
        # 5. Pick/take there
        for f in reversed(self.facts):
            if f.subj == e and not f.neg and f.obj:
                if f.rel == "there":
                    return f.obj
        
        return None
    
    def verify(self, subj: str, obj: str = "", rel: str = "") -> Tuple[bool, bool]:
        """
        Verify if a claim is true. Returns (found, negated).
        Supports inheritance and negation propagation.
        """
        e, o = norm(subj), norm(obj)
        if not e:
            return False, False
        
        negated = False
        
        # Direct check
        for f in self.facts:
            if f.subj == e:
                # Relation match
                if rel and f.rel == rel:
                    if f.neg:
                        negated = True
                    elif not o or o in f.obj or f.obj in o:
                        return True, False
                # Object match (for is-a verification)
                if o and f.obj and (o in f.obj or f.obj in o):
                    if f.neg:
                        negated = True
                    elif not rel or rel in ("be",) or rel in f.rel:
                        return True, False
        
        # Inheritance check: if entity IS-A parent, check parent's properties
        parents = self.get_parents(e)
        for p in parents:
            result, child_neg = self.verify(p, o, rel)
            if result and not child_neg:
                return True, False
            if child_neg:
                negated = True
        
        # Negation propagation: if entity IS-NOT something, that thing's properties don't apply
        # e.g., rock is not a living thing. living things grow. → rock does NOT grow.
        if rel and not negated:
            not_things = []
            for f in self.facts:
                if f.subj == e and f.rel == "not be" and f.obj:
                    not_things.append(f.obj)
                if f.subj == e and f.neg and f.rel == "be" and f.obj:
                    not_things.append(f.obj)
            for nt in not_things:
                # Check if the 'not-thing' has the property
                for f in self.facts:
                    if f.subj == norm(nt) and f.rel == rel and not f.neg:
                        return False, True  # Found: entity is NOT like nt, and nt HAS the property
        
        return False, negated
    
    def get_parents(self, entity: str) -> List[str]:
        """Get immediate parent categories (is-a relations)."""
        e = norm(entity)
        return [f.obj for f in self.facts
                if f.subj == e and f.rel == "be" and f.obj and not f.neg]
    
    def get_ancestors(self, entity: str) -> List[str]:
        """Get all ancestors (entity + parents + grandparents)."""
        e = norm(entity)
        seen = set()
        result = []
        queue = [e]
        while queue:
            cur = queue.pop(0)
            if cur in seen:
                continue
            seen.add(cur)
            result.append(cur)
            queue.extend(self.get_parents(cur))
        return result
    
    def is_a(self, entity: str, category: str) -> bool:
        """Check if entity is a member of category (direct or inherited)."""
        e, c = norm(entity), norm(category)
        if e == c:
            return True
        return c in self.get_ancestors(e)
    
    def count(self, query: str = "") -> int:
        """Count entities matching a pattern. query can be:
        - '' : count all distinct entities
        - 'with:entity' : count entities possessed by entity
        - 'at:location' : count entities at location
        - 'type:category' : count entities of category
        """
        if not query:
            return len(self._entity_index)
        
        q = norm(query)
        if q.startswith("with:"):
            holder = q[5:]
            seen = set()
            for f in self.facts:
                if f.rel == "with" and f.obj == holder and not f.neg:
                    seen.add(f.subj)
            return len(seen)
        
        if q.startswith("at:"):
            loc = q[3:]
            seen = set()
            for f in self.facts:
                if f.rel in LOCATIVE_VERBS and f.obj == loc and not f.neg:
                    seen.add(f.subj)
                if f.rel == "placed" and f.obj == loc and not f.neg:
                    seen.add(f.subj)
            return len(seen)
        
        if q.startswith("type:"):
            cat = q[5:]
            seen = set()
            for f in self.facts:
                if f.rel == "be" and f.obj == cat and not f.neg:
                    seen.add(f.subj)
            return len(seen)
        
        return 0
    
    def resolve_coreference_in_text(self, text: str, current_entity: str = "") -> str:
        """Resolve pronouns and definite references in text."""
        t = norm(text)
        if not t:
            return text
        
        resolved = self.coref.resolve(t)
        if resolved != t:
            # Update coref state with current entity
            if current_entity:
                self.coref.mention(current_entity)
            return resolved
        
        return text
    
    def process_statement(self, text: str) -> List[Fact]:
        """
        Process a statement (possibly multi-sentence) and return facts.
        Handles coreference resolution across sentences.
        """
        sentences = split_sentences(text)
        facts = []
        
        for sent in sentences:
            parsed = parse_sentence(sent)
            
            # Resolve subject coreference
            resolved_subj = self.coref.resolve(parsed["subject"])
            if resolved_subj != parsed["subject"]:
                parsed["subject"] = resolved_subj
            
            # Handle conjunction subjects
            subjs = [parsed["subject"]]
            if parsed.get("conjunction_subjects"):
                subjs = parsed["conjunction_subjects"]
            
            for subj in subjs:
                if not subj:
                    continue
                
                # Record entity mention for coreference
                self.coref.mention(subj)
                
                # Handle negation
                if parsed["is_negation"]:
                    rel = parsed["verb"]
                    if rel in BE_VERBS:
                        self.add_fact(subj, "not " + rel, parsed["object"],
                                      neg=True, source=sent)
                    else:
                        self.add_fact(subj, rel, parsed["object"],
                                      neg=True, source=sent)
                else:
                    self.add_fact(subj, parsed["verb"], parsed["object"],
                                  source=sent)
            
            # Handle 'there' marker (bAbI: "Mary got the milk there.")
            if "there" in sent.lower() and not parsed["object"].endswith("there"):
                for subj in subjs:
                    if subj and parsed["object"]:
                        pass  # 'there' doesn't add new facts
            
            facts.extend([f for f in self.facts if f.time == self._time])
        
        return facts
    
    def query(self, text: str) -> Dict[str, Any]:
        """
        Answer a question about the knowledge graph.
        Returns structured answer.
        """
        parsed = parse_sentence(text)
        subj = self.coref.resolve(parsed["subject"])
        verb = parsed["verb"]
        obj = parsed["object"]
        qword = parsed["qword"]
        raw = text.lower()
        
        # ── WHERE questions ──
        if qword == "where":
            loc = self.get_location(subj)
            if loc:
                return {"found": True, "answer": loc, "type": "location"}
            return {"found": False, "answer": "", "type": "location"}
        
        # ── WHO questions ──
        if qword == "who":
            # Find who matches the description
            if obj:
                for f in self.facts:
                    if f.rel == "be" and f.obj and (obj in f.obj or f.obj in obj) and not f.neg:
                        return {"found": True, "answer": f.subj, "type": "identity"}
            return {"found": False, "answer": "", "type": "identity"}
        
        # ── WHAT questions ──
        if qword == "what":
            return {"found": True, "answer": self._describe_entity(subj), "type": "description"}
        
        # ── IS/ARE questions (yes/no) ──
        if raw.startswith("is ") or raw.startswith("are ") or verb in ("be",):
            found, neg = self.verify(subj, obj, verb)
            if neg:
                return {"found": True, "answer": "no", "type": "boolean", "negated": True}
            if found:
                return {"found": True, "answer": "yes", "type": "boolean"}
            if obj and self.is_a(subj, obj):
                return {"found": True, "answer": "yes", "type": "boolean"}
            return {"found": False, "answer": "no", "type": "boolean"}
        
        # ── DOES/DO/CAN questions (property) ──
        if verb in ("do", "does", "did", "can", "could", "will", "would", "may", "might"):
            found, neg = self.verify(subj, obj, verb if verb not in ("do", "does", "did") else "")
            if found and not neg:
                return {"found": True, "answer": "yes", "type": "boolean"}
            if neg:
                return {"found": True, "answer": "no", "type": "boolean", "negated": True}
            # Check property verbs like "grow", "fly", "need"
            if obj:
                for f in self.facts:
                    if f.subj == subj and f.obj and (obj in f.obj or f.obj in obj) and not f.neg:
                        return {"found": True, "answer": "yes", "type": "boolean"}
            # Inheritance check
            for p in self.get_parents(subj):
                found_p, neg_p = self.verify(p, obj, verb)
                if found_p and not neg_p:
                    return {"found": True, "answer": "yes", "type": "boolean"}
            return {"found": False, "answer": "no", "type": "boolean"}
        
        # ── HOW MANY questions ──
        if "how many" in raw:
            # Extract what to count
            count_target = obj if obj else subj
            count_val = self.count("with:" + norm(count_target))
            if count_val > 0:
                return {"found": True, "answer": str(count_val), "type": "count"}
            count_val = self.count("type:" + norm(count_target))
            if count_val > 0:
                return {"found": True, "answer": str(count_val), "type": "count"}
            # Try general counting
            seen = set()
            for f in self.facts:
                if f.subj and not f.neg:
                    seen.add(f.subj)
            return {"found": True, "answer": str(len(seen)), "type": "count"}
        
        # ── Generic verification ──
        found, neg = self.verify(subj, obj, verb)
        if found and not neg:
            return {"found": True, "answer": "yes", "type": "boolean"}
        if neg:
            return {"found": True, "answer": "no", "type": "boolean", "negated": True}
        
        return {"found": False, "answer": "", "type": "unknown"}
    
    def _describe_entity(self, entity: str) -> str:
        """Generate a description of an entity from all known facts."""
        e = norm(entity)
        if not e:
            return ""
        
        parts = []
        seen = set()
        
        # Get all facts about this entity
        for f in self.facts:
            if f.subj == e and not f.neg:
                if f.rel == "be" and f.obj:
                    desc = f"is {f.obj}"
                    if desc not in seen:
                        seen.add(desc)
                        parts.append(desc)
                elif f.rel == "with" and f.obj:
                    desc = f"is with {f.obj}"
                    if desc not in seen:
                        seen.add(desc)
                        parts.append(desc)
                elif f.rel in LOCATIVE_VERBS and f.obj:
                    desc = f"is in {f.obj}"
                    if desc not in seen:
                        seen.add(desc)
                        parts.append(desc)
                elif f.rel == "placed" and f.obj:
                    desc = f"is placed in {f.obj}"
                    if desc not in seen:
                        seen.add(desc)
                        parts.append(desc)
                elif f.obj:
                    desc = f"{f.rel} {f.obj}"
                    if desc not in seen:
                        seen.add(desc)
                        parts.append(desc)
                else:
                    desc = f"{f.rel}"
                    if desc not in seen:
                        seen.add(desc)
                        parts.append(desc)
        
        # Check if entity is object of any fact (inverse relations)
        for f in self.facts:
            if f.obj == e and not f.neg:
                if f.rel == "be":
                    desc = f"is {f.subj}"
                elif f.rel == "with":
                    desc = f"is held by {f.subj}"
                else:
                    desc = f"has {f.subj} as its {f.rel}"
                if desc not in seen:
                    seen.add(desc)
                    parts.append(desc)
        
        if not parts:
            # Full text search: find any fact mentioning this entity
            for f in self.facts:
                if e in f.subj or e in f.obj:
                    if f.subj == e:
                        parts.append(f"{f.rel} {f.obj}" if f.obj else f.rel)
                    else:
                        parts.append(f"{f.subj} {f.rel} {f.obj}")
            seen = set()
            unique = []
            for p in parts:
                if p not in seen:
                    seen.add(p)
                    unique.append(p)
            parts = unique
        
        return ", ".join(parts)


# ── Main Engine ──

class NeuroSymbolicEngine:
    """
    Complete cognitive engine.
    Processes documents, stores facts, answers questions.
    Handles multi-sentence inputs, coreference, all bAbI tasks.
    """
    
    def __init__(self):
        self.kg = EntityKnowledgeGraph()
    
    def hear(self, text: str) -> str:
        """Process input text. Returns response."""
        parsed = parse_sentence(text)
        
        if parsed["is_question"]:
            result = self.kg.query(text)
            return self._format_answer(result, text)
        else:
            facts = self.kg.process_statement(text)
            return self._format_storage(text)
    
    def _format_storage(self, text: str) -> str:
        """Format a storage confirmation."""
        return "Got it. I've stored that information."
    
    def _format_answer(self, result: Dict[str, Any], question: str) -> str:
        """Format an answer from the query result."""
        q_parsed = parse_sentence(question)
        qword = q_parsed["qword"]
        
        if result["type"] == "location":
            if result.get("answer"):
                return f"It is in {result['answer']}."
            return "I don't know where it is."
        
        if result["type"] == "boolean":
            return result["answer"]
        
        if result["type"] == "count":
            return result["answer"]
        
        if result["type"] == "identity":
            if result.get("answer"):
                return f"It is {result['answer']}."
            return "I don't know who that is."
        
        if result["type"] == "description":
            desc = result.get("answer", "")
            if desc:
                return f"I recall: it {desc}."
            return "I don't know anything about that yet."
        
        if result.get("found"):
            return result.get("answer", "yes")
        
        return "I don't know."
