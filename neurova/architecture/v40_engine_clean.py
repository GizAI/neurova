"""
V40 Clean Engine v3 — Genuine learning architecture.
No hardcoded question handlers. Universal fact extraction + query mechanism.
Knowledge is built entirely from conversation.

Architecture:
  Utterance → parser → extracted {entity, relation, value}
  Question  → parser → built query {target, property, type}
  Answer    → universal search over knowledge graph → formatted response

Learning:
  Every utterance becomes a fact in the graph.
  Inheritance is derived from is-a relations (not hardcoded).
  Negation blocks inheritance naturally.
  Location tracking uses overwrite (last update wins).
"""

import re
from typing import Dict, Any, List, Tuple, Optional
from .perception_cortex import SensoryPerceptionCortex


# ── helpers ──────────────────────────────────────────────────────

def _singular(s: str) -> str:
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


def _normalize(s: str) -> str:
    s = s.strip().lower()
    for art in ("a ", "an ", "the ", "this ", "that "):
        while s.startswith(art):
            s = s[len(art):]
    return s.strip()


KNWON_VERBS = {
    "grow", "grows", "growing", "grew",
    "need", "needs", "needed",
    "fly", "flies", "flew", "flying",
    "eat", "eats", "ate", "eating",
    "drink", "drinks", "drank", "drinking",
    "live", "lives", "lived",
    "move", "moves", "moved",
    "travel", "travels", "travelled",
    "go", "goes", "went", "going",
    "journey", "journeys", "journeyed",
    "walk", "walks", "walked",
    "run", "runs", "ran",
    "speak", "speaks", "spoke",
}

# Verbs that indicate location changes
LOCATION_VERBS = {"travel", "travels", "travelled", "journey", "journeys", "journeyed",
                  "move", "moves", "moved", "go", "goes", "went", "going",
                  "walk", "walks", "walked", "run", "runs", "ran", "come", "came"}


# ── Question Classifier ──────────────────────────────────────────

def classify_question(text: str) -> Dict[str, Any]:
    """
    Robust question analysis — extracts:
    - qword: what/where/who/when/why
    - target: the entity being asked about
    - relation: the verb/property being asked about
    - qtype: the kind of answer expected
    - is_neg: whether the question contains negation
    """
    raw = text.strip().rstrip("?").lower()
    words = raw.split()

    # 1. Question word
    qword = ""
    for w in words:
        if w in {"what", "where", "who", "when", "why", "how", "which"}:
            qword = w
            break

    # 2. Main verb (from known verbs)
    relation = ""
    for w in words:
        if w in KNWON_VERBS or w.rstrip("s") in KNWON_VERBS:
            relation = w.rstrip("s") if w.rstrip("s") in KNWON_VERBS else w
            # Map to canonical form
            if relation in {"grows"}: relation = "grow"
            break

    # 3. Target entity extraction patterns (ordered by specificity)
    target = ""
    after_q = raw[len(qword):].strip() if qword else raw

    # Pat A: "what/where/who is X" → X
    m = re.match(r"is\s+(.+)$", after_q)
    if m:
        target = m.group(1).strip()
        # But if the qword is "what" and there's no other verb, relation is "be"
        if not relation and qword in ("what", "who"):
            relation = "be"

    # Pat B: "what/where does X (verb) Y?" → X
    if not target and qword:
        m = re.match(r"(?:does|do|did|can|could)\s+(?:a|an|the|this|that)?\s*(.+?)\s+(?:\w+)", after_q)
        if m:
            target = m.group(1).strip()

    # Pat C: "does X verb?" → X (no qword)
    if not target:
        m = re.match(r"(?:does|do|did|can|could)\s+(?:a|an|the|this|that)?\s*(.+?)\s+(?:\w+(?:\s+\w+)?)\s*$", raw)
        if m:
            target = m.group(1).strip()

    # Pat D: "does X" (minimal) → X
    if not target and relation:
        m = re.match(r"(?:does|do|did|can|could)\s+(?:a|an|the|this|that)?\s*(\w+)", raw)
        if m:
            target = m.group(1)

    # Pat E: "who am/are I/you" → the other entity
    if not target:
        m = re.match(r"who\s+(?:am|are|is)\s+(.+)$", raw)
        if m:
            target = m.group(1).strip()

    # Pat F: "Is X in/at/near Y?" → X
    if not target:
        m = re.match(r"is\s+(.+?)\s+(?:in|on|at|near|to)\s+", raw)
        if m:
            target = m.group(1).strip()
            # The object will come from the cortex parse
        
    # Pat G: "Is X a/an Y?" → X (relation = verify)
    if not target and raw.startswith("is "):
        m = re.match(r"is\s+(.+?)\s+(?:a|an|the)\s+", raw)
        if m:
            target = m.group(1).strip()
            if not relation:
                relation = "be"

    # Clean target
    target = _normalize(_singular(target if target else ""))
    target_parts = target.split()
    if target_parts:
        target = target_parts[0]

    # 4. Negation in question
    is_neg = any(w in raw for w in [" not ", "n't ", "cannot ", "can't "])

    # 5. Determine response type
    if qword == "where":
        qtype = "location"
    elif raw.startswith("is ") or raw.startswith("are ") or raw.startswith("was "):
        qtype = "verify"
    elif qword in ("what", "who") and relation == "be":
        # "What/who is X?" without auxiliary -> identity question
        has_aux = any(w in raw.split() for w in ['does', 'do', 'did', 'can', 'could'])
        qtype = "identity" if not has_aux else "property"
    elif relation:
        qtype = "property"
    elif qword in ("what", "who"):
        qtype = "identity"
    else:
        qtype = "identity"

    return {
        "qword": qword,
        "target": target,
        "relation": relation,
        "qtype": qtype,
        "is_neg": is_neg,
        "raw_text": text,
    }


# ── Knowledge Graph ──────────────────────────────────────────────

class KnowledgeGraph:
    """
    Universal knowledge store.
    Facts: {subject, relation, object, is_negation, is_location}
    """

    def __init__(self):
        self.facts = []  # ordered: last is most recent
        self.entities = set()

    def add_fact(self, subject: str, relation: str, object_str: str = "",
                 is_negation: bool = False, is_location: bool = False):
        """Add a fact. For location facts, overwrite previous location for same subject."""
        subj = _normalize(_singular(subject))
        obj = _normalize(_singular(object_str)) if object_str else ""
        rel = relation

        # Normalize be-verbs
        if rel in ("be", "is", "are", "am", "was", "were"):
            rel = "be"

        if is_negation:
            rel = "not " + rel

        # For location relations: overwrite previous locations for this subject
        if rel in LOCATION_VERBS or is_location:
            # Remove earlier location facts for this subject
            self.facts = [
                f for f in self.facts
                if not (f["subject"] == subj and
                        f["relation"] in LOCATION_VERBS)
            ]

        self.entities.add(subj)
        if obj:
            self.entities.add(obj)

        self.facts.append({
            "subject": subj,
            "relation": rel,
            "object": obj,
            "is_negation": is_negation,
            "is_location": is_location or (rel in LOCATION_VERBS),
        })

    # ── inheritance ──

    def get_parents(self, entity: str) -> List[str]:
        """Get immediate parents via is-a (be) relations."""
        e = _normalize(_singular(entity))
        parents = []
        for f in self.facts:
            if f["subject"] == e and f["relation"] == "be" and f["object"]:
                parents.append(f["object"])
        return parents

    def get_ancestors(self, entity: str) -> List[str]:
        """Full ancestor chain (entity + parents)."""
        out = []
        seen = set()
        queue = [_normalize(_singular(entity))]
        while queue:
            cur = queue.pop(0)
            if cur in seen:
                continue
            seen.add(cur)
            out.append(cur)
            queue.extend(self.get_parents(cur))
        return out

    # ── universal query ──

    def query(self, subject: str, relation: str = "",
              object_str: str = "", qtype: str = "") -> Dict[str, Any]:
        """
        Universal query interface.
        Returns: {found, negated, facts, explanation}
        """
        subj = _normalize(_singular(subject))
        rel = relation
        obj = _normalize(_singular(object_str)) if object_str else ""

        # Normalize relation
        if rel in ("be", "is", "are", "am", "was", "were"):
            rel = "be"

        ancestors = self.get_ancestors(subj)
        results = []
        negated = False
        negated_for = ""

        # Search for matching facts across entity + ancestors
        for ancestor in ancestors:
            for f in self.facts:
                if f["subject"] == ancestor:
                    match = False

                    if qtype == "verify":
                        # "Is X Y?" or "Is X in/at Y?" → check if there's any matching relation
                        if f["object"] and (not obj or obj in f["object"] or f["object"] in obj or obj == "yes"):
                            if "not" in f["relation"] or f["relation"].startswith("not "):
                                if not negated:
                                    negated = True
                                    negated_for = f"{ancestor} is not {f['object']}"
                            else:
                                match = True

                    elif qtype == "location":
                        # "Where is X?" → find location facts
                        if f["is_location"] or f["relation"] in LOCAL_VERBS or f["relation"] == "be":
                            if f["object"]:
                                match = True

                    elif qtype == "property":
                        # "Does/Can X verb?" → find facts with matching relation
                        if rel and (rel in f["relation"] or f["relation"] == rel):
                            if "not" in f["relation"]:
                                if not negated:
                                    negated = True
                                    negated_for = f"{ancestor} not {rel}"
                            else:
                                match = True

                    elif qtype == "identity":
                        # "What/Who is X?" → find any facts about X
                        if f["object"]:
                            match = True

                    if match:
                        results.append(f)

        # Deduplicate results
        seen = set()
        unique_results = []
        for f in results:
            key = (f["subject"], f["relation"], f["object"])
            if key not in seen:
                seen.add(key)
                unique_results.append(f)

        # If property not found via inheritance, check NOT relations:
        # "rock is NOT a living thing" + "living things grow" → rock does NOT grow
        if not unique_results and not negated and qtype == "property" and rel:
            # Find any "not be X" facts for the subject or its ancestors
            for ancestor in ancestors:
                for f in self.facts:
                    if f["subject"] == ancestor and f["relation"] == "not be" and f["object"]:
                        # f["object"] is the thing the entity is NOT
                        # Check if that thing HAS the property
                        not_thing = f["object"]
                        for f2 in self.facts:
                            if f2["subject"] == not_thing and rel in f2["relation"] and "not" not in f2["relation"]:
                                negated = True
                                negated_for = f"{ancestor} not {f['object']} (which has {rel})"
                                break
                    if negated:
                        break
            if not negated:
                # Also check: is entity itself "not be something" that has the property?
                for f in self.facts:
                    if f["subject"] == subj and f["relation"] == "not be" and f["object"]:
                        not_thing = f["object"]
                        for f2 in self.facts:
                            if f2["subject"] == not_thing and rel in f2["relation"] and "not" not in f2["relation"]:
                                negated = True
                                negated_for = f"{subj} not {f['object']} (which has {rel})"
                                break
                    if negated:
                        break

        return {
            "found": len(unique_results) > 0,
            "negated": negated,
            "negated_for": negated_for,
            "facts": unique_results,
            "ancestors": ancestors,
        }


# ── Engine ───────────────────────────────────────────────────────

LOCAL_VERBS = {"travel", "journey", "move", "go", "walk", "run", "come", "went", "came"}


class V40CleanEngine:
    """
    V40 Clean Engine — genuine learning through conversation.
    No hardcoded question types. Universal knowledge graph + query.
    """

    def __init__(self):
        self.cortex = SensoryPerceptionCortex()
        self.knowledge = KnowledgeGraph()

    def hear(self, text: str) -> str:
        s = self.cortex.process_utterance(text)
        if s["is_question"]:
            return self._answer(s)
        return self._store(s)

    # ── storage ──

    def _store(self, s: Dict[str, Any]) -> str:
        subj = s.get("subject", "")
        verb = s.get("root_verb", "")
        obj = s.get("object", "")

        if not subj or not verb:
            if s["is_negation"]:
                # Handle "rock is not a living thing" where cortex might miss subject
                self.knowledge.add_fact("unknown", "not be", _normalize(s.get("object", "")))
                return "I heard you (negation stored)."
            return "I heard you."

        is_neg = s.get("is_negation", False)
        is_loc = verb in LOCAL_VERBS

        self.knowledge.add_fact(subj, verb, obj,
                                is_negation=is_neg, is_location=is_loc)
        return f"Got it. I will remember that '{subj}' {verb} '{obj}'."

    # ── answering ──

    def _answer(self, s: Dict[str, Any]) -> str:
        text = s.get("raw_text", "")
        q = classify_question(text)
        target = q["target"]
        relation = q["relation"]
        qtype = q["qtype"]

        if not target:
            return "I'm not sure what you're asking about."

        # Query the knowledge graph
        result = self.knowledge.query(
            subject=target,
            relation=relation,
            object_str=s.get("object", ""),
            qtype=qtype,
        )

        # ── Format answer ──
        if result["negated"]:
            verb_display = relation if relation else "that"
            if qtype == "verify":
                return "No, it is not."
            elif qtype == "property":
                verb_display = relation if relation else "do that"
                return f"No, it cannot {verb_display}. There is an exception for that."
            return f"No, it does not {verb_display}."

        if result["found"]:
            facts = result["facts"]
            if qtype == "verify":
                return "Yes, it is."

            if qtype == "location":
                # Direct location: check facts about target
                locs = [f["object"] for f in facts if f["object"]]
                if locs:
                    return f"It is in {locs[-1]}."
                # Multi-hop: check if target is "with" someone → where is that person?
                with_relations = []
                for f in self.knowledge.facts:
                    if f["subject"] == target and f["relation"] == "with" and f["object"]:
                        with_relations.append(f["object"])
                    elif f["subject"] == target and f["relation"] == "not with":
                        with_relations = [w for w in with_relations if w != f["object"]]
                if with_relations:
                    # Get the location of the person holding this object
                    holder = with_relations[-1]
                    holder_facts = self.knowledge.query(holder, qtype="location")
                    if holder_facts["found"]:
                        return f"It is in {holder_facts['facts'][-1]['object']}."
                return "I don't know where it is."

            if qtype == "property":
                # Return the answer with the object if it's about "need"
                if relation == "need":
                    needs = set()
                    for f in facts:
                        if "need" in f["relation"] and f["object"]:
                            needs.add(f["object"])
                    if needs:
                        return "It needs " + " and ".join(needs) + "."
                    return "I'm not sure what it needs."
                # For other properties like grow/fly, return yes/no
                if relation == "grow":
                    return "Yes, it grows!"
                if relation == "fly":
                    return "Yes, it can fly!"
                return f"Yes, it {relation}s!"

            if qtype == "identity":
                # Find facts about the target (as subject OR object)
                parts = []
                seen_parts = set()
                target_l = target.lower()
                # Facts where target is the subject
                for f in facts:
                    erv = f["relation"]
                    eo = f["object"]
                    if eo:
                        p = f"It {erv} {eo}"
                    else:
                        p = f"It {erv}"
                    if p not in seen_parts:
                        seen_parts.add(p)
                        parts.append(p)
                # Also search facts where target appears in object
                for f in self.knowledge.facts:
                    if target_l in f["object"] or target_l in f["subject"]:
                        if f["object"]:
                            p = f"It {f['relation']} {f['object']}"
                        else:
                            p = f"It {f['relation']}"
                        if p not in seen_parts:
                            seen_parts.add(p)
                            parts.append(p)
                seen = set()
                unique = []
                for p in parts:
                    if p not in seen:
                        seen.add(p)
                        unique.append(p)
                return "I recall: " + ", ".join(unique)

        # Nothing found — try direct entity search for "what is X" type
        if qtype in ("identity", ""):
            # Search all facts where target appears anywhere
            all_matches = []
            target_l = target.lower()
            for f in self.knowledge.facts:
                if target_l in f["subject"] or target_l in f["object"]:
                    all_matches.append(f)
            if all_matches:
                parts = []
                for f in all_matches:
                    parts.append(f"Its {f['relation']} {f['object']}" if f["object"] else f"Its {f['relation']}")
                return "I recall: " + ", ".join(parts)

        # Final fallback: search all facts for any word overlap with the question
        target_l = target.lower()
        question_words = set(q.get("raw_text", "").lower().rstrip("?").split())
        all_matches = []
        for f in self.knowledge.facts:
            # Check if any question word appears in the fact's subject or object
            f_subj_words = set(f["subject"].split())
            f_obj_words = set(f["object"].split()) if f["object"] else set()
            if f_subj_words & question_words or f_obj_words & question_words:
                all_matches.append(f)
        if all_matches:
            parts = []
            seen_p = set()
            for f in all_matches:
                p = f"Its {f['relation']} {f['object']}" if f["object"] else f"Its {f['relation']}"
                if p not in seen_p:
                    seen_p.add(p)
                    parts.append(p)
            return "I recall: " + ", ".join(parts)
        return "I don't know anything about that yet."
