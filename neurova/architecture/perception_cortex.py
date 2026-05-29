"""
V40 Perception Cortex — fixed for deep preposition structures ("went back to X").
Uses singleton spaCy model to avoid reload overhead.
"""
from typing import Dict, Any, List
import logging

# Singleton spaCy (lazy loaded)
_NLP_EN = None
_KIWI = None

def _get_nlp():
    global _NLP_EN
    if _NLP_EN is None:
        try:
            import spacy
            _NLP_EN = spacy.load("en_core_web_sm")
        except Exception as e:
            logging.warning(f"spaCy unavailable: {e}")
    return _NLP_EN

def _get_kiwi():
    global _KIWI
    if _KIWI is None:
        try:
            from kiwipiepy import Kiwi
            _KIWI = Kiwi()
        except Exception as e:
            logging.warning(f"Kiwi unavailable: {e}")
    return _KIWI


def _find_prep_objects(token, depth=0):
    """
    Recursively find prepositional objects anywhere in the dependency tree.
    Handles "went back to X" where prep "to" is a child of "back", not ROOT.
    """
    if depth > 5:
        return []
    results = []
    for child in token.children:
        if child.pos_ == "ADP" and child.dep_ == "prep":
            for gc in child.children:
                if gc.dep_ == "pobj":
                    obj_text = " ".join(c.text.lower() for c in gc.subtree)
                    results.append(obj_text)
        results.extend(_find_prep_objects(child, depth + 1))
    return results


class SensoryPerceptionCortex:
    """Enhanced NLP cortex with singleton spaCy and recursive prep search."""

    def process_utterance(self, text: str) -> Dict[str, Any]:
        is_korean = any('\uac00' <= c <= '\ud7a3' for c in text)
        is_question = "?" in text
        raw_lower = text.lower()

        result = {
            "raw_text": text,
            "language": "ko" if is_korean else "en",
            "entities": [],
            "root_verb": "",
            "subject": "",
            "object": "",
            "is_question": is_question,
            "is_negation": bool("not" in raw_lower or "n't" in raw_lower
                                or "cannot" in raw_lower or "can't" in raw_lower),
        }

        if is_korean:
            kiwi = _get_kiwi()
            if kiwi:
                try:
                    tokens = kiwi.tokenize(text)
                    for t in tokens:
                        if t.tag.startswith('N'):
                            result["entities"].append(t.form)
                        elif t.tag.startswith('V'):
                            result["root_verb"] = t.form
                except Exception:
                    pass
            return result

        nlp = _get_nlp()
        if not nlp:
            return result

        try:
            doc = nlp(text)

            for ent in doc.ents:
                result["entities"].append({"text": ent.text, "label": ent.label_})

            subject, obj = "", ""

            for token in doc:
                if token.dep_ == "ROOT":
                    result["root_verb"] = token.lemma_
                    for child in token.children:
                        if child.dep_ in ("nsubj", "nsubjpass", "expl"):
                            subject = " ".join(c.text.lower() for c in child.subtree)
                        elif child.dep_ in ("attr", "acomp", "dobj"):
                            obj = " ".join(c.text.lower() for c in child.subtree)
                        elif child.dep_ == "prep":
                            for gc in child.children:
                                if gc.dep_ == "pobj":
                                    pobj = " ".join(c.text.lower() for c in gc.subtree)
                                    if obj:
                                        obj += " " + pobj
                                    else:
                                        obj = pobj

                    # Deep prep search for "went back to X" patterns
                    if not obj:
                        deep_objs = _find_prep_objects(token)
                        if deep_objs:
                            obj = deep_objs[-1]

            result["subject"] = subject
            result["object"] = obj

        except Exception:
            pass

        return result
