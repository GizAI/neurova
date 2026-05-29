from __future__ import annotations
from typing import Dict, Iterable, List
from .slot_tagger import StructuredPerceptronTagger
from .phrase_segmenter import SurfaceSegmenter
from .tiny_encoder import TinySemanticEncoder
from ..ir import *

class LearnedSemanticParser:
    """Sample-efficient no-LLM phrase-to-IR parser.

    It learns IR templates and slot extraction patterns from small supervised rows.
    It does not generate text and has no autoregressive objective.
    """
    name = "learned_semantic_parser_no_lm"

    def __init__(self, rows: Iterable[dict] | None = None):
        self.tagger = StructuredPerceptronTagger()
        self.encoder = TinySemanticEncoder()
        self.segmenter = SurfaceSegmenter()
        if rows:
            self.fit(rows)

    def fit(self, rows: Iterable[dict]) -> "LearnedSemanticParser":
        rows = list(rows)
        self.tagger.fit(rows)
        self.encoder.fit(rows)
        return self

    @property
    def objective(self) -> str:
        return self.tagger.objective + " + " + self.encoder.objective

    def parse(self, text: str) -> List[IRCandidate]:
        # First try multi-fragment composition; skip prompts better handled by grammar/regex.
        low = text.lower().strip()
        if low.startswith("teach:") or low.startswith("explain ") or low.startswith("why ") or low.endswith("?") or any(k in low for k in ["implement", "research", "write", "essay"]):
            return []
        if True:
            segments = self.segmenter.segment(text)
            if len(segments) > 1:
                items: List[CognitiveIR] = []
                scores: List[float] = []
                for seg in segments:
                    subs = self.parse_one(seg.text)
                    if subs:
                        items.append(subs[0].ir); scores.append(subs[0].confidence)
                if len(items) >= 2:
                    return [IRCandidate(CompositeIR(items=items, source_text=text), min(0.94, sum(scores) / len(scores) + 0.06), "learned_phrase_composer", notes=["phrase-to-IR fragments composed"])]
        return self.parse_one(text)

    def parse_one(self, text: str) -> List[IRCandidate]:
        preds = self.tagger.predict(text)
        cands: List[IRCandidate] = []
        type_scores = {r.ir_type: r.score for r in self.encoder.predict(text, top_k=8)}
        for p in preds:
            ir = self._ir_from_prediction(p.ir_type, p.slots)
            if ir is not None:
                bonus = max(-0.05, min(0.12, 0.015 * type_scores.get(p.ir_type, 0.0)))
                cands.append(IRCandidate(ir, p.score + bonus, "learned_semantic_parser", notes=["structured template learned from NL→IR corpus", "tiny semantic encoder adjusted candidate score"]))
        # Prefer specific semantic atoms over generic ClaimIR when surface cues clearly indicate
        # comparison/causality/time. This is a verifier-style disambiguation, not generation.
        low = text.lower()
        specific_cue = any(x in low for x in [" than ", " above ", " exceeds ", " outranks ", "보다", "leads to", "because of", "causes", "때문에", "원인", "exception to", " not ", "does not", " is no ", "on ", "in ", "during "]) or bool(__import__('re').match(r"^\d{4}:", low))
        if specific_cue and any(type(c.ir) is not ClaimIR for c in cands):
            cands = [c for c in cands if type(c.ir) is not ClaimIR]
        # If a learned alias candidate says "classified as X"/"belongs to X", prefer
        # the normalized object X over the generic "is classified as X" claim.
        if any(type(c.ir) is ClaimIR for c in cands):
            dirty = ("classified as", "belongs to", "counts as", "count as")
            clean_claims = [c for c in cands if type(c.ir) is ClaimIR and not any(d in c.ir.object.lower() for d in dirty)]
            if clean_claims:
                cands = [c for c in cands if type(c.ir) is not ClaimIR or c in clean_claims]
        return cands

    def _ir_from_prediction(self, ir_type: str, s: Dict[str, str]) -> CognitiveIR | None:
        if ir_type == "ClaimIR":
            return ClaimIR(subject=s.get("subject", ""), relation=s.get("relation", "is"), object=s.get("object", ""))
        if ir_type == "NegatedClaimIR":
            return NegatedClaimIR(subject=s.get("subject", ""), relation=s.get("relation", "is"), object=s.get("object", ""))
        if ir_type == "TemporalClaimIR":
            t = s.get("time_expr") or s.get("valid_from") or s.get("time") or ""
            return TemporalClaimIR(subject=s.get("subject", ""), relation=s.get("relation", "is"), object=s.get("object", ""), time_expr=t, valid_from=t, valid_during=t)
        if ir_type == "CausalClaimIR":
            return CausalClaimIR(cause=s.get("cause", ""), effect=s.get("effect", ""))
        if ir_type == "ComparisonIR":
            return ComparisonIR(left=s.get("left", ""), comparator=s.get("comparator", "greater_than"), right=s.get("right", ""))
        if ir_type == "ExceptionIR":
            subj = s.get("exception_subject", s.get("subject", ""))
            cond = s.get("condition_object", s.get("domain", ""))
            rel = s.get("conclusion_relation", "can")
            obj = s.get("conclusion_object", s.get("object", ""))
            sig = f"is|{cond}=>{rel}|{obj}"
            return ExceptionIR(rule_id=s.get("rule_id", sig), exception_subject=subj, exception_text=f"{subj} exception", condition_object=cond, conclusion_relation=rel, conclusion_object=obj)
        return None
