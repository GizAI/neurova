from __future__ import annotations
import re
from typing import List
from ..ir import *
from .v23_parsers import strip_article, _target_from_natural


def _clean(x: str) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip().strip(" .?!'\"")).lower()


def _target_to_tool(surface: str, meaning: str, parser: str) -> List[IRCandidate]:
    target = _target_from_natural(meaning)
    if target:
        return [IRCandidate(ToolCallIR(tool_name="learn_construction", args={"text": surface.strip(), "target": target, "source": parser}), 0.985, parser, notes=["natural_language_feedback", "feature_construction_candidate"])]
    return []


class V24InteractiveCorrectionParser:
    """Natural-language feedback -> feature construction patch.

    Covers interactive semantic parsing feedback forms without requiring a formal
    DSL from the user.  The generated ToolCallIR is handled by the compiler's
    construction grammar engine, not by appending one-off regexes.
    """
    name = "v24_interactive_correction_parser"

    def parse(self, text: str) -> List[IRCandidate]:
        raw = text.strip().strip(" .")
        patterns = [
            r'^when\s+i\s+say\s+["“](.+?)["”],?\s*(?:it\s+means|i\s+mean|please\s+interpret\s+it\s+as)\s+(.+)$',
            r'^by\s+["“](.+?)["”]\s+i\s+mean\s+(.+)$',
            r'^["“](.+?)["”]\s+(?:means|denotes|maps\s+to|should\s+be\s+understood\s+as|is\s+equivalent\s+to)\s+(.+)$',
            r'^(.+?)\s+(?:should\s+be\s+understood\s+as|is\s+equivalent\s+to|means|denotes|maps\s+to)\s+(.+)$',
            r'^(.+?)(?:라는\s+말은|라는\s+표현은|는|은)\s+(.+?)(?:라는\s+뜻이야|로\s+해석해|로\s+이해해)$',
        ]
        for pat in patterns:
            m = re.match(pat, raw, re.I)
            if m:
                surface, meaning = m.groups()
                # Avoid capturing meta-prefix as surface when generic means matches.
                if surface.lower().startswith("when i say"):
                    continue
                return _target_to_tool(surface, meaning, self.name)
        return []


class V24TaxonomyQuestionParser:
    name = "v24_taxonomy_question_parser"
    def parse(self, text: str) -> List[IRCandidate]:
        raw = text.strip().strip(" .")
        low = raw.lower()
        patterns = [
            r"^is\s+it\s+fair\s+to\s+call\s+(.+?)\s+(?:a|an|the)?\s*(.+?)\??$",
            r"^(?:could|can)\s+(.+?)\s+be\s+treated\s+as\s+(?:a\s+kind\s+of\s+|a\s+|an\s+|the\s+)?(.+?)\??$",
            r"^would\s+(.+?)\s+qualify\s+as\s+(?:a\s+|an\s+|the\s+)?(.+?)\??$",
            r"^does\s+(.+?)\s+count\s+as\s+(?:a\s+|an\s+|the\s+)?(.+?)\??$",
        ]
        for pat in patterns:
            m = re.match(pat, low, re.I)
            if m:
                subj, obj = m.groups()
                return [IRCandidate(QuestionIR(target=ClaimIR(subject=subj.strip(), relation="is", object=strip_article(obj)), requested_mode="proof"), 0.93, self.name)]
        # Possession question from event-derived claims.
        m = re.match(r"^does\s+(.+?)\s+have\s+(?:a\s+|an\s+|the\s+)?(.+?)\??$", low)
        if m:
            subj, obj = m.groups()
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=subj.strip(), relation="has", object=strip_article(obj)), requested_mode="proof"), 0.92, self.name)]
        return []


class V24KoreanGrammarParser:
    """Particle/ending-aware Korean mini grammar for comparison and negation."""
    name = "v24_korean_grammar_parser"
    def parse(self, text: str) -> List[IRCandidate]:
        raw = text.strip().strip(" .")
        low = raw.lower()
        # B보다 A가 크다는 말이 맞아?
        m = re.match(r"^(.+?)보다\s+(.+?)(?:가|이|는|은)?\s*(?:더\s*)?(크|높|빠르|작|낮|느리)(?:다는\s+말이\s+맞아\??|니\??|냐\??|다\??)$", low)
        if m:
            right, left, stem = m.groups()
            comp = "greater_than" if stem in {"크", "높", "빠르"} else "less_than"
            ir = ComparisonIR(left=left.strip(), comparator=comp, right=right.strip())
            if "?" in raw or "맞아" in low or low.endswith(("니", "냐")):
                return [IRCandidate(QuestionIR(target=ir, requested_mode="proof"), 0.94, self.name)]
            return [IRCandidate(ir, 0.93, self.name)]
        # A는 B보다 큰 편이다 / 우위에 있다 / 앞서 있니?
        m = re.match(r"^(.+?)(?:는|은|가|이)?\s+(.+?)보다\s+(?:더\s*)?(큰\s+편이다|우위에\s+있다|앞서\s+있니\??|앞서\s+있다)$", low)
        if m:
            left, right, _ = m.groups()
            ir = ComparisonIR(left=left.strip(), comparator="greater_than", right=right.strip())
            if "?" in raw or "있니" in low:
                return [IRCandidate(QuestionIR(target=ir, requested_mode="proof"), 0.93, self.name)]
            return [IRCandidate(ir, 0.92, self.name)]
        m = re.match(r"^(.+?)(?:는|은|가|이)?\s+(.+?)에\s+비해\s+(?:더\s*)?(우위에\s+있다|앞서\s+있다|크다|높다|빠르다)$", low)
        if m:
            left, right, _ = m.groups()
            return [IRCandidate(ComparisonIR(left=left.strip(), comparator="greater_than", right=right.strip()), 0.92, self.name)]
        # A는 B보다 크지 않다 => inverse/negated comparison. Use less_than as conservative operational inverse.
        m = re.match(r"^(.+?)(?:는|은|가|이)?\s+(.+?)보다\s+(크|높|빠르|작|낮|느리)지\s+않다$", low)
        if m:
            left, right, stem = m.groups()
            comp = "less_than" if stem in {"크", "높", "빠르"} else "greater_than"
            return [IRCandidate(ComparisonIR(left=left.strip(), comparator=comp, right=right.strip()), 0.82, self.name, notes=["negated Korean comparison normalized to operational inverse"])]
        return []


class V24TemporalIntervalParser:
    name = "v24_temporal_interval_parser"
    def parse(self, text: str) -> List[IRCandidate]:
        raw = text.strip().strip(" .")
        low = raw.lower().replace(",", "")
        m = re.match(r"^(?:from\s+)?(\d{4})\s+(?:to|through|until|-)\s+(\d{4})\s+(.+?)\s+(?:served\s+as|was|is)\s+(?:the\s+)?(.+)$", low)
        if not m:
            m = re.match(r"^(.+?)\s+(?:served\s+as|was|is)\s+(?:the\s+)?(.+?)\s+from\s+(\d{4})\s+(?:to|through|until|-)\s+(\d{4})$", low)
            if m:
                subj, obj, start, end = m.groups()
                return [IRCandidate(TemporalClaimIR(subject=subj.strip(), relation="is", object=strip_article(obj), time_expr=start, valid_from=start, valid_to=end, valid_during=f"{start}-{end}"), 0.94, self.name)]
        else:
            start, end, subj, obj = m.groups()
            return [IRCandidate(TemporalClaimIR(subject=subj.strip(), relation="is", object=strip_article(obj), time_expr=start, valid_from=start, valid_to=end, valid_during=f"{start}-{end}"), 0.94, self.name)]
        m = re.match(r"^in\s+(\d{4})\s+(.+?)\s+(?:was|is)\s+not\s+(?:the\s+)?(.+)$", low)
        if m:
            t, subj, obj = m.groups()
            return [IRCandidate(TemporalClaimIR(subject=subj.strip(), relation="is", object=strip_article(obj), polarity="negative", time_expr=t, valid_from=t, valid_during=t), 0.94, self.name)]
        m = re.match(r"^(.+?)\s+became\s+(?:the\s+)?(.+?)\s+in\s+(\d{4})$", low)
        if m:
            subj, obj, t = m.groups()
            return [IRCandidate(TemporalClaimIR(subject=subj.strip(), relation="is", object=strip_article(obj), time_expr=t, valid_from=t, valid_during=t), 0.9, self.name)]
        m = re.match(r"^(.+?)\s+stopped\s+being\s+(?:the\s+)?(.+?)\s+in\s+(\d{4})$", low)
        if m:
            subj, obj, t = m.groups()
            return [IRCandidate(TemporalClaimIR(subject=subj.strip(), relation="is", object=strip_article(obj), polarity="negative", time_expr=t, valid_from=t, valid_during=t), 0.88, self.name)]
        return []


class V24EventFrameParser:
    name = "v24_event_frame_parser"
    def parse(self, text: str) -> List[IRCandidate]:
        raw = text.strip().strip(" .")
        low = raw.lower()
        # Alice gave a book to Bob in Seoul yesterday.
        m = re.match(r"^(.+?)\s+(?:gave|handed|sent)\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+to\s+(.+?)(?:\s+in\s+(.+?))?(?:\s+(yesterday|today|tomorrow|in\s+\d{4}|on\s+[0-9\-]+))?$", low)
        if m:
            actor, patient, rec, loc, t = m.groups()
            return [IRCandidate(EventIR(actor=actor.strip(), action="give", patient=strip_article(patient), recipient=rec.strip(), location=(loc or "").strip() or None, time_expr=(t or "").strip() or None), 0.92, self.name)]
        # Bob received a book from Alice yesterday.
        m = re.match(r"^(.+?)\s+received\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+from\s+(.+?)(?:\s+(yesterday|today|tomorrow))?$", low)
        if m:
            rec, patient, actor, t = m.groups()
            return [IRCandidate(EventIR(actor=actor.strip(), action="give", patient=strip_article(patient), recipient=rec.strip(), time_expr=(t or "").strip() or None), 0.91, self.name)]
        return []


class V24ExceptionDiscourseParser:
    name = "v24_exception_discourse_parser"
    def parse(self, text: str) -> List[IRCandidate]:
        raw = text.strip().strip(" .")
        low = raw.lower()
        # Penguins are birds; however/but/although they usually do not fly.
        m = re.match(r"^(?:although\s+)?(?P<subj>\w+)s?\s+are\s+(?P<kind>\w+)s?[,;]?\s*(?:(?:however|but|although),?\s*)?they\s+(?:usually\s+|normally\s+)?(?:do\s+not|cannot|can't)\s+(?P<action>.+)$", low)
        if m:
            subj, cond, action = strip_article(m.group('subj')), strip_article(m.group('kind')), strip_article(m.group('action'))
            return [IRCandidate(CompositeIR(items=[ClaimIR(subject=subj, relation="is", object=cond), ExceptionIR(rule_id=f"is|{cond}=>can|{action}", exception_subject=subj, exception_text=raw, condition_object=cond, conclusion_relation="can", conclusion_object=action)], source_text=raw), 0.93, self.name)]
        # Can a penguin fly even though it is a bird?
        m = re.match(r"^can\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+(.+?)\s+even\s+though\s+it\s+is\s+(?:a\s+|an\s+|the\s+)?(.+?)\??$", low)
        if m:
            subj, action, cond = m.groups()
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=strip_article(subj), relation="can", object=strip_article(action)), requested_mode="proof"), 0.9, self.name, notes=[f"exception_condition={strip_article(cond)}"])]
        return []


__all__ = [
    "V24InteractiveCorrectionParser", "V24TaxonomyQuestionParser", "V24KoreanGrammarParser",
    "V24TemporalIntervalParser", "V24EventFrameParser", "V24ExceptionDiscourseParser"
]
