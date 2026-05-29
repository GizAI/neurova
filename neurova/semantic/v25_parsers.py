from __future__ import annotations
import re
from typing import List
from ..ir import *


def _clean(x: str) -> str:
    x = re.sub(r"\s+", " ", (x or "").strip().lower().strip(" .?!,;:'\"“”"))
    return x


def _strip_article(x: str) -> str:
    x = _clean(x)
    x = re.sub(r"^(a|an|the)\s+", "", x)
    if x.endswith("s") and len(x) > 3 and not x.endswith("ss"):
        x = x[:-1]
    return x


def _target_from_free_meaning(meaning: str) -> str | None:
    m = _clean(meaning)
    # A is greater than B / A greater_than B / A faster than B.
    if re.search(r"\bgreater_than\b|\bgreater\s+than\b|\bfaster\s+than\b|\btaller\s+than\b|\babove\b|\bahead\s+of\b|\bsuperior\s+to\b", m):
        return "compare(A,greater_than,B)"
    if re.search(r"\bless_than\b|\bless\s+than\b|\bslower\s+than\b|\bsmaller\s+than\b|\bbehind\b|\binferior\s+to\b", m):
        return "compare(A,less_than,B)"
    if re.search(r"\bcauses\b|\bcause\b|\bleads\s+to\b|\bbrings\b|\bsparks\b|\bresults\s+in\b|\b원인\b", m):
        return "causal(A,B)"
    mt = re.match(r"^a\s+is\s+not\s+(.+?)\s+b$", m)
    if mt:
        return "not_claim(A,is,B)"
    if re.search(r"\bis\b|\bclassified\s+as\b|\bcounts\s+as\b|\bkind\s+of\b", m):
        return "claim(A,is,B)"
    return None


class V25InteractiveSemanticFeedbackParser:
    """Interactive semantic parsing feedback -> construction update.

    This accepts free-form correction utterances and produces a learn_construction
    tool call. It covers the NL-EDIT / transparent interactive parsing pattern:
    users correct the system in language, not with Python APIs.
    """
    name = "v25_interactive_semantic_feedback"

    def parse(self, text: str) -> List[IRCandidate]:
        raw = text.strip().strip(" .")
        patterns = [
            r'^when\s+i\s+say\s+["“](.+?)["”],?\s*(?:it\s+means|i\s+mean|please\s+interpret\s+it\s+as|understand\s+it\s+as)\s+(.+)$',
            r'^by\s+["“](.+?)["”],?\s+i\s+mean\s+(.+)$',
            r'^["“](.+?)["”]\s+(?:means|denotes|maps\s+to|should\s+be\s+understood\s+as|is\s+equivalent\s+to|should\s+mean)\s+(.+)$',
            r'^(.+?)\s+(?:means|denotes|maps\s+to|should\s+be\s+understood\s+as|is\s+equivalent\s+to)\s+(.+)$',
            r'^["“](.+?)["”]\s*(?:라는\s+말은|라는\s+표현은)\s*(.+?)(?:라는\s+뜻이야|로\s+해석해|로\s+이해해)$',
        ]
        for pat in patterns:
            m = re.match(pat, raw, re.I)
            if not m:
                continue
            surface, meaning = m.groups()
            target = _target_from_free_meaning(meaning)
            if not target:
                continue
            return [IRCandidate(ToolCallIR(tool_name="learn_construction", args={"text": surface.strip(), "target": target, "source": self.name}), 0.995, self.name, notes=["interactive_feedback", "parser_update"])]
        # Step correction style: "No, relation should be greater_than" etc.  This is logged as an active-teacher item by the agent/compiler layer.
        m = re.match(r"^(?:no,?\s*)?(?:the\s+)?(subject|relation|object|time|polarity)\s+should\s+be\s+(.+)$", raw, re.I)
        if m:
            field, value = m.groups()
            return [IRCandidate(ToolCallIR(tool_name="parser_step_correction", args={"field": field.lower(), "value": value.strip(), "source": self.name}), 0.9, self.name, notes=["step_level_correction"])]
        return []


class V25QuestionGeneralizationParser:
    name = "v25_question_generalization_parser"
    def parse(self, text: str) -> List[IRCandidate]:
        raw = text.strip().strip(" .")
        low = _clean(raw)
        # Taxonomy-like questions.
        pats = [
            r"^is\s+it\s+fair\s+to\s+(?:call|say)\s+(.+?)\s+(?:a\s+|an\s+|the\s+)?(.+)$",
            r"^(?:could|can)\s+(.+?)\s+be\s+treated\s+as\s+(?:a\s+kind\s+of\s+|a\s+|an\s+|the\s+)?(.+)$",
            r"^would\s+(.+?)\s+qualify\s+as\s+(?:a\s+|an\s+|the\s+)?(.+)$",
            r"^(?:would\s+you\s+say\s+)?(.+?)\s+counts?\s+as\s+(?:a\s+|an\s+|the\s+)?(.+)$",
        ]
        for p in pats:
            m = re.match(p, low)
            if m:
                a, b = m.groups()
                return [IRCandidate(QuestionIR(target=ClaimIR(subject=_strip_article(a), relation="is", object=_strip_article(b)), requested_mode="proof"), 0.94, self.name)]
        # Generic construction-friendly comparison questions.
        m = re.match(r"^is\s+(.+?)\s+(?:slightly\s+|clearly\s+|far\s+)?(?:ahead\s+of|above|greater\s+than|better\s+than|superior\s+to)\s+(.+)$", low)
        if m:
            a, b = m.groups()
            return [IRCandidate(QuestionIR(target=ComparisonIR(left=_strip_article(a), comparator="greater_than", right=_strip_article(b)), requested_mode="proof"), 0.92, self.name)]
        m = re.match(r"^does\s+(.+?)\s+(?:outrank|dominate|surpass|exceed|outperform|outrun)\s+(.+)$", low)
        if m:
            a, b = m.groups()
            return [IRCandidate(QuestionIR(target=ComparisonIR(left=_strip_article(a), comparator="greater_than", right=_strip_article(b)), requested_mode="proof"), 0.91, self.name)]
        m = re.match(r"^does\s+(.+?)\s+have\s+(?:a\s+|an\s+|the\s+)?(.+)$", low)
        if m:
            a, b = m.groups()
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=_strip_article(a), relation="has", object=_strip_article(b)), requested_mode="proof"), 0.94, self.name)]
        return []


class V25TemporalStateParser:
    name = "v25_temporal_state_parser"
    def parse(self, text: str) -> List[IRCandidate]:
        raw = text.strip().strip(" .")
        low = _clean(raw.replace(",", ""))
        # From 2025 to/through 2026, Alice served as CEO.
        m = re.match(r"^from\s+(\d{4})\s+(?:to|through|until|-)\s+(\d{4})\s+(.+?)\s+(?:served\s+as|was|is)\s+(?:the\s+)?(.+)$", low)
        if m:
            start, end, subj, obj = m.groups()
            return [IRCandidate(TemporalClaimIR(subject=_strip_article(subj), relation="is", object=_strip_article(obj), time_expr=start, valid_from=start, valid_to=end, valid_during=f"{start}-{end}"), 0.96, self.name)]
        m = re.match(r"^(.+?)\s+(?:served\s+as|was|is)\s+(?:the\s+)?(.+?)\s+from\s+(\d{4})\s+(?:to|through|until|-)\s+(\d{4})$", low)
        if m:
            subj, obj, start, end = m.groups()
            return [IRCandidate(TemporalClaimIR(subject=_strip_article(subj), relation="is", object=_strip_article(obj), time_expr=start, valid_from=start, valid_to=end, valid_during=f"{start}-{end}"), 0.96, self.name)]
        m = re.match(r"^in\s+(\d{4})\s+(.+?)\s+(?:was|is)\s+not\s+(?:the\s+)?(.+)$", low)
        if m:
            t, subj, obj = m.groups()
            return [IRCandidate(TemporalClaimIR(subject=_strip_article(subj), relation="is", object=_strip_article(obj), polarity="negative", time_expr=t, valid_from=t, valid_during=t), 0.96, self.name)]
        m = re.match(r"^(.+?)\s+became\s+(?:the\s+)?(.+?)\s+in\s+(\d{4})$", low)
        if m:
            subj, obj, t = m.groups()
            return [IRCandidate(TemporalClaimIR(subject=_strip_article(subj), relation="is", object=_strip_article(obj), time_expr=t, valid_from=t, valid_to="9999", valid_during=f"{t}-9999", source_id="became_event"), 0.94, self.name)]
        m = re.match(r"^(.+?)\s+stopped\s+being\s+(?:the\s+)?(.+?)\s+in\s+(\d{4})$", low)
        if m:
            subj, obj, t = m.groups()
            return [IRCandidate(TemporalClaimIR(subject=_strip_article(subj), relation="is", object=_strip_article(obj), polarity="negative", time_expr=t, valid_from=t, valid_during=t, source_id="stop_event"), 0.93, self.name)]
        m = re.match(r"^who\s+(?:held\s+the\s+)?(.+?)\s+role\s+(?:during|in)\s+(\d{4})$", low)
        if m:
            obj, t = m.groups()
            return [IRCandidate(QuestionIR(target=TemporalClaimIR(subject="?", relation="is", object=_strip_article(obj), time_expr=t, valid_from=t, valid_during=t), requested_mode="proof"), 0.93, self.name)]
        m = re.match(r"^who\s+(?:was|is)\s+(?:the\s+)?(.+?)\s+(?:during|in)\s+(\d{4})$", low)
        if m:
            obj, t = m.groups()
            return [IRCandidate(QuestionIR(target=TemporalClaimIR(subject="?", relation="is", object=_strip_article(obj), time_expr=t, valid_from=t, valid_during=t), requested_mode="proof"), 0.93, self.name)]
        return []


class V25EventWorldFrameParser:
    name = "v25_event_world_frame_parser"
    def parse(self, text: str) -> List[IRCandidate]:
        raw = text.strip().strip(" .")
        low = _clean(raw)
        # Alice gave a book to Bob in Seoul yesterday.  Put this before the double-object frame.
        m = re.match(r"^(.+?)\s+(?:gave|handed|sent)\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+to\s+(.+?)(?:\s+in\s+(.+?))?(?:\s+(yesterday|today|tomorrow|in\s+\d{4}|on\s+[0-9\-]+))?$", low)
        if m:
            actor, patient, rec, loc, t = m.groups()
            return [IRCandidate(EventIR(actor=_strip_article(actor), action="give", patient=_strip_article(patient), recipient=_strip_article(rec), location=_strip_article(loc) if loc else None, time_expr=t), 0.95, self.name)]
        # Alice gave Bob a book in Seoul yesterday.
        m = re.match(r"^(.+?)\s+(?:gave|handed|sent)\s+(.+?)\s+(?:a\s+|an\s+|the\s+)?(.+?)(?:\s+in\s+(.+?))?(?:\s+(yesterday|today|tomorrow|in\s+\d{4}|on\s+[0-9\-]+))?$", low)
        if m:
            actor, rec, patient, loc, t = m.groups()
            return [IRCandidate(EventIR(actor=_strip_article(actor), action="give", patient=_strip_article(patient), recipient=_strip_article(rec), location=_strip_article(loc) if loc else None, time_expr=t), 0.94, self.name)]
        # Bob received a book from Alice yesterday.
        m = re.match(r"^(.+?)\s+received\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+from\s+(.+?)(?:\s+(yesterday|today|tomorrow))?$", low)
        if m:
            rec, patient, actor, t = m.groups()
            return [IRCandidate(EventIR(actor=_strip_article(actor), action="give", patient=_strip_article(patient), recipient=_strip_article(rec), time_expr=t), 0.94, self.name)]
        # action/state frames
        m = re.match(r"^(.+?)\s+(opened|closed|moved|collected)\s+(?:a\s+|an\s+|the\s+)?(.+?)(?:\s+in\s+(.+?))?(?:\s+(yesterday|today|tomorrow))?$", low)
        if m:
            actor, action, patient, loc, t = m.groups()
            return [IRCandidate(EventIR(actor=_strip_article(actor), action=action[:-2] if action.endswith("ed") else action, patient=_strip_article(patient), location=_strip_article(loc) if loc else None, time_expr=t), 0.9, self.name)]
        # Speech acts.
        m = re.match(r"^(.+?)\s+(asked|ordered|told)\s+(.+?)\s+to\s+(.+)$", low)
        if m:
            speaker, verb, target, action = m.groups()
            content = EventIR(actor=_strip_article(target), action=_strip_article(action))
            return [IRCandidate(SpeechActIR(speaker=_strip_article(speaker), act_type="order" if verb == "ordered" else "request", content=content), 0.9, self.name)]
        return []


class V25MentalStateParser:
    name = "v25_mental_state_parser"
    def parse(self, text: str) -> List[IRCandidate]:
        raw = text.strip().strip(" .")
        low = _clean(raw)
        m = re.match(r"^(?:does\s+)?(.+?)\s+(?:believe|believes|think|thinks|know|knows)\s+(?:that\s+)?(.+?)\s+(?:is|was)\s+not\s+(?:the\s+)?(.+)$", low)
        if m:
            holder, subj, obj = m.groups()
            prop = NegatedClaimIR(subject=_strip_article(subj), relation="is", object=_strip_article(obj))
            belief = BeliefIR(holder=_strip_article(holder), proposition=prop)
            if low.startswith("does "):
                return [IRCandidate(QuestionIR(target=ClaimIR(subject=_strip_article(holder), relation="believes", object=prop.text()), requested_mode="proof"), 0.86, self.name)]
            return [IRCandidate(belief, 0.91, self.name)]
        m = re.match(r"^(.+?)\s+(?:plans\s+on|plans\s+to|intends\s+to|wants\s+to)\s+(.+)$", low)
        if m:
            agent, goal = m.groups()
            return [IRCandidate(GoalIR(agent=_strip_article(agent), desired_state=_strip_article(goal)), 0.9, self.name)]
        return []


class V25KoreanParticleGrammarParser:
    name = "v25_korean_particle_grammar"
    def parse(self, text: str) -> List[IRCandidate]:
        raw = text.strip().strip(" .")
        s = raw.strip()
        # 영희보다 철수가 더 크다는 말이 맞아?
        m = re.match(r"^(.+?)보다\s+(.+?)(?:가|이|는|은)?\s*(?:더\s*)?(크|높|빠르)(?:다는\s+말이\s+맞아\??|니\??|냐\??|다\??)$", s)
        if m:
            right, left, _ = m.groups()
            ir = ComparisonIR(left=left.strip(), comparator="greater_than", right=right.strip())
            if "?" in raw or "맞아" in s or s.endswith(("니", "냐")):
                return [IRCandidate(QuestionIR(target=ir, requested_mode="proof"), 0.95, self.name)]
            return [IRCandidate(ir, 0.94, self.name)]
        # 철수가 영희보다 더 큰 편이다 / 앞서 있니 / 크지 않다
        m = re.match(r"^(.+?)(?:는|은|가|이)?\s+(.+?)보다\s+(?:더\s*)?(큰\s+편이다|우위에\s+있다|앞서\s+있니\??|앞서\s+있다|크지\s+않다)$", s)
        if m:
            left, right, form = m.groups()
            comp = "less_than" if "지 않" in form else "greater_than"
            ir = ComparisonIR(left=left.strip(), comparator=comp, right=right.strip())
            if "?" in raw or "있니" in form:
                return [IRCandidate(QuestionIR(target=ir, requested_mode="proof"), 0.94, self.name)]
            return [IRCandidate(ir, 0.93, self.name)]
        m = re.match(r"^(.+?)(?:는|은|가|이)?\s+(.+?)에\s+비해\s+(?:더\s*)?(우위에\s+있다|앞서\s+있다|크다|높다|빠르다)$", s)
        if m:
            left, right, _ = m.groups()
            return [IRCandidate(ComparisonIR(left=left.strip(), comparator="greater_than", right=right.strip()), 0.92, self.name)]
        return []


class V25ExceptionAndDiscourseParser:
    name = "v25_exception_discourse_parser"
    def parse(self, text: str) -> List[IRCandidate]:
        raw = text.strip().strip(" .")
        low = _clean(raw)
        # Penguins are birds; however, they usually do not fly.
        m = re.match(r"^(?:although\s+)?(?P<subj>\w+)s?\s+are\s+(?P<kind>\w+)s?[,;]?\s*(?:(?:however|but|although),?\s*)?they\s+(?:usually\s+|normally\s+)?(?:do\s+not|cannot|can't)\s+(?P<action>.+)$", low)
        if m:
            subj, kind, action = _strip_article(m.group("subj")), _strip_article(m.group("kind")), _strip_article(m.group("action"))
            return [IRCandidate(CompositeIR(items=[ClaimIR(subject=subj, relation="is", object=kind), ExceptionIR(rule_id=f"is|{kind}=>can|{action}", exception_subject=subj, exception_text=raw, condition_object=kind, conclusion_relation="can", conclusion_object=action)], source_text=raw), 0.94, self.name)]
        # Can a penguin fly even though it is a bird?
        m = re.match(r"^can\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+(.+?)\s+even\s+though\s+it\s+is\s+(?:a\s+|an\s+|the\s+)?(.+)$", low)
        if m:
            subj, action, cond = m.groups()
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=_strip_article(subj), relation="can", object=_strip_article(action)), requested_mode="proof"), 0.92, self.name, notes=[f"exception_condition={_strip_article(cond)}"])]
        # Complex causality: because/since X makes Y, Z becomes W.
        m = re.match(r"^(?:because|since)\s+(.+?)\s+(?:wets|makes|causes)\s+(?:the\s+)?(.+?),\s*(?:the\s+)?(.+?)\s+(?:may\s+|can\s+)?(?:become|get|turn)\s+(.+)$", low)
        if m:
            cause, mid, subj, eff = m.groups()
            mid_state = _strip_article(mid if "wet" in mid else "wet " + mid)
            effect_state = _strip_article(subj + " " + eff)
            return [IRCandidate(CompositeIR(items=[CausalClaimIR(cause=_strip_article(cause), effect=mid_state), CausalClaimIR(cause=mid_state, effect=effect_state)], source_text=raw), 0.92, self.name)]
        return []


__all__ = [
    "V25InteractiveSemanticFeedbackParser", "V25QuestionGeneralizationParser", "V25TemporalStateParser",
    "V25EventWorldFrameParser", "V25MentalStateParser", "V25KoreanParticleGrammarParser", "V25ExceptionAndDiscourseParser"
]
