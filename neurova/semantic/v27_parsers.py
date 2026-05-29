from __future__ import annotations
import re
from typing import List, Optional
from .grammar_engine import _strip_articles
from ..ir import *


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().strip(" .!?\"'“”").lower())


def _lemma_plural(s: str) -> str:
    s = _clean(s)
    s = re.sub(r"^(a|an|the)\s+", "", s)
    irregular = {"children": "child", "people": "person", "mice": "mouse", "geese": "goose"}
    if s in irregular:
        return irregular[s]
    if s.endswith("ies") and len(s) > 4:
        return s[:-3] + "y"
    if s.endswith("ches") or s.endswith("shes") or s.endswith("xes") or s.endswith("zes"):
        return s[:-2]
    if s.endswith("ses") and len(s) > 4:
        return s[:-2]
    if s.endswith("s") and len(s) > 3 and not s.endswith("ss"):
        return s[:-1]
    return s


def _strip_correction_prefix(s: str) -> str:
    s = s.strip()
    # These are discourse/meta-correction prefixes, not part of the construction surface.
    prefixes = [
        r"no[,，]\s*", r"actually[,，]\s*", r"correction[:：]\s*", r"in\s+this\s+domain[,，]?\s*",
        r"for\s+(?:our|this)\s+(?:task|domain)[,，]?\s*", r"here[,，]?\s*", r"what\s+i\s+meant\s+is[,，]?\s*",
        r"아니[,，]?\s*", r"여기서는\s*", r"이\s+도메인에서는\s*"
    ]
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            ns = re.sub(r"^" + p, "", s, flags=re.I).strip()
            if ns != s:
                s = ns; changed = True
    return s


def _meaning_to_target(meaning: str) -> Optional[str]:
    m = _clean(meaning)
    m = re.sub(r"^(it\s+means|i\s+mean|means|as)\s+", "", m).strip()
    # A is greater/faster/bigger/etc than B.
    mm = re.match(r"^a\s+(?:is\s+)?(?:greater|faster|larger|bigger|higher|ahead|above|dominant|superior)\s+(?:than|over)?\s*b$", m)
    if mm: return "compare(A,greater_than,B)"
    mm = re.match(r"^a\s+(?:is\s+)?(?:less|slower|smaller|lower|behind|below|inferior)\s+(?:than)?\s*b$", m)
    if mm: return "compare(A,less_than,B)"
    mm = re.match(r"^a\s+(greater_than|less_than|equal_to)\s+b$", m)
    if mm: return f"compare(A,{mm.group(1)},B)"
    mm = re.match(r"^a\s+(?:causes|leads\s+to|sparks|brings|results\s+in)\s+b$", m)
    if mm: return "causal(A,B)"
    mm = re.match(r"^a\s+(?:is|counts\s+as|falls\s+under|is\s+regarded\s+as|is\s+classified\s+as)\s+b$", m)
    if mm: return "claim(A,is,B)"
    mm = re.match(r"^a\s+(?:is\s+not|cannot\s+be\s+classified\s+as|is\s+unlikely\s+to\s+be)\s+b$", m)
    if mm: return "not_claim(A,is,B)"
    if re.match(r"^(compare|causal|claim|not_claim|temporal)\(", m):
        return meaning.strip().strip(" .")
    return None


class V27InteractiveCorrectionParser:
    """Natural-language feedback -> construction patch.

    This parser treats meta-correction words (No/Actually/In this domain/Correction)
    as discourse, strips them, then extracts a form-meaning pairing. It outputs a
    ToolCallIR instead of directly answering, so the construction grammar can learn.
    """
    def parse(self, text: str) -> List[IRCandidate]:
        raw0 = text.strip().strip(" .")
        raw = _strip_correction_prefix(raw0)
        patterns = [
            r'^when\s+i\s+say\s+["“](.+?)["”],?\s*(?:it\s+means|i\s+mean)\s+(.+)$',
            r'^(?:by\s+)?["“](.+?)["”]\s*(?:means|should\s+be\s+understood\s+as|is\s+equivalent\s+to|maps\s+to)\s+(.+)$',
            r'^["“](.+?)["”]\s*(?:라는\s+말은|은|는)\s*(.+?)\s*(?:라는\s+뜻|뜻이야|로\s+해석해)$',
        ]
        for pat in patterns:
            m = re.match(pat, raw, re.I)
            if not m: continue
            surface, meaning = m.groups()
            target = _meaning_to_target(meaning)
            if target:
                return [IRCandidate(ToolCallIR(tool_name="learn_construction", args={"text": surface.strip(), "target": target, "source": "v27_nl_feedback"}), 0.98, "v27_interactive_correction")]
        # Relation shorthand with meta prefixes: dominates means greater_than.
        m = re.match(r"^([a-zA-Z_][\w\- ]*?)\s+(?:means|denotes|maps\s+to)\s+(greater_than|less_than|equal_to|causes|is|not_is)$", raw, re.I)
        if m:
            phrase, op = m.groups(); op = op.lower()
            if op in {"greater_than", "less_than", "equal_to"}:
                target = f"compare(A,{op},B)"
            elif op == "causes":
                target = "causal(A,B)"
            elif op == "not_is":
                target = "not_claim(A,is,B)"
            else:
                target = "claim(A,is,B)"
            return [IRCandidate(ToolCallIR(tool_name="learn_construction", args={"text": f"A {phrase.strip()} B", "target": target, "source": "v27_relation_feedback"}), 0.97, "v27_relation_feedback")]
        return []


class V27GeneralLanguageParser:
    """General front-end for patterns the chart/construction layer should see as IR.

    It is not meant as a bag of answers. It maps families of utterances into typed IR:
    question wrappers, modality/negation, event frames, belief questions, Korean markers,
    temporal intervals, and world-state queries.
    """
    def parse(self, text: str) -> List[IRCandidate]:
        raw = text.strip().strip(" .")
        low = _clean(raw)
        out: List[IRCandidate] = []

        # Taxonomy/class membership paraphrase questions.
        m = re.match(r"^(?:can|could)\s+(.+?)\s+be\s+(?:regarded|seen|treated|classified)\s+as\s+(?:a\s+kind\s+of\s+|an\s+|a\s+|the\s+)?(.+?)\??$", low)
        if m:
            subj, obj = m.groups()
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=subj, relation="is", object=_strip_articles(obj)), requested_mode="proof"), 0.93, "v27_taxonomy_paraphrase")]
        m = re.match(r"^does\s+(.+?)\s+fall\s+under\s+(?:a\s+|an\s+|the\s+)?(.+?)\??$", low)
        if m:
            subj, obj = m.groups()
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=subj, relation="is", object=_strip_articles(obj)), requested_mode="proof"), 0.93, "v27_taxonomy_fall_under")]

        # Negation + modality. Keep modality in ClaimIR.modality where uncertainty is not pure negative.
        m = re.match(r"^(.+?)\s+(?:is\s+)?unlikely\s+to\s+be\s+(?:a\s+|an\s+|the\s+)?(.+)$", low)
        if m:
            subj, obj = m.groups()
            return [IRCandidate(NegatedClaimIR(subject=subj, relation="is", object=_strip_articles(obj), modality="unlikely"), 0.9, "v27_unlikely_negation")]
        m = re.match(r"^(.+?)\s+cannot\s+be\s+classified\s+as\s+(?:a\s+|an\s+|the\s+)?(.+)$", low)
        if m:
            subj, obj = m.groups()
            return [IRCandidate(NegatedClaimIR(subject=subj, relation="is", object=_strip_articles(obj), modality="cannot_classify"), 0.91, "v27_cannot_classified")]
        m = re.match(r"^(.+?)\s+is\s+not\s+exactly\s+(?:a\s+|an\s+|the\s+)?(.+)$", low)
        if m:
            subj, obj = m.groups()
            return [IRCandidate(NegatedClaimIR(subject=subj, relation="is", object=_strip_articles(obj), modality="not_exactly"), 0.86, "v27_not_exactly")]

        # Belief questions and statements. Proposition may be negated.
        m = re.match(r"^does\s+(.+?)\s+(?:believe|think)\s+(.+?)\s+is\s+not\s+(?:a\s+|an\s+|the\s+)?(.+?)\??$", low)
        if m:
            holder, subj, obj = m.groups(); prop = NegatedClaimIR(subject=subj, relation="is", object=_strip_articles(obj))
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=holder, relation="believes", object=prop.text()), requested_mode="proof"), 0.92, "v27_belief_neg_question")]
        m = re.match(r"^does\s+(.+?)\s+(?:believe|think)\s+(.+?)\s+is\s+(?:a\s+|an\s+|the\s+)?(.+?)\??$", low)
        if m:
            holder, subj, obj = m.groups(); prop = ClaimIR(subject=subj, relation="is", object=_strip_articles(obj))
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=holder, relation="believes", object=prop.text()), requested_mode="proof"), 0.92, "v27_belief_question")]
        m = re.match(r"^(.+?)\s+(?:believes|thinks)\s+(?:that\s+)?(.+?)\s+is\s+not\s+(?:a\s+|an\s+|the\s+)?(.+)$", low)
        if m:
            holder, subj, obj = m.groups()
            return [IRCandidate(BeliefIR(holder=holder, proposition=NegatedClaimIR(subject=subj, relation="is", object=_strip_articles(obj))), 0.9, "v27_belief_neg")]
        m = re.match(r"^(.+?)\s+(?:believes|thinks)\s+(?:that\s+)?(.+?)\s+is\s+(?:a\s+|an\s+|the\s+)?(.+)$", low)
        if m:
            holder, subj, obj = m.groups()
            return [IRCandidate(BeliefIR(holder=holder, proposition=ClaimIR(subject=subj, relation="is", object=_strip_articles(obj))), 0.9, "v27_belief_pos")]

        # Goal/intent variants.
        m = re.match(r"^(.+?)\s+(?:plans\s+on|plans\s+to|intends\s+to|wants\s+to)\s+(.+)$", low)
        if m:
            agent, goal = m.groups()
            return [IRCandidate(GoalIR(agent=agent, desired_state=goal), 0.88, "v27_goal")]

        # Speech acts.
        m = re.match(r"^(.+?)\s+(?:asked|ordered|told)\s+(.+?)\s+to\s+(.+)$", low)
        if m:
            speaker, listener, action = m.groups()
            content = EventIR(actor=listener, action=action.split()[0], patient=" ".join(action.split()[1:]) or None)
            return [IRCandidate(SpeechActIR(speaker=speaker, act_type="request", content=content), 0.88, "v27_speech_request")]

        # Possession and location queries.
        m = re.match(r"^(?:does|do)\s+(.+?)\s+have\s+(?:a\s+|an\s+|the\s+)?(.+?)\??$", low)
        if m:
            subj, obj = m.groups()
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=subj, relation="has", object=_strip_articles(obj)), requested_mode="proof"), 0.91, "v27_possession_question")]
        m = re.match(r"^where\s+is\s+(?:the\s+)?(.+?)\??$", low)
        if m:
            obj = _strip_articles(m.group(1))
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=obj, relation="located_at", object="?"), requested_mode="answer"), 0.89, "v27_location_question")]

        # Event frames: buy/sell/move/put/take/open/close plus transfer variants.
        # buy: Alice bought a book from Bob yesterday.
        m = re.match(r"^(.+?)\s+(?:bought|purchased)\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+from\s+(.+?)(?:\s+(yesterday|today|tomorrow|in\s+[0-9]{4}))?$", low)
        if m:
            actor, patient, seller, t = m.groups()
            return [IRCandidate(EventIR(actor=actor, action="buy", patient=_strip_articles(patient), recipient=seller, time_expr=(t or "").strip() or None), 0.9, "v27_event_buy")]
        m = re.match(r"^(.+?)\s+sold\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+to\s+(.+?)(?:\s+(yesterday|today|tomorrow|in\s+[0-9]{4}))?$", low)
        if m:
            seller, patient, buyer, t = m.groups()
            return [IRCandidate(EventIR(actor=seller, action="sell", patient=_strip_articles(patient), recipient=buyer, time_expr=(t or "").strip() or None), 0.9, "v27_event_sell")]
        m = re.match(r"^(.+?)\s+moved\s+(?:the\s+|a\s+|an\s+)?(.+?)\s+from\s+(.+?)\s+to\s+(.+)$", low)
        if m:
            actor, patient, src, dst = m.groups()
            return [IRCandidate(EventIR(actor=actor, action="move", patient=_strip_articles(patient), location=_strip_articles(dst)), 0.91, "v27_event_move")]
        m = re.match(r"^(.+?)\s+(?:put|placed)\s+(?:the\s+|a\s+|an\s+)?(.+?)\s+(?:in|on|at)\s+(.+)$", low)
        if m:
            actor, patient, loc = m.groups()
            return [IRCandidate(EventIR(actor=actor, action="put", patient=_strip_articles(patient), location=_strip_articles(loc)), 0.89, "v27_event_put")]
        m = re.match(r"^(.+?)\s+(opened|closed)\s+(?:the\s+|a\s+|an\s+)?(.+)$", low)
        if m:
            actor, action, patient = m.groups(); act = "open" if action == "opened" else "close"
            return [IRCandidate(EventIR(actor=actor, action=act, patient=_strip_articles(patient)), 0.88, "v27_event_open_close")]

        # Temporal robust parsing.
        m = re.match(r"^(?:on|in|during)\s+([0-9]{4})[, ]+(.+?)\s+(?:is|was)\s+(?:a\s+|an\s+|the\s+)?(.+)$", low)
        if m and " not " not in low:
            t, subj, obj = m.groups()
            return [IRCandidate(TemporalClaimIR(subject=subj, relation="is", object=_strip_articles(obj), time_expr=t, valid_from=t, valid_during=t), 0.9, "v27_temporal_year_prefix")]
        m = re.match(r"^(.+?)\s+was\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+between\s+([0-9]{4})\s+and\s+([0-9]{4})$", low)
        if m:
            subj, obj, start, end = m.groups()
            return [IRCandidate(TemporalClaimIR(subject=subj, relation="is", object=_strip_articles(obj), time_expr=f"{start}-{end}", valid_from=start, valid_to=end, valid_during=f"{start}-{end}"), 0.9, "v27_temporal_between")]
        m = re.match(r"^(.+?)\s+was\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+before\s+([0-9]{4})$", low)
        if m:
            subj, obj, end = m.groups()
            return [IRCandidate(TemporalClaimIR(subject=subj, relation="is", object=_strip_articles(obj), time_expr=f"before {end}", valid_to=end), 0.86, "v27_temporal_before")]
        m = re.match(r"^(.+?)\s+is\s+no\s+longer\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+(?:in|during)\s+([0-9]{4})$", low)
        if m:
            subj, obj, t = m.groups()
            return [IRCandidate(TemporalClaimIR(subject=subj, relation="is", object=_strip_articles(obj), polarity="negative", time_expr=t, valid_from=t, valid_during=t), 0.9, "v27_temporal_no_longer")]

        # Exception discourse with normalization.
        m = re.match(r"^(?:although\s+)?(.+?)\s+are\s+(.+?)[,;]?\s*(?:however|but|although)?[,]?\s*they\s+(?:usually\s+|normally\s+)?(?:do\s+not|cannot|can't)\s+(?:usually\s+|normally\s+)?(.+)$", low)
        if m:
            subj_plural, kind, action = m.groups()
            subj = _lemma_plural(subj_plural); cond = _lemma_plural(kind); action = re.sub(r"\b(usually|normally|generally|typically)\b\s*", "", _strip_articles(action)).strip()
            return [IRCandidate(CompositeIR(items=[ClaimIR(subject=subj, relation="is", object=cond), ExceptionIR(rule_id=f"is|{cond}=>can|{action}", exception_subject=subj, exception_text=raw, condition_object=cond, conclusion_relation="can", conclusion_object=action)], source_text=raw), 0.91, "v27_exception_discourse")]
        m = re.match(r"^can\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+(.+?)\s+even\s+though\s+it\s+is\s+(?:a\s+|an\s+|the\s+)?(.+?)\??$", low)
        if m:
            subj, action, kind = m.groups()
            subj = _lemma_plural(subj)
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=subj, relation="can", object=_strip_articles(action)), requested_mode="proof"), 0.91, "v27_exception_question")]

        # English if/and causal chain: If rain falls, the ground gets wet and the road becomes slippery.
        m = re.match(r"^if\s+(.+?),\s*(?:the\s+)?(.+?)\s+gets\s+(.+?)\s+and\s+(?:the\s+)?(.+?)\s+becomes\s+(.+)$", low)
        if m:
            cause, mid_subj, mid_state, end_subj, end_state = m.groups()
            mid = f"{_strip_articles(mid_subj)} {mid_state.strip()}"
            end = f"{_strip_articles(end_subj)} {end_state.strip()}"
            return [IRCandidate(CompositeIR(items=[CausalClaimIR(cause=cause, effect=mid), CausalClaimIR(cause=mid, effect=end)], source_text=raw), 0.9, "v27_if_causal_chain")]

        # Korean comparison variants and uncertainty/negation.
        m = re.match(r"^(.+?)(?:는|은)?\s+(.+?)보다\s+크지\s+않은\s+것\s+같다$", low)
        if m:
            left, right = m.groups(); return [IRCandidate(ComparisonIR(left=left.strip(), comparator="less_than", right=right.strip()), 0.84, "v27_ko_uncertain_negative_comparison")]
        m = re.match(r"^(.+?)(?:가|이)?\s+(.+?)보다\s+크다고\s+볼\s+수\s+있나\??$", low)
        if m:
            left, right = m.groups(); return [IRCandidate(QuestionIR(target=ComparisonIR(left=left.strip(), comparator="greater_than", right=right.strip()), requested_mode="proof"), 0.88, "v27_ko_considered_comparison_question")]
        m = re.match(r"^(.+?)에\s+비해\s+(.+?)(?:가|이)?\s+앞선다$", low)
        if m:
            right, left = m.groups(); return [IRCandidate(ComparisonIR(left=left.strip(), comparator="greater_than", right=right.strip()), 0.88, "v27_ko_relative_ahead")]
        m = re.match(r"^(.+?)보다\s+(.+?)(?:가|이)?\s+뒤처진다$", low)
        if m:
            left, right = m.groups(); return [IRCandidate(ComparisonIR(left=left.strip(), comparator="greater_than", right=right.strip()), 0.88, "v27_ko_lagging_inverse")]
        return out
