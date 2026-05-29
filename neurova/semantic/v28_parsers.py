from __future__ import annotations
import re
from typing import List
from .v27_parsers import _clean, _strip_correction_prefix, _meaning_to_target, _lemma_plural
from .grammar_engine import _strip_articles
from ..ir import *


def _norm(x: str) -> str:
    return _strip_articles(re.sub(r"\s+", " ", (x or "").strip().strip(" .?!,'\"“”").lower()))


def _lemma_action(a: str) -> str:
    a = _norm(a)
    if a.endswith("ing") and len(a) > 5:
        a = a[:-3]
    if a.endswith("ed") and len(a) > 4:
        a = a[:-2]
    return a


class V28InteractiveFeedbackParser:
    """Richer natural-language correction parser.

    Converts ordinary feedback utterances into construction patches while stripping
    discourse prefixes. This is still deterministic, but it is structured as an
    interactive semantic parsing update rather than a one-off task answer.
    """
    def parse(self, text: str) -> List[IRCandidate]:
        raw0 = text.strip().strip(" .")
        raw = _strip_correction_prefix(raw0)
        # Normalize common conversational wrappers before the actual quoted pattern.
        raw = re.sub(r"^(?:no|actually|correction|right|okay|ok)[,：:]?\s*", "", raw, flags=re.I).strip()
        raw = re.sub(r"^(?:in\s+this\s+domain|for\s+this\s+task|for\s+our\s+task|here)[,：:]?\s*", "", raw, flags=re.I).strip()
        patterns = [
            r'^when\s+i\s+say\s+["“](.+?)["”],?\s*(?:it\s+means|i\s+mean|understand\s+it\s+as)\s+(.+)$',
            r'^(?:by\s+)?["“](.+?)["”]\s*(?:means|should\s+be\s+understood\s+as|is\s+equivalent\s+to|is\s+the\s+same\s+as|maps\s+to|denotes|stands\s+for)\s+(.+)$',
            r'^interpret\s+["“](.+?)["”]\s+as\s+(.+)$',
            r'^treat\s+["“](.+?)["”]\s+as\s+meaning\s+(.+)$',
            r'^["“](.+?)["”]\s*(?:라는\s+말은|은|는)\s*(.+?)\s*(?:라는\s+뜻|뜻이야|로\s+해석해)$',
        ]
        for pat in patterns:
            m = re.match(pat, raw, re.I)
            if not m:
                continue
            surface, meaning = m.groups()
            target = _meaning_to_target(meaning)
            if target:
                return [IRCandidate(ToolCallIR(tool_name="learn_construction", args={"text": surface.strip(), "target": target, "source": "v28_interactive_feedback"}), 0.99, "v28_interactive_feedback")]
        # Free relation shorthand: "dominates means greater_than".
        m = re.match(r"^([a-zA-Z_][\w\- ]*?)\s+(?:means|denotes|maps\s+to|stands\s+for)\s+(greater_than|less_than|equal_to|causes|is|not_is)$", raw, re.I)
        if m:
            phrase, op = m.groups(); op = op.lower(); surface = f"A {phrase.strip()} B"
            if op in {"greater_than", "less_than", "equal_to"}: target = f"compare(A,{op},B)"
            elif op == "causes": target = "causal(A,B)"
            elif op == "not_is": target = "not_claim(A,is,B)"
            else: target = "claim(A,is,B)"
            return [IRCandidate(ToolCallIR(tool_name="learn_construction", args={"text": surface, "target": target, "source": "v28_relation_feedback"}), 0.98, "v28_relation_feedback")]
        return []


class V28GeneralizationParser:
    """General language layer for non-leaky held-out evaluation.

    It handles wrappers and families not tied to a particular benchmark entity:
    taxonomy paraphrase, nested belief questions, modality/negation, temporal
    intervals, world-state questions, Korean variants, and discourse causal chains.
    """
    def parse(self, text: str) -> List[IRCandidate]:
        raw = text.strip().strip(" .")
        low = _clean(raw)
        out: List[IRCandidate] = []

        # Question wrappers around learned constructions: "would you say A verb B".
        for prefix in [r"would\s+you\s+say", r"is\s+it\s+true\s+that", r"is\s+it\s+fair\s+to\s+say\s+that", r"do\s+you\s+think"]:
            m = re.match(rf"^{prefix}\s+(.+?)\??$", low)
            if m:
                inner = m.group(1).strip()
                # Do not decide here; let constructions parse the inner form first by returning a low-priority ToolCall marker is not ideal.
                # Instead cover taxonomy wrappers below and construction questions are handled by grammar variants.
                pass

        # Negated class-membership must run before counts-as question patterns.
        m = re.match(r"^(.+?)\s+hardly\s+counts\s+as\s+(?:a\s+|an\s+|the\s+)?(.+?)\??$", low)
        if m:
            subj, obj = m.groups()
            return [IRCandidate(NegatedClaimIR(subject=_norm(subj), relation="is", object=_norm(obj), modality="hardly_counts"), 0.98, "v28_hardly_counts_negation")]

        # Taxonomy / class membership paraphrase questions.
        m = re.match(r"^(?:would\s+you\s+say\s+)?(.+?)\s+(?:counts\s+as|qualifies\s+as)\s+(?:a\s+|an\s+|the\s+)?(.+?)\??$", low)
        if m:
            subj, obj = m.groups()
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=_norm(subj), relation="is", object=_norm(obj)), requested_mode="proof"), 0.96, "v28_counts_qualifies_question")]
        tax_patterns = [
            r"^(?:can|could|may)\s+(.+?)\s+be\s+(?:regarded|viewed|seen|treated|classified)\s+as\s+(?:a\s+member\s+of\s+|a\s+kind\s+of\s+|a\s+type\s+of\s+|an\s+|a\s+|the\s+)?(.+?)\??$",
            r"^(?:does|do)\s+(.+?)\s+(?:fall\s+under|belong\s+to|fit\s+within)\s+(?:a\s+|an\s+|the\s+)?(.+?)\??$",
            r"^(?:is|are)\s+(.+?)\s+(?:part\s+of|within)\s+(?:a\s+|an\s+|the\s+)?(.+?)\??$",
            r"^(?:would\s+you\s+say|is\s+it\s+fair\s+to\s+call)\s+(.+?)\s+(?:a\s+|an\s+|the\s+)?(.+?)\??$",
        ]
        for pat in tax_patterns:
            m = re.match(pat, low)
            if m:
                subj, obj = m.groups()
                return [IRCandidate(QuestionIR(target=ClaimIR(subject=_norm(subj), relation="is", object=_norm(obj)), requested_mode="proof"), 0.94, "v28_taxonomy_question_family")]

        # Modality/negation with scope.
        neg_patterns = [
            (r"^(.+?)\s+(?:is\s+)?(?:unlikely|not\s+likely)\s+to\s+be\s+(?:a\s+|an\s+|the\s+)?(.+)$", "unlikely"),
            (r"^(.+?)\s+(?:cannot|can't|can\s+not)\s+be\s+(?:classified|regarded|viewed|treated)\s+as\s+(?:a\s+|an\s+|the\s+)?(.+)$", "cannot_classify"),
            (r"^(.+?)\s+is\s+not\s+(?:exactly|really|truly)\s+(?:a\s+|an\s+|the\s+)?(.+)$", "hedged_negative"),
            (r"^(.+?)\s+hardly\s+counts\s+as\s+(?:a\s+|an\s+|the\s+)?(.+)$", "hardly_counts"),
        ]
        for pat, mod in neg_patterns:
            m = re.match(pat, low)
            if m:
                subj, obj = m.groups()
                return [IRCandidate(NegatedClaimIR(subject=_norm(subj), relation="is", object=_norm(obj), modality=mod), 0.91, "v28_modality_negation")]

        # Belief questions/statements with nested proposition and pronoun-resolved text supplied by agent.
        m = re.match(r"^(?:does|do)\s+(.+?)\s+(?:believe|think|suppose)\s+(.+?)\s+is\s+not\s+(?:a\s+|an\s+|the\s+)?(.+?)\??$", low)
        if m:
            h, s, o = m.groups(); prop = NegatedClaimIR(subject=_norm(s), relation="is", object=_norm(o))
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=_norm(h), relation="believes", object=prop.text()), requested_mode="proof"), 0.93, "v28_belief_neg_question")]
        m = re.match(r"^(?:does|do)\s+(.+?)\s+(?:believe|think|suppose)\s+(.+?)\s+is\s+(?:a\s+|an\s+|the\s+)?(.+?)\??$", low)
        if m:
            h, s, o = m.groups(); prop = ClaimIR(subject=_norm(s), relation="is", object=_norm(o))
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=_norm(h), relation="believes", object=prop.text()), requested_mode="proof"), 0.93, "v28_belief_question")]
        m = re.match(r"^(.+?)\s+(?:believes|thinks|supposes)\s+(?:that\s+)?(.+?)\s+is\s+not\s+(?:a\s+|an\s+|the\s+)?(.+)$", low)
        if m:
            h, s, o = m.groups()
            return [IRCandidate(BeliefIR(holder=_norm(h), proposition=NegatedClaimIR(subject=_norm(s), relation="is", object=_norm(o))), 0.91, "v28_belief_neg")]
        m = re.match(r"^(.+?)\s+(?:believes|thinks|supposes)\s+(?:that\s+)?(.+?)\s+is\s+(?:a\s+|an\s+|the\s+)?(.+)$", low)
        if m:
            h, s, o = m.groups()
            return [IRCandidate(BeliefIR(holder=_norm(h), proposition=ClaimIR(subject=_norm(s), relation="is", object=_norm(o))), 0.91, "v28_belief_pos")]

        # Temporal algebra front-end.
        m = re.match(r"^(.+?)\s+(?:became|started\s+being)\s+(?:the\s+|a\s+|an\s+)?(.+?)\s+in\s+([0-9]{4})$", low)
        if m:
            subj, obj, t = m.groups(); return [IRCandidate(TemporalClaimIR(subject=_norm(subj), relation="is", object=_norm(obj), time_expr=t, valid_from=t, valid_during=t), 0.92, "v28_temporal_became")]
        m = re.match(r"^(.+?)\s+stopped\s+being\s+(?:the\s+|a\s+|an\s+)?(.+?)\s+in\s+([0-9]{4})$", low)
        if m:
            subj, obj, t = m.groups(); return [IRCandidate(TemporalClaimIR(subject=_norm(subj), relation="is", object=_norm(obj), polarity="negative", time_expr=t, valid_from=t, valid_during=t, source_id="stop_event"), 0.92, "v28_temporal_stopped")]
        m = re.match(r"^(.+?)\s+was\s+(?:the\s+|a\s+|an\s+)?(.+?)\s+(?:between|from)\s+([0-9]{4})\s+(?:and|to|through)\s+([0-9]{4})$", low)
        if m:
            subj, obj, a, b = m.groups(); return [IRCandidate(TemporalClaimIR(subject=_norm(subj), relation="is", object=_norm(obj), time_expr=f"{a}-{b}", valid_from=a, valid_to=b, valid_during=f"{a}-{b}"), 0.93, "v28_temporal_interval")]
        m = re.match(r"^(?:who|which\s+person)\s+(?:was|served\s+as|held\s+the\s+role\s+of)\s+(?:the\s+)?(.+?)\s+(?:in|during)\s+([0-9]{4})\??$", low)
        if m:
            obj, t = m.groups(); return [IRCandidate(QuestionIR(target=TemporalClaimIR(subject="?", relation="is", object=_norm(obj), time_expr=t, valid_during=t), requested_mode="answer"), 0.93, "v28_temporal_who_question")]
        m = re.match(r"^(?:in|during|on)\s+([0-9]{4})[, ]+(.+?)\s+(?:was|is)\s+not\s+(?:the\s+|a\s+|an\s+)?(.+)$", low)
        if m:
            t, subj, obj = m.groups(); return [IRCandidate(TemporalClaimIR(subject=_norm(subj), relation="is", object=_norm(obj), polarity="negative", time_expr=t, valid_from=t, valid_during=t), 0.93, "v28_temporal_negative")]

        # Event frame families.
        m = re.match(r"^(.+?)\s+(?:bought|purchased)\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+from\s+(.+?)(?:\s+(yesterday|today|tomorrow|in\s+[0-9]{4}))?$", low)
        if m:
            buyer, item, seller, t = m.groups(); return [IRCandidate(EventIR(actor=_norm(buyer), action="buy", patient=_norm(item), recipient=_norm(seller), time_expr=(t or None)), 0.93, "v28_event_buy")]
        m = re.match(r"^(.+?)\s+sold\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+to\s+(.+?)(?:\s+(yesterday|today|tomorrow|in\s+[0-9]{4}))?$", low)
        if m:
            seller, item, buyer, t = m.groups(); return [IRCandidate(EventIR(actor=_norm(seller), action="sell", patient=_norm(item), recipient=_norm(buyer), time_expr=(t or None)), 0.93, "v28_event_sell")]
        m = re.match(r"^(.+?)\s+(?:moved|carried|transported)\s+(?:the\s+|a\s+|an\s+)?(.+?)\s+from\s+(.+?)\s+to\s+(.+)$", low)
        if m:
            actor, obj, src, dst = m.groups(); return [IRCandidate(EventIR(actor=_norm(actor), action="move", patient=_norm(obj), location=_norm(dst)), 0.93, "v28_event_move")]
        m = re.match(r"^(.+?)\s+(?:put|placed|set)\s+(?:the\s+|a\s+|an\s+)?(.+?)\s+(?:in|on|at|inside)\s+(.+)$", low)
        if m:
            actor, obj, loc = m.groups(); return [IRCandidate(EventIR(actor=_norm(actor), action="put", patient=_norm(obj), location=_norm(loc)), 0.92, "v28_event_put")]

        # World-state questions.
        m = re.match(r"^(?:does|do)\s+(.+?)\s+have\s+(?:a\s+|an\s+|the\s+)?(.+?)\??$", low)
        if m:
            subj, obj = m.groups(); return [IRCandidate(QuestionIR(target=ClaimIR(subject=_norm(subj), relation="has", object=_norm(obj)), requested_mode="proof"), 0.94, "v28_has_question")]
        m = re.match(r"^where\s+(?:is|are)\s+(?:the\s+)?(.+?)\??$", low)
        if m:
            obj = _norm(m.group(1)); return [IRCandidate(QuestionIR(target=ClaimIR(subject=obj, relation="located_at", object="?"), requested_mode="answer"), 0.93, "v28_where_question")]

        # Exception + causal discourse.
        m = re.match(r"^(?:although|even\s+though)\s+(.+?)\s+are\s+(.+?)[,;]?\s*they\s+(?:usually\s+|normally\s+)?(?:do\s+not|cannot|can't)\s+(?:usually\s+|normally\s+)?(.+)$", low)
        if m:
            subj, kind, action = m.groups(); subj = _lemma_plural(subj); kind = _lemma_plural(kind); action = re.sub(r"\b(usually|normally|generally|typically)\b\s*", "", _norm(action))
            return [IRCandidate(CompositeIR(items=[ClaimIR(subject=subj, relation="is", object=kind), ExceptionIR(rule_id=f"is|{kind}=>can|{action}", exception_subject=subj, exception_text=raw, condition_object=kind, conclusion_relation="can", conclusion_object=action)], source_text=raw), 0.93, "v28_exception_discourse")]
        m = re.match(r"^(.+?)\s+are\s+(.+?)[,;]?\s*(?:however|but)\s+they\s+(?:usually\s+|normally\s+)?(?:do\s+not|cannot|can't)\s+(?:usually\s+|normally\s+)?(.+)$", low)
        if m:
            subj, kind, action = m.groups(); subj = _lemma_plural(subj); kind = _lemma_plural(kind); action = re.sub(r"\b(usually|normally|generally|typically)\b\s*", "", _norm(action))
            return [IRCandidate(CompositeIR(items=[ClaimIR(subject=subj, relation="is", object=kind), ExceptionIR(rule_id=f"is|{kind}=>can|{action}", exception_subject=subj, exception_text=raw, condition_object=kind, conclusion_relation="can", conclusion_object=action)], source_text=raw), 0.93, "v28_exception_discourse")]
        m = re.match(r"^can\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+(.+?)\s+even\s+though\s+it\s+is\s+(?:a\s+|an\s+|the\s+)?(.+?)\??$", low)
        if m:
            subj, action, _kind = m.groups(); return [IRCandidate(QuestionIR(target=ClaimIR(subject=_lemma_plural(subj), relation="can", object=_norm(action)), requested_mode="proof"), 0.93, "v28_exception_question")]
        m = re.match(r"^if\s+(.+?),\s*(?:the\s+)?(.+?)\s+(?:gets|becomes)\s+(.+?)\s+and\s+(?:the\s+)?(.+?)\s+(?:gets|becomes)\s+(.+)$", low)
        if m:
            cause, mid_subj, mid_state, end_subj, end_state = m.groups(); mid = f"{_norm(mid_subj)} {mid_state.strip()}"; end = f"{_norm(end_subj)} {end_state.strip()}"
            return [IRCandidate(CompositeIR(items=[CausalClaimIR(cause=_norm(cause), effect=mid), CausalClaimIR(cause=mid, effect=end)], source_text=raw), 0.92, "v28_if_and_chain")]

        # Korean broader comparison / negation / questions.
        m = re.match(r"^(.+?)(?:는|은)?\s+(.+?)보다\s+크지\s+않은\s+것\s+같다$", low)
        if m:
            left, right = m.groups(); return [IRCandidate(ComparisonIR(left=left.strip(), comparator="less_than", right=right.strip()), 0.9, "v28_ko_uncertain_neg_comp")]
        m = re.match(r"^(.+?)(?:가|이)?\s+(.+?)보다\s+크다고\s+볼\s+수\s+있나\??$", low)
        if m:
            left, right = m.groups(); return [IRCandidate(QuestionIR(target=ComparisonIR(left=left.strip(), comparator="greater_than", right=right.strip()), requested_mode="proof"), 0.9, "v28_ko_considered_comp_q")]
        m = re.match(r"^(.+?)에\s+비해\s+(.+?)(?:가|이)?\s+(?:앞선다|우위에\s+있다)$", low)
        if m:
            right, left = m.groups(); return [IRCandidate(ComparisonIR(left=left.strip(), comparator="greater_than", right=right.strip()), 0.9, "v28_ko_relative_comp")]
        m = re.match(r"^(.+?)보다\s+(.+?)(?:가|이)?\s+뒤처진다$", low)
        if m:
            left, right = m.groups(); return [IRCandidate(ComparisonIR(left=left.strip(), comparator="greater_than", right=right.strip()), 0.9, "v28_ko_lag_inverse")]
        return out
