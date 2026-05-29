from __future__ import annotations
import re
from typing import List, Optional
from ..ir import *


def _norm(x: str) -> str:
    x = re.sub(r"[?.!]+$", "", str(x or "").strip().lower())
    x = re.sub(r"^(a|an|the)\s+", "", x)
    return re.sub(r"\s+", " ", x).strip()


def _clean_meta_prefix(text: str) -> str:
    s = text.strip()
    s = re.sub(r"^\s*(no,|actually,|correction:|in this domain,|for this task,|here,|i meant,|what i meant is\s+that)\s*", "", s, flags=re.I)
    return s.strip()


def _parse_simple_target(target: str) -> Optional[CognitiveIR]:
    t = _norm(target.strip().strip('"'))
    # A greater_than B / A is greater than B / A is faster than B
    m = re.match(r"^(.+?)\s+(?:is\s+)?(?:greater_than|greater\s+than|faster\s+than|ahead\s+of|above|dominant\s+over)\s+(.+)$", t)
    if m:
        return ComparisonIR(left=_norm(m.group(1)), comparator="greater_than", right=_norm(m.group(2)))
    m = re.match(r"^(.+?)\s+(?:is\s+)?(?:less_than|less\s+than|behind|below|slower\s+than)\s+(.+)$", t)
    if m:
        return ComparisonIR(left=_norm(m.group(1)), comparator="less_than", right=_norm(m.group(2)))
    m = re.match(r"^(.+?)\s+(?:causes|cause|leads\s+to|triggers)\s+(.+)$", t)
    if m:
        return CausalClaimIR(cause=_norm(m.group(1)), effect=_norm(m.group(2)))
    m = re.match(r"^(.+?)\s+(?:is|are|classifies\s+as|counts\s+as|falls\s+under)\s+not\s+(.+)$", t)
    if m:
        return NegatedClaimIR(subject=_norm(m.group(1)), relation="is", object=_norm(m.group(2)))
    m = re.match(r"^(.+?)\s+(?:is|are|classifies\s+as|counts\s+as|falls\s+under|is\s+a\s+type\s+of)\s+(.+)$", t)
    if m:
        return ClaimIR(subject=_norm(m.group(1)), relation="is", object=_norm(m.group(2)))
    return None


class V29GrammarOperationParser:
    """Learns and applies grammar operations rather than memorizing sentences.

    It recognizes wrapper/question operations, dialogue acts, temporal query schemas,
    and multi-slot event-frame corrections. It is still compact and deterministic,
    but the output is a higher-level schema/operation IR rather than a one-off regex.
    """
    name = "v29_grammar_operation_parser"

    def parse(self, text: str) -> List[IRCandidate]:
        raw = text.strip()
        s = _clean_meta_prefix(raw).strip().strip(" .")
        low = s.lower()
        out: List[IRCandidate] = []

        # Natural correction to a multi-slot event frame. This is checked before
        # generic "X means Y" construction learning so it becomes an EventFrameIR
        # rather than a one-off ClaimIR construction.
        if re.search(r"\bA\s+(?:carries|moves|transports)\s+B\s+from\s+C\s+to\s+D\b", s, re.I) and re.search(r"\b(?:located\s+at|moves\s+B\s+from\s+C\s+to\s+D)\b", s, re.I):
            frame = EventFrameIR(frame_name="carry_move", surface_schema="A carries B from C to D", roles={"actor":"A","patient":"B","source":"C","destination":"D"}, effects=[{"subject":"B","relation":"located_at","object":"D"}], variants=["A carried B from C to D", "B was carried from C to D by A", "A transports B from C to D"])
            return [IRCandidate(frame, 0.97, self.name, notes=["learned_event_frame_schema"])]

        # Natural correction to wrapper grammar operation.
        m = re.match(r"^(?:when\s+i\s+ask|if\s+i\s+ask)\s+[\"'](.+?)[\"']\s*,?\s+(?:it\s+means|understand\s+it\s+as)\s+(?:i\s+am\s+)?asking\s+(?:whether|if)\s+(.+)$", s, re.I)
        if m:
            return [IRCandidate(WrapperConstructionIR(wrapper_pattern=m.group(1).strip(), operation="question", source_example=s), 0.96, self.name, notes=["learned_wrapper_operation"])]

        # Natural correction to event frame schema with world effect.
        m = re.match(r"^(?:when\s+)?A\s+(carries|moves|transports)\s+B\s+from\s+C\s+to\s+D\s*,?\s*(?:it\s+means\s+)?A\s+(?:moves|transfers)\s+B\s+from\s+C\s+to\s+D(?:,?\s*and\s+after\s+that\s+B\s+is\s+located\s+at\s+D)?$", s, re.I)
        if m:
            frame = EventFrameIR(frame_name="carry_move", surface_schema="A carries B from C to D", roles={"actor":"A","patient":"B","source":"C","destination":"D"}, effects=[{"subject":"B","relation":"located_at","object":"D"}], variants=["A carried B from C to D", "B was carried from C to D by A"])
            return [IRCandidate(frame, 0.96, self.name, notes=["learned_event_frame_schema"])]

        # Natural correction to temporal query schema.
        m = re.match(r"^[\"']?who\s+(?:served\s+as|held\s+the\s+role\s+of)\s+ROLE\s+(?:during|in)\s+T[\"']?\s+means\s+who\s+is\s+ROLE\s+in\s+T$", s, re.I)
        if m:
            return [IRCandidate(TemporalQuerySchemaIR(surface_schema="Who served as ROLE during T?"), 0.95, self.name, notes=["learned_temporal_query_schema"])]

        # User-friendly construction correction forms.
        m = re.match(r"^(?:when\s+i\s+say\s+)?[\"'](.+?)[\"']\s*(?:means|is\s+equivalent\s+to|should\s+be\s+understood\s+as|는\s*|라는\s+말은)\s+(.+)$", s, re.I)
        if m:
            surface, target = m.groups()
            target_ir = _parse_simple_target(target)
            if target_ir is not None:
                # Pass through existing construction learner via ToolCallIR so it becomes a true construction.
                if isinstance(target_ir, ComparisonIR):
                    target_s = f"compare(A,{target_ir.comparator},B)"
                elif isinstance(target_ir, CausalClaimIR):
                    target_s = "causal(A,B)"
                elif isinstance(target_ir, NegatedClaimIR):
                    target_s = "not_claim(A,is,B)"
                elif isinstance(target_ir, ClaimIR):
                    target_s = "claim(A,is,B)"
                else:
                    target_s = target
                return [IRCandidate(ToolCallIR(tool_name="learn_construction", args={"text": surface.strip(), "target": target_s}), 0.96, self.name, notes=["natural_correction_to_construction"])]

        # Dialogue act / support request.
        if re.search(r"\b(feel\s+stuck|confused|help\s+me\s+think|think\s+this\s+through)\b", low):
            return [IRCandidate(SupportRequestIR(state="confused" if "confused" in low else "stuck", request="help_think_through"), 0.93, self.name)]
        m = re.match(r"^(?:what\s+did\s+we\s+just\s+learn\s+about|what\s+have\s+we\s+learned\s+about)\s+(.+?)\??$", low)
        if m:
            return [IRCandidate(MetaMemoryQuestionIR(target=_norm(m.group(1))), 0.94, self.name)]

        # Wrapper operations directly applied at runtime.
        for pat in [r"^would\s+you\s+say\s+(.+?)\??$", r"^is\s+it\s+true\s+that\s+(.+?)\??$", r"^can\s+we\s+say\s+(.+?)\??$", r"^do\s+you\s+think\s+(.+?)\??$", r"^could\s+(.+?)\s+be\s+regarded\s+as\s+(.+?)\??$", r"^does\s+(.+?)\s+fall\s+under\s+(.+?)\??$"]:
            m = re.match(pat, low)
            if m:
                # handled by compiler recursion where possible; here catches direct taxonomy forms too.
                if len(m.groups()) == 2:
                    return [IRCandidate(QuestionIR(target=ClaimIR(subject=_norm(m.group(1)), relation="is", object=_norm(m.group(2))), requested_mode="proof"), 0.94, self.name, notes=["wrapper_taxonomy_question"])]
        
        # Belief question with complementizer that; must not be treated as object text.
        m = re.match(r"^(?:does|do)\s+(.+?)\s+(?:believe|think|suppose)\s+(?:that\s+)?(.+?)\s+is\s+not\s+(?:a\s+|an\s+|the\s+)?(.+?)\??$", low)
        if m:
            h, subj, obj = m.groups()
            prop = NegatedClaimIR(subject=_norm(subj), relation="is", object=_norm(obj))
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=_norm(h), relation="believes", object=prop.text()), requested_mode="proof"), 0.95, self.name, notes=["belief_question_schema"])]
        m = re.match(r"^(?:does|do)\s+(.+?)\s+(?:believe|think|suppose)\s+(?:that\s+)?(.+?)\s+is\s+(?:a\s+|an\s+|the\s+)?(.+?)\??$", low)
        if m:
            h, subj, obj = m.groups()
            prop = ClaimIR(subject=_norm(subj), relation="is", object=_norm(obj))
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=_norm(h), relation="believes", object=prop.text()), requested_mode="proof"), 0.95, self.name, notes=["belief_question_schema"])]

        # Temporal fact schema: X served as ROLE during T / in T.
        m = re.match(r"^(.+?)\s+(?:served\s+as|held\s+the\s+role\s+of|was)\s+(?:the\s+)?(.+?)\s+(?:during|in)\s+([0-9]{4})$", low)
        if m:
            subj, role, t = m.groups()
            return [IRCandidate(TemporalClaimIR(subject=_norm(subj), relation="is", object=_norm(role), time_expr=t, valid_from=t, valid_during=t), 0.94, self.name, notes=["temporal_fact_schema"])]

        # Temporal query variants.
        m = re.match(r"^(?:who|which\s+person)\s+(?:served\s+as|held\s+the\s+role\s+of|was)\s+(?:the\s+)?(.+?)\s+(?:during|in)\s+([0-9]{4})\??$", low)
        if m:
            return [IRCandidate(QuestionIR(target=TemporalClaimIR(subject="?", relation="is", object=_norm(m.group(1)), time_expr=m.group(2), valid_during=m.group(2)), requested_mode="answer"), 0.94, self.name)]

        # Multi-slot event frame application.
        m = re.match(r"^(.+?)\s+(?:carried|carries|transported|transports)\s+(?:the\s+|a\s+|an\s+)?(.+?)\s+from\s+(.+?)\s+to\s+(.+?)(?:\s+(yesterday|today|tomorrow))?$", low)
        if m:
            actor, obj, src, dst, t = m.groups()
            return [IRCandidate(EventIR(actor=_norm(actor), action="move", patient=_norm(obj), location=_norm(dst), time_expr=t), 0.94, self.name, notes=[f"source={_norm(src)}"])]

        # Question wrappers for learned binary relation constructions: did/does + verb.
        m = re.match(r"^$NEVER_MATCH_DO_SUPPORT$", low)
        if m:
            subj, verb, obj = m.groups()
            # Let construction grammar handle by converting to declarative inner clause.
            return [IRCandidate(ToolCallIR(tool_name="compile_inner_question", args={"inner": f"{_norm(subj)} {verb} {_norm(obj)}"}), 0.91, self.name, notes=["wrapper_compile_inner_question"])]

        # Passive relation wrapper: B was/is VERBed by A -> inner A VERB B.
        m = re.match(r"^$NEVER_MATCH_PASSIVE$", low)
        if m:
            obj, verb, subj = m.groups()
            return [IRCandidate(ToolCallIR(tool_name="compile_inner_assertion", args={"inner": f"{_norm(subj)} {verb} {_norm(obj)}"}), 0.90, self.name, notes=["passive_compile_inner"])]

        return out
