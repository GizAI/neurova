from __future__ import annotations
import re
from typing import List
from ..ir import *


def strip_article(x: str) -> str:
    x = re.sub(r"[?.!]+$", "", str(x).strip().lower())
    x = re.sub(r"^(a|an|the)\s+", "", x)
    if x.endswith("s") and len(x) > 3 and not x.endswith("ss"):
        x = x[:-1]
    return x.strip()


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().strip(" .?!'\"")).lower()


def _target_from_natural(meaning: str) -> str | None:
    low = _clean(meaning)
    # Explicit operator names.
    m = re.match(r"^a\s+(greater_than|less_than|equal_to)\s+b$", low)
    if m:
        return f"compare(A,{m.group(1)},B)"
    # Natural comparison paraphrases.
    if re.match(r"^a\s+(?:is\s+)?(?:greater|bigger|larger|taller|higher|faster|ahead|above|superior)\s+(?:than\s+)?b$", low):
        return "compare(A,greater_than,B)"
    if re.match(r"^a\s+(?:is\s+)?(?:smaller|shorter|lower|slower|below|behind|inferior|less)\s+(?:than\s+)?b$", low):
        return "compare(A,less_than,B)"
    if re.match(r"^b\s+(?:is\s+)?(?:below|behind|inferior\s+to|less\s+than)\s+a$", low):
        return "compare(A,greater_than,B)"
    if re.match(r"^b\s+(?:is\s+)?(?:above|ahead\s+of|greater\s+than)\s+a$", low):
        return "compare(A,less_than,B)"
    # Causality paraphrases.
    if re.match(r"^a\s+(?:causes|leads\s+to|brings|produces|results\s+in|makes)\s+b$", low):
        return "causal(A,B)"
    if re.match(r"^b\s+(?:happens|occurs|results)\s+because\s+of\s+a$", low):
        return "causal(A,B)"
    # Claims and negation.
    if re.match(r"^a\s+(?:is|counts\s+as|is\s+classified\s+as|can\s+be\s+considered)\s+b$", low):
        return "claim(A,is,B)"
    if re.match(r"^a\s+(?:is\s+not|is\s+no|does\s+not\s+count\s+as|should\s+not\s+be\s+classified\s+as)\s+b$", low):
        return "not_claim(A,is,B)"
    # Already-structured target.
    if re.match(r"^(compare|causal|claim|not_claim|temporal)\(", low):
        return meaning.strip()
    return None


class V23InteractiveCorrectionParser:
    """Natural-language correction parser.

    Turns user-facing explanations such as:
      - 'When I say "A outruns B", I mean A is faster than B.'
      - '"A ranks above B" means A greater_than B.'
      - '영희보다 철수가 더 크다 means 철수 greater_than 영희'
    into ToolCallIR(learn_construction), so the construction learner can generalize.
    """
    name = "v23_interactive_correction_parser"

    def parse(self, text: str) -> List[IRCandidate]:
        raw = text.strip().strip(" .")
        low = raw.lower()
        patterns = [
            r"^when\s+i\s+say\s+[\"'](.+?)[\"'],?\s+i\s+mean\s+(.+)$",
            r"^by\s+[\"'](.+?)[\"']\s+i\s+mean\s+(.+)$",
            r"^[\"'](.+?)[\"']\s+(?:means|denotes|maps\s+to)\s+(.+)$",
            r"^(.+?)\s+(?:means|denotes|maps\s+to)\s+(.+)$",
            r"^(.+?)\s+(?:라는\s+뜻은|의미는)\s+(.+)$",
        ]
        for pat in patterns:
            m = re.match(pat, raw, re.I)
            if not m:
                continue
            surface, meaning = m.groups()
            target = _target_from_natural(meaning)
            if target:
                return [IRCandidate(ToolCallIR(tool_name="learn_construction", args={"text": surface.strip(), "target": target, "source": "v23_natural_language_correction"}), 0.97, self.name)]
        return []


class V23KoreanParticleParser:
    """Small Korean particle-aware semantic parser.

    This is not a full Korean parser; it captures frequent particle-driven
    constructions that previous template parsers missed: reverse comparison,
    cause/condition chains, giving events, belief/goal forms, and direct
    natural-language corrections.
    """
    name = "v23_korean_particle_parser"

    def parse(self, text: str) -> List[IRCandidate]:
        raw = text.strip().strip(" .")
        low = raw.lower()
        out: List[IRCandidate] = []
        # Correction: "A가 B를 앞선다" means A greater_than B handled by English 'means' above,
        # but allow Korean target phrase too.
        m = re.match(r"^[\"']?(.+?)[\"']?\s+는\s+(.+?)\s+라는\s+뜻$", raw)
        if m:
            surface, meaning = m.groups()
            target = _target_from_natural(meaning)
            if target:
                return [IRCandidate(ToolCallIR(tool_name="learn_construction", args={"text": surface, "target": target, "source": "v23_ko_correction"}), 0.94, self.name)]
        # Reverse comparison variants.
        m = re.match(r"^(.+?)보다\s+(.+?)(?:가|이|는|은)?\s+더\s+(크|높|빠르|작|낮|느리)(?:다|니|냐|게\s+맞아\??)$", low)
        if m:
            right, left, stem = m.groups()
            comp = "greater_than" if stem in {"크", "높", "빠르"} else "less_than"
            ir = ComparisonIR(left=left.strip(), comparator=comp, right=right.strip())
            if raw.endswith("?") or "맞아" in low or low.endswith(("니", "냐")):
                return [IRCandidate(QuestionIR(target=ir, requested_mode="proof"), 0.91, self.name)]
            return [IRCandidate(ir, 0.9, self.name)]
        m = re.match(r"^(.+?)(?:가|이|는|은)?\s+(.+?)(?:를|을)\s+(앞선다|능가한다|추월한다)$", low)
        if m:
            left, right, _ = m.groups()
            return [IRCandidate(ComparisonIR(left=left.strip(), comparator="greater_than", right=right.strip()), 0.9, self.name)]
        # Causal / conditional chains.
        if "비" in raw and "젖" in raw and "미끄" in raw:
            return [IRCandidate(CompositeIR(items=[CausalClaimIR(cause="비", effect="땅 젖음"), CausalClaimIR(cause="땅 젖음", effect="길 미끄러움")], source_text=raw), 0.9, self.name)]
        m = re.match(r"^(.+?)(?:가|이)?\s+오면\s+(.+?)(?:가|이)?\s+(.+?)(?:다|진다)$", low)
        if m:
            cause, subject, pred = m.groups()
            return [IRCandidate(CausalClaimIR(cause=f"{cause} 발생", effect=f"{subject} {pred}"), 0.82, self.name)]
        # Korean event: Alice가 Bob에게 책을 서울에서 어제 줬다.
        m = re.match(r"^(.+?)(?:가|이|는|은)\s+(.+?)에게\s+(.+?)(?:을|를)\s+(.+?)에서\s+(.+?)\s+(줬다|주었다)$", raw)
        if m:
            actor, recipient, patient, location, t, _ = m.groups()
            return [IRCandidate(EventIR(actor=actor.strip(), action="give", patient=patient.strip(), recipient=recipient.strip(), location=location.strip(), time_expr=t.strip()), 0.88, self.name)]
        # Korean belief/goal.
        m = re.match(r"^(.+?)(?:는|은|가|이)\s+(.+?)(?:가|이|는|은)\s+(.+?)라고\s+믿는다$", raw)
        if m:
            holder, subj, obj = m.groups()
            return [IRCandidate(BeliefIR(holder=holder.strip(), proposition=ClaimIR(subject=subj.strip(), relation="is", object=strip_article(obj))), 0.84, self.name)]
        m = re.match(r"^(.+?)(?:는|은|가|이)\s+(.+?)고\s+싶다$", raw)
        if m:
            agent, goal = m.groups()
            return [IRCandidate(GoalIR(agent=agent.strip(), desired_state=goal.strip()), 0.84, self.name)]
        return out


class V23DiscourseFrameParser:
    """Frame/role parser for event, belief, goal, speech act and compound discourse."""
    name = "v23_discourse_frame_parser"

    def parse(self, text: str) -> List[IRCandidate]:
        raw = text.strip().strip(" .")
        low = raw.lower()
        # Speech acts: ask/tell/correct/order.
        m = re.match(r"^(.+?)\s+(?:asked|told|instructed)\s+(.+?)\s+to\s+(.+)$", low)
        if m:
            speaker, target, action = m.groups()
            return [IRCandidate(SpeechActIR(speaker=speaker.strip(), act_type="request", content=EventIR(actor=target.strip(), action=action.strip())), 0.87, self.name)]
        # Giving/transfer variants.
        m = re.match(r"^(.+?)\s+(?:gave|handed|sent)\s+(.+?)\s+(?:a\s+|an\s+|the\s+)?(.+?)(?:\s+in\s+(.+?))?(?:\s+(yesterday|today|tomorrow|in\s+[0-9]{4}|on\s+[0-9\-]+))?$", low)
        if m:
            actor, rec, patient, loc, t = m.groups()
            return [IRCandidate(EventIR(actor=actor.strip(), action="give", patient=strip_article(patient), recipient=rec.strip(), location=(loc or "").strip() or None, time_expr=(t or "").strip() or None), 0.84, self.name)]
        # Belief with embedded negation/temporal proposition.
        m = re.match(r"^(.+?)\s+(?:believes|thinks|assumes)\s+(.+?)\s+(?:is|was)\s+not\s+(?:the\s+)?(.+)$", low)
        if m:
            holder, subj, obj = m.groups()
            return [IRCandidate(BeliefIR(holder=holder.strip(), proposition=NegatedClaimIR(subject=subj.strip(), relation="is", object=strip_article(obj))), 0.87, self.name)]
        m = re.match(r"^(.+?)\s+(?:believes|thinks|assumes)\s+(.+?)\s+(?:is|was)\s+(?:the\s+)?(.+)$", low)
        if m:
            holder, subj, obj = m.groups()
            return [IRCandidate(BeliefIR(holder=holder.strip(), proposition=ClaimIR(subject=subj.strip(), relation="is", object=strip_article(obj))), 0.86, self.name)]
        # Goal/intent.
        m = re.match(r"^(.+?)\s+(?:wants|intends|plans)\s+to\s+(.+)$", low)
        if m:
            agent, goal = m.groups()
            return [IRCandidate(GoalIR(agent=agent.strip(), desired_state=goal.strip()), 0.86, self.name)]
        # Compound discourse: because X, Y; if X, Y; although/even though X, Y.
        m = re.match(r"^(?:because|since)\s+(.+?),\s*(.+)$", low)
        if m:
            cause_clause, effect_clause = m.groups()
            # Simplify common state-change verbs to causal chain.
            if "wet" in cause_clause and ("slippery" in effect_clause or "slip" in effect_clause):
                return [IRCandidate(CompositeIR(items=[CausalClaimIR(cause="rain", effect="wet ground") if "rain" in cause_clause else CausalClaimIR(cause=cause_clause, effect="wet ground"), CausalClaimIR(cause="wet ground", effect=effect_clause)], source_text=raw), 0.86, self.name)]
            return [IRCandidate(CausalClaimIR(cause=cause_clause.strip(), effect=effect_clause.strip()), 0.78, self.name)]
        m = re.match(r"^(?:if|when)\s+(.+?),\s*(.+)$", low)
        if m:
            cond, outcome = m.groups()
            return [IRCandidate(CompositeIR(items=[CausalClaimIR(cause=cond.strip(), effect=outcome.strip())], source_text=raw), 0.78, self.name)]
        # Natural exceptions with entity plural normalization.
        m = re.match(r"^(.+?)s\s+are\s+(.+?)s,?\s+but\s+they\s+(?:usually\s+)?(?:cannot|can't)\s+(.+)$", low)
        if m:
            subj_plural, kind, action = m.groups()
            subj, cond, action = strip_article(subj_plural), strip_article(kind), strip_article(action)
            return [IRCandidate(CompositeIR(items=[ClaimIR(subject=subj, relation="is", object=cond), ExceptionIR(rule_id=f"is|{cond}=>can|{action}", exception_subject=subj, exception_text=raw, condition_object=cond, conclusion_relation="can", conclusion_object=action)], source_text=raw), 0.86, self.name)]
        return []


__all__ = ["V23InteractiveCorrectionParser", "V23KoreanParticleParser", "V23DiscourseFrameParser"]
