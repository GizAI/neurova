from __future__ import annotations
import re
from typing import List
from .cognitive_model import NeuralCognitiveCompiler
from .datasets import generate_nl_ir_examples
from .semantic import LearnedSemanticParser, MeaningAtomTable, MeaningAtomCalculus, ActiveTeacher, SemanticBeam, ConstructionLearner, CognitiveConstructionGrammar
from .semantic.v23_parsers import V23InteractiveCorrectionParser, V23KoreanParticleParser, V23DiscourseFrameParser
from .semantic.v24_parsers import V24InteractiveCorrectionParser, V24TaxonomyQuestionParser, V24KoreanGrammarParser, V24TemporalIntervalParser, V24EventFrameParser, V24ExceptionDiscourseParser
from .semantic.v25_parsers import (V25InteractiveSemanticFeedbackParser, V25QuestionGeneralizationParser, V25TemporalStateParser, V25EventWorldFrameParser, V25MentalStateParser, V25KoreanParticleGrammarParser, V25ExceptionAndDiscourseParser)
from .semantic.v26_parsers import V26DevelopmentalCorrectionParser, V26GrammarVariantParser, V26WorldAndElementaryParser, V26CoreferenceParser
from .semantic.v27_parsers import V27InteractiveCorrectionParser, V27GeneralLanguageParser
from .semantic.v28_parsers import V28InteractiveFeedbackParser, V28GeneralizationParser
from .semantic.v29_parsers import V29GrammarOperationParser
from .semantic.v30_unified import V30UnifiedFrontEnd
from .semantic.neural_perception import NeuralSemanticPerception
from .ir import *


def strip_article(x: str) -> str:
    x = re.sub(r"[?.!]+$", "", x.strip().lower())
    x = re.sub(r"^(a|an|the)\s+", "", x)
    if x.endswith("s") and len(x) > 3 and not x.endswith("ss"):
        x = x[:-1]
    return x.strip()


def _clean_clause(s: str) -> str:
    return s.strip(" ,.;")


class PhraseFragmentParser:
    """Phrase-to-IR fragment parser.

    This is not a full natural-language parser. It is a no-LLM structured parser that
    splits simple discourse into independently validated IR fragments.
    """
    def __init__(self):
        self.regex = None

    def parse(self, text: str) -> List[IRCandidate]:
        raw = text.strip()
        # Avoid splitting rule/queries/programs where conjunctions are part of prose.
        low = raw.lower()
        if any(k in low for k in ["if ", "implement", "research", "write", "essay", "?"]):
            return []
        parts = re.split(r"\s*(?:,?\s+and\s+|,?\s+but\s+|그리고|하지만|,|;|\.\s+)\s*", raw)
        parts = [_clean_clause(p) for p in parts if _clean_clause(p)]
        if len(parts) < 2:
            return []
        regex = RegexParser()
        items = []
        notes = []
        scores = []
        for p in parts:
            sub = regex.parse(p)
            if not sub:
                notes.append(f"unparsed fragment: {p}")
                continue
            ir = sub[0].ir
            if isinstance(ir, (ClaimIR, NegatedClaimIR, TemporalClaimIR, CausalClaimIR, ComparisonIR, RuleIR, QuantifiedRuleIR, ExceptionIR)):
                items.append(ir)
                scores.append(sub[0].confidence)
        if len(items) >= 2:
            return [IRCandidate(CompositeIR(items=items, source_text=raw), min(0.97, sum(scores)/len(scores) + 0.12), "phrase_fragment_parser", ambiguity=0.0, notes=notes)]
        return []


class V22AdaptiveLanguageParser:
    """High-precision no-LLM parser for correction-driven language learning.

    It is intentionally small: it recognizes natural-language teaching/correction
    acts and a few high-value semantic patterns, then delegates generalization to
    ConstructionLearner. This is the bridge from a new expression heard once to a
    reusable IR pattern.
    """
    def parse(self, text: str) -> List[IRCandidate]:
        raw = text.strip().strip(" .")
        low = raw.lower().strip()
        out: List[IRCandidate] = []

        # Correction-to-parser patch: 'dominates means greater_than'.
        m = re.match(r"^([a-zA-Z_][\w\- ]*?)\s+(?:means|maps\s+to|denotes)\s+(greater_than|less_than|equal_to|causes|is|not_is)\s*$", raw, re.I)
        if m:
            phrase, op = m.groups()
            phrase = phrase.strip()
            op = op.lower()
            if op in {"greater_than", "less_than", "equal_to"}:
                surface = f"A {phrase} B"
                target = f"compare(A,{op},B)"
            elif op == "causes":
                surface = f"A {phrase} B"
                target = "causal(A,B)"
            elif op == "not_is":
                surface = f"A {phrase} B"
                target = "not_claim(A,is,B)"
            else:
                surface = f"A {phrase} B"
                target = "claim(A,is,B)"
            return [IRCandidate(ToolCallIR(tool_name="learn_construction", args={"text": surface, "target": target, "source": "natural_language_correction"}), 0.95, "v22_relation_correction")]

        # Correction-to-parser patch: '"A dominates B" means A greater_than B'.
        m = re.match(r"^[\"']?(.+?)[\"']?\s+(?:means|뜻은|의미는)\s+(.+)$", raw, re.I)
        if m:
            surface, meaning = m.groups()
            target = self._meaning_to_target(meaning.strip())
            if target:
                return [IRCandidate(ToolCallIR(tool_name="learn_construction", args={"text": surface.strip(), "target": target, "source": "natural_language_correction"}), 0.95, "v22_surface_correction")]

        # Natural taxonomy question paraphrases.
        m = re.match(r"^(?:would\s+you\s+say\s+)?(.+?)\s+counts\s+as\s+(?:a\s+|an\s+|the\s+)?(.+?)\??$", low)
        if m:
            subj, obj = m.groups()
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=subj.strip(), relation="is", object=strip_article(obj)), requested_mode="proof"), 0.87, "v22_counts_as_question")]
        m = re.match(r"^can\s+(.+?)\s+be\s+considered\s+(?:a\s+|an\s+|the\s+)?(.+?)\??$", low)
        if m:
            subj, obj = m.groups()
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=subj.strip(), relation="is", object=strip_article(obj)), requested_mode="proof"), 0.87, "v22_considered_question")]

        # Modal / classification negation.
        m = re.match(r"^(.+?)\s+should\s+not\s+be\s+classified\s+as\s+(?:a\s+|an\s+|the\s+)?(.+)$", low)
        if m:
            subj, obj = m.groups()
            return [IRCandidate(NegatedClaimIR(subject=subj.strip(), relation="is", object=strip_article(obj)), 0.87, "v22_modal_negated_classification")]

        # Temporal negation and service intervals.
        m = re.match(r"^(?:in|during)\s+([0-9]{4})\s+(.+?)\s+(?:is|was)\s+not\s+(?:the\s+)?(.+)$", low)
        if m:
            t, subj, obj = m.groups()
            return [IRCandidate(TemporalClaimIR(subject=subj.strip(), relation="is", object=strip_article(obj), polarity="negative", time_expr=t, valid_during=t, valid_from=t), 0.88, "v22_temporal_negation_prefix")]
        m = re.match(r"^(.+?)\s+(?:is|was)\s+not\s+(?:the\s+)?(.+?)\s+(?:in|during)\s+([0-9]{4})$", low)
        if m:
            subj, obj, t = m.groups()
            return [IRCandidate(TemporalClaimIR(subject=subj.strip(), relation="is", object=strip_article(obj), polarity="negative", time_expr=t, valid_during=t, valid_from=t), 0.88, "v22_temporal_negation_suffix")]
        m = re.match(r"^from\s+([0-9]{4})\s+to\s+([0-9]{4}),?\s+(.+?)\s+(?:served\s+as|was)\s+(?:the\s+)?(.+)$", low)
        if m:
            start, end, subj, obj = m.groups()
            return [IRCandidate(TemporalClaimIR(subject=subj.strip(), relation="is", object=strip_article(obj), time_expr=f"{start}-{end}", valid_during=f"{start}-{end}", valid_from=start, valid_to=end), 0.88, "v22_temporal_interval")]
        m = re.match(r"^who\s+held\s+(?:the\s+)?(.+?)\s+role\s+during\s+([0-9]{4})\??$", low)
        if m:
            role, t = m.groups()
            return [IRCandidate(QuestionIR(target=TemporalClaimIR(subject="?", relation="is", object=strip_article(role), time_expr=t, valid_during=t), requested_mode="answer"), 0.88, "v22_temporal_role_question")]

        # Compound natural causal expressions.
        m = re.match(r"^because\s+(.+?)\s+wets\s+(?:the\s+)?(.+?),\s*(.+?)\s+may\s+become\s+(.+)$", low)
        if m:
            cause, medium, downstream, effect = m.groups()
            wet = f"wet {strip_article(medium)}"
            return [IRCandidate(CompositeIR(items=[CausalClaimIR(cause=cause.strip(), effect=wet), CausalClaimIR(cause=wet, effect=f"{strip_article(downstream)} {effect.strip()}")], source_text=raw), 0.86, "v22_compound_causal")]
        if "비" in raw and "젖" in raw and "미끄" in raw:
            return [IRCandidate(CompositeIR(items=[CausalClaimIR(cause="비", effect="땅 젖음"), CausalClaimIR(cause="땅 젖음", effect="길 미끄러움")], source_text=raw), 0.86, "v22_ko_compound_causal")]

        # Korean reverse/variant comparison.
        m = re.match(r"^(.+?)보다\s+(.+?)(?:가|이)?\s+더\s+(크다|높다|빠르다|작다|낮다|느리다)$", low)
        if m:
            right, left, word = m.groups()
            comp = "greater_than" if word in {"크다", "높다", "빠르다"} else "less_than"
            return [IRCandidate(ComparisonIR(left=left.strip(), comparator=comp, right=right.strip()), 0.88, "v22_ko_reverse_comparison")]
        m = re.match(r"^(.+?)(?:가|이|는|은)?\s+(.+?)(?:를|을)\s+앞선다$", low)
        if m:
            left, right = m.groups()
            return [IRCandidate(ComparisonIR(left=left.strip(), comparator="greater_than", right=right.strip()), 0.88, "v22_ko_ahead_comparison")]
        m = re.match(r"^(.+?)보다\s+(.+?)(?:가|이)?\s+큰\s+게\s+맞아\??$", low)
        if m:
            right, left = m.groups()
            return [IRCandidate(QuestionIR(target=ComparisonIR(left=left.strip(), comparator="greater_than", right=right.strip()), requested_mode="proof"), 0.88, "v22_ko_reverse_comparison_question")]

        # Natural exception statements.
        m = re.match(r"^(.+?)s\s+are\s+(.+?)s,?\s+but\s+they\s+usually\s+cannot\s+(.+)$", low)
        if m:
            subj_plural, kind, action = m.groups()
            subj = strip_article(subj_plural)
            cond = strip_article(kind)
            action = strip_article(action)
            return [IRCandidate(CompositeIR(items=[ClaimIR(subject=subj, relation="is", object=cond), ExceptionIR(rule_id=f"is|{cond}=>can|{action}", exception_subject=subj, exception_text=raw, condition_object=cond, conclusion_relation="can", conclusion_object=action)], source_text=raw), 0.86, "v22_natural_exception")]
        m = re.match(r"^can\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+(.+?)\s+even\s+though\s+it\s+is\s+(?:a\s+|an\s+|the\s+)?(.+?)\??$", low)
        if m:
            subj, action, _kind = m.groups()
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=strip_article(subj), relation="can", object=strip_article(action)), requested_mode="proof"), 0.87, "v22_exception_question")]

        # Event / belief / goal language.
        m = re.match(r"^(.+?)\s+gave\s+(.+?)\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+in\s+(.+?)\s+(yesterday|today|tomorrow|[0-9]{4}.*?)$", low)
        if m:
            actor, rec, patient, loc, t = m.groups()
            return [IRCandidate(EventIR(actor=actor.strip(), action="give", patient=strip_article(patient), recipient=rec.strip(), location=loc.strip(), time_expr=t.strip()), 0.86, "v22_event_give")]
        m = re.match(r"^(.+?)\s+(?:believes|thinks)\s+(.+?)\s+(is|means|can|has)\s+(?:a\s+|an\s+|the\s+)?(.+)$", low)
        if m:
            holder, subj, rel, obj = m.groups()
            return [IRCandidate(BeliefIR(holder=holder.strip(), proposition=ClaimIR(subject=subj.strip(), relation=rel, object=strip_article(obj))), 0.86, "v22_belief")]
        m = re.match(r"^(.+?)\s+wants\s+to\s+(.+)$", low)
        if m:
            agent, goal = m.groups()
            return [IRCandidate(GoalIR(agent=agent.strip(), desired_state=goal.strip()), 0.86, "v22_goal")]
        return out

    @staticmethod
    def _meaning_to_target(meaning: str) -> str | None:
        m = meaning.strip().strip(" .")
        low = m.lower()
        mm = re.match(r"^a\s+(greater_than|less_than|equal_to)\s+b$", low)
        if mm:
            return f"compare(A,{mm.group(1)},B)"
        mm = re.match(r"^a\s+causes\s+b$", low)
        if mm:
            return "causal(A,B)"
        mm = re.match(r"^a\s+is\s+not\s+b$", low)
        if mm:
            return "not_claim(A,is,B)"
        mm = re.match(r"^a\s+is\s+b$", low)
        if mm:
            return "claim(A,is,B)"
        if re.match(r"^(compare|causal|claim|not_claim|temporal)\(", low):
            return m
        return None


class RegexParser:
    def parse(self, text: str) -> List[IRCandidate]:
        raw = text.strip()
        low = raw.lower().strip()
        body = low.split(":", 1)[1].strip() if low.startswith("teach:") else low
        body = body.strip()

        m = re.match(r"(?:learn|teach)\s+construction:\s*(.+?)\s*=>\s*(.+)$", raw, re.I)
        if m:
            return [IRCandidate(ToolCallIR(tool_name="learn_construction", args={"text": m.group(1).strip(), "target": m.group(2).strip()}), 0.93, "regex_learn_construction")]

        if any(k in low for k in ["implement", "code", "function", "python", "test", "구현", "코드"]):
            return [IRCandidate(self._program(low), 0.86, "regex_program")]
        if any(k in low for k in ["write", "essay", "article", "report", "장문", "보고서", "글"]):
            return [IRCandidate(self._writing(raw), 0.84, "regex_writing")]
        if any(k in low for k in ["research", "paper", "논문", "심층 검색", "분석"]):
            return [IRCandidate(ResearchTaskIR(question=raw, requires_sources=True), 0.8, "regex_research")]

        # Temporal questions.
        m = re.match(r"who\s+(?:is|was)\s+(?:the\s+)?(.+?)\s+(?:in|on|during)\s+(.+?)\??$", body)
        if m:
            role, t = m.groups()
            return [IRCandidate(QuestionIR(target=TemporalClaimIR(subject="?", relation="is", object=strip_article(role), time_expr=t.strip(), valid_during=t.strip()), requested_mode="answer"), 0.86, "regex_temporal_who")]
        m = re.match(r"what\s+(?:is|was)\s+(.+?)\s+(?:in|on|during)\s+(.+?)\??$", body)
        if m:
            subj, t = m.groups()
            return [IRCandidate(QuestionIR(target=TemporalClaimIR(subject=subj.strip(), relation="is", object="?", time_expr=t.strip(), valid_during=t.strip()), requested_mode="answer"), 0.82, "regex_temporal_what")]

        # Causal question / causal fact.
        m = re.match(r"what\s+happens\s+after\s+(.+?)\??$", body)
        if m:
            return [IRCandidate(QuestionIR(target=CausalClaimIR(cause=m.group(1).strip(), effect="?"), requested_mode="answer"), 0.84, "regex_causal_question")]
        m = re.match(r"(.+?)\s+causes\s+(.+?)\??$", body)
        if m and raw.endswith("?"):
            return [IRCandidate(QuestionIR(target=CausalClaimIR(cause=m.group(1).strip(), effect=strip_article(m.group(2))), requested_mode="proof"), 0.84, "regex_causal_proof")]
        if m:
            return [IRCandidate(CausalClaimIR(cause=m.group(1).strip(), effect=strip_article(m.group(2))), 0.82, "regex_causal")]

        # World transition: state {json-ish} action -> next. Keep minimal, no eval.
        m = re.match(r"world:\s*state=(.+?);\s*action=(.+?);\s*next=(.+)$", raw, re.I)
        if m:
            return [IRCandidate(ToolCallIR(tool_name="world_transition", args={"state": m.group(1).strip(), "action": m.group(2).strip(), "next": m.group(3).strip()}), 0.92, "grammar_world_transition")]

        # Temporal fact.
        m = re.match(r"(?:on|at|during)\s+([0-9]{4}(?:-[0-9]{2})?(?:-[0-9]{2})?|[^,]+),?\s+(.+?)\s+(is|means|can|has)\s+(.+)$", body)
        if m:
            t, subj, rel, obj = m.groups()
            return [IRCandidate(TemporalClaimIR(subject=subj.strip(), relation=rel, object=strip_article(obj), time_expr=t.strip(), valid_during=t.strip(), valid_from=t.strip()), 0.82, "regex_temporal")]

        # Korean comparison question/fact.
        m = re.match(r"(.+?)(?:는|은)\s+(.+?)보다\s+(크니|크냐|작니|작냐|높니|낮니|빠르니|느리니)\??$", body)
        if m:
            left, right, comp = m.groups()
            mapping = {"크니":"greater_than", "크냐":"greater_than", "높니":"greater_than", "빠르니":"greater_than", "작니":"less_than", "작냐":"less_than", "낮니":"less_than", "느리니":"less_than"}
            return [IRCandidate(QuestionIR(target=ComparisonIR(left=left.strip(), comparator=mapping[comp], right=right.strip()), requested_mode="proof"), 0.84, "regex_ko_comparison_question")]
        m = re.match(r"(.+?)(?:는|은)\s+(.+?)보다\s+(크다|작다|높다|낮다|빠르다|느리다)$", body)
        if m:
            left, right, comp = m.groups()
            mapping = {"크다":"greater_than", "높다":"greater_than", "빠르다":"greater_than", "작다":"less_than", "낮다":"less_than", "느리다":"less_than"}
            return [IRCandidate(ComparisonIR(left=left.strip(), comparator=mapping[comp], right=right.strip()), 0.86, "regex_ko_comparison")]

        # English comparison question/fact.
        m = re.match(r"is\s+(.+?)\s+(taller|larger|greater|bigger|smaller|less|shorter)\s+than\s+(.+?)\??$", body)
        if m and raw.endswith("?"):
            left, word, right = m.groups()
            comp = "greater_than" if word in {"taller", "larger", "greater", "bigger"} else "less_than"
            return [IRCandidate(QuestionIR(target=ComparisonIR(left=left.strip(), comparator=comp, right=right.strip()), requested_mode="proof"), 0.86, "regex_comparison_question")]
        m = re.match(r"(.+?)\s+is\s+(taller|larger|greater|bigger|smaller|less|shorter)\s+than\s+(.+)$", body)
        if m:
            left, word, right = m.groups()
            comp = "greater_than" if word in {"taller", "larger", "greater", "bigger"} else "less_than"
            return [IRCandidate(ComparisonIR(left=left.strip(), comparator=comp, right=right.strip()), 0.84, "regex_comparison")]

        # Explicit exceptions.
        m = re.match(r"(.+?)\s+is\s+(?:an?\s+)?exception\s+to\s+(.+?)\s+(is|means|can|has)\s+(.+)$", body)
        if m:
            subj, cond, rel, obj = m.groups()
            cond = strip_article(cond)
            sig = f"is|{cond}=>{rel}|{strip_article(obj)}"
            return [IRCandidate(ExceptionIR(rule_id=sig, exception_subject=strip_article(subj), exception_text=body), 0.84, "regex_exception")]

        # Rules.
        m = re.match(r"if\s+(.+?)\s+(is|means|can|has)\s+(.+?)\s+then\s+(.+?)\s+(is|means|can|has)\s+(.+)$", body)
        if m:
            _, cr, co, _, rr, ro = m.groups()
            return [IRCandidate(RuleIR(condition_relation=cr, condition_object=strip_article(co), conclusion_relation=rr, conclusion_object=strip_article(ro)), 0.9, "regex_rule")]
        m = re.match(r"all\s+(.+?)s?\s+(?:are|is)\s+(.+?)s?$", body)
        if m:
            return [IRCandidate(QuantifiedRuleIR(condition_relation="is", condition_object=strip_article(m.group(1)), conclusion_relation="is", conclusion_object=strip_article(m.group(2)), quantifier="all"), 0.78, "regex_quantified_is_rule")]
        m = re.match(r"all\s+(.+?)s?\s+(can|has|means)\s+(.+?)$", body)
        if m:
            cond, rr, ro = m.groups()
            return [IRCandidate(QuantifiedRuleIR(condition_relation="is", condition_object=strip_article(cond), conclusion_relation=rr, conclusion_object=strip_article(ro), quantifier="all"), 0.8, "regex_quantified_relation_rule")]

        # Questions.
        m = re.match(r"does\s+(.+?)\s+belong\s+to\s+(?:the\s+)?(.+?)s?\??$", body)
        if m:
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=m.group(1).strip(), relation="is", object=strip_article(m.group(2))), requested_mode="proof"), 0.84, "regex_belong_question")]
        m = re.match(r"explain\s+why\s+(.+?)\s+is\s+(?:(?:a|an|the)\s+)?(.+?)\??$", body)
        if m:
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=m.group(1).strip(), relation="is", object=strip_article(m.group(2))), requested_mode="explain"), 0.88, "regex_explain_why")]
        m = re.match(r"why\s+is\s+(.+?)\s+(?:(?:a|an|the)\s+)?(.+?)\??$", body)
        if m:
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=m.group(1).strip(), relation="is", object=strip_article(m.group(2))), requested_mode="explain"), 0.88, "regex_why")]
        m = re.match(r"(is|am|are|can|has)\s+(.+?)\s+(?:(?:a|an|the)\s+)?(.+?)\??$", body)
        if m and raw.endswith("?"):
            rel, subj, obj = m.groups()
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=subj.strip(), relation=rel, object=strip_article(obj)), requested_mode="proof"), 0.88, "regex_question")]
        m = re.match(r"what\s+do\s+you\s+know\s+about\s+(.+?)\??$", body)
        if m:
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=m.group(1).strip(), relation="is", object="?"), requested_mode="answer"), 0.78, "regex_what_know")]
        m = re.match(r"what\s+is\s+(.+?)\??$", body)
        if m:
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=m.group(1).strip(), relation="is", object="?"), requested_mode="answer"), 0.78, "regex_what")]

        # Negation: multiple candidates.
        m = re.match(r"(.+?)\s+(is|am|are|means|can|has)\s+not\s+(.+)$", body)
        if m:
            subj, rel, obj = m.groups()
            clean = strip_article(obj)
            return [
                IRCandidate(NegatedClaimIR(subject=subj.strip(), relation=rel.strip(), object=clean), 0.88, "regex_negated_claim"),
                IRCandidate(ClaimIR(subject=subj.strip(), relation=f"{rel.strip()}_not", object=clean), 0.45, "regex_claim_relation_negated", ambiguity=0.35, notes=["noncanonical negation candidate"]),
            ]

        # Wh-question catch-all: before claim pattern so questions don't become claims.
        wh_words = r"(who|what|where|why|when|how|whom|whose)"
        m_wh = re.match(wh_words + r"\s+(.+?)\??$", body, re.I)
        if m_wh and raw.strip().endswith("?"):
            wh_type = m_wh.group(1).lower()
            rest = m_wh.group(2).strip()
            # Extract the target entity: for "who are you" -> "you", "where is X" -> "X"
            target = rest
            copula_m = re.match(r"(?:is|am|are|was|were|do|does|did)\s+(.+)$", rest, re.I)
            if copula_m:
                target = copula_m.group(1).strip()
            # Further clean: for "you" -> subject="you", for "X called Y" -> keep as is
            target = re.sub(r"\s+called\s+", " ", target)
            target = re.sub(r"\s+known\s+as\s+", " ", target)
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=target, relation="is", object="?"), requested_mode="answer"), 0.5, "regex_wh_question", ambiguity=0.5, notes=["wh-question fallback"])]
        # Also catch questions ending with ? that contain a verb: "are you a robot?", "am I human?"
        m_yn = re.match(r"(is|am|are|can|has|do|does|did|will|would|could|should|may|might)\s+(.+?)\s+(?:(?:a|an|the)\s+)?(.+?)\??$", body, re.I)
        if m_yn and raw.strip().endswith("?"):
            rel, subj, obj = m_yn.groups()
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=subj.strip(), relation=rel, object=strip_article(obj)), requested_mode="proof"), 0.7, "regex_yn_question", ambiguity=0.3, notes=["yes/no question"])]

        # Claim.
        m = re.match(r"(.+?)\s+(is|am|are|means|can|has)\s+(.+)$", body)
        if m:
            subj, rel, obj = m.groups()
            return [IRCandidate(ClaimIR(subject=subj.strip(), relation=rel.strip(), object=strip_article(obj)), 0.82, "regex_claim")]
        # Generic SVO fallback: catch any <word> <word> <word>... pattern as a claim with low confidence.
        m_svo = re.match(r"(.+?)\s+([a-zA-Z]+)\s+(.+)$", body, re.I)
        if m_svo:
            subj, verb, obj = m_svo.groups()
            return [IRCandidate(ClaimIR(subject=subj.strip(), relation=verb.strip(), object=strip_article(obj)), 0.35, "regex_generic_svo", ambiguity=0.6, notes=["Broad SVO fallback; low confidence."])]
        return []

    def _program(self, low: str) -> ProgramSpecIR:
        if "factorial" in low:
            return ProgramSpecIR(name="factorial", function_name="factorial", inputs=["n:int"], outputs="int", tests=["factorial(5)==120"])
        if "gcd" in low:
            return ProgramSpecIR(name="gcd", function_name="gcd", inputs=["a:int", "b:int"], outputs="int", tests=["gcd(54,24)==6"])
        if "prime" in low:
            return ProgramSpecIR(name="is_prime", function_name="is_prime", inputs=["n:int"], outputs="bool", tests=["is_prime(17)", "not is_prime(21)"])
        if "fib" in low or "fibonacci" in low:
            return ProgramSpecIR(name="fibonacci", function_name="fib", inputs=["n:int"], outputs="int", tests=["fib(7)==13"])
        return ProgramSpecIR(name="add", function_name="add", tests=["add(2,3)==5"])

    def _writing(self, raw: str) -> WritingTaskIR:
        low = raw.lower()
        m = re.search(r"(\d+)\s*words?", low)
        target = int(m.group(1)) if m else 500
        m2 = re.search(r"(?:about|on|대한|대해)\s+(.+?)(?:\s+\d+\s*words?)?$", low)
        topic = m2.group(1).strip() if m2 else re.sub(r"write|essay|article|report|long|장문|보고서|글", " ", raw, flags=re.I).strip(" :.,")
        return WritingTaskIR(topic=topic or "learned topic", target_words=target)


class GrammarParser:
    def parse(self, text: str) -> List[IRCandidate]:
        s = text.strip()
        out: List[IRCandidate] = []
        m = re.match(r"claim\((.+?),(.+?),(.+?)\)$", s, re.I)
        if m:
            out.append(IRCandidate(ClaimIR(subject=m.group(1).strip(), relation=m.group(2).strip(), object=m.group(3).strip()), 0.95, "grammar_claim"))
        m = re.match(r"not_claim\((.+?),(.+?),(.+?)\)$", s, re.I)
        if m:
            out.append(IRCandidate(NegatedClaimIR(subject=m.group(1).strip(), relation=m.group(2).strip(), object=m.group(3).strip()), 0.95, "grammar_negated_claim"))
        m = re.match(r"compare\((.+?),(greater_than|less_than|equal_to),(.+?)\)$", s, re.I)
        if m:
            out.append(IRCandidate(ComparisonIR(left=m.group(1).strip(), comparator=m.group(2).strip(), right=m.group(3).strip()), 0.95, "grammar_comparison"))
        m = re.match(r"causal\((.+?),(.+?)\)$", s, re.I)
        if m:
            out.append(IRCandidate(CausalClaimIR(cause=m.group(1).strip(), effect=m.group(2).strip()), 0.95, "grammar_causal"))
        m = re.match(r"temporal\((.+?),(.+?),(.+?),(.+?)\)$", s, re.I)
        if m:
            out.append(IRCandidate(TemporalClaimIR(subject=m.group(1).strip(), relation=m.group(2).strip(), object=m.group(3).strip(), time_expr=m.group(4).strip(), valid_from=m.group(4).strip()), 0.95, "grammar_temporal"))
        m = re.match(r"rule\((.+?),(.+?)=>(.+?),(.+?)\)$", s, re.I)
        if m:
            out.append(IRCandidate(RuleIR(condition_relation=m.group(1).strip(), condition_object=m.group(2).strip(), conclusion_relation=m.group(3).strip(), conclusion_object=m.group(4).strip()), 0.95, "grammar_rule"))
        return out


class IRValidator:
    def validate(self, cand: IRCandidate) -> IRCandidate:
        ir = cand.ir
        checks = {
            ClaimIR: ["subject", "relation", "object"],
            NegatedClaimIR: ["subject", "relation", "object"],
            TemporalClaimIR: ["subject", "relation", "object", "time_expr"],
            CausalClaimIR: ["cause", "effect"],
            ComparisonIR: ["left", "comparator", "right"],
            RuleIR: ["condition_relation", "condition_object", "conclusion_relation", "conclusion_object"],
            QuantifiedRuleIR: ["condition_relation", "condition_object", "conclusion_relation", "conclusion_object"],
            ExceptionIR: ["rule_id", "exception_subject"],
            ContradictionIR: ["claim_a", "claim_b"],
            ProofStepIR: ["conclusion"],
            ProgramSpecIR: ["name", "function_name"],
            WritingTaskIR: ["topic"],
            ResearchTaskIR: ["question"],
            ExperimentIR: ["hypothesis", "intervention", "metric"],
            EventIR: ["actor", "action"],
            BeliefIR: ["holder", "proposition"],
            GoalIR: ["agent", "desired_state"],
            SpeechActIR: ["speaker", "act_type"],
            ToolCallIR: ["tool_name"],
            CompositeIR: ["items"],
        }
        for cls, fields in checks.items():
            if isinstance(ir, cls):
                for f in fields:
                    if not getattr(ir, f, None):
                        cand.missing_fields.append(f)
                break
        if isinstance(ir, QuestionIR) and ir.target is None:
            cand.missing_fields.append("target")
        if cand.missing_fields:
            cand.validation_errors.append("missing required fields: " + ",".join(cand.missing_fields))
            cand.confidence *= 0.35
            cand.ambiguity += 0.35
        return cand


class CandidateRanker:
    def rank(self, candidates: List[IRCandidate]) -> List[IRCandidate]:
        return sorted(candidates, key=lambda c: c.total_score, reverse=True)


class HybridSemanticCompiler:
    def __init__(self, cognitive_model: NeuralCognitiveCompiler | None = None, memory=None):
        self.learned_parser = LearnedSemanticParser(generate_nl_ir_examples(500, seed=17))
        self.meaning_atoms = MeaningAtomTable()
        self.meaning_calculus = MeaningAtomCalculus()
        self.construction_learner = ConstructionLearner()  # legacy simple construction learner
        self.construction_grammar = CognitiveConstructionGrammar()
        self.beam = SemanticBeam(width=20)
        self.active_teacher = ActiveTeacher()
        self.neural_perception = NeuralSemanticPerception().fit(generate_nl_ir_examples(1200, seed=25), epochs=2)
        self.v30_frontend = V30UnifiedFrontEnd()
        # V30 order: a unified wrapper/event/dialogue/temporal front-end first; legacy cascade only as fallback.
        # V25 order: interactive feedback and feature constructions first, then broad semantic front-ends, then legacy fallbacks.
        self.parsers = [self.v30_frontend, GrammarParser(), V29GrammarOperationParser(), self.construction_grammar, V28InteractiveFeedbackParser(), V28GeneralizationParser(), V27InteractiveCorrectionParser(), V27GeneralLanguageParser(), V26DevelopmentalCorrectionParser(), V26GrammarVariantParser(), V26WorldAndElementaryParser(), V26CoreferenceParser(), V25InteractiveSemanticFeedbackParser(), V25QuestionGeneralizationParser(), V25TemporalStateParser(), V25EventWorldFrameParser(), V25MentalStateParser(), V25KoreanParticleGrammarParser(), V25ExceptionAndDiscourseParser(), V24InteractiveCorrectionParser(), V24TaxonomyQuestionParser(), V24TemporalIntervalParser(), V24KoreanGrammarParser(), V24EventFrameParser(), V24ExceptionDiscourseParser(), V23InteractiveCorrectionParser(), V23KoreanParticleParser(), V23DiscourseFrameParser(), V22AdaptiveLanguageParser(), self.construction_learner, self.learned_parser, PhraseFragmentParser(), RegexParser()]
        self.validator = IRValidator()
        self.ranker = CandidateRanker()
        self.cognitive_model = cognitive_model or NeuralCognitiveCompiler()
        self.memory = memory

    def learn_construction(self, surface_text: str, target_ir: CognitiveIR) -> bool:
        # If the user teaches only a relation label ("dominates means greater_than"),
        # generalize it into a binary construction instead of storing a zero-slot patch.
        st = surface_text.strip()
        st_low = st.lower()
        if isinstance(target_ir, ComparisonIR) and "{" not in st and not re.search(r"\bA\b|\bB\b", st, re.I):
            has_values = target_ir.left.lower() in st_low and target_ir.right.lower() in st_low
            if not has_values:
                st = f"A {st} B"
        if isinstance(target_ir, CausalClaimIR) and "{" not in st and not re.search(r"\bA\b|\bB\b", st, re.I):
            has_values = target_ir.cause.lower() in st_low and target_ir.effect.lower() in st_low
            if not has_values:
                st = f"A {st} B"
        learned_fcg = self.construction_grammar.learn(st, target_ir)
        learned_legacy = self.construction_learner.learn(st, target_ir)
        learned = learned_fcg or learned_legacy
        if learned:
            self.active_teacher.queue.append(__import__('neurova.semantic.active_teacher', fromlist=['ActiveLearningItem']).ActiveLearningItem(st, 'feature_construction_learned', getattr(learned, 'ir_type', type(target_ir).__name__), 0.0))
            return True
        return False

    def parse_target_ir(self, target: str) -> CognitiveIR | None:
        target_s = target.strip()
        # Direct, no-LLM symbolic target forms used by dialogue/correction learning.
        m = re.match(r"^(.+?)\s+(greater_than|less_than|equal_to)\s+(.+)$", target_s, re.I)
        if m:
            return ComparisonIR(left=m.group(1).strip(), comparator=m.group(2).strip(), right=m.group(3).strip())
        m = re.match(r"^(.+?)\s+causes\s+(.+)$", target_s, re.I)
        if m:
            return CausalClaimIR(cause=m.group(1).strip(), effect=m.group(2).strip())
        parsers = [GrammarParser(), RegexParser()]
        for p in parsers:
            cands = p.parse(target)
            for c in cands:
                if not isinstance(c.ir, (ToolCallIR, ProgramSpecIR, WritingTaskIR, ResearchTaskIR, QuestionIR)):
                    return c.ir
        return None


    def _inner_variants(self, inner: str) -> List[str]:
        inner = re.sub(r"[?.!]+$", "", inner.strip().lower())
        variants = [inner]
        m = re.match(r"^(.+?)\s+([a-z][a-z\-]+)\s+(.+?)$", inner)
        if m:
            subj, verb, obj = m.groups()
            stems = {verb}
            # Generate common finite/past/passive forms without relying on a lexicon.
            if verb.endswith("ed") and len(verb) > 3:
                stems.add(verb[:-2]); stems.add(verb[:-1])
            base = verb
            if base.endswith("es") and len(base) > 4:
                stems.add(base[:-2])
            if base.endswith("s") and len(base) > 3:
                stems.add(base[:-1])
            for b in list(stems):
                if b.endswith("y"):
                    stems.add(b[:-1] + "ies")
                elif b.endswith(("s", "x", "z", "ch", "sh", "o")):
                    stems.add(b + "es")
                else:
                    stems.add(b + "s")
                stems.add(b + "ed")
                if b.endswith("e"):
                    stems.add(b + "d")
            for v in stems:
                variants.append(f"{subj} {v} {obj}")
                variants.append(f"does {subj} {v} {obj}")
                variants.append(f"did {subj} {v} {obj}")
        seen = set(); out=[]
        for v in variants:
            if v not in seen:
                seen.add(v); out.append(v)
        return out

    def _compile_inner(self, inner: str, as_question: bool = True) -> List[IRCandidate]:
        # Typed wrapper operation: parse inner clause, then lift to QuestionIR if requested.
        inner = re.sub(r"[?.!]+$", "", inner.strip())
        if not inner:
            return []
        inner_cands: List[IRCandidate] = []
        inner_variants = self._inner_variants(inner)
        for inner_v in inner_variants:
            for p in [self.v30_frontend, self.construction_grammar, V28GeneralizationParser(), V27GeneralLanguageParser(), V26GrammarVariantParser(), V25QuestionGeneralizationParser(), RegexParser()]:
                for c in p.parse(inner_v):
                    if isinstance(c.ir, ToolCallIR):
                        continue
                    c.notes.append(f"inner_variant={inner_v}")
                    inner_cands.append(c)
        # Drop degenerate partial matches produced by over-permissive legacy regexes,
        # e.g. interpreting "nova outclassed mira" as right="ed mira".
        filtered: List[IRCandidate] = []
        for c in inner_cands:
            ir = c.ir
            bad = False
            check_ir = ir.target if isinstance(ir, QuestionIR) and ir.target is not None else ir
            if isinstance(check_ir, ComparisonIR):
                bad = str(check_ir.left).lower().startswith(("ed ", "d ", "s ")) or str(check_ir.right).lower().startswith(("ed ", "d ", "s "))
            elif isinstance(check_ir, CausalClaimIR):
                bad = str(check_ir.cause).lower().startswith(("ed ", "d ", "s ")) or str(check_ir.effect).lower().startswith(("ed ", "d ", "s "))
            if not bad:
                filtered.append(c)
        inner_cands = filtered
        if not inner_cands:
            return []
        inner_cands = self.ranker.rank([self.validator.validate(c) for c in inner_cands])
        best = inner_cands[0].ir
        if as_question:
            if isinstance(best, QuestionIR):
                return [IRCandidate(best, 0.93, "v29_wrapper_inner_chart", notes=["wrapper_operation_identity"])]
            if isinstance(best, (ClaimIR, NegatedClaimIR, TemporalClaimIR, CausalClaimIR, ComparisonIR)):
                return [IRCandidate(QuestionIR(target=best, requested_mode="proof"), 0.94, "v29_wrapper_inner_chart", notes=["wrapper_operation_question"])]
        if isinstance(best, QuestionIR) and best.target is not None:
            best = best.target
        return [IRCandidate(best, 0.92, "v29_wrapper_inner_chart", notes=["wrapper_operation_assertion"])]

    def _compile_wrapper_operations(self, text: str) -> List[IRCandidate]:
        low = text.strip().lower().strip()
        out: List[IRCandidate] = []
        # Generic learned grammar operations over an inner proposition.
        wrapper_patterns = [
            r"^would\s+you\s+say\s+(.+?)\??$",
            r"^is\s+it\s+true\s+that\s+(.+?)\??$",
            r"^can\s+we\s+say\s+(.+?)\??$",
            r"^do\s+you\s+think\s+(.+?)\??$",
        ]
        for pat in wrapper_patterns:
            m = re.match(pat, low, re.I)
            if m:
                out.extend(self._compile_inner(m.group(1), as_question=True))
        # Do/did question over learned two-slot verb constructions.
        m = re.match(r"^(?:did|does|do)\s+(.+?)\s+([a-z][a-z\-]+)\s+(.+?)\??$", low, re.I)
        if m:
            subj, verb, obj = m.groups()
            before_len = len(out)
            out.extend(self._compile_inner(f"{subj} {verb} {obj}", as_question=True))
            # If the learned construction was stored with a 3rd-person -s anchor
            # (e.g. "A glarns B"), recover it from do/did-support questions.
            if len(out) == before_len and not verb.endswith("s"):
                out.extend(self._compile_inner(f"{subj} {verb}s {obj}", as_question=True))
        # Negative do-support: A does/did not VERB B.
        m = re.match(r"^(.+?)\s+(?:does|did|do)\s+not\s+([a-z][a-z\-]+)\s+(.+?)$", low, re.I)
        if m:
            subj, verb, obj = m.groups()
            for cand in self._compile_inner(f"{subj} {verb} {obj}", as_question=False):
                ir = cand.ir
                if isinstance(ir, ComparisonIR):
                    inv = "less_than" if ir.comparator == "greater_than" else "greater_than" if ir.comparator == "less_than" else ir.comparator
                    out.append(IRCandidate(ComparisonIR(left=ir.left, comparator=inv, right=ir.right), 0.93, "v29_negation_operation", notes=["do_support_negation_over_construction"]))
                elif isinstance(ir, CausalClaimIR):
                    out.append(IRCandidate(NegatedClaimIR(subject=ir.cause, relation="causes", object=ir.effect), 0.90, "v29_negation_operation"))
                elif isinstance(ir, ClaimIR):
                    out.append(IRCandidate(NegatedClaimIR(subject=ir.subject, relation=ir.relation, object=ir.object), 0.90, "v29_negation_operation"))
        # Passive: B was/is VERBed by A -> A VERB B.
        m = re.match(r"^(.+?)\s+(?:was|is)\s+([a-z][a-z\-]+?)(?:ed)?\s+by\s+(.+?)\??$", low, re.I)
        if m:
            obj, verb, subj = m.groups()
            out.extend(self._compile_inner(f"{subj} {verb} {obj}", as_question=False))
        return out

    def compile(self, text: str) -> List[IRCandidate]:
        cands: List[IRCandidate] = []
        cands.extend(self._compile_wrapper_operations(text))
        for p in self.parsers:
            pcands = p.parse(text)
            if isinstance(p, (V30UnifiedFrontEnd, V29GrammarOperationParser, V28InteractiveFeedbackParser, V28GeneralizationParser, V27InteractiveCorrectionParser, V27GeneralLanguageParser, V26DevelopmentalCorrectionParser, V26GrammarVariantParser, V26WorldAndElementaryParser, V26CoreferenceParser, V25InteractiveSemanticFeedbackParser, CognitiveConstructionGrammar, V25QuestionGeneralizationParser, V25TemporalStateParser, V25EventWorldFrameParser, V25MentalStateParser, V25KoreanParticleGrammarParser, V25ExceptionAndDiscourseParser, V24InteractiveCorrectionParser, V24TaxonomyQuestionParser, V24TemporalIntervalParser, V24KoreanGrammarParser, V24EventFrameParser, V24ExceptionDiscourseParser, V23InteractiveCorrectionParser, V23KoreanParticleParser, V23DiscourseFrameParser, V22AdaptiveLanguageParser, ConstructionLearner)):
                for c in pcands:
                    if isinstance(p, V30UnifiedFrontEnd):
                        c.model_score += 6.0
                    elif isinstance(p, V29GrammarOperationParser):
                        c.model_score += 1.55
                    elif isinstance(p, (V28InteractiveFeedbackParser, V27InteractiveCorrectionParser)):
                        c.model_score += 0.88
                    elif isinstance(p, V28GeneralizationParser):
                        c.model_score += 1.35
                    elif isinstance(p, V27GeneralLanguageParser):
                        c.model_score += 1.05
                    elif isinstance(p, V26DevelopmentalCorrectionParser):
                        c.model_score += 0.75
                    elif isinstance(p, CognitiveConstructionGrammar):
                        c.model_score += 0.46
                    else:
                        c.model_score += 0.28
                    c.notes.append("adaptive_construction_priority")
            for c in pcands:
                c.model_score += self.neural_perception.candidate_bias(text, type(c.ir).__name__)
                if isinstance(c.ir, (WrapperConstructionIR, EventFrameIR, TemporalQuerySchemaIR, MetaMemoryQuestionIR, SupportRequestIR)):
                    c.model_score += 1.25
            cands.extend(pcands)
        if not cands:
            self.active_teacher.add_failed_parse(text, "no_candidate", "add supervised NL→IR row or grammar fragment")
            # Generic fallback: instead of ResearchTaskIR, try to extract meaning from any text.
            low = text.strip().lower()
            if low.endswith("?"):
                cands = [IRCandidate(QuestionIR(target=ClaimIR(subject=text.strip().rstrip("?"), relation="is", object="?"), requested_mode="answer"), 0.25, "fallback_question", ambiguity=0.75, notes=["Question not recognized; queued for active learning."])]
            elif re.match(r".+\s+.+\s+.+", low):  # 3+ words = likely a statement
                parts = low.split(None, 2)
                cands = [IRCandidate(ClaimIR(subject=parts[0], relation=parts[1], object=parts[2]), 0.3, "fallback_statement", ambiguity=0.7, notes=["No reliable parse; stored as generic statement."])]
            else:
                cands = [IRCandidate(SpeechActIR(speaker="user", act_type="modal_nonassertive", content=text.strip()), 0.3, "fallback_smalltalk", ambiguity=0.6, notes=["Not recognized; treated as non-asserted context."])]
        cands = [self.validator.validate(c) for c in cands]
        cands = self.cognitive_model.rank_ir_candidates(text, cands, memory=self.memory)
        return self.beam.prune(self.ranker.rank(cands))
