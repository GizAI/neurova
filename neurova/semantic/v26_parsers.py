from __future__ import annotations
import re
from typing import List
from ..ir import *


def _clean(x: str) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip().lower().strip(" .?!,'\""))


def _strip(x: str) -> str:
    x = re.sub(r"[?.!]+$", "", str(x or "").strip().lower())
    x = re.sub(r"^(a|an|the)\s+", "", x)
    if x.endswith("s") and len(x) > 3 and not x.endswith("ss"):
        x = x[:-1]
    return x.strip()


class V26DevelopmentalCorrectionParser:
    name = "v26_developmental_correction_parser"
    PREFIX = r"(?:no,\s*|actually,\s*|in\s+this\s+domain,\s*|for\s+our\s+task,\s*|here,\s*|correction:\s*|i\s+meant,\s*)?"
    def parse(self, text: str) -> List[IRCandidate]:
        raw = text.strip().strip(" .")
        # Meta-correction with quoted construction.
        pats = [
            rf"^{self.PREFIX}when\s+i\s+say\s+[\"'](.+?)[\"'],?\s+(?:it\s+means|i\s+mean)\s+(.+)$",
            rf"^{self.PREFIX}by\s+[\"'](.+?)[\"']\s+i\s+mean\s+(.+)$",
            rf"^{self.PREFIX}[\"'](.+?)[\"']\s+(?:means|should\s+be\s+understood\s+as|is\s+equivalent\s+to)\s+(.+)$",
            rf"^{self.PREFIX}interpret\s+[\"'](.+?)[\"']\s+as\s+(.+)$",
        ]
        for p in pats:
            m = re.match(p, raw, re.I)
            if m:
                surface, meaning = m.groups()
                target = self._meaning_to_target(meaning)
                if target:
                    return [IRCandidate(ToolCallIR(tool_name="learn_construction", args={"text": surface.strip(), "target": target, "source": "v26_natural_feedback"}), 0.999, self.name)]
        # Korean informal correction: "X라는 말은 Y라는 뜻".
        m = re.match(r"^(.+?)라는\s+말은\s+(.+?)라는\s+뜻", raw)
        if m:
            surface, meaning = m.groups()
            target = self._meaning_to_target(meaning)
            if target:
                return [IRCandidate(ToolCallIR(tool_name="learn_construction", args={"text": surface.strip(), "target": target, "source": "v26_korean_feedback"}), 0.997, self.name)]
        return []

    def _meaning_to_target(self, meaning: str) -> str | None:
        m = _clean(meaning)
        # formal-style targets
        mm = re.match(r"^a\s+(greater_than|less_than|equal_to)\s+b$", m)
        if mm: return f"compare(A,{mm.group(1)},B)"
        if re.match(r"^a\s+(?:is\s+)?(?:greater|faster|larger|higher)\s+than\s+b$", m): return "compare(A,greater_than,B)"
        if re.match(r"^a\s+(?:is\s+)?(?:less|smaller|lower|slower)\s+than\s+b$", m): return "compare(A,less_than,B)"
        if re.match(r"^a\s+causes\s+b$", m): return "causal(A,B)"
        if re.match(r"^a\s+(?:is\s+)?not\s+b$", m): return "not_claim(A,is,B)"
        if re.match(r"^a\s+is\s+b$", m): return "claim(A,is,B)"
        # Korean semantic glosses.
        if "크" in m or "빠르" in m or "우위" in m: return "compare(A,greater_than,B)"
        if "작" in m or "느리" in m or "열위" in m: return "compare(A,less_than,B)"
        if "원인" in m or "때문" in m: return "causal(A,B)"
        if re.match(r"^(compare|causal|claim|not_claim|temporal)\(", m): return meaning.strip()
        return None


class V26GrammarVariantParser:
    name = "v26_grammar_variant_parser"
    def parse(self, text: str) -> List[IRCandidate]:
        low = _clean(text)
        # Question wrappers around learned comparison forms: would you say/did/does/passive.
        m = re.match(r"^(?:would\s+you\s+say\s+|is\s+it\s+true\s+that\s+|did\s+|does\s+)(.+?)\??$", low)
        if m:
            inner = m.group(1)
            # return a tool call that will be handled after construction parse?  Use common surface variants directly.
            # A narrowly beats B / A outruns B / A dominates B etc are learned via construction grammar;
            # keep as text by wrapping into learnable comparison if common auxiliary question.
            # Compile-time recursion is avoided; cover broad lexical comparison here.
            c = self._comparison_from_clause(inner, question=True)
            if c: return [IRCandidate(QuestionIR(target=c, requested_mode="proof"), 0.88, self.name)]
        m = re.match(r"^(.+?)\s+does\s+not\s+(.+?)\s+(.+)$", low)
        if m:
            left, verb, right = m.groups()
            comp = self._verb_to_comp(verb)
            if comp:
                return [IRCandidate(ComparisonIR(left=_strip(left), comparator="less_than" if comp == "greater_than" else "greater_than", right=_strip(right)), 0.86, self.name)]
        m = re.match(r"^(.+?)\s+is\s+(?:narrowly\s+|slightly\s+|barely\s+)?(?:beaten|outpaced|outrun)\s+by\s+(.+)$", low)
        if m:
            right, left = m.groups()
            return [IRCandidate(ComparisonIR(left=_strip(left), comparator="greater_than", right=_strip(right)), 0.86, self.name)]
        return []

    def _verb_to_comp(self, verb: str) -> str | None:
        v = _clean(verb)
        if v in {"beat", "beats", "outpace", "outpaces", "outrun", "outruns", "dominate", "dominates", "exceed", "exceeds"}:
            return "greater_than"
        if v in {"trail", "trails", "lag", "lags"}:
            return "less_than"
        return None

    def _comparison_from_clause(self, clause: str, question: bool=False) -> ComparisonIR | None:
        c = _clean(clause)
        # strip auxiliaries introduced by did/does.
        m = re.match(r"^(.+?)\s+(?:narrowly\s+|slightly\s+|barely\s+|clearly\s+)?(beats?|outpaces?|outruns?|dominates?|exceeds?|trails?|lags?)\s+(?:behind\s+)?(.+)$", c)
        if m:
            left, verb, right = m.groups()
            comp = self._verb_to_comp(verb)
            if comp:
                return ComparisonIR(left=_strip(left), comparator=comp, right=_strip(right))
        return None


class V26WorldAndElementaryParser:
    name = "v26_world_elementary_parser"
    def parse(self, text: str) -> List[IRCandidate]:
        raw = text.strip().strip(" .")
        low = _clean(raw)
        out: List[IRCandidate] = []
        # Arithmetic word problems.  The parser extracts operation from semantic cues; it does not memorize answers.
        nums = [int(n) for n in re.findall(r"\b\d+\b", low)]
        if len(nums) >= 2 and any(k in low for k in ["how many", "얼마", "몇"]):
            op = None
            if any(k in low for k in ["more", "total", "altogether", "in all", "gets"]): op = "add"
            if any(k in low for k in ["gave", "left", "remaining", "lost", "spent"]): op = "subtract"
            if any(k in low for k in ["each", "boxes", "rows", "groups of", "씩"]): op = "multiply"
            if any(k in low for k in ["shared equally", "each child", "divided", "per"]): op = "divide"
            if op:
                return [IRCandidate(ToolCallIR(tool_name="solve_arithmetic", args={"numbers": nums, "operation": op, "text": raw}), 0.995, self.name)]
        # Buy/sell/move/put/take event frames.
        m = re.match(r"^(.+?)\s+bought\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+from\s+(.+?)(?:\s+(yesterday|today|tomorrow))?$", low)
        if m:
            buyer, obj, seller, t = m.groups()
            return [IRCandidate(EventIR(actor=_strip(buyer), action="buy", patient=_strip(obj), recipient=_strip(buyer), location=None, time_expr=t), 0.9, self.name)]
        m = re.match(r"^(.+?)\s+sold\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+to\s+(.+?)(?:\s+(yesterday|today|tomorrow))?$", low)
        if m:
            seller, obj, buyer, t = m.groups()
            return [IRCandidate(EventIR(actor=_strip(seller), action="sell", patient=_strip(obj), recipient=_strip(buyer), time_expr=t), 0.9, self.name)]
        m = re.match(r"^(.+?)\s+(?:moved|put)\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+from\s+(.+?)\s+to\s+(.+)$", low)
        if m:
            actor, obj, _src, dst = m.groups()
            return [IRCandidate(EventIR(actor=_strip(actor), action="move", patient=_strip(obj), location=_strip(dst)), 0.9, self.name)]
        m = re.match(r"^where\s+(?:is|was)\s+(?:a\s+|an\s+|the\s+)?(.+?)\??$", low)
        if m:
            obj = m.group(1)
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=_strip(obj), relation="located_at", object="?"), requested_mode="answer"), 0.86, self.name)]
        # Possession question common in elementary reading.
        m = re.match(r"^(?:does|do)\s+(.+?)\s+have\s+(?:a\s+|an\s+|the\s+)?(.+?)\??$", low)
        if m:
            subj, obj = m.groups()
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=_strip(subj), relation="has", object=_strip(obj)), requested_mode="proof"), 0.88, self.name)]
        return out


class V26CoreferenceParser:
    """Fallback parser for pronoun-based belief questions after the agent resolves text."""
    name = "v26_coreference_parser"
    def parse(self, text: str) -> List[IRCandidate]:
        low = _clean(text)
        m = re.match(r"^does\s+(.+?)\s+believe\s+(.+?)\s+is\s+not\s+(?:the\s+)?(.+)$", low)
        if m:
            holder, subj, obj = m.groups()
            prop = NegatedClaimIR(subject=_strip(subj), relation="is", object=_strip(obj))
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=_strip(holder), relation="believes", object=prop.text()), requested_mode="proof"), 0.9, self.name)]
        m = re.match(r"^does\s+(.+?)\s+believe\s+(.+?)\s+is\s+(?:the\s+)?(.+)$", low)
        if m:
            holder, subj, obj = m.groups()
            prop = ClaimIR(subject=_strip(subj), relation="is", object=_strip(obj))
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=_strip(holder), relation="believes", object=prop.text()), requested_mode="proof"), 0.9, self.name)]
        return []


__all__ = ["V26DevelopmentalCorrectionParser", "V26GrammarVariantParser", "V26WorldAndElementaryParser", "V26CoreferenceParser"]
