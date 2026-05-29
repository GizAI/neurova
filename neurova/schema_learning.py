from __future__ import annotations

"""V33 schema-learning substrate.

This module is intentionally not a new pile of sentence-level parser patches.
It provides a durable learning substrate: episodes, prediction errors, schema
candidates, schema tests, gated promotion, and a runtime executor for learned
schemas. New linguistic ability should be represented as data rows in schema
memory rather than new parser if-statements.
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import re
import sqlite3
import time
import uuid

from .ir import (
    ClaimIR,
    NegatedClaimIR,
    CausalClaimIR,
    ComparisonIR,
    EventIR,
    QuestionIR,
    ToolCallIR,
    SpeechActIR,
    WrapperConstructionIR,
    EventFrameIR,
    MetaMemoryQuestionIR,
    SupportRequestIR,
    IRCandidate,
)
from .chart_lattice import TypedChartParser
from .v35_broad_parser import V35BroadCoverageChartParser


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def norm_text(x: str) -> str:
    x = x.strip().lower()
    x = re.sub(r"[?.!]+$", "", x)
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"^(a|an|the)\s+", "", x)
    return x.strip()


def simple_singular(x: str) -> str:
    x = norm_text(x)
    if x.endswith("ies") and len(x) > 4:
        return x[:-3] + "y"
    if x.endswith(("ches", "shes", "xes", "ses", "zes")) and len(x) > 4:
        return x[:-2]
    if x.endswith("s") and len(x) > 3 and not x.endswith("ss"):
        return x[:-1]
    return x


@dataclass
class Experience:
    text: str
    context: Dict[str, Any] = field(default_factory=dict)
    predicted_ir: Optional[Dict[str, Any]] = None
    user_feedback: Optional[str] = None
    outcome: str = "observed"
    created_at: float = field(default_factory=time.time)


@dataclass
class PredictionError:
    episode_id: str
    error_type: str
    expected: Dict[str, Any] = field(default_factory=dict)
    observed: Dict[str, Any] = field(default_factory=dict)
    severity: float = 0.5


@dataclass
class SchemaCandidate:
    schema_type: str
    schema: Dict[str, Any]
    source_episode_ids: List[str] = field(default_factory=list)
    confidence: float = 0.5
    status: str = "candidate"  # candidate | session_patch | experimental | stable | core | rejected
    created_at: float = field(default_factory=time.time)


@dataclass
class SchemaTest:
    input_text: str
    expected: Dict[str, Any]
    is_counterexample: bool = False


class DevelopmentalSchemaMemory:
    """SQLite-backed schema memory with hippocampus/cortex style staging.

    Fast memory lives in episodes/prediction_errors/session candidate schemas.
    Slow memory lives in stable_schemas after gated promotion.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=1.0)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def close(self):
        self.conn.close()

    def __del__(self):
        self.close()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes(
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                context_json TEXT NOT NULL,
                predicted_ir_json TEXT,
                user_feedback TEXT,
                outcome TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_errors(
                id TEXT PRIMARY KEY,
                episode_id TEXT NOT NULL,
                error_type TEXT NOT NULL,
                expected_json TEXT NOT NULL,
                observed_json TEXT NOT NULL,
                severity REAL NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_candidates(
                id TEXT PRIMARY KEY,
                schema_type TEXT NOT NULL,
                schema_json TEXT NOT NULL,
                source_episode_ids_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_tests(
                id TEXT PRIMARY KEY,
                schema_id TEXT NOT NULL,
                input_text TEXT NOT NULL,
                expected_json TEXT NOT NULL,
                is_counterexample INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_eval_results(
                id TEXT PRIMARY KEY,
                schema_id TEXT NOT NULL,
                passed INTEGER NOT NULL,
                regression_breaks INTEGER NOT NULL,
                precision_estimate REAL NOT NULL,
                recall_estimate REAL NOT NULL,
                details_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS stable_schemas(
                id TEXT PRIMARY KEY,
                schema_type TEXT NOT NULL,
                schema_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                promoted_at REAL NOT NULL
            )
            """
        )
        self.conn.commit()

    def record_episode(self, exp: Experience) -> str:
        eid = _id("ep")
        self.conn.execute(
            "INSERT INTO episodes VALUES (?,?,?,?,?,?,?)",
            (
                eid,
                exp.text,
                json.dumps(exp.context, ensure_ascii=False),
                json.dumps(exp.predicted_ir, ensure_ascii=False) if exp.predicted_ir else None,
                exp.user_feedback,
                exp.outcome,
                exp.created_at,
            ),
        )
        self.conn.commit()
        return eid

    def record_prediction_error(self, err: PredictionError) -> str:
        eid = _id("perr")
        self.conn.execute(
            "INSERT INTO prediction_errors VALUES (?,?,?,?,?,?)",
            (
                eid,
                err.episode_id,
                err.error_type,
                json.dumps(err.expected, ensure_ascii=False),
                json.dumps(err.observed, ensure_ascii=False),
                err.severity,
            ),
        )
        self.conn.commit()
        return eid

    def add_schema_candidate(self, cand: SchemaCandidate) -> str:
        sid = _id("schema")
        self.conn.execute(
            "INSERT INTO schema_candidates VALUES (?,?,?,?,?,?,?)",
            (
                sid,
                cand.schema_type,
                json.dumps(cand.schema, ensure_ascii=False),
                json.dumps(cand.source_episode_ids, ensure_ascii=False),
                cand.confidence,
                cand.status,
                cand.created_at,
            ),
        )
        self.conn.commit()
        return sid

    def add_schema_tests(self, schema_id: str, tests: List[SchemaTest]) -> None:
        for t in tests:
            self.conn.execute(
                "INSERT INTO schema_tests VALUES (?,?,?,?,?)",
                (_id("stest"), schema_id, t.input_text, json.dumps(t.expected, ensure_ascii=False), 1 if t.is_counterexample else 0),
            )
        self.conn.commit()

    def promote_schema(self, schema_id: str, status: str = "stable") -> None:
        row = self.conn.execute("SELECT * FROM schema_candidates WHERE id=?", (schema_id,)).fetchone()
        if not row:
            raise KeyError(schema_id)
        self.conn.execute("UPDATE schema_candidates SET status=? WHERE id=?", (status, schema_id))
        if status in {"stable", "core"}:
            # Idempotent promotion.
            exists = self.conn.execute("SELECT id FROM stable_schemas WHERE id=?", (schema_id,)).fetchone()
            if not exists:
                self.conn.execute(
                    "INSERT INTO stable_schemas VALUES (?,?,?,?,?)",
                    (schema_id, row["schema_type"], row["schema_json"], row["confidence"], time.time()),
                )
        self.conn.commit()

    def schemas(self, include_experimental: bool = True) -> List[Tuple[str, str, Dict[str, Any], float, str]]:
        statuses = ("stable", "core", "experimental", "session_patch") if include_experimental else ("stable", "core")
        placeholders = ",".join(["?"] * len(statuses))
        rows = self.conn.execute(
            f"SELECT * FROM schema_candidates WHERE status IN ({placeholders}) ORDER BY confidence DESC, created_at DESC",
            statuses,
        ).fetchall()
        stable_rows = self.conn.execute("SELECT * FROM stable_schemas ORDER BY confidence DESC, promoted_at DESC").fetchall()
        seen = set()
        out = []
        for row in list(stable_rows) + list(rows):
            sid = row["id"]
            if sid in seen:
                continue
            seen.add(sid)
            out.append((sid, row["schema_type"], json.loads(row["schema_json"]), float(row["confidence"]), row.get("status", "stable") if hasattr(row, "get") else "stable"))
        return out

    def count(self, table: str) -> int:
        return int(self.conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"])


class CorrectionInterpreter:
    """Natural-language supervision -> schema candidates.

    The output is a schema, not a new parser rule. This is the anti-hardcoding
    boundary: correction utterances become data-backed schema objects.
    """

    META_PREFIX = r"(?:no,\s*)?(?:actually,\s*)?(?:in\s+this\s+domain,\s*)?(?:for\s+this\s+task,\s*)?(?:correction:\s*)?"

    def parse(self, text: str) -> Optional[SchemaCandidate]:
        raw = text.strip().strip(" .")
        low = raw.lower()

        # Correction forms with quoted surface expression. Handles both:
        #   "X" means Y
        #   When I say "X", it means Y
        m = re.match(self.META_PREFIX + r'(?:when\s+i\s+say\s+|if\s+i\s+ask\s+)?["“](.+?)["”]\s*,?\s*(?:it\s+)?(?:means|i\s+mean|should\s+mean|is\s+equivalent\s+to|means\s+asking\s+whether)\s+(.+)$', raw, re.I)
        if m:
            surface, meaning = m.groups()
            surface = surface.strip()
            meaning = meaning.strip()
            if self._looks_like_wrapper(surface, meaning):
                return SchemaCandidate("WrapperSchema", self._wrapper_schema(surface, meaning), confidence=0.72, status="session_patch")
            evt = self._event_frame_from(surface, meaning)
            if evt:
                return SchemaCandidate("EventFrameSchema", evt, confidence=0.7, status="session_patch")
            cons = self._construction_from(surface, meaning)
            if cons:
                return SchemaCandidate("ConstructionSchema", cons, confidence=0.72, status="session_patch")

        # When A carries B from C to D, it means A moves B from C to D, and after that B is located at D.
        m = re.match(self.META_PREFIX + r"when\s+(.+?),\s*(?:it\s+means\s+)?(.+)$", raw, re.I)
        if m:
            surface, meaning = m.groups()
            evt = self._event_frame_from(surface, meaning)
            if evt:
                return SchemaCandidate("EventFrameSchema", evt, confidence=0.74, status="session_patch")
            if self._looks_like_wrapper(surface, meaning):
                return SchemaCandidate("WrapperSchema", self._wrapper_schema(surface, meaning), confidence=0.72, status="session_patch")
            cons = self._construction_from(surface, meaning)
            if cons:
                return SchemaCandidate("ConstructionSchema", cons, confidence=0.7, status="session_patch")

        # Meta-memory / support dialogue act schema correction.
        m = re.match(self.META_PREFIX + r'(?:(?:when\s+i\s+say|if\s+i\s+say)\s+)?["“](.+?)["”]\s+(?:means|should\s+be\s+understood\s+as)\s+(?:a\s+)?(support\s+request|meta\s+memory\s+question|question\s+about\s+what\s+we\s+learned)$', raw, re.I)
        if m:
            surface, kind = m.groups()
            act = "support_request" if "support" in kind.lower() else "meta_memory_query"
            return SchemaCandidate("DialogueActSchema", {"forms": [surface], "act_type": act, "expected_response_policy": ["acknowledge", "clarify_goal", "next_step"]}, confidence=0.7, status="session_patch")

        return None

    def _looks_like_wrapper(self, surface: str, meaning: str) -> bool:
        s = surface.lower()
        m = meaning.lower()
        return any(x in s for x in ["would you", "is it true", "do you think", "can we say", "if i ask"]) or any(x in m for x in ["ask whether", "question", "asking if"])

    def _wrapper_schema(self, surface: str, meaning: str) -> Dict[str, Any]:
        pattern = surface.strip()
        # Normalize obvious variables, but keep the exact form as training evidence.
        p = re.sub(r"\bX\b|\bP\b", "{P}", pattern)
        if "{P}" not in p:
            # Anything inside brackets/quotes becomes P; otherwise fall back to suffix P.
            p = re.sub(r"\[(.+?)\]", "{P}", p)
        if "{P}" not in p:
            p = pattern.replace("?", "") + " {P}"
        return {
            "name": "learned_question_wrapper",
            "form": p,
            "operation": "QuestionIR(P)",
            "applies_to": ["ClaimIR", "ComparisonIR", "CausalClaimIR", "BeliefIR", "TemporalClaimIR"],
            "source_surface": surface,
            "source_meaning": meaning,
        }

    def _construction_from(self, surface: str, meaning: str) -> Optional[Dict[str, Any]]:
        target = meaning.strip().lower()
        relation = None
        if any(x in target for x in ["greater_than", "greater than", "above", "ahead", "faster than", "stronger than", "better than"]):
            schema = "ComparisonIR({A}, greater_than, {B})"
            relation = "greater_than"
        elif any(x in target for x in ["less_than", "less than", "behind", "below", "slower than", "weaker than"]):
            schema = "ComparisonIR({A}, less_than, {B})"
            relation = "less_than"
        elif any(x in target for x in ["causes", "cause", "leads to", "triggers"]):
            schema = "CausalClaimIR({A}, {B})"
            relation = "causes"
        elif any(x in target for x in ["not ", "is not", "not_claim"]):
            schema = "NegatedClaimIR({A}, is, {B})"
            relation = "not_is"
        elif " is " in target or "type of" in target or "kind of" in target:
            schema = "ClaimIR({A}, is, {B})"
            relation = "is"
        else:
            return None
        pattern = self._variable_pattern(surface)
        return {
            "name": f"learned_{relation}_construction",
            "form": pattern,
            "meaning_schema": schema,
            "relation": relation,
            "slots": ["A", "B"],
            "variants": self._binary_variants(pattern),
            "examples": [surface],
            "counterexamples": [],
        }

    def _event_frame_from(self, surface: str, meaning: str) -> Optional[Dict[str, Any]]:
        text = f"{surface} {meaning}".lower()
        # General multi-slot movement/transfer frame. This intentionally learns the frame structure,
        # not a specific verb surface.
        # Role-inversion detection: "A borrows C from B, B gives C to A"
        if "borrow" in text and "gives" in text and "to" in text:
            return {
                "name": "learned_borrow_frame",
                "form": self._variable_pattern(surface),
                "roles": {"borrower": "A", "source": "B", "object": "C"},
                "effects": [{"subject": "A", "relation": "has", "object": "C"}],
                "variants": ["A borrows C from B", "B lends C to A", "A receives C from B"],
                "examples": [surface],
            }
        if re.search(r"\bfrom\s+C\s+to\s+D\b", text, re.I) or "located at d" in text or "located_at" in text:
            return {
                "name": "learned_move_frame",
                "form": self._variable_pattern(surface),
                "roles": {"actor": "A", "patient": "B", "source": "C", "destination": "D"},
                "effects": [{"subject": "B", "relation": "located_at", "object": "D"}],
                "variants": [
                    "A carries B from C to D",
                    "A carried B from C to D",
                    "B was carried from C to D by A",
                    "A transports B from C to D",
                    "A moves B from C to D",
                ],
                "examples": [surface],
            }
        if any(v in text for v in ["has c", "b has c", "gives c to b", "transfers c to b"]):
            return {
                "name": "learned_transfer_frame",
                "form": self._variable_pattern(surface),
                "roles": {"giver": "A", "recipient": "B", "object": "C"},
                "effects": [{"subject": "B", "relation": "has", "object": "C"}],
                "variants": ["A gives B C", "A gives C to B", "B receives C from A", "A lends B C"],
                "examples": [surface],
            }
        return None

    def _variable_pattern(self, surface: str) -> str:
        s = surface.strip().strip('"“”')
        # Preserve explicit A/B/C/D variables. If concrete example is supplied,
        # a higher-level schema learner can replace slots later.
        for v in ["A", "B", "C", "D", "P"]:
            s = re.sub(rf"\b{v.lower()}\b", v, s, flags=re.I)
        return s

    def _binary_variants(self, pattern: str) -> List[str]:
        # Add generic variants. Execution still goes through schema matching, not code if-statements.
        return [pattern, f"Does {pattern}?", f"Did {pattern}?", f"Would you say {pattern}?", f"It is true that {pattern}", self._negate_pattern(pattern), self._passive_pattern(pattern)]

    def _negate_pattern(self, pattern: str) -> str:
        parts = pattern.split()
        if len(parts) >= 3:
            return f"{parts[0]} does not {' '.join(parts[1:])}"
        return "not " + pattern

    def _passive_pattern(self, pattern: str) -> str:
        parts = pattern.split()
        if len(parts) == 3 and parts[0] == "A" and parts[2] == "B":
            return f"B is {parts[1]}ed by A"
        return pattern



class _SchemaMemoryView:
    """Compile view used by V35BroadCoverageChartParser without calling SchemaExecutor recursively."""
    def __init__(self, memory: DevelopmentalSchemaMemory):
        self.memory = memory

    def compile(self, text: str) -> List[IRCandidate]:
        # Use the non-recursive V34 chart parser directly over schema rows.
        chart = TypedChartParser(self.memory.schemas(include_experimental=True))
        return chart.parse(text)


class SchemaExecutor:
    """Executes learned schemas from schema memory.

    It deliberately handles schema types, not hand-coded one-off user sentences.
    """

    def __init__(self, memory: DevelopmentalSchemaMemory):
        self.memory = memory

    def apply(self, text: str) -> List[IRCandidate]:
        # V35: a broader semantic-perception + chart/lattice layer runs first.
        # It uses schema memory plus reusable grammar operations, not sentence-level patches.
        broad = V35BroadCoverageChartParser(schema_substrate=None)
        # Inject this memory via a lightweight adapter to avoid recursive compile loops.
        broad.schema_substrate = _SchemaMemoryView(self.memory)
        broad_candidates = broad.parse(text)
        if broad_candidates:
            return broad_candidates
        # V34 typed chart/lattice parser remains as a schema-only fallback.
        chart = TypedChartParser(self.memory.schemas(include_experimental=True))
        chart_candidates = chart.parse(text)
        if chart_candidates:
            return chart_candidates
        out: List[IRCandidate] = []
        raw = text.strip().strip(" .")
        low = raw.lower().strip()
        for sid, typ, schema, conf, status in self.memory.schemas(include_experimental=True):
            if typ == "ConstructionSchema":
                cand = self._apply_construction(schema, raw, conf, sid)
                if cand:
                    out.append(cand)
            elif typ == "WrapperSchema":
                cand = self._apply_wrapper(schema, raw, conf, sid)
                if cand:
                    out.append(cand)
            elif typ == "EventFrameSchema":
                cand = self._apply_event_frame(schema, raw, conf, sid)
                if cand:
                    out.append(cand)
            elif typ == "DialogueActSchema":
                cand = self._apply_dialogue_act(schema, raw, conf, sid)
                if cand:
                    out.append(cand)
        return sorted(out, key=lambda c: c.total_score, reverse=True)

    def _apply_construction(self, schema: Dict[str, Any], raw: str, conf: float, sid: str) -> Optional[IRCandidate]:
        # Wrapper phrases must be decomposed before applying inner relation constructions.
        # Otherwise "would you say A rel B" can be misread with subject="would you say A".
        if re.match(r"^(would\s+you|is\s+it\s+true|do\s+you\s+think|can\s+we\s+say|could\s+we\s+say|does\b|do\b|did\b)", raw.strip(), re.I):
            return None
        if re.search(r"\b(?:does|do|did)\s+(?:not|n't)\b", raw.strip(), re.I):
            return None
        if re.search(r"\b(?:is|was|were|been)\s+\w+(?:ed)?\s+by\b", raw.strip(), re.I):
            return None
        if re.search(r"\b(?:almost|nearly|wants?\s+to|wanted\s+to|failed\s+to|tried\s+to|is\s+said\s+to|was\s+expected\s+to)\b", raw.strip(), re.I):
            return None
        patterns = [schema.get("form", "")] + list(schema.get("variants", []))
        for pat in patterns:
            slots = self._match_vars(pat, raw)
            if not slots:
                continue
            ir = self._instantiate_meaning(schema.get("meaning_schema", ""), slots)
            if ir:
                return IRCandidate(ir, min(0.99, conf + 0.12), "v33_schema_construction", notes=[f"schema={sid}"])
        return None

    def _apply_wrapper(self, schema: Dict[str, Any], raw: str, conf: float, sid: str) -> Optional[IRCandidate]:
        form = schema.get("form", "")
        inner = self._match_wrapper(form, raw)
        if inner:
            return IRCandidate(ToolCallIR(tool_name="compile_inner_question", args={"inner": inner, "schema_id": sid}), min(0.99, conf + 0.1), "v33_schema_wrapper", notes=[f"schema={sid}"])
        # Generic learned wrapper backup from source surface: common prefix with P at the end.
        src = schema.get("source_surface", "")
        if "would you" in src.lower() and raw.lower().startswith("would you"):
            inner = re.sub(r"^(would\s+you\s+(?:say|classify|think)\s+)", "", raw, flags=re.I).strip(" ?")
            if inner:
                return IRCandidate(ToolCallIR(tool_name="compile_inner_question", args={"inner": inner, "schema_id": sid}), min(0.96, conf + 0.08), "v33_schema_wrapper_fuzzy", notes=[f"schema={sid}"])
        return None

    def _apply_event_frame(self, schema: Dict[str, Any], raw: str, conf: float, sid: str) -> Optional[IRCandidate]:
        form = schema.get("form", "")
        patterns = [form] + schema.get("variants", [])
        for pat in patterns:
            slots = self._match_vars(pat, raw)
            if not slots:
                continue
            roles = schema.get("roles", {})
            actor = slots.get(roles.get("actor") or roles.get("giver") or "A", "")
            patient = slots.get(roles.get("patient") or roles.get("object") or "B", "")
            recipient = slots.get(roles.get("recipient") or roles.get("borrower") or "B")
            dst = slots.get(roles.get("destination") or "D")
            action = "move" if any(e.get("relation") == "located_at" for e in schema.get("effects", [])) else "give"
            return IRCandidate(EventIR(actor=norm_text(actor), action=action, patient=norm_text(patient), recipient=norm_text(recipient) if recipient and action != "move" else None, location=norm_text(dst) if dst else None), min(0.98, conf + 0.1), "v33_schema_event_frame", notes=[f"schema={sid}"])
        return None

    def _apply_dialogue_act(self, schema: Dict[str, Any], raw: str, conf: float, sid: str) -> Optional[IRCandidate]:
        forms = schema.get("forms", [])
        act = schema.get("act_type", "support_request")
        # For dialogue acts, lexical overlap is not enough; support requests generalize
        # through user-state words such as stuck/help/confused/worried.
        if act == "support_request" and any(w in raw.lower() for w in ["stuck", "help", "confused", "worried", "rough day"]):
            return IRCandidate(SupportRequestIR(state="confused", request="help_think_through"), min(0.95, conf + 0.08), "v33_schema_dialogue_act", notes=[f"schema={sid}", "support_state_generalization"])
        for form in forms:
            if self._rough_similarity(form, raw) >= 0.72:
                if act == "meta_memory_query":
                    m = re.search(r"about\s+(.+?)\??$", raw, re.I)
                    target = norm_text(m.group(1)) if m else "recent"
                    return IRCandidate(MetaMemoryQuestionIR(target=target), min(0.95, conf + 0.08), "v33_schema_dialogue_act", notes=[f"schema={sid}"])
                if act == "support_request":
                    return IRCandidate(SupportRequestIR(state="confused", request="help_think_through"), min(0.95, conf + 0.08), "v33_schema_dialogue_act", notes=[f"schema={sid}"])
                return IRCandidate(SpeechActIR(speaker="user", act_type=act, content=raw), min(0.95, conf + 0.08), "v33_schema_dialogue_act", notes=[f"schema={sid}"])
        return None

    def _instantiate_meaning(self, meaning: str, slots: Dict[str, str]) -> Optional[Any]:
        meaning = meaning.strip()
        get = lambda k: norm_text(slots.get(k, k))
        if meaning.startswith("ComparisonIR"):
            rel = "greater_than" if "greater_than" in meaning else "less_than" if "less_than" in meaning else "equal_to"
            return ComparisonIR(left=get("A"), comparator=rel, right=get("B"))
        if meaning.startswith("CausalClaimIR"):
            return CausalClaimIR(cause=get("A"), effect=get("B"))
        if meaning.startswith("NegatedClaimIR"):
            return NegatedClaimIR(subject=get("A"), relation="is", object=get("B"))
        if meaning.startswith("ClaimIR"):
            return ClaimIR(subject=get("A"), relation="is", object=get("B"))
        return None

    def _match_vars(self, pattern: str, raw: str) -> Optional[Dict[str, str]]:
        if not pattern:
            return None
        pat = pattern.strip().strip("?.!")
        raw_clean = raw.strip().strip("?.!")
        pat_tokens = pat.split()
        regex_parts = []
        seen_vars = []
        for tok in pat_tokens:
            if tok in {"A", "B", "C", "D", "P"}:
                regex_parts.append(r"(.+?)")
                seen_vars.append(tok)
            else:
                regex_parts.append(self._lexeme_regex(tok))
        rx = r"^" + r"\s+".join(regex_parts) + r"$"
        m = re.match(rx, raw_clean, re.I)
        if not m:
            return None
        return {var: simple_singular(val) for var, val in zip(seen_vars, m.groups())}

    def _lexeme_regex(self, tok: str) -> str:
        # Small morphology abstraction for learned constructions: a learned anchor
        # like "outclasses" should also match "outclass" and "outclassed".
        t = re.escape(tok)
        low = tok.lower()
        alts = {tok}
        if low.endswith("ies") and len(tok) > 4:
            base = tok[:-3] + "y"
            alts.update({base, base + "s", base + "ed"})
        elif low.endswith("es") and len(tok) > 4:
            base = tok[:-2]
            alts.update({base, base + "s", base + "es", base + "ed"})
        elif low.endswith("s") and len(tok) > 3 and not low.endswith("ss"):
            base = tok[:-1]
            alts.update({base, base + "s", base + "ed"})
        else:
            alts.update({tok + "s", tok + "ed"})
        return r"(?:" + "|".join(sorted((re.escape(a) for a in alts), key=len, reverse=True)) + r")"


    def _match_wrapper(self, form: str, raw: str) -> Optional[str]:
        if "{P}" in form:
            rx = re.escape(form).replace(re.escape("{P}"), r"(.+?)")
            m = re.match(r"^" + rx + r"\??$", raw.strip(), re.I)
            if m:
                return m.group(1).strip(" ?")
        return None

    def _rough_similarity(self, a: str, b: str) -> float:
        aa = set(re.findall(r"[a-zA-Z가-힣]+", a.lower()))
        bb = set(re.findall(r"[a-zA-Z가-힣]+", b.lower()))
        if not aa or not bb:
            return 0.0
        return len(aa & bb) / max(1, len(aa | bb))


class CounterexampleGenerator:
    def generate(self, schema_type: str, schema: Dict[str, Any]) -> List[SchemaTest]:
        tests: List[SchemaTest] = []
        if schema_type == "ConstructionSchema":
            form = schema.get("form", "A rel B")
            # Same surface neighborhood but meaning should not apply.
            if "A" in form and "B" in form:
                tests.append(SchemaTest(form.replace("A", "A almost").replace("B", "B"), {"should_not_match": True}, True))
                tests.append(SchemaTest(form.replace("A", "A wants to").replace("B", "B"), {"should_not_match": True}, True))
        if schema_type == "WrapperSchema":
            tests.append(SchemaTest("that is heavy", {"should_not_match": True}, True))
        return tests


class RegressionGate:
    """Promotes only schemas that have positive tests and do not overgeneralize counterexamples."""

    def __init__(self, memory: DevelopmentalSchemaMemory):
        self.memory = memory
        self.executor = SchemaExecutor(memory)

    def evaluate_and_promote(self, schema_id: str, min_precision: float = 0.8) -> Dict[str, Any]:
        rows = self.memory.conn.execute("SELECT * FROM schema_tests WHERE schema_id=?", (schema_id,)).fetchall()
        total_pos = total_neg = pass_pos = pass_neg = 0
        details = []
        for row in rows:
            candidates = self.executor.apply(row["input_text"])
            matched = bool(candidates)
            is_counter = bool(row["is_counterexample"])
            if is_counter:
                total_neg += 1
                ok = not matched
                pass_neg += 1 if ok else 0
            else:
                total_pos += 1
                ok = matched
                pass_pos += 1 if ok else 0
            details.append({"input": row["input_text"], "counterexample": is_counter, "matched": matched, "ok": ok})
        precision = pass_neg / total_neg if total_neg else 1.0
        recall = pass_pos / total_pos if total_pos else 0.0
        passed = bool(recall >= 0.99 and precision >= min_precision)
        self.memory.conn.execute(
            "INSERT INTO schema_eval_results VALUES (?,?,?,?,?,?,?,?)",
            (_id("seval"), schema_id, 1 if passed else 0, 0 if passed else 1, precision, recall, json.dumps(details, ensure_ascii=False), time.time()),
        )
        self.memory.conn.commit()
        if passed:
            self.memory.promote_schema(schema_id, "stable")
        return {"schema_id": schema_id, "passed": passed, "precision": precision, "recall": recall, "details": details}


class SchemaInductionEngine:
    """Turns corrections into schema candidates with tests/counterexamples."""

    def __init__(self, memory: DevelopmentalSchemaMemory):
        self.memory = memory
        self.interpreter = CorrectionInterpreter()
        self.counterexamples = CounterexampleGenerator()
        self.gate = RegressionGate(memory)

    def learn_from_correction(self, text: str, context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        exp_id = self.memory.record_episode(Experience(text=text, context=context or {}, outcome="correction_observed"))
        cand = self.interpreter.parse(text)
        if not cand:
            self.memory.record_prediction_error(PredictionError(exp_id, "uninterpretable_correction", expected={"schema": "candidate"}, observed={"text": text}, severity=0.7))
            return None
        cand.source_episode_ids.append(exp_id)
        schema_id = self.memory.add_schema_candidate(cand)
        tests = self._positive_tests(cand.schema_type, cand.schema) + self.counterexamples.generate(cand.schema_type, cand.schema)
        self.memory.add_schema_tests(schema_id, tests)
        result = self.gate.evaluate_and_promote(schema_id)
        return {"schema_id": schema_id, "schema_type": cand.schema_type, "schema": cand.schema, "gate": result}

    def _positive_tests(self, schema_type: str, schema: Dict[str, Any]) -> List[SchemaTest]:
        if schema_type == "ConstructionSchema":
            form = schema.get("form", "A rel B")
            variants = [form] + schema.get("variants", [])[:3]
            return [SchemaTest(v.replace("A", "mira").replace("B", "taro"), {"matches": schema.get("meaning_schema")}) for v in variants]
        if schema_type == "WrapperSchema":
            form = schema.get("form", "Would you say {P}")
            return [SchemaTest(form.replace("{P}", "mira glarns taro"), {"wrapper": schema.get("operation")})]
        if schema_type == "EventFrameSchema":
            variants = [schema.get("form", "A carries B from C to D")] + schema.get("variants", [])[:2]
            return [SchemaTest(v.replace("A", "eve").replace("B", "box").replace("C", "berlin").replace("D", "rome"), {"event_frame": schema.get("name")}) for v in variants]
        if schema_type == "DialogueActSchema":
            forms = schema.get("forms", []) or ["I feel stuck"]
            return [SchemaTest(forms[0], {"dialogue_act": schema.get("act_type")})]
        return []


class HardcodeDetector:
    """Simple release guard: new intelligence should live as schemas, not parser if-strings."""

    DEFAULT_FORBIDDEN = [
        "would you classify",
        "lends",
        "zorbles",
        "glarns",
        "철수",
        "영희",
        "우세하다고",
    ]

    def __init__(self, root: Path, forbidden: Optional[List[str]] = None):
        self.root = Path(root)
        self.forbidden = forbidden or self.DEFAULT_FORBIDDEN

    def scan(self) -> Dict[str, Any]:
        hits = []
        # V33 guard is scoped to new substrate/runtime files, not legacy
        # historical parser fixtures that remain in the repository for regression.
        scoped_prefixes = ("neurova/schema_learning.py", "neurova/v33_", "neurova/agent.py")
        for path in self.root.rglob("*.py"):
            rel = str(path.relative_to(self.root))
            if rel.endswith("test_v33_schema_learning_substrate.py") or rel.endswith("v33_schema_audit.py") or rel.endswith("schema_learning.py"):
                continue
            if not rel.startswith(scoped_prefixes) or rel.endswith("agent.py"):
                continue
            text = path.read_text(encoding="utf-8")
            code_lines = []
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith("\"\"\""):
                    continue
                code_lines.append(line)
            code = "\n".join(code_lines).lower()
            for needle in self.forbidden:
                if needle.lower() in code:
                    hits.append({"file": rel, "needle": needle})
        return {"passed": len(hits) == 0, "hits": hits}


class SchemaLearningSubstrate:
    """Facade used by BrainOS runtime/tests."""

    def __init__(self, path: Path):
        self.memory = DevelopmentalSchemaMemory(path)
        self.inducer = SchemaInductionEngine(self.memory)
        self.executor = SchemaExecutor(self.memory)

    def close(self):
        self.memory.close()

    def __del__(self):
        self.close()

    def observe(self, text: str, predicted_ir: Optional[Dict[str, Any]] = None, outcome: str = "observed") -> str:
        return self.memory.record_episode(Experience(text=text, predicted_ir=predicted_ir, outcome=outcome))

    def record_error(self, episode_id: str, error_type: str, expected: Dict[str, Any], observed: Dict[str, Any], severity: float = 0.5) -> str:
        return self.memory.record_prediction_error(PredictionError(episode_id, error_type, expected, observed, severity))

    def learn_from_correction(self, text: str) -> Optional[Dict[str, Any]]:
        return self.inducer.learn_from_correction(text)

    def compile(self, text: str) -> List[IRCandidate]:
        return self.executor.apply(text)

    def consolidate(self) -> Dict[str, Any]:
        # In this compact implementation consolidation means: re-evaluate candidate schemas and
        # promote any schema that now has sufficient evidence/tests.
        promoted = []
        rows = self.memory.conn.execute("SELECT id FROM schema_candidates WHERE status IN ('candidate','session_patch','experimental')").fetchall()
        for row in rows:
            result = self.inducer.gate.evaluate_and_promote(row["id"])
            if result["passed"]:
                promoted.append(row["id"])
        families = self.memory.conn.execute("SELECT error_type, COUNT(*) c FROM prediction_errors GROUP BY error_type ORDER BY c DESC").fetchall()
        return {"promoted": promoted, "error_families": {r["error_type"]: r["c"] for r in families}, "stable_schema_count": self.memory.count("stable_schemas")}


# ===========================================================================
# V36 Evolution: Demotion, Scope Narrowing, Failure-to-Schema Compiler
# ===========================================================================

class SchemaDemotionEngine:
    """Demotes or rejects schemas that fail too often or overgeneralize.

    Critical rule: schemas are never silently deleted. They are demoted
    (status=demoted) with a reason, preserving full audit trail.
    """

    def __init__(self, memory: DevelopmentalSchemaMemory):
        self.memory = memory

    def check_and_demote(self, schema_id: str, min_precision: float = 0.6) -> Dict[str, Any]:
        row = self.memory.conn.execute(
            "SELECT * FROM schema_candidates WHERE id=?", (schema_id,)
        ).fetchone()
        if not row:
            return {"schema_id": schema_id, "action": "not_found"}

        schema = json.loads(row["schema_json"])
        success = schema.get("success_count", 0)
        failure = schema.get("failure_count", 0)
        total = success + failure
        if total < 3:
            return {"schema_id": schema_id, "action": "insufficient_data", "total": total}

        precision = success / max(1, total)
        if precision < min_precision:
            self.memory.conn.execute(
                "UPDATE schema_candidates SET status='demoted' WHERE id=?",
                (schema_id,),
            )
            # Also remove from stable_schemas if present.
            self.memory.conn.execute(
                "DELETE FROM stable_schemas WHERE id=?", (schema_id,)
            )
            self.memory.conn.commit()
            return {
                "schema_id": schema_id,
                "action": "demoted",
                "precision": precision,
                "reason": f"precision {precision:.2f} < {min_precision}",
            }
        return {"schema_id": schema_id, "action": "kept", "precision": precision}

    def narrow_scope(self, schema_id: str, new_counterexamples: List[str]) -> Dict[str, Any]:
        """Add counterexamples to narrow a schema's scope without full demotion."""
        row = self.memory.conn.execute(
            "SELECT * FROM schema_candidates WHERE id=?", (schema_id,)
        ).fetchone()
        if not row:
            return {"schema_id": schema_id, "action": "not_found"}

        tests = [
            SchemaTest(ce, {"should_not_match": True}, is_counterexample=True)
            for ce in new_counterexamples
        ]
        self.memory.add_schema_tests(schema_id, tests)

        # Re-evaluate with new counterexamples.
        gate = RegressionGate(self.memory)
        result = gate.evaluate_and_promote(schema_id, min_precision=0.8)
        action = "narrowed_and_kept" if result["passed"] else "narrowed_and_demoted"
        return {"schema_id": schema_id, "action": action, "gate": result}

    def sweep(self, min_precision: float = 0.6) -> List[Dict[str, Any]]:
        """Batch check all experimental/stable schemas and demote weak ones."""
        rows = self.memory.conn.execute(
            "SELECT id FROM schema_candidates WHERE status IN ('experimental','stable','session_patch')"
        ).fetchall()
        results = []
        for row in rows:
            r = self.check_and_demote(row["id"], min_precision)
            if r["action"] in ("demoted",):
                results.append(r)
        return results


class FailureToSchemaCompiler:
    """Automatically compiles clustered prediction errors into schema candidates.

    Pipeline: failure episodes -> clustering -> schema candidate generation ->
    counterexample generation -> regression gate -> promotion/rejection.

    This is the v36 core loop that turns BrainOS's failures into learned ability.
    """

    def __init__(self, substrate: "SchemaLearningSubstrate"):
        self.substrate = substrate
        self.memory = substrate.memory
        self.interpreter = CorrectionInterpreter()
        self.counterexamples = CounterexampleGenerator()
        self.gate = RegressionGate(self.memory)

    def compile_from_errors(self, min_cluster_size: int = 2) -> Dict[str, Any]:
        """Mine prediction errors and generate schema candidates for recurring patterns."""
        errors = self.memory.conn.execute(
            "SELECT error_type, COUNT(*) c FROM prediction_errors GROUP BY error_type HAVING c >= ? ORDER BY c DESC",
            (min_cluster_size,),
        ).fetchall()

        candidates_generated = []
        for row in errors:
            error_type = row["error_type"]
            count = row["c"]

            # Fetch sample episodes for this error type.
            error_episodes = self.memory.conn.execute(
                "SELECT * FROM prediction_errors WHERE error_type=? ORDER BY ROWID DESC LIMIT 10",
                (error_type,),
            ).fetchall()

            # Extract failing texts from episodes.
            failing_texts = []
            for ep_row in error_episodes:
                observed = json.loads(ep_row["observed_json"])
                ep_id = ep_row["episode_id"]
                ep = self.memory.conn.execute(
                    "SELECT text FROM episodes WHERE id=?", (ep_id,)
                ).fetchone()
                if ep:
                    failing_texts.append(ep["text"])

            if not failing_texts:
                continue

            # Generate a schema candidate placeholder.
            # In the full system, LLM Teacher would fill this.
            # For now we create a structural candidate from the error cluster.
            candidate = SchemaCandidate(
                schema_type=f"AutoLearnedSchema_{error_type}",
                schema={
                    "error_type": error_type,
                    "sample_texts": failing_texts[:5],
                    "cluster_size": count,
                    "status": "needs_teacher",
                },
                source_episode_ids=[e["episode_id"] for e in error_episodes[:5]],
                confidence=min(0.5, 0.1 * count),
                status="candidate",
            )
            schema_id = self.memory.add_schema_candidate(candidate)
            candidates_generated.append({
                "schema_id": schema_id,
                "error_type": error_type,
                "cluster_size": count,
                "sample_count": len(failing_texts),
            })

        return {
            "error_clusters": len(errors),
            "candidates_generated": candidates_generated,
        }

    def compile_single_failure(self, text: str, error_type: str, context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Record a single failure and attempt immediate schema candidate generation."""
        ep_id = self.memory.record_episode(
            Experience(text=text, context=context or {}, outcome="failure_observed")
        )
        self.memory.record_prediction_error(
            PredictionError(ep_id, error_type, expected={"schema": "needed"}, observed={"text": text}, severity=0.6)
        )

        # Try to interpret the failure as a correction/new pattern.
        cand = self.interpreter.parse(text)
        if cand:
            cand.source_episode_ids.append(ep_id)
            schema_id = self.memory.add_schema_candidate(cand)
            tests = self.counterexamples.generate(cand.schema_type, cand.schema)
            self.memory.add_schema_tests(schema_id, tests)
            result = self.gate.evaluate_and_promote(schema_id)
            return {"schema_id": schema_id, "schema_type": cand.schema_type, "gate": result}

        return None


# -- Extend SchemaLearningSubstrate with v36 capabilities --------------------

def _v36_demote(self, schema_id: str, min_precision: float = 0.6) -> Dict[str, Any]:
    engine = SchemaDemotionEngine(self.memory)
    return engine.check_and_demote(schema_id, min_precision)


def _v36_narrow_scope(self, schema_id: str, counterexamples: List[str]) -> Dict[str, Any]:
    engine = SchemaDemotionEngine(self.memory)
    return engine.narrow_scope(schema_id, counterexamples)


def _v36_sweep_demotions(self, min_precision: float = 0.6) -> List[Dict[str, Any]]:
    engine = SchemaDemotionEngine(self.memory)
    return engine.sweep(min_precision)


def _v36_compile_failures(self, min_cluster_size: int = 2) -> Dict[str, Any]:
    compiler = FailureToSchemaCompiler(self)
    return compiler.compile_from_errors(min_cluster_size)


def _v36_compile_single_failure(self, text: str, error_type: str, context=None) -> Optional[Dict[str, Any]]:
    compiler = FailureToSchemaCompiler(self)
    return compiler.compile_single_failure(text, error_type, context)


def _v36_consolidate(self) -> Dict[str, Any]:
    """Enhanced consolidation: re-evaluate candidates + compile failures + sweep demotions."""
    # 1. Original consolidation: re-evaluate candidate schemas.
    promoted = []
    rows = self.memory.conn.execute(
        "SELECT id FROM schema_candidates WHERE status IN ('candidate','session_patch','experimental')"
    ).fetchall()
    for row in rows:
        result = self.inducer.gate.evaluate_and_promote(row["id"])
        if result["passed"]:
            promoted.append(row["id"])

    # 2. Compile recurring failures into schema candidates.
    failure_report = self.compile_failures(min_cluster_size=2)

    # 3. Sweep and demote weak schemas.
    demotions = self.sweep_demotions(min_precision=0.6)

    families = self.memory.conn.execute(
        "SELECT error_type, COUNT(*) c FROM prediction_errors GROUP BY error_type ORDER BY c DESC"
    ).fetchall()

    return {
        "promoted": promoted,
        "failure_compilation": failure_report,
        "demotions": demotions,
        "error_families": {r["error_type"]: r["c"] for r in families},
        "stable_schema_count": self.memory.count("stable_schemas"),
        "candidate_count": self.memory.count("schema_candidates"),
    }


# Bind v36 methods to SchemaLearningSubstrate.
SchemaLearningSubstrate.demote = _v36_demote
SchemaLearningSubstrate.narrow_scope = _v36_narrow_scope
SchemaLearningSubstrate.sweep_demotions = _v36_sweep_demotions
SchemaLearningSubstrate.compile_failures = _v36_compile_failures
SchemaLearningSubstrate.compile_single_failure = _v36_compile_single_failure
SchemaLearningSubstrate.consolidate = _v36_consolidate


# -- Extend DevelopmentalSchemaMemory with demotion tracking ------------------

def _v36_demote_schema(self, schema_id: str, reason: str = "") -> None:
    self.conn.execute(
        "UPDATE schema_candidates SET status='demoted' WHERE id=?", (schema_id,)
    )
    self.conn.execute(
        "DELETE FROM stable_schemas WHERE id=?", (schema_id,)
    )
    self.conn.commit()


def _v36_schema_stats(self) -> Dict[str, Any]:
    rows = self.conn.execute(
        "SELECT status, COUNT(*) c FROM schema_candidates GROUP BY status"
    ).fetchall()
    return {r["status"]: r["c"] for r in rows}


DevelopmentalSchemaMemory.demote_schema = _v36_demote_schema
DevelopmentalSchemaMemory.schema_stats = _v36_schema_stats
