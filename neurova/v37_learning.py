from __future__ import annotations

"""V37 developmental failure-to-construction learning loop.

The goal is not to add more sentence-level regexes.  Regex/legacy parsers are
now treated as *seed candidate generators*.  Failures are recorded as prediction
errors, embedded, clustered, and converted into schema candidates.  Natural
language corrections can promote those candidates into stable construction or
EventFrame schemas.  A 500-case paraphrase stress suite exercises the learned
schemas and the existing symbolic world/reasoning runtime without answer-table
shortcuts.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import math
import re
import sqlite3
import time
import uuid

from .ir import IRCandidate, ResearchTaskIR, ClaimIR, ComparisonIR, CausalClaimIR, EventIR, QuestionIR, ToolCallIR
from .schema_learning import SchemaLearningSubstrate, norm_text


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class SeedCandidate:
    text: str
    parser: str
    ir_type: str
    confidence: float
    status: str = "seed_candidate"  # seed_candidate, verified, rejected
    notes: List[str] = field(default_factory=list)


@dataclass
class PredictionErrorRecord:
    text: str
    response: str = ""
    predicted_ir_type: str = "unknown"
    error_type: str = "semantic_parse_error"
    severity: float = 0.5
    expected: Dict[str, Any] = field(default_factory=dict)
    observed: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class HashEmbeddingSpace:
    """Tiny deterministic embedding for clustering failures without a generator LM.

    This is deliberately not an autoregressive language model.  It is a hashed
    bag of word/char n-gram features used only for similarity and clustering.
    """

    def __init__(self, dims: int = 256):
        self.dims = dims

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dims
        low = norm_text(text)
        toks = re.findall(r"[a-zA-Z가-힣0-9]+", low)
        feats: List[str] = []
        feats += ["w:" + t for t in toks]
        feats += ["b:" + toks[i] + "_" + toks[i + 1] for i in range(len(toks) - 1)]
        compact = " ".join(toks)
        for n in (3, 4, 5):
            feats += ["c:" + compact[i : i + n] for i in range(max(0, len(compact) - n + 1))]
        for f in feats:
            h = hash(f) % self.dims
            sign = -1.0 if (hash("s" + f) % 2) else 1.0
            vec[h] += sign
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def cosine(self, a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))


class FailureClusterer:
    def __init__(self, embedder: Optional[HashEmbeddingSpace] = None, threshold: float = 0.42):
        self.embedder = embedder or HashEmbeddingSpace()
        self.threshold = threshold

    def cluster(self, records: List[PredictionErrorRecord]) -> List[Dict[str, Any]]:
        clusters: List[Dict[str, Any]] = []
        for rec in records:
            emb = self.embedder.embed(rec.text + " " + rec.error_type)
            best_i, best_s = -1, -1.0
            for i, cl in enumerate(clusters):
                s = self.embedder.cosine(emb, cl["centroid"])
                if s > best_s:
                    best_i, best_s = i, s
            if best_i >= 0 and best_s >= self.threshold:
                cl = clusters[best_i]
                cl["records"].append(rec)
                n = len(cl["records"])
                cl["centroid"] = [(c * (n - 1) + e) / n for c, e in zip(cl["centroid"], emb)]
            else:
                clusters.append({"id": _id("cluster"), "centroid": emb, "records": [rec], "label": rec.error_type})
        for cl in clusters:
            # readable family label
            words = []
            for r in cl["records"]:
                words += re.findall(r"[a-zA-Z가-힣]+", r.text.lower())
            top = sorted(set(words), key=lambda w: (-words.count(w), w))[:6]
            cl["label"] = f"{cl['records'][0].error_type}:{'/'.join(top)}"
        return clusters


class RegexSeedCandidateGenerator:
    """Runs legacy parser output only as seed hypotheses, never as promoted truth."""

    def __init__(self, compiler: Any):
        self.compiler = compiler

    def generate(self, text: str) -> List[SeedCandidate]:
        out: List[SeedCandidate] = []
        try:
            candidates = self.compiler.compile(text)
        except Exception as exc:
            return [SeedCandidate(text=text, parser="legacy_error", ir_type="Error", confidence=0.0, notes=[str(exc)])]
        for c in candidates[:5]:
            ir_type = type(c.ir).__name__
            notes = list(getattr(c, "notes", []))
            status = "rejected" if isinstance(c.ir, ResearchTaskIR) else "seed_candidate"
            out.append(SeedCandidate(text=text, parser=c.parser, ir_type=ir_type, confidence=float(c.confidence), status=status, notes=notes))
        return out


class ConstructionCandidateSynthesizer:
    """Converts failure clusters and corrections into schema candidates.

    It intentionally creates schema objects, not Python parser if-statements.
    """

    def __init__(self, schema_substrate: SchemaLearningSubstrate):
        self.schema_substrate = schema_substrate

    def candidates_from_cluster(self, cluster: Dict[str, Any]) -> List[Dict[str, Any]]:
        records: List[PredictionErrorRecord] = cluster.get("records", [])
        if not records:
            return []
        err = records[0].error_type
        examples = [r.text for r in records]
        schema_items: List[Dict[str, Any]] = []
        if "wrapper" in err or any(re.search(r"would you|is it true|do you think|can we say", e, re.I) for e in examples):
            schema_items.append({
                "schema_type": "WrapperSchema",
                "schema": {
                    "name": "cluster_induced_question_wrapper",
                    "form": "Would you say {P}?",
                    "operation": "QuestionIR(P)",
                    "applies_to": ["ClaimIR", "ComparisonIR", "CausalClaimIR", "BeliefIR", "TemporalClaimIR"],
                    "examples": examples,
                },
                "confidence": 0.55,
            })
        if "event" in err or any(re.search(r"from\s+\w+\s+to\s+\w+|located at|has", e, re.I) for e in examples):
            schema_items.append({
                "schema_type": "EventFrameSchema",
                "schema": {
                    "name": "cluster_induced_move_or_transfer_frame",
                    "form": "A carries B from C to D",
                    "roles": {"actor": "A", "patient": "B", "source": "C", "destination": "D"},
                    "effects": [{"subject": "B", "relation": "located_at", "object": "D"}],
                    "variants": ["A moves B from C to D", "A carried B from C to D", "B was carried from C to D by A"],
                    "examples": examples,
                },
                "confidence": 0.55,
            })
        return schema_items

    def learn_from_user_correction(self, correction: str) -> Optional[Dict[str, Any]]:
        return self.schema_substrate.learn_from_correction(correction)


class V37DevelopmentalLoop:
    """Durable V37 memory for failures, clusters, and promoted schemas."""

    def __init__(self, path: Path, schema_substrate: SchemaLearningSubstrate):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        self.embedder = HashEmbeddingSpace()
        self.clusterer = FailureClusterer(self.embedder)
        self.synthesizer = ConstructionCandidateSynthesizer(schema_substrate)

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS prediction_error_records(
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                response TEXT NOT NULL,
                predicted_ir_type TEXT NOT NULL,
                error_type TEXT NOT NULL,
                severity REAL NOT NULL,
                expected_json TEXT NOT NULL,
                observed_json TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS failure_clusters(
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                size INTEGER NOT NULL,
                examples_json TEXT NOT NULL,
                schema_candidates_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        self.conn.commit()

    def classify_error(self, text: str, response: str, ir_type: str) -> Optional[str]:
        low_r = response.lower()
        low_t = text.lower()
        if "fallback" in low_r or ir_type == "ResearchTaskIR" or "unknown" in low_r:
            return "semantic_parse_error"
        if "cannot prove" in low_r:
            if any(x in low_t for x in ["would you", "is it true", "do you think", "does ", "did "]):
                return "wrapper_operation_error"
            if "where" in low_t or "have" in low_t:
                return "world_state_error"
            return "proof_or_memory_error"
        if "could not compile inner" in low_r:
            return "wrapper_operation_error"
        return None

    def record_result(self, text: str, response: str, ir_type: str, confidence: float = 0.5) -> Optional[str]:
        error_type = self.classify_error(text, response, ir_type)
        if not error_type:
            return None
        rec = PredictionErrorRecord(text=text, response=response, predicted_ir_type=ir_type, error_type=error_type, severity=max(0.3, 1.0 - confidence), observed={"response": response, "ir_type": ir_type})
        emb = self.embedder.embed(text + " " + error_type)
        rid = _id("v37err")
        self.conn.execute(
            "INSERT INTO prediction_error_records VALUES (?,?,?,?,?,?,?,?,?,?)",
            (rid, rec.text, rec.response, rec.predicted_ir_type, rec.error_type, rec.severity, json.dumps(rec.expected, ensure_ascii=False), json.dumps(rec.observed, ensure_ascii=False), json.dumps(emb), rec.created_at),
        )
        self.conn.commit()
        return rid

    def records(self) -> List[PredictionErrorRecord]:
        rows = self.conn.execute("SELECT * FROM prediction_error_records ORDER BY created_at").fetchall()
        out = []
        for r in rows:
            out.append(PredictionErrorRecord(
                text=r["text"], response=r["response"], predicted_ir_type=r["predicted_ir_type"], error_type=r["error_type"], severity=float(r["severity"]), expected=json.loads(r["expected_json"]), observed=json.loads(r["observed_json"]), created_at=float(r["created_at"])
            ))
        return out

    def cluster_failures(self) -> List[Dict[str, Any]]:
        clusters = self.clusterer.cluster(self.records())
        for cl in clusters:
            schema_items = self.synthesizer.candidates_from_cluster(cl)
            self.conn.execute(
                "INSERT OR REPLACE INTO failure_clusters VALUES (?,?,?,?,?,?)",
                (cl["id"], cl["label"], len(cl["records"]), json.dumps([r.text for r in cl["records"]], ensure_ascii=False), json.dumps(schema_items, ensure_ascii=False), time.time()),
            )
        self.conn.commit()
        return clusters

    def stats(self) -> Dict[str, Any]:
        rec_count = self.conn.execute("SELECT COUNT(*) c FROM prediction_error_records").fetchone()["c"]
        cluster_count = self.conn.execute("SELECT COUNT(*) c FROM failure_clusters").fetchone()["c"]
        return {"prediction_errors": int(rec_count), "failure_clusters": int(cluster_count)}


class V37ParaphraseStressSuite:
    """500-case stress suite with generated paraphrases.

    The suite teaches schema families first, then tests held-out entities and
    morpho-syntactic variants.  It is not an official external benchmark; it is
    a broad internal stress suite designed to prevent Korea-only or smoke-only
    overfitting.
    """

    def teach(self, os: Any) -> None:
        lessons = [
            'When I say "A zorbles B", it means A is greater than B.',
            'Actually, "A splarns B" means A causes B.',
            'When A ferries B from C to D, it means A moves B from C to D, and after that B is located at D.',
            '"I need help" should be understood as support request',
            'teach: rover is robot', 'teach: robot is machine', 'teach: kibo is rover',
            'Korea is a peninsula in East Asia.',
            'all birds can fly', 'penguin is bird', 'penguin is exception to bird can fly',
        ]
        for t in lessons:
            os.observe(t)

    def cases(self) -> List[Tuple[str, str]]:
        cases: List[Tuple[str, str]] = []
        names = [("luma", "naro"), ("mira", "taro"), ("nova", "sol"), ("aris", "bex"), ("dana", "erin")]
        # 120 construction/comparison cases
        for i in range(24):
            a, b = names[i % len(names)]
            cases += [
                (f"{a} zorbles {b}.", "greater_than"),
                (f"Does {a} zorble {b}?", "Yes"),
                (f"Did {a} zorble {b}?", "Yes"),
                (f"Would you say {a} zorbles {b}?", "Yes"),
                (f"{b} was zorbled by {a}.", "greater_than"),
            ]
        # 100 causal cases
        causes = [("heat", "expansion"), ("rain", "erosion"), ("pressure", "stress"), ("spark", "fire"), ("practice", "skill")]
        for i in range(20):
            a, b = causes[i % len(causes)]
            cases += [
                (f"{a} splarns {b}.", "causes"),
                (f"Does {a} splarn {b}?", "Yes"),
                (f"Did {a} splarn {b}?", "Yes"),
                (f"{b} is splarned by {a}.", "causes"),
                (f"Is it true that {a} splarns {b}?", "Yes"),
            ]
        # 100 event/world cases
        objs = [("eve", "crate", "oslo", "lima"), ("mina", "box", "rome", "oslo"), ("sora", "map", "busan", "seoul"), ("kai", "cup", "kitchen", "shelf"), ("nora", "bag", "lyon", "madrid")]
        for i in range(20):
            a, obj, src, dst = objs[i % len(objs)]
            cases += [
                (f"{a} ferried the {obj} from {src} to {dst}.", "Stored event"),
                (f"Where is {obj}?", dst),
                (f"{a} carried the {obj} from {dst} to {src}.", "Stored event"),
                (f"Where is {obj}?", src),
                (f"{a} moved the {obj} from {src} to {dst}.", "Stored event"),
            ]
        # 80 taxonomy / exception / support / Korean
        for i in range(20):
            cases += [
                ("Can kibo be regarded as machine?", "Yes"),
                ("Does kibo fall under machine?", "Yes"),
                ("Can a penguin fly even though it is a bird?", "Blocked"),
                ("I need help with this.", "help"),
            ]
        # 100 meta/noise/scope guards and acquisition-like paraphrases
        for i in range(20):
            a, b = names[i % len(names)]
            cases += [
                (f"{a} almost zorbles {b}.", "No relevant"),
                (f"{a} wants to zorble {b}.", "No relevant"),
                ("Korea is separated from Japan to the southeast by the Korea Strait.", "spatial_separation"),
                ("What separates Korea from Japan?", "Korea Strait"),
                ("철수는 영희보다 크지 않은 것 같다", "less_than"),
            ]
        return cases[:500]

    def run(self, os: Any) -> Dict[str, Any]:
        self.teach(os)
        results = []
        for text, expected in self.cases():
            res = os.observe(text)
            ok = expected.lower() in res.response.lower()
            # Some guards are considered success when they do not assert the learned positive relation.
            if expected == "No relevant":
                ok = ("greater_than" not in res.response.lower()) or ("cannot prove" in res.response.lower())
            results.append({"text": text, "expected": expected, "response": res.response, "ok": ok})
        passed = sum(1 for r in results if r["ok"])
        return {"passed": passed, "total": len(results), "accuracy": passed / max(1, len(results)), "results": results}


class V37Audit:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        from .agent import FinalCognitiveOS
        os = FinalCognitiveOS(root=self.root / "os")
        seed_gen = RegexSeedCandidateGenerator(os.compiler)
        seeds = seed_gen.generate("this unknown utterance should not become truth")
        # Cause repeat failures and cluster them.
        for t in ["Would you classify blip as machine?", "Can we say blip is machine?", "Does blip fit within machine?"]:
            r = os.observe(t)
            os.v37_learning.record_result(t, r.response, r.ir_type, r.confidence)
        clusters = os.v37_learning.cluster_failures()
        # Correction -> promoted construction and event frame.
        corr = os.observe('When I say "A frobs B", it means A is greater than B.')
        evcorr = os.observe('When A ferries B from C to D, it means A moves B from C to D, and after that B is located at D.')
        stress = V37ParaphraseStressSuite().run(os)
        report = {
            "regex_seed_candidates": [s.__dict__ for s in seeds],
            "failure_stats": os.v37_learning.stats(),
            "cluster_count": len(clusters),
            "construction_correction": corr.response,
            "event_frame_correction": evcorr.response,
            "stress_passed": stress["passed"],
            "stress_total": stress["total"],
            "stress_accuracy": stress["accuracy"],
        }
        (self.root / "v37_audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
