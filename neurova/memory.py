from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from time import time
from typing import Dict, List, Optional
import json, sqlite3, uuid
from .ir import *


def _norm(x: Optional[str]) -> str:
    return (x or "").strip().lower()


def _time_matches(query_time: Optional[str], valid_from: Optional[str], valid_to: Optional[str], valid_during: Optional[str] = None) -> bool:
    if not query_time:
        return True
    q = str(query_time)
    vf = valid_from or valid_during or ""
    vt = valid_to or ""
    if vf and (q == vf or q.startswith(vf) or vf.startswith(q)):
        return True
    if vf and vt:
        return vf <= q <= vt
    if vf and len(q) == 4 and vf.startswith(q):
        return True
    return False


def _time_ranges_overlap(a_from: Optional[str], a_to: Optional[str], b_from: Optional[str], b_to: Optional[str]) -> bool:
    # Unknown/untimed claims are treated as globally overlapping.
    if not (a_from or a_to or b_from or b_to):
        return True
    if not (a_from or a_to) or not (b_from or b_to):
        return True
    af, at = a_from or a_to, a_to or a_from
    bf, bt = b_from or b_to, b_to or b_from
    return str(af) <= str(bt) and str(bf) <= str(at)

class EvidenceGraphMemory:
    """Versioned evidence graph.

    Claim identity is separated from claim versions. Positive and negative claims no longer
    overwrite each other, and temporal/source/scope variants are stored independently.
    """
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=1.0)
        self.conn.row_factory = sqlite3.Row
        self._init()

    def close(self):
        self.conn.close()

    def __del__(self):
        self.close()

    def _init(self):
        c = self.conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, kind TEXT, raw_text TEXT, ir_json TEXT, created_at REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS evidence(id TEXT PRIMARY KEY, source TEXT, quote TEXT, reliability REAL, created_at REAL)")
        c.execute("""
        CREATE TABLE IF NOT EXISTS claims(
          claim_id TEXT PRIMARY KEY,
          normalized_key TEXT UNIQUE,
          subject TEXT, relation TEXT, object TEXT,
          created_at REAL
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS claim_versions(
          version_id TEXT PRIMARY KEY,
          claim_id TEXT,
          version_key TEXT UNIQUE,
          polarity TEXT,
          valid_from TEXT,
          valid_to TEXT,
          source_id TEXT,
          scope TEXT,
          confidence REAL,
          reliability REAL,
          status TEXT,
          support_count INTEGER,
          refute_count INTEGER,
          exceptions_json TEXT,
          last_verified_at REAL,
          updated_at REAL,
          FOREIGN KEY(claim_id) REFERENCES claims(claim_id)
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS rules(
          rule_id TEXT PRIMARY KEY,
          signature TEXT UNIQUE,
          condition_relation TEXT,
          condition_object TEXT,
          conclusion_relation TEXT,
          conclusion_object TEXT,
          quantifier TEXT,
          scope TEXT,
          confidence REAL,
          exceptions_json TEXT,
          created_at REAL
        )""")
        c.execute("CREATE TABLE IF NOT EXISTS exceptions(id TEXT PRIMARY KEY, rule_signature TEXT, exception_subject TEXT, exception_text TEXT, confidence REAL, created_at REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS transitions(id TEXT PRIMARY KEY, cause TEXT, effect TEXT, confidence REAL, updated_at REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS comparisons(id TEXT PRIMARY KEY, left_value TEXT, comparator TEXT, right_value TEXT, confidence REAL, updated_at REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS contradictions(id TEXT PRIMARY KEY, claim_a TEXT, claim_b TEXT, reason TEXT, created_at REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS benchmarks(id TEXT PRIMARY KEY, name TEXT, score REAL, detail_json TEXT, created_at REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS graph_nodes(id TEXT PRIMARY KEY, kind TEXT, ref_key TEXT, label TEXT, created_at REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS graph_edges(id TEXT PRIMARY KEY, src TEXT, dst TEXT, kind TEXT, confidence REAL, created_at REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS memory_actions(id TEXT PRIMARY KEY, action TEXT, target TEXT, detail_json TEXT, reward REAL, created_at REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS trajectories(id TEXT PRIMARY KEY, task TEXT, status TEXT, failure_type TEXT, detail_json TEXT, lesson TEXT, created_at REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS world_transitions(id TEXT PRIMARY KEY, state_json TEXT, action TEXT, next_state_json TEXT, confidence REAL, created_at REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS fluents(key TEXT PRIMARY KEY, subject TEXT, relation TEXT, value TEXT, confidence REAL, updated_at REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS fluent_history(id TEXT PRIMARY KEY, key TEXT, old_value TEXT, new_value TEXT, confidence REAL, created_at REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS promotion_candidates(id TEXT PRIMARY KEY, kind TEXT, payload_json TEXT, status TEXT, score REAL, detail_json TEXT, created_at REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS immune_events(id TEXT PRIMARY KEY, target TEXT, status TEXT, reason TEXT, created_at REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS sleep_reports(id TEXT PRIMARY KEY, report_json TEXT, created_at REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS learned_strategies(id TEXT PRIMARY KEY, failure_type TEXT, strategy TEXT, support_count INTEGER, created_at REAL)")
        self.conn.commit()

    def add_node(self, node_id: str, kind: str, ref_key: str, label: str):
        self.conn.execute("INSERT OR IGNORE INTO graph_nodes VALUES (?, ?, ?, ?, ?)", (node_id, kind, ref_key, label, time()))

    def add_edge(self, src: str, dst: str, kind: str, confidence: float = 0.7):
        self.conn.execute("INSERT INTO graph_edges VALUES (?, ?, ?, ?, ?, ?)", ("edge_" + uuid.uuid4().hex[:12], src, dst, kind, confidence, time()))

    def log_action(self, action: str, target: str, detail: dict | None = None, reward: float = 0.0):
        self.conn.execute("INSERT INTO memory_actions VALUES (?, ?, ?, ?, ?, ?)", ("act_" + uuid.uuid4().hex[:12], action, target, json.dumps(detail or {}, ensure_ascii=False), reward, time()))
        self.conn.commit()

    def log_trajectory(self, task: str, status: str, failure_type: str = "", detail: dict | None = None, lesson: str = ""):
        self.conn.execute("INSERT INTO trajectories VALUES (?, ?, ?, ?, ?, ?, ?)", ("traj_" + uuid.uuid4().hex[:12], task, status, failure_type, json.dumps(detail or {}, ensure_ascii=False), lesson, time()))
        self.conn.commit()

    def append_event(self, kind: str, raw_text: str, ir: CognitiveIR) -> str:
        eid = "evt_" + uuid.uuid4().hex[:12]
        try:
            payload = asdict(ir)
        except Exception:
            payload = {"repr": repr(ir)}
        payload["_type"] = type(ir).__name__
        self.conn.execute("INSERT INTO events VALUES (?, ?, ?, ?, ?)", (eid, kind, raw_text, json.dumps(payload, ensure_ascii=False), time()))
        self.conn.commit()
        return eid

    def add_evidence(self, source: str, quote: str, reliability: float = 0.7) -> str:
        eid = "ev_" + uuid.uuid4().hex[:12]
        self.conn.execute("INSERT INTO evidence VALUES (?, ?, ?, ?, ?)", (eid, source, quote, reliability, time()))
        self.add_node(eid, "evidence", eid, source)
        self.conn.commit()
        return eid

    def _ensure_claim_identity(self, claim: ClaimIR) -> str:
        key = claim.normalized_key()
        row = self.conn.execute("SELECT claim_id FROM claims WHERE normalized_key=?", (key,)).fetchone()
        if row:
            return row["claim_id"]
        cid = "claim_" + uuid.uuid4().hex[:12]
        self.conn.execute("INSERT INTO claims VALUES (?, ?, ?, ?, ?, ?)", (cid, key, _norm(claim.subject), _norm(claim.relation), _norm(claim.object), time()))
        self.add_node("claim:" + cid, "claim_identity", key, claim.text())
        return cid

    def _version_key(self, claim_id: str, claim: ClaimIR) -> str:
        vf = claim.valid_from or getattr(claim, "valid_during", None) or ""
        return "|".join([claim_id, claim.polarity, vf, claim.valid_to or "", claim.source_id or "user", claim.scope or "global"])

    def upsert_claim(self, claim: ClaimIR, evidence_id: Optional[str] = None) -> str:
        if isinstance(claim, TemporalClaimIR):
            claim.valid_from = claim.valid_from or claim.time_expr or claim.valid_during
        cid = self._ensure_claim_identity(claim)
        version_key = self._version_key(cid, claim)
        vid = "claimv_" + uuid.uuid4().hex[:12]
        support = 1 if claim.polarity == "positive" else 0
        refute = 1 if claim.polarity == "negative" else 0
        self.conn.execute("""
        INSERT INTO claim_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(version_key) DO UPDATE SET
          confidence=max(confidence, excluded.confidence),
          reliability=max(reliability, excluded.reliability),
          support_count=support_count+excluded.support_count,
          refute_count=refute_count+excluded.refute_count,
          exceptions_json=excluded.exceptions_json,
          last_verified_at=excluded.last_verified_at,
          updated_at=excluded.updated_at
        """, (
            vid, cid, version_key, claim.polarity, claim.valid_from or getattr(claim, "valid_during", None), claim.valid_to,
            claim.source_id or "user", claim.scope, claim.confidence, claim.reliability, "active", support, refute,
            json.dumps(claim.exceptions, ensure_ascii=False), time(), time()
        ))
        row = self.find_claim(claim.subject, claim.relation, claim.object, claim.polarity, claim.valid_from or getattr(claim, "valid_during", None))
        node_id = "claimv:" + row["version_id"]
        self.add_node(node_id, "claim_version", claim.normalized_key() + "|" + claim.polarity, claim.text())
        self.add_edge("claim:" + cid, node_id, "has_version", claim.confidence)
        if evidence_id:
            self.add_edge(evidence_id, node_id, "supports" if claim.polarity == "positive" else "refutes", claim.confidence)
        self.log_action("ADD_CLAIM" if claim.polarity == "positive" else "ADD_NEGATED_CLAIM", claim.normalized_key(), {"text": claim.text(), "polarity": claim.polarity, "valid_from": claim.valid_from}, claim.confidence)
        self._detect_contradiction(cid)
        self.conn.commit()
        return row["version_id"]

    def find_claim(self, subject: str, relation: str, obj: str, polarity: str = "positive", at: Optional[str] = None):
        key = f"{_norm(subject)}|{_norm(relation)}|{_norm(obj)}"
        rows = self.conn.execute("""
          SELECT c.claim_id, c.normalized_key, c.subject, c.relation, c.object, v.*
          FROM claims c JOIN claim_versions v ON c.claim_id=v.claim_id
          WHERE c.normalized_key=? AND v.polarity=? AND v.status='active'
          ORDER BY v.updated_at DESC
        """, (key, polarity)).fetchall()
        for r in rows:
            if _time_matches(at, r["valid_from"], r["valid_to"]):
                return r
        return None

    def claim_versions(self, relation: Optional[str] = None, polarity: Optional[str] = None, at: Optional[str] = None):
        q = """
          SELECT c.claim_id, c.normalized_key, c.subject, c.relation, c.object, v.*
          FROM claims c JOIN claim_versions v ON c.claim_id=v.claim_id
          WHERE v.status='active'
        """
        args = []
        if relation:
            q += " AND c.relation=?"; args.append(_norm(relation))
        if polarity:
            q += " AND v.polarity=?"; args.append(polarity)
        rows = self.conn.execute(q, args).fetchall()
        return [r for r in rows if _time_matches(at, r["valid_from"], r["valid_to"])]

    def claims_about(self, subject: str) -> List[ClaimIR]:
        rows = self.conn.execute("""
          SELECT c.subject, c.relation, c.object, v.confidence, v.reliability, v.polarity, v.valid_from, v.valid_to
          FROM claims c JOIN claim_versions v ON c.claim_id=v.claim_id
          WHERE c.subject=? AND v.polarity='positive' AND v.status='active'
          ORDER BY c.relation, c.object
        """, (_norm(subject),)).fetchall()
        return [ClaimIR(subject=r["subject"], relation=r["relation"], object=r["object"], confidence=r["confidence"], reliability=r["reliability"], polarity=r["polarity"], valid_from=r["valid_from"], valid_to=r["valid_to"]) for r in rows]

    def add_rule(self, rule: RuleIR):
        rid = "rule_" + uuid.uuid4().hex[:12]
        quant = getattr(rule, "quantifier", "some")
        sig = rule.signature()
        self.conn.execute("""
        INSERT INTO rules VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(signature) DO UPDATE SET confidence=max(confidence, excluded.confidence), exceptions_json=excluded.exceptions_json
        """, (rid, sig, _norm(rule.condition_relation), _norm(rule.condition_object), _norm(rule.conclusion_relation), _norm(rule.conclusion_object), quant, rule.scope, rule.confidence, json.dumps(rule.exceptions, ensure_ascii=False), time()))
        row = self.conn.execute("SELECT rule_id FROM rules WHERE signature=?", (sig,)).fetchone()
        self.add_node(row["rule_id"], "rule", sig, rule.text())
        self.log_action("ADD_RULE", sig, {"text": rule.text()}, rule.confidence)
        self.conn.commit()
        return row["rule_id"]

    def rules(self):
        return [dict(r) for r in self.conn.execute("SELECT * FROM rules").fetchall()]

    def add_exception(self, exc: ExceptionIR):
        eid = "exc_" + uuid.uuid4().hex[:12]
        self.conn.execute("INSERT INTO exceptions VALUES (?, ?, ?, ?, ?, ?)", (eid, exc.rule_id, _norm(exc.exception_subject), exc.exception_text, exc.confidence, time()))
        self.add_node(eid, "exception", exc.rule_id, exc.exception_text or exc.exception_subject)
        self.log_action("ADD_EXCEPTION", exc.rule_id, {"subject": exc.exception_subject, "text": exc.exception_text}, exc.confidence)
        self.conn.commit()
        return eid

    def exception_blocks(self, subject: str, rule: dict) -> Optional[dict]:
        sig = rule.get("signature") or f"{rule['condition_relation']}|{rule['condition_object']}=>{rule['conclusion_relation']}|{rule['conclusion_object']}"
        rows = self.conn.execute("SELECT * FROM exceptions WHERE rule_signature=?", (sig,)).fetchall()
        s = _norm(subject)
        for r in rows:
            if r["exception_subject"] == s:
                return dict(r)
        return None

    def direct_exception_blocks(self, subject: str, relation: str, obj: str) -> Optional[dict]:
        suffix = f"=>{_norm(relation)}|{_norm(obj)}"
        rows = self.conn.execute("SELECT * FROM exceptions WHERE exception_subject=?", (_norm(subject),)).fetchall()
        for r in rows:
            if str(r["rule_signature"]).endswith(suffix):
                return dict(r)
        return None

    def add_causal(self, causal: CausalClaimIR, evidence_id: Optional[str] = None):
        if causal.polarity == "negative":
            self.log_action("SKIP_NEGATED_CAUSAL", "", {"cause": causal.cause, "effect": causal.effect}, 0.0)
            return None
        tid = "causal_" + uuid.uuid4().hex[:12]
        self.conn.execute("INSERT INTO transitions VALUES (?, ?, ?, ?, ?)", (tid, _norm(causal.cause), _norm(causal.effect), causal.confidence, time()))
        self.add_node(tid, "causal", causal.normalized_key(), causal.text())
        if evidence_id:
            self.add_edge(evidence_id, tid, "supports", causal.confidence)
        self.log_action("ADD_CAUSAL", tid, {"text": causal.text()}, causal.confidence)
        self.conn.commit()
        return tid

    def all_transitions(self):
        return [dict(r) for r in self.conn.execute("SELECT * FROM transitions").fetchall()]

    def add_comparison(self, comp: ComparisonIR):
        cid = "cmp_" + uuid.uuid4().hex[:12]
        self.conn.execute("INSERT INTO comparisons VALUES (?, ?, ?, ?, ?, ?)", (cid, _norm(comp.left), comp.comparator, _norm(comp.right), comp.confidence, time()))
        self.add_node(cid, "comparison", comp.normalized_key(), comp.text())
        self.log_action("ADD_COMPARISON", cid, {"text": comp.text()}, comp.confidence)
        self.conn.commit()
        return cid

    def all_comparisons(self):
        return [dict(r) for r in self.conn.execute("SELECT * FROM comparisons").fetchall()]

    def add_world_transition(self, state: dict, action: str, next_state: dict, confidence: float = 0.7):
        wid = "world_" + uuid.uuid4().hex[:12]
        self.conn.execute("INSERT INTO world_transitions VALUES (?, ?, ?, ?, ?, ?)", (wid, json.dumps(state, ensure_ascii=False, sort_keys=True), _norm(action), json.dumps(next_state, ensure_ascii=False, sort_keys=True), confidence, time()))
        self.log_action("ADD_WORLD_TRANSITION", action, {"state": state, "next_state": next_state}, confidence)
        self.conn.commit()
        return wid

    def world_predictions(self, action: str) -> List[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM world_transitions WHERE action=?", (_norm(action),)).fetchall()]

    def retrieve_broad(self, query: str, limit: int = 8) -> List[str]:
        terms = [t.lower() for t in query.split() if len(t) > 1]
        hits = []
        for t in terms:
            rows = self.conn.execute("""
              SELECT c.subject, c.relation, c.object, v.polarity
              FROM claims c JOIN claim_versions v ON c.claim_id=v.claim_id
              WHERE c.subject LIKE ? OR c.object LIKE ? LIMIT ?
            """, (f"%{t}%", f"%{t}%", limit)).fetchall()
            hits.extend([f"{r['subject']} {r['relation']} {'not ' if r['polarity']=='negative' else ''}{r['object']}" for r in rows])
        hits = list(dict.fromkeys(hits))[:limit]
        self.log_action("RETRIEVE_EVIDENCE", query, {"hits": hits}, 0.1 * len(hits))
        return hits

    def store_proof(self, proof: ProofIR):
        pid = "proof_" + uuid.uuid4().hex[:12]
        self.add_node(pid, "proof", proof.query, proof.status)
        for step in proof.steps:
            sid = "proofstep_" + uuid.uuid4().hex[:12]
            self.add_node(sid, "proof_step", step.conclusion, step.rule_applied)
            self.add_edge(sid, pid, "supports" if proof.success else "refutes", step.confidence)
        self.log_action("STORE_PROOF", proof.query, {"success": proof.success, "status": proof.status, "steps": len(proof.steps)}, 1.0 if proof.success else 0.2)
        self.conn.commit()

    def _detect_contradiction(self, claim_id: str):
        positives = self.conn.execute("SELECT * FROM claim_versions WHERE claim_id=? AND polarity='positive' AND status='active'", (claim_id,)).fetchall()
        negatives = self.conn.execute("SELECT * FROM claim_versions WHERE claim_id=? AND polarity='negative' AND status='active'", (claim_id,)).fetchall()
        for pos in positives:
            for neg in negatives:
                if not _time_ranges_overlap(pos["valid_from"], pos["valid_to"], neg["valid_from"], neg["valid_to"]):
                    continue
                exists = self.conn.execute("SELECT 1 FROM contradictions WHERE claim_a=? AND claim_b=?", (pos["version_id"], neg["version_id"])).fetchone()
                if not exists:
                    cid = "contr_" + uuid.uuid4().hex[:12]
                    self.conn.execute("INSERT INTO contradictions VALUES (?, ?, ?, ?, ?)", (cid, pos["version_id"], neg["version_id"], "positive/negative polarity conflict over overlapping time scope", time()))
                    self.add_edge("claimv:" + pos["version_id"], "claimv:" + neg["version_id"], "contradicts", 1.0)

    def contradictions_for_key(self, normalized_key: str) -> List[dict]:
        row = self.conn.execute("SELECT claim_id FROM claims WHERE normalized_key=?", (normalized_key,)).fetchone()
        if not row:
            return []
        versions = [r["version_id"] for r in self.conn.execute("SELECT version_id FROM claim_versions WHERE claim_id=?", (row["claim_id"],)).fetchall()]
        if not versions:
            return []
        marks = ",".join("?" for _ in versions)
        rows = self.conn.execute(f"SELECT * FROM contradictions WHERE claim_a IN ({marks}) OR claim_b IN ({marks})", versions + versions).fetchall()
        return [dict(r) for r in rows]


    def set_fluent(self, subject: str, relation: str, value: str, confidence: float = 0.7):
        subject, relation, value = _norm(subject), _norm(relation), _norm(value)
        key = f"{subject}|{relation}"
        old = self.conn.execute("SELECT value FROM fluents WHERE key=?", (key,)).fetchone()
        old_value = old["value"] if old else None
        self.conn.execute("INSERT INTO fluents VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value, confidence=max(confidence, excluded.confidence), updated_at=excluded.updated_at", (key, subject, relation, value, confidence, time()))
        self.conn.execute("INSERT INTO fluent_history VALUES (?, ?, ?, ?, ?, ?)", ("flh_" + uuid.uuid4().hex[:12], key, old_value or "", value, confidence, time()))
        self.conn.commit()
        return key

    def get_fluent(self, subject: str, relation: str):
        return self.conn.execute("SELECT * FROM fluents WHERE subject=? AND relation=?", (_norm(subject), _norm(relation))).fetchone()

    def fluents_about(self, subject: str):
        return list(self.conn.execute("SELECT * FROM fluents WHERE subject=? ORDER BY updated_at DESC", (_norm(subject),)))

    def record_benchmark(self, name: str, score: float, detail: dict):
        self.conn.execute("INSERT INTO benchmarks VALUES (?, ?, ?, ?, ?)", ("bench_" + uuid.uuid4().hex[:12], name, score, json.dumps(detail, ensure_ascii=False), time()))
        self.conn.commit()

    def add_promotion_candidate(self, kind: str, payload: dict, status: str = "candidate", score: float = 0.0, detail: dict | None = None) -> str:
        pid = "promo_" + uuid.uuid4().hex[:12]
        self.conn.execute("INSERT INTO promotion_candidates VALUES (?, ?, ?, ?, ?, ?, ?)", (pid, kind, json.dumps(payload, ensure_ascii=False), status, score, json.dumps(detail or {}, ensure_ascii=False), time()))
        self.conn.commit()
        return pid


    def set_claim_version_status(self, version_id: str, status: str, reason: str = ""):
        self.conn.execute("UPDATE claim_versions SET status=?, updated_at=? WHERE version_id=?", (status, time(), version_id))
        self.conn.execute("INSERT INTO immune_events VALUES (?, ?, ?, ?, ?)", ("immune_" + uuid.uuid4().hex[:12], version_id, status, reason, time()))
        self.log_action("SET_CLAIM_STATUS", version_id, {"status": status, "reason": reason}, 0.0)
        self.conn.commit()

    def claim_version_status_counts(self) -> Dict[str, int]:
        rows = self.conn.execute("SELECT status, COUNT(*) AS n FROM claim_versions GROUP BY status").fetchall()
        return {r["status"]: r["n"] for r in rows}

    def record_sleep_report(self, report: dict) -> str:
        sid = "sleep_" + uuid.uuid4().hex[:12]
        self.conn.execute("INSERT INTO sleep_reports VALUES (?, ?, ?)", (sid, json.dumps(report, ensure_ascii=False), time()))
        self.conn.commit()
        return sid

    def add_learned_strategy(self, failure_type: str, strategy: str, support_count: int = 1) -> str:
        sid = "strategy_" + uuid.uuid4().hex[:12]
        self.conn.execute("INSERT INTO learned_strategies VALUES (?, ?, ?, ?, ?)", (sid, failure_type, strategy, support_count, time()))
        self.log_action("ADD_LEARNED_STRATEGY", failure_type, {"strategy": strategy, "support_count": support_count}, min(1.0, 0.2 * support_count))
        self.conn.commit()
        return sid

    def stats(self) -> Dict[str, int]:
        tables = ["events", "evidence", "claims", "claim_versions", "rules", "exceptions", "transitions", "comparisons", "contradictions", "benchmarks", "graph_nodes", "graph_edges", "memory_actions", "trajectories", "world_transitions", "promotion_candidates", "immune_events", "sleep_reports", "learned_strategies"]
        return {t: self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}

# --- V25 dynamic extensions -------------------------------------------------
def _v25_prev_year(year: str) -> str:
    try:
        return str(int(str(year)[:4]) - 1)
    except Exception:
        return year


def close_positive_claim_before(self, subject: str, relation: str, obj: str, stop_time: str) -> int:
    """Close open-ended positive temporal claim versions before a stop event.

    Example: Alice became CEO in 2025; Alice stopped being CEO in 2027.
    The positive version is rewritten to valid_to=2026 so 2026 queries work and
    2027 queries are refuted/unknown rather than incorrectly proved.
    """
    key = f"{_norm(subject)}|{_norm(relation)}|{_norm(obj)}"
    row = self.conn.execute("SELECT claim_id FROM claims WHERE normalized_key=?", (key,)).fetchone()
    if not row:
        return 0
    close_to = _v25_prev_year(stop_time)
    cur = self.conn.execute("""
        UPDATE claim_versions SET valid_to=?, updated_at=?
        WHERE claim_id=? AND polarity='positive' AND status='active' AND (valid_to IS NULL OR valid_to='' OR valid_to='9999')
    """, (close_to, time(), row['claim_id']))
    self.log_action("CLOSE_TEMPORAL_CLAIM", key, {"stop_time": stop_time, "valid_to": close_to}, 0.8)
    self.conn.commit()
    return cur.rowcount


EvidenceGraphMemory.close_positive_claim_before = close_positive_claim_before

