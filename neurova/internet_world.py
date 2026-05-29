"""Internet World -- document ingestion and source-grounded memory.

The internet is BrainOS's environment, not a search engine fallback.
All internet-sourced information starts as SourcedClaimCandidate --
never directly as a stable fact.

Ingestion pipeline:
URL/page/PDF/wiki -> structured parse -> section-aware chunks ->
BM25+vector index -> entity/date/source metadata -> claim candidates ->
source-backed memory (always candidate status).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path


@dataclass
class SourcedClaimCandidate:
    """A claim extracted from an external source. Always starts as candidate."""
    id: str = ""
    claim_text: str = ""
    evidence_span: str = ""
    source_url: str = ""
    source_type: str = ""  # web, pdf, wiki, api
    fetched_at: float = field(default_factory=time.time)
    published_at: Optional[str] = None
    freshness: str = "fresh"  # fresh | stale | expired
    confidence: float = 0.3
    contradictions: List[str] = field(default_factory=list)
    status: str = "candidate"  # candidate | verified | rejected | expired

    def __post_init__(self):
        if not self.id:
            h = hashlib.blake2b(
                f"{self.claim_text}:{self.source_url}".encode(), digest_size=8
            ).hexdigest()
            self.id = f"src_{h}"


@dataclass
class DocumentChunk:
    """A section-aware chunk of an ingested document."""
    chunk_id: str = ""
    doc_id: str = ""
    section: str = ""
    text: str = ""
    position: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class InternetWorldMemory:
    """SQLite-backed storage for source-grounded claims and document chunks."""

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
        c = self.conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS sourced_claims(
                id TEXT PRIMARY KEY,
                claim_text TEXT NOT NULL,
                evidence_span TEXT DEFAULT '',
                source_url TEXT DEFAULT '',
                source_type TEXT DEFAULT '',
                fetched_at REAL NOT NULL,
                published_at TEXT,
                freshness TEXT DEFAULT 'fresh',
                confidence REAL DEFAULT 0.3,
                contradictions_json TEXT DEFAULT '[]',
                status TEXT DEFAULT 'candidate'
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks(
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                section TEXT DEFAULT '',
                text TEXT NOT NULL,
                position INTEGER DEFAULT 0,
                metadata_json TEXT DEFAULT '{}'
            )
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_claims_status ON sourced_claims(status)
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON document_chunks(doc_id)
        """)
        self.conn.commit()

    def add_claim(self, claim: SourcedClaimCandidate) -> str:
        self.conn.execute(
            """INSERT OR REPLACE INTO sourced_claims
               (id, claim_text, evidence_span, source_url, source_type,
                fetched_at, published_at, freshness, confidence, contradictions_json, status)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                claim.id, claim.claim_text, claim.evidence_span,
                claim.source_url, claim.source_type, claim.fetched_at,
                claim.published_at, claim.freshness, claim.confidence,
                json.dumps(claim.contradictions, ensure_ascii=False),
                claim.status,
            ),
        )
        self.conn.commit()
        return claim.id

    def add_chunk(self, chunk: DocumentChunk) -> str:
        if not chunk.chunk_id:
            h = hashlib.blake2b(
                f"{chunk.doc_id}:{chunk.position}:{chunk.text[:100]}".encode(),
                digest_size=8,
            ).hexdigest()
            chunk.chunk_id = f"chk_{h}"
        self.conn.execute(
            """INSERT OR REPLACE INTO document_chunks
               (chunk_id, doc_id, section, text, position, metadata_json)
               VALUES (?,?,?,?,?,?)""",
            (
                chunk.chunk_id, chunk.doc_id, chunk.section,
                chunk.text, chunk.position,
                json.dumps(chunk.metadata, ensure_ascii=False),
            ),
        )
        self.conn.commit()
        return chunk.chunk_id

    def search_claims(self, query: str, limit: int = 10) -> List[SourcedClaimCandidate]:
        tokens = set(re.findall(r"[a-zA-Z가-힣]+", query.lower()))
        rows = self.conn.execute(
            "SELECT * FROM sourced_claims WHERE status IN ('candidate','verified') ORDER BY confidence DESC LIMIT ?",
            (limit * 3,),
        ).fetchall()
        scored = []
        for r in rows:
            claim_tokens = set(re.findall(r"[a-zA-Z가-힣]+", r["claim_text"].lower()))
            overlap = len(tokens & claim_tokens) / max(1, len(tokens | claim_tokens))
            scored.append((r, overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [self._row_to_claim(r) for r, _ in scored[:limit]]

    def search_chunks(self, query: str, limit: int = 10) -> List[DocumentChunk]:
        tokens = set(re.findall(r"[a-zA-Z가-힣]+", query.lower()))
        rows = self.conn.execute(
            "SELECT * FROM document_chunks ORDER BY position LIMIT ?",
            (limit * 3,),
        ).fetchall()
        scored = []
        for r in rows:
            chunk_tokens = set(re.findall(r"[a-zA-Z가-힣]+", r["text"].lower()))
            overlap = len(tokens & chunk_tokens) / max(1, len(tokens | chunk_tokens))
            scored.append((r, overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            DocumentChunk(
                chunk_id=r["chunk_id"], doc_id=r["doc_id"],
                section=r["section"], text=r["text"],
                position=r["position"],
                metadata=json.loads(r["metadata_json"]),
            )
            for r, _ in scored[:limit]
        ]

    def verify_claim(self, claim_id: str) -> None:
        self.conn.execute(
            "UPDATE sourced_claims SET status='verified' WHERE id=?", (claim_id,)
        )
        self.conn.commit()

    def expire_stale(self, max_age_seconds: float = 86400 * 30) -> int:
        cutoff = time.time() - max_age_seconds
        cur = self.conn.execute(
            "UPDATE sourced_claims SET freshness='expired', status='expired' WHERE fetched_at < ? AND status='candidate'",
            (cutoff,),
        )
        self.conn.commit()
        return cur.rowcount

    def stats(self) -> Dict[str, int]:
        claims = self.conn.execute(
            "SELECT status, COUNT(*) c FROM sourced_claims GROUP BY status"
        ).fetchall()
        chunks = int(self.conn.execute("SELECT COUNT(*) FROM document_chunks").fetchone()[0])
        result = {r["status"]: r["c"] for r in claims}
        result["chunks"] = chunks
        return result

    def _row_to_claim(self, row: sqlite3.Row) -> SourcedClaimCandidate:
        return SourcedClaimCandidate(
            id=row["id"],
            claim_text=row["claim_text"],
            evidence_span=row["evidence_span"],
            source_url=row["source_url"],
            source_type=row["source_type"],
            fetched_at=row["fetched_at"],
            published_at=row["published_at"],
            freshness=row["freshness"],
            confidence=row["confidence"],
            contradictions=json.loads(row["contradictions_json"]),
            status=row["status"],
        )


class DocumentIngester:
    """Ingests raw text documents into section-aware chunks with claim extraction."""

    def __init__(self, world_memory: InternetWorldMemory):
        self.memory = world_memory

    def ingest_text(
        self,
        text: str,
        source_url: str = "",
        source_type: str = "text",
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not doc_id:
            doc_id = hashlib.blake2b(text[:500].encode(), digest_size=8).hexdigest()

        sections = self._split_sections(text)
        chunk_ids = []
        claim_ids = []

        for i, (section_title, section_text) in enumerate(sections):
            chunk = DocumentChunk(
                doc_id=doc_id,
                section=section_title,
                text=section_text,
                position=i,
                metadata={"source_url": source_url, "source_type": source_type},
            )
            cid = self.memory.add_chunk(chunk)
            chunk_ids.append(cid)

            # Extract simple claims from section.
            claims = self._extract_claims(section_text, source_url, source_type)
            for claim in claims:
                claim_id = self.memory.add_claim(claim)
                claim_ids.append(claim_id)

        return {
            "doc_id": doc_id,
            "chunks": len(chunk_ids),
            "claims": len(claim_ids),
        }

    def _split_sections(self, text: str) -> List[tuple[str, str]]:
        """Split text into (title, body) sections."""
        lines = text.split("\n")
        sections: List[tuple[str, str]] = []
        current_title = ""
        current_body: List[str] = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#") or (stripped.isupper() and len(stripped) > 3 and len(stripped) < 100):
                if current_body:
                    sections.append((current_title, "\n".join(current_body)))
                current_title = stripped.lstrip("#").strip()
                current_body = []
            else:
                current_body.append(line)

        if current_body:
            sections.append((current_title, "\n".join(current_body)))

        if not sections:
            sections = [("", text)]

        return sections

    def _extract_claims(
        self, text: str, source_url: str, source_type: str
    ) -> List[SourcedClaimCandidate]:
        """Extract simple factual claims from text. Intentionally conservative."""
        sentences = re.split(r"[.!?]\s+", text)
        claims = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10 or len(sent) > 500:
                continue
            if not re.search(r"\b(is|are|was|were|has|have|can|will)\b", sent, re.I):
                continue
            claims.append(
                SourcedClaimCandidate(
                    claim_text=sent,
                    evidence_span=sent,
                    source_url=source_url,
                    source_type=source_type,
                    confidence=0.25,
                )
            )
        return claims
