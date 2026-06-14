from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class EvidencePointer:
    source: str
    byte_start: int
    byte_end: int


@dataclass(frozen=True)
class LedgerPage:
    page_id: str
    summary: str
    evidence: EvidencePointer
    embedding_key: list[float]
    timestamp: float
    source_confidence: float = 1.0
    contradiction_hash: str = ""


class JsonlLedger:
    """Append-only raw-byte evidence ledger for proof memory experiments."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        *,
        summary: str,
        evidence: EvidencePointer,
        embedding_key: Iterable[float],
        source_confidence: float = 1.0,
        contradiction_hash: str = "",
    ) -> LedgerPage:
        page = LedgerPage(
            page_id=f"{int(time.time() * 1_000_000)}-{evidence.byte_start}-{evidence.byte_end}",
            summary=summary,
            evidence=evidence,
            embedding_key=[float(x) for x in embedding_key],
            timestamp=time.time(),
            source_confidence=float(source_confidence),
            contradiction_hash=contradiction_hash,
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self._to_json(page), ensure_ascii=False) + "\n")
        return page

    def pages(self) -> list[LedgerPage]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(self._from_json(json.loads(line)))
        return out

    @staticmethod
    def _to_json(page: LedgerPage) -> dict:
        row = asdict(page)
        row["evidence"] = asdict(page.evidence)
        return row

    @staticmethod
    def _from_json(row: dict) -> LedgerPage:
        return LedgerPage(
            page_id=row["page_id"],
            summary=row["summary"],
            evidence=EvidencePointer(**row["evidence"]),
            embedding_key=[float(x) for x in row.get("embedding_key", [])],
            timestamp=float(row["timestamp"]),
            source_confidence=float(row.get("source_confidence", 1.0)),
            contradiction_hash=row.get("contradiction_hash", ""),
        )
