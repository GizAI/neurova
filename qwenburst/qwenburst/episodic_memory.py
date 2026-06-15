from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import hashlib

import torch
import torch.nn.functional as F


@dataclass
class MemoryRecord:
    record_id: str
    text: str
    embedding: torch.Tensor
    state_delta_path: str | None = None
    start_pos: int = 0
    end_pos: int = 0
    score: float = 0.0


class HashEmbedding:
    """Dependency-free lexical embedding fallback for tests/offline demos.

    Production should replace this with a real embedding model. Keeping this
    tiny embedder in-tree lets state-RAG tests run without internet or GPUs.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, text: str) -> torch.Tensor:
        vec = torch.zeros(self.dim, dtype=torch.float32)
        for raw in text.lower().replace("/", " ").replace("_", " ").split():
            token = raw.strip(".,:;()[]{}<>!?\"'")
            if not token:
                continue
            h = int(hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest(), 16)
            idx = h % self.dim
            sign = -1.0 if (h >> 63) & 1 else 1.0
            vec[idx] += sign
        return F.normalize(vec, dim=0) if vec.norm() > 0 else vec


class EpisodicMemory:
    """Small local state-RAG index.

    Each record can point to a DecodeState snapshot or state-delta file. The
    engine retrieves text spans for exact recall and optionally reloads/merges
    state snapshots for model-native compressed context.
    """

    def __init__(self, embedder: HashEmbedding | None = None):
        self.embedder = embedder or HashEmbedding()
        self.records: list[MemoryRecord] = []

    def add_text(
        self,
        text: str,
        *,
        record_id: str | None = None,
        state_delta_path: str | None = None,
        start_pos: int = 0,
        end_pos: int = 0,
    ) -> MemoryRecord:
        record_id = record_id or hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
        rec = MemoryRecord(
            record_id=record_id,
            text=text,
            embedding=self.embedder.encode(text),
            state_delta_path=state_delta_path,
            start_pos=start_pos,
            end_pos=end_pos,
        )
        self.records.append(rec)
        return rec

    def search(self, query: str, *, top_k: int = 4) -> list[MemoryRecord]:
        if top_k < 1:
            return []
        q = self.embedder.encode(query)
        scored: list[MemoryRecord] = []
        for rec in self.records:
            score = float(torch.dot(q, rec.embedding).item()) if q.norm() > 0 and rec.embedding.norm() > 0 else 0.0
            scored.append(MemoryRecord(**{**rec.__dict__, "score": score}))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dim": self.embedder.dim,
            "records": [
                {
                    "record_id": r.record_id,
                    "text": r.text,
                    "embedding": r.embedding.tolist(),
                    "state_delta_path": r.state_delta_path,
                    "start_pos": r.start_pos,
                    "end_pos": r.end_pos,
                }
                for r in self.records
            ],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "EpisodicMemory":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        mem = cls(HashEmbedding(dim=int(payload["dim"])))
        for item in payload["records"]:
            mem.records.append(
                MemoryRecord(
                    record_id=item["record_id"],
                    text=item["text"],
                    embedding=torch.tensor(item["embedding"], dtype=torch.float32),
                    state_delta_path=item.get("state_delta_path"),
                    start_pos=int(item.get("start_pos", 0)),
                    end_pos=int(item.get("end_pos", 0)),
                )
            )
        return mem
