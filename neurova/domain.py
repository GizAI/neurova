from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


@dataclass
class DomainShard:
    name: str
    confidence: float
    reason: str


class DomainShardRouter:
    """Routes language into specialized shard policies without using an LLM."""
    SHARDS: Dict[str, tuple[str, ...]] = {
        "code": ("code", "python", "function", "test", "bug", "구현", "코드"),
        "policy": ("policy", "contract", "rule", "terms", "must", "shall", "정책", "계약"),
        "ops_log": ("error", "log", "trace", "deploy", "incident", "root cause", "장애", "로그"),
        "quant": ("return", "sharpe", "cagr", "mdd", "portfolio", "factor", "퀀트", "수익률"),
        "world": ("state=", "action=", "world:", "next=", "causes", "happens", "상태"),
        "conversation": ("why", "what", "explain", "?", "왜", "무엇"),
    }

    def route(self, text: str) -> DomainShard:
        low = text.lower()
        best = ("conversation", 0, "default")
        for name, keys in self.SHARDS.items():
            hits = [k for k in keys if k in low]
            score = len(hits) / max(1, len(keys))
            if score > best[1]:
                best = (name, score, ",".join(hits))
        conf = min(0.95, 0.35 + 2.0 * best[1]) if best[1] else 0.35
        return DomainShard(best[0], conf, best[2])
