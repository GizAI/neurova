from __future__ import annotations

from dataclasses import dataclass, field
import time
import threading
import uuid
from typing import Any

from ...core.features import RuntimeFeatures


def _features_key(features: RuntimeFeatures) -> tuple[tuple[str, object], ...]:
    return tuple(sorted(features.summary().items()))


@dataclass
class SessionStateRecord:
    session_id: str
    model_name: str
    features_key: tuple[tuple[str, object], ...]
    state: Any
    created_monotonic: float = field(default_factory=time.monotonic)
    last_used_monotonic: float = field(default_factory=time.monotonic)
    prompt_tokens: int = 0
    generated_tokens: int = 0
    turns: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock)

    def touch(self) -> None:
        self.last_used_monotonic = time.monotonic()

    def summary(self, now: float | None = None) -> dict[str, object]:
        now = time.monotonic() if now is None else now
        return {
            "session_id": self.session_id,
            "model_name": self.model_name,
            "age_s": max(0.0, now - self.created_monotonic),
            "idle_s": max(0.0, now - self.last_used_monotonic),
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "turns": self.turns,
        }


class SessionStateStore:
    """Explicit conversation-state store.

    This is intentionally separate from state pooling. The state pool recycles
    resettable objects for stateless requests; this store preserves model state
    across requests only when the caller supplies an explicit session id.
    """

    def __init__(self, *, max_sessions: int = 16, ttl_s: float | None = 3600.0) -> None:
        if max_sessions < 0:
            raise ValueError("max_sessions must be >= 0")
        if ttl_s is not None and ttl_s <= 0:
            raise ValueError("ttl_s must be positive when set")
        self.max_sessions = int(max_sessions)
        self.ttl_s = float(ttl_s) if ttl_s is not None else None
        self._records: dict[tuple[str, str, tuple[tuple[str, object], ...]], SessionStateRecord] = {}
        self._lock = threading.RLock()

    def new_session_id(self) -> str:
        return f"sess-{uuid.uuid4().hex}"

    def get_or_create(self, *, session_id: str, engine: Any, features: RuntimeFeatures) -> SessionStateRecord:
        if not session_id:
            raise ValueError("session_id must not be empty")
        if self.max_sessions == 0:
            raise RuntimeError("stateful sessions are disabled")
        key = (engine.model_name, str(session_id), _features_key(features))
        with self._lock:
            self.evict_expired()
            record = self._records.get(key)
            if record is not None:
                record.touch()
                return record
            self._evict_until_capacity_locked(reserve=1)
            record = SessionStateRecord(
                session_id=str(session_id),
                model_name=engine.model_name,
                features_key=key[2],
                state=engine.new_state(features),
            )
            self._records[key] = record
            return record

    def delete(self, session_id: str, *, model_name: str | None = None) -> int:
        with self._lock:
            keys = [
                key
                for key in self._records
                if key[1] == session_id and (model_name is None or key[0] == model_name)
            ]
            for key in keys:
                del self._records[key]
            return len(keys)

    def delete_model(self, model_name: str) -> int:
        with self._lock:
            keys = [key for key in self._records if key[0] == model_name]
            for key in keys:
                del self._records[key]
            return len(keys)

    def clear(self) -> int:
        with self._lock:
            n = len(self._records)
            self._records.clear()
            return n

    def evict_expired(self) -> int:
        if self.ttl_s is None:
            return 0
        now = time.monotonic()
        with self._lock:
            keys = [
                key
                for key, record in self._records.items()
                if now - record.last_used_monotonic > float(self.ttl_s)
            ]
            for key in keys:
                del self._records[key]
            return len(keys)

    def summary(self) -> dict[str, object]:
        with self._lock:
            self.evict_expired()
            now = time.monotonic()
            rows = [record.summary(now) for record in self._records.values()]
            return {
                "max_sessions": self.max_sessions,
                "ttl_s": self.ttl_s,
                "active_sessions": len(rows),
                "sessions": sorted(rows, key=lambda row: (str(row["model_name"]), str(row["session_id"]))),
            }

    def _evict_until_capacity_locked(self, *, reserve: int) -> None:
        while len(self._records) + reserve > self.max_sessions:
            if not self._records:
                break
            victim_key = min(self._records.items(), key=lambda item: item[1].last_used_monotonic)[0]
            del self._records[victim_key]
