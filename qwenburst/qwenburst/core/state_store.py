from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .features import RuntimeFeatures
from ..state import DecodeStateArena


@dataclass(frozen=True)
class StateStoreStats:
    allocated_states: int
    active_state_indices: tuple[int, ...]

    def summary(self) -> dict[str, object]:
        return {
            "allocated_states": self.allocated_states,
            "active_state_indices": list(self.active_state_indices),
        }


class BatchStateStore:
    """vLLM-style request-state table.

    The scheduler owns request rows and integer state indices. This store owns
    the actual adapter DecodeState objects behind those indices, so runners,
    future CUDA graph buckets, and paged KV code all consume the same boundary.
    """

    def __init__(self, *, engine: Any, features: RuntimeFeatures, max_slots: int | None = None) -> None:
        self.engine = engine
        self.features = features
        self._states: dict[int, Any] = {}
        self._arena = self._maybe_create_arena(max_slots=max_slots)
        self._state_to_slot: dict[int, int] = {}

    @property
    def states(self) -> dict[int, Any]:
        return self._states

    def allocate(self, state_index: int) -> Any:
        idx = int(state_index)
        if idx in self._states:
            raise ValueError(f"state_index already allocated: {idx}")
        if self._arena is not None:
            slot, state = self._arena.allocate()
            self._state_to_slot[idx] = slot
        else:
            state = self.engine.new_state(self.features)
        self._states[idx] = state
        return state

    def release(self, state_index: int) -> Any | None:
        idx = int(state_index)
        state = self._states.pop(idx, None)
        slot = self._state_to_slot.pop(idx, None)
        if slot is not None and self._arena is not None:
            self._arena.release(slot)
        return state

    def get(self, state_index: int) -> Any:
        idx = int(state_index)
        try:
            return self._states[idx]
        except KeyError as exc:
            raise KeyError(f"state_index is not allocated: {idx}") from exc

    def get_many(self, state_indices: Iterable[int]) -> list[Any]:
        return [self.get(int(idx)) for idx in state_indices]

    def clear(self) -> None:
        if self._arena is not None:
            for slot in list(self._state_to_slot.values()):
                self._arena.release(slot)
            self._state_to_slot.clear()
        self._states.clear()

    def stats(self) -> StateStoreStats:
        return StateStoreStats(
            allocated_states=len(self._states),
            active_state_indices=tuple(sorted(self._states)),
        )

    def arena_summary(self) -> dict[str, int] | None:
        return self._arena.summary() if self._arena is not None else None

    def _maybe_create_arena(self, *, max_slots: int | None) -> DecodeStateArena | None:
        cfg = getattr(self.engine, "cfg", None)
        if cfg is None or not hasattr(cfg, "gdn_layers") or not hasattr(cfg, "attention_layers"):
            return None
        slots = int(max_slots or 0)
        if slots < 1:
            return None
        return DecodeStateArena(
            cfg=cfg,
            max_seq_len=int(getattr(self.engine, "recent_window")),
            num_slots=slots,
            device=getattr(self.engine, "device"),
            kv_window_policy=self.features.kv_window_policy,
        )
