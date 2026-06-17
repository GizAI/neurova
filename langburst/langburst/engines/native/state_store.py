from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Iterable

from ...core.features import RuntimeFeatures


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
    """continuous-serving request-state table.

    The scheduler owns request rows and integer state indices. This store owns
    the actual adapter DecodeState objects behind those indices, so runners,
    future CUDA graph buckets, and paged KV code all consume the same boundary.
    """

    def __init__(
        self,
        *,
        engine: Any,
        features: RuntimeFeatures,
        max_slots: int | None = None,
        kv_num_blocks: int | None = None,
        kv_block_size: int | None = None,
    ) -> None:
        self.engine = engine
        self.features = features
        self._states: dict[int, Any] = {}
        self._arena = self._maybe_create_arena(max_slots=max_slots, kv_num_blocks=kv_num_blocks, kv_block_size=kv_block_size)
        self._state_to_slot: dict[int, int] = {}
        self._reuse_pool: list[Any] = []
        self._reuse_pool_size = int(os.environ.get("LANGBURST_STATE_REUSE_POOL_SIZE", "1"))

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
        elif self._reuse_pool:
            state = self._reuse_pool.pop()
            reset = getattr(state, "reset", None)
            if callable(reset):
                reset(reset_attention=True)
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
        elif state is not None and len(self._reuse_pool) < self._reuse_pool_size:
            reset = getattr(state, "reset", None)
            if callable(reset):
                reset(reset_attention=True)
            self._reuse_pool.append(state)
        return state

    def get(self, state_index: int) -> Any:
        idx = int(state_index)
        try:
            return self._states[idx]
        except KeyError as exc:
            raise KeyError(f"state_index is not allocated: {idx}") from exc

    def get_many(self, state_indices: Iterable[int]) -> list[Any]:
        return [self.get(int(idx)) for idx in state_indices]

    def physical_index(self, state_index: int) -> int:
        idx = int(state_index)
        if self._arena is None:
            return idx
        try:
            return int(self._state_to_slot[idx])
        except KeyError as exc:
            raise KeyError(f"state_index is not allocated in arena: {idx}") from exc

    def clear(self) -> None:
        if self._arena is not None:
            for slot in list(self._state_to_slot.values()):
                self._arena.release(slot)
            self._state_to_slot.clear()
        self._states.clear()
        self._reuse_pool.clear()

    def stats(self) -> StateStoreStats:
        return StateStoreStats(
            allocated_states=len(self._states),
            active_state_indices=tuple(sorted(self._states)),
        )

    def arena_summary(self) -> dict[str, object] | None:
        return self._arena.summary() if self._arena is not None else None

    def reuse_pool_summary(self) -> dict[str, int]:
        return {
            "reuse_pool_size": self._reuse_pool_size,
            "cached_states": len(self._reuse_pool),
        }

    def _maybe_create_arena(
        self,
        *,
        max_slots: int | None,
        kv_num_blocks: int | None,
        kv_block_size: int | None,
    ) -> Any | None:
        slots = int(max_slots or 0)
        if slots < 1:
            return None
        arena_mode = os.environ.get("LANGBURST_BATCH_STATE_ARENA", "auto").strip().lower()
        if arena_mode in {"0", "false", "off", "no"}:
            return None
        if (
            arena_mode == "auto"
            and slots == 1
            and not self.features.speculative_decoding
        ):
            # Single-request serving gets batch worker queueing/streaming without
            # taking the paged-arena hot path. The canonical state path is the
            # quality champion and matches plain RuntimeEngine generation; paged
            # arena remains available for explicit multi-slot experiments.
            #
            # Speculative verification is different: even a single request must
            # use the arena/paged contract so target verify can commit the
            # reducer-selected prefix without replay or rollback.
            return None
        create_arena = getattr(self.engine, "create_state_arena", None)
        if not callable(create_arena):
            return None
        return create_arena(
            features=self.features,
            max_slots=slots,
            kv_num_blocks=kv_num_blocks,
            kv_block_size=kv_block_size,
        )
