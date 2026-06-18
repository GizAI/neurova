from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch


@dataclass(frozen=True)
class KVBlockRef:
    block_id: int
    offset: int


@dataclass
class RequestBlockTable:
    request_id: str
    block_ids: list[int] = field(default_factory=list)

    def logical_to_physical(self, token_pos: int, *, block_size: int) -> KVBlockRef:
        if token_pos < 0:
            raise ValueError("token_pos must be >= 0")
        token_pos = int(token_pos)
        block_size = int(block_size)
        capacity = len(self.block_ids) * block_size
        if capacity > 0:
            token_pos %= capacity
        block_index, offset = divmod(token_pos, block_size)
        if block_index >= len(self.block_ids):
            raise IndexError("token position is not allocated")
        return KVBlockRef(block_id=self.block_ids[block_index], offset=offset)


class KVBlockTable:
    """Paged KV block allocator.

    The table is model-agnostic.  Attention kernels consume physical block IDs
    and offsets; schedulers mutate request tables as sequences grow.
    """

    def __init__(self, *, num_blocks: int, block_size: int) -> None:
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        self._free_blocks = list(range(self.num_blocks - 1, -1, -1))
        self._refcounts: dict[int, int] = {}
        self._tables: dict[str, RequestBlockTable] = {}

    @property
    def free_block_count(self) -> int:
        return len(self._free_blocks)

    @property
    def used_block_count(self) -> int:
        return self.num_blocks - len(self._free_blocks)

    def get(self, request_id: str) -> RequestBlockTable:
        return self._tables.setdefault(request_id, RequestBlockTable(request_id=request_id))

    def _acquire_block(self) -> int:
        if not self._free_blocks:
            raise MemoryError("KV block table exhausted")
        block_id = self._free_blocks.pop()
        self._refcounts[block_id] = 1
        return block_id

    def _incref_block(self, block_id: int) -> None:
        block_id = int(block_id)
        if block_id < 0 or block_id >= self.num_blocks:
            raise ValueError("block_id out of range")
        if block_id not in self._refcounts:
            raise ValueError(f"block_id is not allocated: {block_id}")
        self._refcounts[block_id] += 1

    def _decref_block(self, block_id: int) -> None:
        block_id = int(block_id)
        count = self._refcounts.get(block_id, 0)
        if count <= 0:
            return
        if count == 1:
            del self._refcounts[block_id]
            self._free_blocks.append(block_id)
            return
        self._refcounts[block_id] = count - 1

    def ensure_tokens(self, request_id: str, token_count: int) -> RequestBlockTable:
        if token_count < 0:
            raise ValueError("token_count must be >= 0")
        table = self.get(request_id)
        required_blocks = (int(token_count) + self.block_size - 1) // self.block_size
        while len(table.block_ids) < required_blocks:
            table.block_ids.append(self._acquire_block())
        return table

    def truncate_tokens(self, request_id: str, token_count: int) -> RequestBlockTable:
        """Release speculative blocks beyond the committed token length."""

        if token_count < 0:
            raise ValueError("token_count must be >= 0")
        table = self.get(request_id)
        required_blocks = (int(token_count) + self.block_size - 1) // self.block_size
        while len(table.block_ids) > required_blocks:
            self._decref_block(table.block_ids.pop())
        return table

    def attach_prefix_blocks(self, request_id: str, block_ids: Sequence[int]) -> RequestBlockTable:
        """Attach immutable cached prefix blocks to a new request table."""

        table = self.get(request_id)
        if table.block_ids:
            raise ValueError("request already has KV blocks")
        for block_id in block_ids:
            self._incref_block(int(block_id))
            table.block_ids.append(int(block_id))
        return table

    def reset_to_prefix_blocks(self, request_id: str, block_ids: Sequence[int]) -> RequestBlockTable:
        """Replace a request's uncomputed blocks with cached immutable prefix blocks."""

        table = self.get(request_id)
        for block_id in reversed(table.block_ids):
            self._decref_block(block_id)
        table.block_ids.clear()
        for block_id in block_ids:
            self._incref_block(int(block_id))
            table.block_ids.append(int(block_id))
        return table

    def pin_prefix_blocks(self, request_id: str, token_count: int) -> tuple[int, ...]:
        """Pin full prefix blocks for cache ownership and return block IDs."""

        if token_count < 0:
            raise ValueError("token_count must be >= 0")
        full_blocks = int(token_count) // self.block_size
        table = self.get(request_id)
        if full_blocks > len(table.block_ids):
            raise IndexError("prefix token_count is not fully allocated")
        out = tuple(table.block_ids[:full_blocks])
        for block_id in out:
            self._incref_block(block_id)
        return out

    def release_pinned_blocks(self, block_ids: Sequence[int]) -> None:
        for block_id in block_ids:
            self._decref_block(int(block_id))

    def release(self, request_id: str) -> int:
        table = self._tables.pop(request_id, None)
        if table is None:
            return 0
        released = len(table.block_ids)
        for block_id in reversed(table.block_ids):
            self._decref_block(block_id)
        return released

    def clear(self) -> None:
        self._tables.clear()
        self._refcounts.clear()
        self._free_blocks = list(range(self.num_blocks - 1, -1, -1))

    def block_table_tensor(self, request_ids: list[str], *, device: torch.device | str = "cpu") -> torch.Tensor:
        max_blocks = max((len(self.get(request_id).block_ids) for request_id in request_ids), default=0)
        out = torch.zeros((len(request_ids), max_blocks), dtype=torch.int32, device=device)
        for row, request_id in enumerate(request_ids):
            blocks = self.get(request_id).block_ids
            if blocks:
                out[row, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32, device=device)
        return out

    def slot_mapping_tensor(
        self,
        request_ids: list[str],
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        *,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        out = torch.empty((int(positions.numel()),), dtype=torch.long, device=device)
        for row, request_id in enumerate(request_ids):
            start = int(query_start_loc[row].detach().cpu().item())
            end = int(query_start_loc[row + 1].detach().cpu().item())
            table = self.get(request_id)
            for idx in range(start, end):
                pos = int(positions[idx].detach().cpu().item())
                ref = table.logical_to_physical(pos, block_size=self.block_size)
                out[idx] = ref.block_id * self.block_size + ref.offset
        return out

    def summary(self) -> dict[str, int]:
        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "used_blocks": self.used_block_count,
            "free_blocks": self.free_block_count,
            "requests": len(self._tables),
            "pinned_or_shared_blocks": sum(1 for count in self._refcounts.values() if count > 1),
        }
