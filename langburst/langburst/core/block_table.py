from __future__ import annotations

from dataclasses import dataclass, field

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
        block_index, offset = divmod(int(token_pos), int(block_size))
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
        self._tables: dict[str, RequestBlockTable] = {}

    @property
    def free_block_count(self) -> int:
        return len(self._free_blocks)

    @property
    def used_block_count(self) -> int:
        return self.num_blocks - len(self._free_blocks)

    def get(self, request_id: str) -> RequestBlockTable:
        return self._tables.setdefault(request_id, RequestBlockTable(request_id=request_id))

    def ensure_tokens(self, request_id: str, token_count: int) -> RequestBlockTable:
        if token_count < 0:
            raise ValueError("token_count must be >= 0")
        table = self.get(request_id)
        required_blocks = (int(token_count) + self.block_size - 1) // self.block_size
        while len(table.block_ids) < required_blocks:
            if not self._free_blocks:
                raise MemoryError("KV block table exhausted")
            table.block_ids.append(self._free_blocks.pop())
        return table

    def release(self, request_id: str) -> int:
        table = self._tables.pop(request_id, None)
        if table is None:
            return 0
        released = len(table.block_ids)
        self._free_blocks.extend(reversed(table.block_ids))
        return released

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
        }
