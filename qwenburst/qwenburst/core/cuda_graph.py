from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_left


@dataclass(frozen=True)
class CudaGraphBucket:
    batch_size: int
    query_len: int
    speculative_tokens: int = 0

    @property
    def target_tokens_per_request(self) -> int:
        return self.query_len + self.speculative_tokens


class CudaGraphBucketPlanner:
    """Static shape planner for decode/spec-decode graph capture."""

    def __init__(self, *, batch_sizes: tuple[int, ...] = (1, 2, 4, 8), query_lens: tuple[int, ...] = (1, 2, 4, 8)) -> None:
        if not batch_sizes or not query_lens:
            raise ValueError("bucket lists must not be empty")
        self.batch_sizes = tuple(sorted(set(int(v) for v in batch_sizes)))
        self.query_lens = tuple(sorted(set(int(v) for v in query_lens)))
        if self.batch_sizes[0] < 1 or self.query_lens[0] < 1:
            raise ValueError("bucket sizes must be positive")

    def select(self, *, batch_size: int, query_len: int, speculative_tokens: int = 0) -> CudaGraphBucket:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if query_len < 1:
            raise ValueError("query_len must be >= 1")
        return CudaGraphBucket(
            batch_size=self._ceil(self.batch_sizes, batch_size, "batch_size"),
            query_len=self._ceil(self.query_lens, query_len, "query_len"),
            speculative_tokens=max(0, int(speculative_tokens)),
        )

    @staticmethod
    def _ceil(values: tuple[int, ...], target: int, name: str) -> int:
        idx = bisect_left(values, int(target))
        if idx >= len(values):
            raise ValueError(f"{name}={target} exceeds largest CUDA graph bucket {values[-1]}")
        return values[idx]
