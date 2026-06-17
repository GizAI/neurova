from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_left
from collections.abc import Callable

import torch


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


@dataclass(frozen=True)
class CudaGraphKey:
    batch_size: int
    query_len: int
    speculative_tokens: int
    context_bucket: int
    buffer_signature: tuple[int, ...] = ()


class CudaGraphExecutable:
    """Captured CUDA graph around a static-buffer decode callable.

    The callable must close over stable input/output tensors and mutate those
    tensors in place. This class deliberately has no tensor allocation or shape
    padding logic; callers must prepare fixed-shape buffers before capture.
    """

    def __init__(self, *, key: CudaGraphKey, replay: Callable[[], None]) -> None:
        self.key = key
        self._replay = replay

    def replay(self) -> None:
        self._replay()


class CudaGraphReplayCache:
    """Small capture/replay registry for fixed-shape native hot paths."""

    def __init__(self, *, enabled: bool = True, warmup_steps: int = 3) -> None:
        self.enabled = bool(enabled)
        self.warmup_steps = max(0, int(warmup_steps))
        self._graphs: dict[CudaGraphKey, CudaGraphExecutable] = {}

    def get(self, key: CudaGraphKey) -> CudaGraphExecutable | None:
        return self._graphs.get(key)

    def disable(self) -> None:
        self.enabled = False
        self._graphs.clear()

    def capture(
        self,
        key: CudaGraphKey,
        fn: Callable[[], None],
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> CudaGraphExecutable:
        if not self.enabled:
            raise RuntimeError("CUDA graph cache is disabled")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA graph capture requires CUDA")
        if key in self._graphs:
            return self._graphs[key]

        capture_stream = stream or torch.cuda.Stream()
        current = torch.cuda.current_stream()
        capture_stream.wait_stream(current)
        with torch.cuda.stream(capture_stream):
            for _ in range(self.warmup_steps):
                fn()
        current.wait_stream(capture_stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        executable = CudaGraphExecutable(key=key, replay=graph.replay)
        self._graphs[key] = executable
        return executable

    def replay_or_capture(self, key: CudaGraphKey, fn: Callable[[], None]) -> None:
        graph = self.get(key)
        if graph is None:
            graph = self.capture(key, fn)
        graph.replay()
