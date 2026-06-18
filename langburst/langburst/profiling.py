from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import atexit
import os
import time
from typing import Iterator


class _DecodeProfiler:
    def __init__(self) -> None:
        self.calls: dict[str, int] = defaultdict(int)
        self.total_ms: dict[str, float] = defaultdict(float)
        self._registered_atexit = False

    def enabled(self) -> bool:
        raw = os.environ.get("LANGBURST_DECODE_PROFILE", "0").strip().lower()
        return raw in {"1", "true", "on", "yes"}

    def record(self, name: str, elapsed_ms: float, calls: int = 1) -> None:
        if not name:
            return
        self.calls[name] += int(calls)
        self.total_ms[name] += float(elapsed_ms)

    def snapshot(self, *, reset: bool = False) -> dict[str, dict[str, float | int]]:
        total = sum(self.total_ms.values()) or 0.0
        rows: dict[str, dict[str, float | int]] = {}
        for name in sorted(self.total_ms, key=lambda key: self.total_ms[key], reverse=True):
            calls = int(self.calls.get(name, 0))
            ms = float(self.total_ms[name])
            rows[name] = {
                "calls": calls,
                "total_ms": ms,
                "avg_us": (ms * 1000.0 / calls) if calls > 0 else 0.0,
                "pct_measured": (ms * 100.0 / total) if total > 0.0 else 0.0,
            }
        if reset:
            self.calls.clear()
            self.total_ms.clear()
        return rows

    def format(self, *, reset: bool = False) -> str:
        rows = self.snapshot(reset=reset)
        if not rows:
            return "category            calls   total_ms   avg_us   pct_measured\n"
        lines = ["category            calls   total_ms   avg_us   pct_measured"]
        for name, row in rows.items():
            lines.append(
                f"{name:<18} {int(row['calls']):5d} "
                f"{float(row['total_ms']):9.3f} {float(row['avg_us']):8.2f} {float(row['pct_measured']):8.2f}"
            )
        return "\n".join(lines)

    def maybe_register_atexit(self) -> None:
        if self._registered_atexit:
            return
        raw = os.environ.get("LANGBURST_DECODE_PROFILE_PRINT", "0").strip().lower()
        if raw not in {"1", "true", "on", "yes"}:
            return
        self._registered_atexit = True
        atexit.register(lambda: print(self.format(reset=False)))


_PROFILER = _DecodeProfiler()


def decode_profile_enabled() -> bool:
    enabled = _PROFILER.enabled()
    if enabled:
        _PROFILER.maybe_register_atexit()
    return enabled


@contextmanager
def decode_profile_scope(name: str | None) -> Iterator[None]:
    if not name or not decode_profile_enabled():
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        _PROFILER.record(name, (time.perf_counter() - start) * 1000.0)


def decode_profile_mark(name: str | None, *, elapsed_ms: float = 0.0, calls: int = 1) -> None:
    if not name or not decode_profile_enabled():
        return
    _PROFILER.record(name, float(elapsed_ms), calls=int(calls))


def decode_profile_snapshot(*, reset: bool = False) -> dict[str, dict[str, float | int]]:
    return _PROFILER.snapshot(reset=reset)


def decode_profile_format(*, reset: bool = False) -> str:
    return _PROFILER.format(reset=reset)
