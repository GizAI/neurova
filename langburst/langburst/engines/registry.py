from __future__ import annotations

from importlib.metadata import entry_points
from typing import Iterable

from .base import EngineBackend, EngineDescriptor, EngineKind, EngineModelSpec, EngineProvider


class EngineRegistry:
    """Single extension seam for serving engines.

    Providers can be built in or exposed by third-party packages through the
    `langburst.engines` entry-point group. Downstream server/CLI code resolves
    one provider here and never branches on vLLM/SGLang/EXL3/native directly.
    """

    def __init__(self) -> None:
        self._providers: dict[str, EngineProvider] = {}

    def register(self, provider: EngineProvider) -> None:
        engine_id = provider.descriptor.engine_id
        if engine_id in self._providers:
            raise ValueError(f"engine already registered: {engine_id}")
        self._providers[engine_id] = provider

    def get(self, engine_id: str | None = None) -> EngineProvider:
        resolved = engine_id or self.default_engine_id()
        try:
            return self._providers[resolved]
        except KeyError as exc:
            known = ", ".join(sorted(self._providers)) or "<none>"
            raise KeyError(f"unknown engine {resolved!r}; known engines: {known}") from exc

    def create(self, spec: EngineModelSpec, *, engine_id: str | None = None) -> EngineBackend:
        return self.get(engine_id).create(spec)

    def list(self) -> list[EngineDescriptor]:
        return [provider.descriptor for provider in self._providers.values()]

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))

    def default_engine_id(self) -> EngineKind:
        for provider in self._providers.values():
            if provider.descriptor.default:
                return provider.descriptor.engine_id
        if "vllm" in self._providers:
            return "vllm"
        try:
            return next(iter(self._providers))  # type: ignore[return-value]
        except StopIteration as exc:
            raise RuntimeError("no LangBurst engines are registered") from exc

    def load_entry_points(self, group: str = "langburst.engines") -> None:
        for ep in entry_points(group=group):
            contribution = ep.load()
            if isinstance(contribution, type):
                contribution = contribution()
            elif callable(contribution) and not hasattr(contribution, "descriptor"):
                contribution = contribution()
            providers: Iterable[EngineProvider] = (
                contribution if isinstance(contribution, (list, tuple)) else (contribution,)
            )
            for provider in providers:
                if provider.descriptor.engine_id in self._providers:
                    continue
                self.register(provider)


engine_registry = EngineRegistry()
