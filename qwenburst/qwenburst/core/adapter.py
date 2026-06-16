from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import torch
from .features import RuntimeCapabilities, RuntimeFeatures


@dataclass(frozen=True)
class AdapterDescriptor:
    adapter_id: str
    family: str
    default_model_name: str
    capabilities: RuntimeCapabilities = field(default_factory=RuntimeCapabilities)
    supports_state: bool = True
    supports_mtp: bool = False


class ModelAdapter(Protocol):
    descriptor: AdapterDescriptor

    def load_config(self, hf_model: Path) -> Any: ...

    def load_tokenizer(self, hf_model: Path) -> Any: ...

    def create_model(
        self,
        *,
        qb_model: Path,
        cfg: Any,
        device: str,
        weight_device: str,
        cpu_embed: bool = False,
    ) -> Any: ...

    def allocate_state(self, cfg: Any, *, recent_window: int, device: str, features: RuntimeFeatures) -> Any: ...

    def encode_messages(self, tokenizer: Any, messages: Sequence[dict[str, Any]]) -> list[int]: ...

    def encode_prompt(self, tokenizer: Any, prompt: str, system: str | None = None) -> list[int]: ...

    def eos_token_ids(self, tokenizer: Any) -> tuple[int, ...]: ...


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, ModelAdapter] = {}

    def register(self, adapter: ModelAdapter) -> None:
        adapter_id = adapter.descriptor.adapter_id
        if adapter_id in self._adapters:
            raise ValueError(f"adapter already registered: {adapter_id}")
        self._adapters[adapter_id] = adapter

    def get(self, adapter_id: str) -> ModelAdapter:
        try:
            return self._adapters[adapter_id]
        except KeyError as exc:
            known = ", ".join(sorted(self._adapters)) or "<none>"
            raise KeyError(f"unknown adapter {adapter_id!r}; known adapters: {known}") from exc

    def list(self) -> list[AdapterDescriptor]:
        return [a.descriptor for a in self._adapters.values()]


adapter_registry = AdapterRegistry()
