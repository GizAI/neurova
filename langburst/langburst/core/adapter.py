from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Protocol, Sequence
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

    def create_state_arena(
        self,
        cfg: Any,
        *,
        max_seq_len: int,
        num_slots: int,
        device: str,
        features: RuntimeFeatures,
        kv_num_blocks: int | None = None,
        kv_block_size: int | None = None,
    ) -> Any | None: ...

    def encode_messages(self, tokenizer: Any, messages: Sequence[dict[str, Any]], **kwargs: Any) -> list[int]: ...

    def encode_prompt(self, tokenizer: Any, prompt: str, system: str | None = None) -> list[int]: ...

    def eos_token_ids(self, tokenizer: Any) -> tuple[int, ...]: ...

    def create_speculative_proposer(self, model: Any) -> Any | None: ...


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

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def load_entry_points(self, group: str = "langburst.adapters") -> None:
        """Load optional third-party adapters registered as package entry points.

        Entry points may expose either a ModelAdapter instance, a ModelAdapter
        class, or a callable returning one adapter or an iterable of adapters.
        """

        for ep in entry_points(group=group):
            contribution = ep.load()
            if isinstance(contribution, type):
                contribution = contribution()
            elif callable(contribution) and not hasattr(contribution, "descriptor"):
                contribution = contribution()
            adapters = contribution if isinstance(contribution, (list, tuple)) else (contribution,)
            for adapter in adapters:
                self.register(adapter)


adapter_registry = AdapterRegistry()
