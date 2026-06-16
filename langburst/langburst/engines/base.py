from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Protocol, Sequence


EngineKind = Literal["vllm", "sglang", "exl3", "native"]
FeatureSupport = Literal["native", "engine", "host", "bridge", "unsupported"]


@dataclass(frozen=True)
class EngineCapabilities:
    """What an engine provider can execute without LangBurst reimplementing it."""

    openai_chat: bool = True
    streaming: bool = True
    continuous_batching: bool = False
    paged_kv: bool = False
    prefix_cache: bool = False
    structured_output: bool = False
    speculative_decoding: bool = False
    stateful_sessions: bool = False
    ring_kv: bool = False
    recurrent_state: bool = False
    infinite_context: bool = False
    episodic_memory: bool = False
    ttt_sidecar: bool = False
    qwen36_lowbit: bool = False
    custom_model: bool = False
    host_state: bool = False
    quantization: tuple[str, ...] = ()

    def summary(self) -> dict[str, object]:
        return {
            "openai_chat": self.openai_chat,
            "streaming": self.streaming,
            "continuous_batching": self.continuous_batching,
            "paged_kv": self.paged_kv,
            "prefix_cache": self.prefix_cache,
            "structured_output": self.structured_output,
            "speculative_decoding": self.speculative_decoding,
            "stateful_sessions": self.stateful_sessions,
            "ring_kv": self.ring_kv,
            "recurrent_state": self.recurrent_state,
            "infinite_context": self.infinite_context,
            "episodic_memory": self.episodic_memory,
            "ttt_sidecar": self.ttt_sidecar,
            "qwen36_lowbit": self.qwen36_lowbit,
            "custom_model": self.custom_model,
            "host_state": self.host_state,
            "quantization": list(self.quantization),
        }


@dataclass(frozen=True)
class EngineDescriptor:
    engine_id: EngineKind
    display_name: str
    module: str
    default: bool = False
    capabilities: EngineCapabilities = field(default_factory=EngineCapabilities)

    def summary(self) -> dict[str, object]:
        return {
            "engine_id": self.engine_id,
            "display_name": self.display_name,
            "module": self.module,
            "default": self.default,
            "capabilities": self.capabilities.summary(),
        }


@dataclass(frozen=True)
class EngineModelSpec:
    """Normalized model declaration consumed by every engine provider."""

    model: str
    served_model_name: str | None = None
    tokenizer: str | None = None
    dtype: str = "auto"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float | None = None
    max_model_len: int | None = None
    quantization: str | None = None
    trust_remote_code: bool = True
    features: "EngineFeatureRequest" = field(default_factory=lambda: EngineFeatureRequest())
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def public_name(self) -> str:
        return self.served_model_name or self.model


@dataclass(frozen=True)
class EngineFeatureRequest:
    """LangBurst-specific feature request shared by all providers."""

    qwen36_lowbit: bool = False
    stateful_sessions: bool = False
    ring_kv: bool = False
    recurrent_state: bool = False
    infinite_context: bool = False
    episodic_memory: bool = False
    ttt_sidecar: bool = False

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "EngineFeatureRequest":
        data = data or {}
        return cls(
            qwen36_lowbit=bool(data.get("qwen36_lowbit", False)),
            stateful_sessions=bool(data.get("stateful_sessions", data.get("stateful_chat", False))),
            ring_kv=bool(data.get("ring_kv", data.get("kv_window_policy") == "ring")),
            recurrent_state=bool(data.get("recurrent_state", False)),
            infinite_context=bool(data.get("infinite_context", data.get("infinite_streaming", False))),
            episodic_memory=bool(data.get("episodic_memory", False)),
            ttt_sidecar=bool(data.get("ttt_sidecar", False)),
        )

    def summary(self) -> dict[str, bool]:
        return {
            "qwen36_lowbit": self.qwen36_lowbit,
            "stateful_sessions": self.stateful_sessions,
            "ring_kv": self.ring_kv,
            "recurrent_state": self.recurrent_state,
            "infinite_context": self.infinite_context,
            "episodic_memory": self.episodic_memory,
            "ttt_sidecar": self.ttt_sidecar,
        }


@dataclass(frozen=True)
class EngineFeaturePlan:
    """Resolved feature ownership for one engine instance."""

    requested: EngineFeatureRequest
    support: dict[str, FeatureSupport]

    def summary(self) -> dict[str, object]:
        return {
            "requested": self.requested.summary(),
            "support": dict(self.support),
            "unsupported": [name for name, status in self.support.items() if status == "unsupported"],
        }

    def require_supported(self) -> None:
        unsupported = [name for name, status in self.support.items() if status == "unsupported"]
        if unsupported:
            raise RuntimeError("engine does not support requested LangBurst features: " + ", ".join(sorted(unsupported)))


def resolve_engine_feature_plan(
    capabilities: EngineCapabilities,
    request: EngineFeatureRequest,
    *,
    prefer_bridge: bool = True,
) -> EngineFeaturePlan:
    support: dict[str, FeatureSupport] = {}
    mapping = request.summary()
    for name, enabled in mapping.items():
        if not enabled:
            continue
        if bool(getattr(capabilities, name)):
            support[name] = "engine"
        elif name in {"stateful_sessions", "episodic_memory", "ttt_sidecar"} and capabilities.host_state:
            support[name] = "host"
        elif prefer_bridge and capabilities.custom_model and name in {
            "qwen36_lowbit",
            "recurrent_state",
            "ring_kv",
            "infinite_context",
            "episodic_memory",
            "ttt_sidecar",
        }:
            support[name] = "bridge"
        else:
            support[name] = "unsupported"
    return EngineFeaturePlan(requested=request, support=support)


@dataclass(frozen=True)
class EngineSamplingParams:
    max_tokens: int = 256
    min_tokens: int = 0
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    stop: tuple[str, ...] = ()
    stop_token_ids: tuple[int, ...] = ()
    seed: int | None = None
    ignore_eos: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EngineChatRequest:
    request_id: str
    model: str | None
    messages: Sequence[dict[str, Any]]
    sampling: EngineSamplingParams
    stream: bool = False
    user: str | None = None
    metadata: dict[str, Any] | None = None
    raw_request: Any | None = None


@dataclass(frozen=True)
class EngineUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    def openai_usage(self) -> dict[str, Any]:
        total = self.total_tokens or self.prompt_tokens + self.completion_tokens
        return {
            "prompt_tokens": int(self.prompt_tokens),
            "completion_tokens": int(self.completion_tokens),
            "total_tokens": int(total),
            "prompt_tokens_details": {
                "cached_tokens": int(self.cached_tokens),
                "uncached_tokens": max(0, int(self.prompt_tokens) - int(self.cached_tokens)),
            },
        }


@dataclass(frozen=True)
class EngineChatResult:
    text: str
    model: str
    finish_reason: str = "stop"
    usage: EngineUsage = field(default_factory=EngineUsage)
    raw: Any | None = None


@dataclass(frozen=True)
class EngineChatChunk:
    text: str
    model: str
    finish_reason: str | None = None
    usage: EngineUsage | None = None
    raw: Any | None = None


class EngineBackend(Protocol):
    descriptor: EngineDescriptor
    feature_plan: EngineFeaturePlan

    def start(self) -> None: ...

    def shutdown(self) -> None: ...

    def list_models(self) -> list[dict[str, Any]]: ...

    def health(self) -> dict[str, Any]: ...

    def generate_chat(self, request: EngineChatRequest) -> EngineChatResult: ...

    def stream_chat(self, request: EngineChatRequest) -> Iterable[EngineChatChunk]: ...


class EngineProvider(Protocol):
    descriptor: EngineDescriptor

    def create(self, spec: EngineModelSpec) -> EngineBackend: ...
