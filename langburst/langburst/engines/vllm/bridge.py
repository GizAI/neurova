from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..base import EngineFeaturePlan, EngineModelSpec


LOWBIT_MAMBA_MIN_BATCHED_TOKENS = 1568
LOWBIT_MTP_MIN_BATCHED_TOKENS = 1600
DEFAULT_MTP_KV_CACHE_MEMORY_BYTES = 760_000_000


VLLM_OWNED_FEATURES: tuple[str, ...] = (
    "openai_server",
    "scheduler",
    "continuous_batching",
    "paged_attention",
    "prefix_cache",
    "sampling",
    "standard_hf_loading",
    "attention_kernels",
)

LANGBURST_CUSTOM_MODEL_BRIDGE_FEATURES: tuple[str, ...] = (
    "custom_config",
    "lowbit_checkpoint_loader",
    "stateful_hybrid_blocks",
    "recurrent_state",
    "gdn_recurrent_kernel",
    "episodic_memory_sidecar",
    "ttt_sidecar",
)

LANGBURST_NATIVE_RUNTIME_EXCLUDED_FROM_VLLM: tuple[str, ...] = (
    "RuntimeEngine",
    "EngineManager",
    "BatchGenerationWorker",
    "BatchedModelRunner",
    "ContinuousBatchScheduler",
    "KVBlockTable",
    "RadixPrefixCache",
    "GenerationConfig",
    "sample_next",
)

VLLM_EXTRA_KWARGS: frozenset[str] = frozenset(
    {
        "block_size",
        "cpu_offload_gb",
        "disable_custom_all_reduce",
        "download_dir",
        "enable_chunked_prefill",
        "enable_lora",
        "enforce_eager",
        "gpu_memory_utilization",
        "kv_cache_dtype",
        "kv_cache_memory_bytes",
        "limit_mm_per_prompt",
        "load_format",
        "mamba_block_size",
        "mamba_cache_mode",
        "max_lora_rank",
        "max_num_batched_tokens",
        "max_num_seqs",
        "max_seq_len_to_capture",
        "model_loader_extra_config",
        "pipeline_parallel_size",
        "reasoning_parser",
        "reasoning_parser_plugin",
        "revision",
        "rope_scaling",
        "seed",
        "speculative_config",
        "swap_space",
        "tokenizer_mode",
        "tokenizer_revision",
    }
)


def lowbit_min_batched_tokens(*, enable_mtp: bool) -> int:
    return LOWBIT_MTP_MIN_BATCHED_TOKENS if enable_mtp else LOWBIT_MAMBA_MIN_BATCHED_TOKENS


def resolve_lowbit_enable_mtp(extra: dict[str, Any]) -> bool:
    """Low-bit bridge models use native MTP unless explicitly disabled."""

    return bool(extra.get("enable_mtp", True))


def resolve_lowbit_max_num_batched_tokens(spec: EngineModelSpec, *, enable_mtp: bool) -> int:
    import os

    floor = lowbit_min_batched_tokens(enable_mtp=enable_mtp)
    configured = spec.extra.get("max_num_batched_tokens")
    if configured is None:
        configured = os.environ.get("LANGBURST_VLLM_MAX_NUM_BATCHED_TOKENS")
    if configured is None:
        configured = spec.max_model_len or 0
    return max(floor, int(configured or 0))


def resolve_mtp_kv_cache_memory_bytes(extra: dict[str, Any]) -> int:
    import os

    return int(extra.get("kv_cache_memory_bytes") or os.environ.get("LANGBURST_VLLM_MTP_KV_CACHE_MEMORY_BYTES") or DEFAULT_MTP_KV_CACHE_MEMORY_BYTES)


@dataclass(frozen=True)
class VLLMBridgeConfig:
    """Single translation point from LangBurst features to vLLM engine kwargs."""

    engine_kwargs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def requires_custom_model(self) -> bool:
        return bool(self.metadata.get("requires_custom_model", False))

    def summary(self) -> dict[str, Any]:
        return {
            "engine_kwargs": dict(self.engine_kwargs),
            "metadata": dict(self.metadata),
            "requires_custom_model": self.requires_custom_model,
        }


def build_vllm_bridge_config(spec: EngineModelSpec, feature_plan: EngineFeaturePlan) -> VLLMBridgeConfig:
    """Resolve all LangBurst-to-vLLM integration knobs in one place.

    vLLM owns generic scheduling, PagedAttention, prefix caching, and sampling.
    LangBurst owns custom low-bit/recurrent semantics until a provider-native
    model implementation is installed. The bridge metadata is forwarded through
    HF config overrides so that custom model code can consume
    the same feature request without server-side branching.
    """

    requested = feature_plan.requested
    metadata: dict[str, Any] = {
        "features": requested.summary(),
        "support": dict(feature_plan.support),
        "vllm_owned": VLLM_OWNED_FEATURES,
        "langburst_custom_model_bridge": LANGBURST_CUSTOM_MODEL_BRIDGE_FEATURES,
        "excluded_native_runtime": LANGBURST_NATIVE_RUNTIME_EXCLUDED_FROM_VLLM,
    }
    kwargs: dict[str, Any] = {}
    hf_overrides: dict[str, Any] = {}

    if requested.ring_kv or requested.infinite_context or requested.stateful_sessions:
        kwargs["enable_prefix_caching"] = True
        hf_overrides["langburst_kv_policy"] = "ring" if requested.ring_kv else "paged_prefix"
        hf_overrides["langburst_infinite_context"] = bool(requested.infinite_context)

    if requested.custom_model_bridge:
        metadata["requires_custom_model"] = False
        metadata["lowbit_checkpoint"] = spec.extra.get("qb_model")
        kwargs["enforce_eager"] = True
        kwargs["dtype"] = str(spec.extra.get("dtype", "float16"))
        kwargs["language_model_only"] = True
        kwargs["load_format"] = "langburst_lowbit"
        enable_mtp = resolve_lowbit_enable_mtp(spec.extra)
        # Hybrid recurrent bridges need a minimum token budget for aligned
        # recurrent cache blocks. MTP pads it slightly higher again.
        kwargs["max_num_batched_tokens"] = resolve_lowbit_max_num_batched_tokens(spec, enable_mtp=enable_mtp)
        kwargs["max_num_seqs"] = int(spec.extra.get("max_num_seqs", 1))
        kwargs["kv_cache_dtype"] = str(spec.extra.get("kv_cache_dtype", "fp8"))
        if kv_cache_memory_bytes := spec.extra.get("kv_cache_memory_bytes"):
            kwargs["kv_cache_memory_bytes"] = int(kv_cache_memory_bytes)
        kwargs["quantization"] = "langburst_lowbit"
        if reasoning_parser := spec.extra.get("reasoning_parser"):
            kwargs["reasoning_parser"] = str(reasoning_parser)
        if enable_mtp and "speculative_config" not in spec.extra:
            kwargs["speculative_config"] = {
                "method": "mtp",
                "num_speculative_tokens": int(spec.extra.get("mtp_speculative_tokens", 2)),
            }
        hf_overrides["langburst_custom_model_bridge"] = True
        if qb_model := spec.extra.get("qb_model"):
            kwargs["model_loader_extra_config"] = {"qb_model": str(qb_model)}
            hf_overrides["langburst_qb_model"] = str(qb_model)
            hf_overrides["quantization_config"] = {
                "quant_method": "langburst_lowbit",
                "qb_model": str(qb_model),
            }
        if quantization_config := spec.extra.get("langburst_quantization_config"):
            hf_overrides["langburst_quantization_config"] = quantization_config

    if requested.recurrent_state:
        kwargs.setdefault("mamba_cache_mode", "align")
        hf_overrides["langburst_recurrent_state"] = True

    if requested.episodic_memory or requested.ttt_sidecar:
        hf_overrides["langburst_sidecars"] = {
            "episodic_memory": requested.episodic_memory,
            "ttt_sidecar": requested.ttt_sidecar,
        }

    if hf_overrides:
        kwargs["hf_overrides"] = {"langburst": metadata, **hf_overrides}

    model_impl = spec.extra.get("vllm_model_impl")
    if model_impl:
        kwargs["model_impl"] = str(model_impl)
    if custom_architecture := spec.extra.get("vllm_custom_model"):
        kwargs.setdefault("hf_overrides", {})
        kwargs["hf_overrides"]["architectures"] = [str(custom_architecture)]
        metadata["custom_architecture"] = str(custom_architecture)

    return VLLMBridgeConfig(engine_kwargs=kwargs, metadata=metadata)


def vllm_engine_extra_kwargs(extra: dict[str, Any]) -> dict[str, Any]:
    """Return only real vLLM engine kwargs from a mixed LangBurst extra map."""

    out: dict[str, Any] = {}
    for key, value in extra.items():
        if key in VLLM_EXTRA_KWARGS:
            out[key] = value
        elif key.startswith("vllm_arg_"):
            out[key.removeprefix("vllm_arg_")] = value
    return out


class VLLMConversationStore:
    """Host-side stateful chat shim for vLLM's stateless request model."""

    def __init__(self, *, max_sessions: int = 128, max_messages_per_session: int = 64) -> None:
        self.max_sessions = int(max_sessions)
        self.max_messages_per_session = int(max_messages_per_session)
        self._sessions: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    def resolve(
        self,
        *,
        session_id: str | None,
        messages: Sequence[dict[str, Any]],
        assistant_text: str | None = None,
        reset: bool = False,
    ) -> list[dict[str, Any]]:
        current = [dict(m) for m in messages]
        if not session_id:
            return current
        if reset:
            self._sessions.pop(session_id, None)
        history = self._sessions.get(session_id, [])
        merged = [*history, *current]
        if assistant_text is not None:
            next_history = [*merged, {"role": "assistant", "content": assistant_text}]
            self._sessions[session_id] = _tail_messages(next_history, self.max_messages_per_session)
            self._sessions.move_to_end(session_id)
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
        return merged

    def summary(self) -> dict[str, Any]:
        return {
            "active_sessions": len(self._sessions),
            "max_sessions": self.max_sessions,
            "max_messages_per_session": self.max_messages_per_session,
            "sessions": [
                {"id": session_id, "messages": len(messages)}
                for session_id, messages in self._sessions.items()
            ],
        }


def _tail_messages(messages: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(messages) <= limit:
        return messages
    system = [m for m in messages if m.get("role") == "system"][:1]
    tail = messages[-limit:]
    if system and tail and tail[0] is not system[0] and system[0] not in tail:
        tail = [system[0], *tail[1:]]
    return tail
