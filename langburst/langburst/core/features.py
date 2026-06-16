from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Literal

RuntimeProfile = Literal["original", "stateful", "research"]
KVPolicy = Literal["error", "shift", "ring"]

RUNTIME_PROFILES: tuple[RuntimeProfile, ...] = ("original", "stateful", "research")
KV_POLICIES: tuple[KVPolicy, ...] = ("error", "shift", "ring")
BOOL_FEATURE_KEYS = (
    "stateful_chat",
    "state_pool",
    "snapshots",
    "gpu_sampling",
    "speculative_decoding",
    "cuda_graph",
    "block_prefill",
    "infinite_streaming",
    "episodic_memory",
    "ttt_sidecar",
)
FEATURE_OVERRIDE_KEYS = (
    "kv_window_policy",
    *BOOL_FEATURE_KEYS,
    "boundary_decay",
    "prefill_chunk_size",
)


@dataclass(frozen=True)
class RuntimeFeatureOverride:
    kv_window_policy: KVPolicy | None = None
    stateful_chat: bool | None = None
    state_pool: bool | None = None
    snapshots: bool | None = None
    boundary_decay: float | None = None
    gpu_sampling: bool | None = None
    speculative_decoding: bool | None = None
    cuda_graph: bool | None = None
    block_prefill: bool | None = None
    infinite_streaming: bool | None = None
    episodic_memory: bool | None = None
    ttt_sidecar: bool | None = None
    prefill_chunk_size: int | None = None

    @classmethod
    def from_obj(cls, obj: Any) -> "RuntimeFeatureOverride":
        return cls(**{key: getattr(obj, key, None) for key in FEATURE_OVERRIDE_KEYS})

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "RuntimeFeatureOverride":
        return cls(**{key: data.get(key) for key in FEATURE_OVERRIDE_KEYS})

    def compact(self) -> dict[str, object]:
        return {field.name: value for field in fields(self) if (value := getattr(self, field.name)) is not None}


@dataclass(frozen=True)
class RuntimeFeatures:
    """Single source of truth for optional runtime behavior.

    The model math stays the same across profiles.  These flags control runtime
    state policy, persistence helpers, and experimental acceleration paths so a
    user can run close to ordinary Qwen decode or enable LangBurst's stateful
    long-streaming features without changing code paths ad hoc.
    """

    profile: RuntimeProfile = "stateful"
    kv_window_policy: KVPolicy = "ring"
    stateful_chat: bool = True
    state_pool: bool = True
    snapshots: bool = False
    boundary_decay: float = 1.0
    gpu_sampling: bool = True
    speculative_decoding: bool = True
    cuda_graph: bool = False
    block_prefill: bool = True
    infinite_streaming: bool = False
    episodic_memory: bool = False
    ttt_sidecar: bool = False
    prefill_chunk_size: int = 64

    @classmethod
    def from_profile(cls, profile: RuntimeProfile) -> "RuntimeFeatures":
        if profile == "original":
            return cls(
                profile="original",
                kv_window_policy="error",
                stateful_chat=False,
                state_pool=True,
                snapshots=False,
                boundary_decay=1.0,
                gpu_sampling=True,
                speculative_decoding=False,
                cuda_graph=False,
                block_prefill=True,
                prefill_chunk_size=64,
            )
        if profile == "stateful":
            return cls(profile="stateful")
        if profile == "research":
            return cls(
                profile="research",
                kv_window_policy="ring",
                stateful_chat=True,
                state_pool=True,
                snapshots=True,
                boundary_decay=1.0,
                gpu_sampling=True,
                speculative_decoding=True,
                cuda_graph=False,
                block_prefill=True,
                infinite_streaming=True,
                episodic_memory=True,
                ttt_sidecar=True,
                prefill_chunk_size=64,
            )
        raise ValueError(f"unknown runtime profile: {profile}")

    def with_overrides(
        self,
        override: RuntimeFeatureOverride | None = None,
        *,
        kv_window_policy: KVPolicy | None = None,
        stateful_chat: bool | None = None,
        state_pool: bool | None = None,
        snapshots: bool | None = None,
        boundary_decay: float | None = None,
        gpu_sampling: bool | None = None,
        speculative_decoding: bool | None = None,
        cuda_graph: bool | None = None,
        block_prefill: bool | None = None,
        infinite_streaming: bool | None = None,
        episodic_memory: bool | None = None,
        ttt_sidecar: bool | None = None,
        prefill_chunk_size: int | None = None,
    ) -> "RuntimeFeatures":
        values = override.compact() if override is not None else {}
        for key, value in {
            "kv_window_policy": kv_window_policy,
            "stateful_chat": stateful_chat,
            "state_pool": state_pool,
            "snapshots": snapshots,
            "boundary_decay": boundary_decay,
            "gpu_sampling": gpu_sampling,
            "speculative_decoding": speculative_decoding,
            "cuda_graph": cuda_graph,
            "block_prefill": block_prefill,
            "infinite_streaming": infinite_streaming,
            "episodic_memory": episodic_memory,
            "ttt_sidecar": ttt_sidecar,
            "prefill_chunk_size": prefill_chunk_size,
        }.items():
            if value is not None:
                values[key] = value
        out = replace(self, **values)
        if out.kv_window_policy not in KV_POLICIES:
            raise ValueError(f"unknown kv_window_policy: {out.kv_window_policy}")
        if not 0.0 <= out.boundary_decay <= 1.0:
            raise ValueError("boundary_decay must be in [0, 1]")
        if out.prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be positive")
        return out

    def summary(self) -> dict[str, object]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def with_overrides_from_mapping(self, data: dict[str, Any]) -> "RuntimeFeatures":
        return self.with_overrides(RuntimeFeatureOverride.from_mapping(data))


@dataclass(frozen=True)
class RuntimeCapabilities:
    """Adapter-declared runtime support.

    `RuntimeFeatures` says what a caller requested.  Capabilities say what a
    model adapter can execute without changing model semantics.  The resolved
    `RuntimePlan` is the only object downstream orchestration should trust.
    """

    kv_window_policies: tuple[KVPolicy, ...] = ("error",)
    stateful_chat: bool = False
    state_pool: bool = True
    snapshots: bool = False
    boundary_decay: bool = False
    gpu_sampling: bool = True
    speculative_decoding: bool = True
    cuda_graph: bool = False
    block_prefill: bool = True
    infinite_streaming: bool = False
    episodic_memory: bool = False
    ttt_sidecar: bool = False
    max_concurrency: int = 1

    @classmethod
    def transformer_decoder(
        cls,
        *,
        kv_window_policies: tuple[KVPolicy, ...] = ("error",),
        max_concurrency: int = 1,
    ) -> "RuntimeCapabilities":
        """Capabilities for generic Hugging Face-style decoder adapters.

        A generic transformer cache can keep state and be snapshotted, but it
        does not expose LangBurst's physical ring-KV contract.  Model-specific
        adapters can opt into `shift`/`ring` once they own cache layout.
        """
        return cls(
            kv_window_policies=kv_window_policies,
            stateful_chat=True,
            state_pool=True,
            snapshots=True,
            boundary_decay=False,
            gpu_sampling=True,
            speculative_decoding=False,
            cuda_graph=False,
            block_prefill=True,
            infinite_streaming=False,
            episodic_memory=False,
            ttt_sidecar=False,
            max_concurrency=max_concurrency,
        )

    @classmethod
    def stateful_hybrid(
        cls,
        *,
        kv_window_policies: tuple[KVPolicy, ...] = ("error", "shift", "ring"),
        max_concurrency: int = 1,
    ) -> "RuntimeCapabilities":
        return cls(
            kv_window_policies=kv_window_policies,
            stateful_chat=True,
            state_pool=True,
            snapshots=True,
            boundary_decay=True,
            gpu_sampling=True,
            speculative_decoding=True,
            cuda_graph=False,
            block_prefill=True,
            infinite_streaming=True,
            episodic_memory=True,
            ttt_sidecar=True,
            max_concurrency=max_concurrency,
        )

    def summary(self) -> dict[str, object]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True)
class RuntimePlan:
    """Resolved execution contract for one engine/request."""

    requested: RuntimeFeatures
    effective: RuntimeFeatures
    capabilities: RuntimeCapabilities
    disabled: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def features(self) -> RuntimeFeatures:
        return self.effective

    def summary(self) -> dict[str, object]:
        return {
            "requested": self.requested.summary(),
            "effective": self.effective.summary(),
            "capabilities": self.capabilities.summary(),
            "disabled": list(self.disabled),
            "notes": list(self.notes),
        }


def resolve_runtime_plan(features: RuntimeFeatures, capabilities: RuntimeCapabilities) -> RuntimePlan:
    values: dict[str, object] = {}
    disabled: list[str] = []
    notes: list[str] = []
    bool_caps = (
        "stateful_chat",
        "state_pool",
        "snapshots",
        "gpu_sampling",
        "speculative_decoding",
        "cuda_graph",
        "block_prefill",
        "infinite_streaming",
        "episodic_memory",
        "ttt_sidecar",
    )
    for key in bool_caps:
        requested = bool(getattr(features, key))
        supported = bool(getattr(capabilities, key))
        if requested and not supported:
            values[key] = False
            disabled.append(key)
    if features.boundary_decay != 1.0 and not capabilities.boundary_decay:
        values["boundary_decay"] = 1.0
        disabled.append("boundary_decay")
    if features.kv_window_policy not in capabilities.kv_window_policies:
        if "ring" in capabilities.kv_window_policies:
            values["kv_window_policy"] = "ring"
        else:
            values["kv_window_policy"] = capabilities.kv_window_policies[0]
        disabled.append("kv_window_policy")
    effective = features.with_overrides(RuntimeFeatureOverride.from_mapping(values))
    if disabled:
        notes.append("unsupported requested features were disabled by adapter capabilities")
    return RuntimePlan(
        requested=features,
        effective=effective,
        capabilities=capabilities,
        disabled=tuple(disabled),
        notes=tuple(notes),
    )
