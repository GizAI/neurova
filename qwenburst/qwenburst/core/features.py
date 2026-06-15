from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Literal

RuntimeProfile = Literal["original", "stateful", "research"]
KVPolicy = Literal["error", "shift", "ring"]

RUNTIME_PROFILES: tuple[RuntimeProfile, ...] = ("original", "stateful", "research")
KV_POLICIES: tuple[KVPolicy, ...] = ("error", "shift", "ring")
BOOL_FEATURE_KEYS = (
    "stateful_chat",
    "infinite_streaming",
    "snapshots",
    "episodic_memory",
    "ttt_sidecar",
    "speculative_mtp",
    "cuda_graph",
    "block_prefill",
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
    infinite_streaming: bool | None = None
    snapshots: bool | None = None
    boundary_decay: float | None = None
    episodic_memory: bool | None = None
    ttt_sidecar: bool | None = None
    speculative_mtp: bool | None = None
    cuda_graph: bool | None = None
    block_prefill: bool | None = None
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
    user can run close to ordinary Qwen decode or enable QwenBurst's stateful
    long-streaming features without changing code paths ad hoc.
    """

    profile: RuntimeProfile = "stateful"
    kv_window_policy: KVPolicy = "ring"
    stateful_chat: bool = True
    infinite_streaming: bool = True
    snapshots: bool = False
    boundary_decay: float = 1.0
    episodic_memory: bool = False
    ttt_sidecar: bool = False
    speculative_mtp: bool = False
    cuda_graph: bool = False
    block_prefill: bool = True
    prefill_chunk_size: int = 64

    @classmethod
    def from_profile(cls, profile: RuntimeProfile) -> "RuntimeFeatures":
        if profile == "original":
            return cls(
                profile="original",
                kv_window_policy="error",
                stateful_chat=False,
                infinite_streaming=False,
                snapshots=False,
                boundary_decay=1.0,
                episodic_memory=False,
                ttt_sidecar=False,
                speculative_mtp=False,
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
                infinite_streaming=True,
                snapshots=True,
                boundary_decay=1.0,
                episodic_memory=True,
                ttt_sidecar=True,
                speculative_mtp=False,
                cuda_graph=False,
                block_prefill=True,
                prefill_chunk_size=64,
            )
        raise ValueError(f"unknown runtime profile: {profile}")

    def with_overrides(
        self,
        override: RuntimeFeatureOverride | None = None,
        *,
        kv_window_policy: KVPolicy | None = None,
        stateful_chat: bool | None = None,
        infinite_streaming: bool | None = None,
        snapshots: bool | None = None,
        boundary_decay: float | None = None,
        episodic_memory: bool | None = None,
        ttt_sidecar: bool | None = None,
        speculative_mtp: bool | None = None,
        cuda_graph: bool | None = None,
        block_prefill: bool | None = None,
        prefill_chunk_size: int | None = None,
    ) -> "RuntimeFeatures":
        values = override.compact() if override is not None else {}
        for key, value in {
            "kv_window_policy": kv_window_policy,
            "stateful_chat": stateful_chat,
            "infinite_streaming": infinite_streaming,
            "snapshots": snapshots,
            "boundary_decay": boundary_decay,
            "episodic_memory": episodic_memory,
            "ttt_sidecar": ttt_sidecar,
            "speculative_mtp": speculative_mtp,
            "cuda_graph": cuda_graph,
            "block_prefill": block_prefill,
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
