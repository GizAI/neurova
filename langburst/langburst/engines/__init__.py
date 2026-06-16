from __future__ import annotations

from .base import (
    EngineBackend,
    EngineCapabilities,
    EngineChatChunk,
    EngineChatRequest,
    EngineChatResult,
    EngineDescriptor,
    EngineModelSpec,
    EngineProvider,
    EngineSamplingParams,
    EngineUsage,
)
from .native import NativeProvider
from .registry import EngineRegistry, engine_registry
from .unavailable import exl3_provider, sglang_provider
from .vllm import VLLMProvider


def ensure_engines_loaded() -> None:
    if engine_registry.ids():
        return
    engine_registry.register(VLLMProvider())
    engine_registry.register(sglang_provider())
    engine_registry.register(exl3_provider())
    engine_registry.register(NativeProvider())
    engine_registry.load_entry_points()


__all__ = [
    "EngineBackend",
    "EngineCapabilities",
    "EngineChatChunk",
    "EngineChatRequest",
    "EngineChatResult",
    "EngineDescriptor",
    "EngineModelSpec",
    "EngineProvider",
    "EngineRegistry",
    "EngineSamplingParams",
    "EngineUsage",
    "ensure_engines_loaded",
    "engine_registry",
]
