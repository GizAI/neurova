from __future__ import annotations

from typing import Iterable

from .base import EngineBackend, EngineCapabilities, EngineChatChunk, EngineChatRequest, EngineChatResult, EngineDescriptor, EngineModelSpec, EngineProvider


class UnavailableBackend:
    def __init__(self, descriptor: EngineDescriptor, spec: EngineModelSpec, install_hint: str) -> None:
        self.descriptor = descriptor
        self.spec = spec
        self.install_hint = install_hint

    def start(self) -> None:
        raise RuntimeError(f"{self.descriptor.display_name} backend is registered but not implemented in this checkout. {self.install_hint}")

    def shutdown(self) -> None:
        return

    def list_models(self) -> list[dict[str, object]]:
        return [{"id": self.spec.public_name, "object": "model", "owned_by": f"langburst-{self.descriptor.engine_id}"}]

    def health(self) -> dict[str, object]:
        return {
            "ok": False,
            "engine": self.descriptor.summary(),
            "model": self.spec.public_name,
            "error": self.install_hint,
        }

    def generate_chat(self, request: EngineChatRequest) -> EngineChatResult:
        del request
        self.start()
        raise AssertionError("unreachable")

    def stream_chat(self, request: EngineChatRequest) -> Iterable[EngineChatChunk]:
        del request
        self.start()
        raise AssertionError("unreachable")


class StaticUnavailableProvider:
    def __init__(self, descriptor: EngineDescriptor, install_hint: str) -> None:
        self.descriptor = descriptor
        self.install_hint = install_hint

    def create(self, spec: EngineModelSpec) -> EngineBackend:
        return UnavailableBackend(self.descriptor, spec, self.install_hint)


def sglang_provider() -> EngineProvider:
    return StaticUnavailableProvider(
        EngineDescriptor(
            engine_id="sglang",
            display_name="SGLang",
            module="langburst.engines.sglang",
            capabilities=EngineCapabilities(
                continuous_batching=True,
                paged_kv=True,
                prefix_cache=True,
                structured_output=True,
                speculative_decoding=True,
                quantization=("awq", "gptq", "fp8", "bitsandbytes"),
            ),
        ),
        "Add langburst.engines.sglang provider wiring when SGLang is installed.",
    )


def exl3_provider() -> EngineProvider:
    return StaticUnavailableProvider(
        EngineDescriptor(
            engine_id="exl3",
            display_name="ExLlamaV3/EXL3",
            module="langburst.engines.exl3",
            capabilities=EngineCapabilities(
                continuous_batching=False,
                paged_kv=False,
                prefix_cache=False,
                structured_output=False,
                speculative_decoding=False,
                quantization=("exl3",),
            ),
        ),
        "Add the EXL3 provider around TabbyAPI/ExLlamaV3 for local EXL3 deployments.",
    )
