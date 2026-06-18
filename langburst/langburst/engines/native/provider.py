from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any
from typing import Iterable

from ..base import EngineBackend, EngineCapabilities, EngineChatChunk, EngineChatRequest, EngineChatResult, EngineDescriptor, EngineModelSpec, EngineProvider, EngineUsage, resolve_engine_feature_plan
from .batch_worker import BatchGenerationHandle, BatchGenerationWorker
from .block_table import KVBlockTable
from .manager import EngineManager, EngineResourcePolicy, ModelResourceSpec, load_model_specs
from .model_runner import BatchedModelRunner
from .runtime import GenerationConfig, RuntimeEngine, sample_next, sample_next_tensor
from .scheduler import AdmissionController, ContinuousBatchScheduler


class NativeBackend:
    """Adapter around the primary LangBurst in-process engine.

    Native is the reference implementation for Qwen3.6/GDN low-bit serving.
    Optional providers such as vLLM, SGLang, and EXL3 plug into the same host
    contract when their execution substrate is a better fit.
    """

    descriptor = EngineDescriptor(
        engine_id="native",
        display_name="LangBurst Native",
        module="langburst.engines.native",
        default=True,
        capabilities=EngineCapabilities(
            continuous_batching=True,
            paged_kv=True,
            prefix_cache=True,
            speculative_decoding=True,
            stateful_sessions=False,
            ring_kv=True,
            recurrent_state=True,
            infinite_context=True,
            episodic_memory=True,
            ttt_sidecar=True,
            custom_model_bridge=True,
            custom_model=True,
            quantization=("langburst-lowbit", "langburst-marlin"),
        ),
    )

    def __init__(self, spec: EngineModelSpec) -> None:
        self.spec = spec
        self.feature_plan = resolve_engine_feature_plan(self.descriptor.capabilities, spec.features, prefer_bridge=False)
        self._engine: RuntimeEngine | None = None
        self._worker: BatchGenerationWorker | None = None
        self._resource_policy = EngineResourcePolicy.from_env()

    def start(self) -> None:
        if self._engine is not None:
            return
        from pathlib import Path

        from ...adapters import ensure_adapters_loaded
        from ...core.adapter import adapter_registry
        from ...core.features import RuntimeFeatures
        ensure_adapters_loaded()
        adapter_id = self.spec.extra.get("adapter")
        if not adapter_id:
            raise RuntimeError("native engine requires adapter in EngineModelSpec.extra")
        adapter_id = str(adapter_id)
        qb_model = self.spec.extra.get("qb_model")
        if not qb_model:
            raise RuntimeError("native engine requires qb_model in EngineModelSpec.extra")
        engine = RuntimeEngine(
            adapter=adapter_registry.get(adapter_id),
            hf_model=Path(self.spec.model),
            qb_model=Path(str(qb_model)),
            device=str(self.spec.extra.get("device", "cuda")),
            recent_window=int(self.spec.max_model_len or self.spec.extra.get("recent_window", 8192)),
            weight_device=str(self.spec.extra.get("weight_device", "auto")),
            cpu_embed=bool(self.spec.extra.get("cpu_embed", False)),
            model_name=self.spec.public_name,
            features=RuntimeFeatures.from_profile(str(self.spec.extra.get("runtime_profile", "stateful"))).with_overrides(
                kv_window_policy="ring" if self.spec.features.ring_kv else None,
                infinite_streaming=True if self.spec.features.infinite_context else None,
                episodic_memory=True if self.spec.features.episodic_memory else None,
                ttt_sidecar=True if self.spec.features.ttt_sidecar else None,
            ),
        )
        block_table = KVBlockTable(
            num_blocks=self._resource_policy.kv_blocks,
            block_size=self._resource_policy.kv_block_size,
        )
        scheduler = ContinuousBatchScheduler(
            max_num_requests=self._resource_policy.max_active_requests,
            max_num_batched_tokens=self._resource_policy.max_num_batched_tokens,
            prefill_chunk_size=self._resource_policy.prefill_chunk_size,
            max_prefill_rows_per_batch=self._resource_policy.max_prefill_rows_per_batch,
            decode_prefill_interleave_steps=self._resource_policy.decode_prefill_interleave_steps,
            block_table=block_table,
            kv_window_tokens=engine.recent_window,
        )
        runner = BatchedModelRunner(
            engine=engine,
            scheduler=scheduler,
            features=engine.resolve_plan(engine.features).effective,
            max_state_pool_size=self._resource_policy.max_state_pool_size,
        )
        self._engine = engine
        self._worker = BatchGenerationWorker(
            runner=runner,
            device=engine.device,
            max_wait_s=float(self._resource_policy.batch_wait_ms) / 1000.0,
        )

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.shutdown()
            self._worker = None
        self._engine = None

    def list_models(self) -> list[dict[str, object]]:
        return [{"id": self.spec.public_name, "object": "model", "owned_by": "langburst-native"}]

    def health(self) -> dict[str, object]:
        return {
            "ok": self._engine is not None,
            "engine": self.descriptor.summary(),
            "model": self.spec.public_name,
            "feature_plan": self.feature_plan.summary(),
            "resource_policy": self._resource_policy.summary(),
            "batch_worker": self._worker.stats() if self._worker is not None else None,
        }

    def generate_chat(self, request: EngineChatRequest) -> EngineChatResult:
        self.start()
        assert self._engine is not None
        assert self._worker is not None
        t0 = time.perf_counter()
        prompt_ids = self._engine.encode_messages(request.messages)
        encode_s = time.perf_counter() - t0
        cfg = _native_generation_config(self._engine, request)
        gen_start = time.perf_counter()
        handle = self._worker.submit(
            prompt_ids,
            max_new_tokens=cfg.max_new_tokens,
            eos_token_ids=tuple(cfg.eos_token_ids),
            generation_config=cfg,
            stop_sequences=tuple(_encode_stop_strings(self._engine, request)),
            include_stop_str_in_output=bool(request.sampling.extra.get("include_stop_str_in_output", False)),
            request_id=request.request_id,
        )
        ids = handle.wait_ids()
        gen_s = time.perf_counter() - gen_start
        e2e_s = time.perf_counter() - t0
        text = self._engine.tokenizer.decode(ids, skip_special_tokens=True)
        usage = EngineUsage(prompt_tokens=len(prompt_ids), completion_tokens=len(ids))
        metrics = {
            "stream": False,
            "request_id": request.request_id,
            "model": self.spec.public_name,
            "prompt_tokens": len(prompt_ids),
            "message_chars": _message_chars(request.messages),
            "completion_tokens": len(ids),
            "encode_s": encode_s,
            "queue_wait_s": handle.metrics().get("queue_wait_s"),
            "generate_s": gen_s,
            "e2e_s": e2e_s,
            "tok_s": len(ids) / max(gen_s, 1e-9),
            **_cuda_memory_snapshot(),
        }
        _profile_log(metrics)
        return EngineChatResult(text=text, model=self.spec.public_name, usage=usage, raw=metrics)

    def stream_chat(self, request: EngineChatRequest) -> Iterable[EngineChatChunk]:
        self.start()
        assert self._engine is not None
        assert self._worker is not None
        t0 = time.perf_counter()
        prompt_ids = self._engine.encode_messages(request.messages)
        encode_s = time.perf_counter() - t0
        cfg = _native_generation_config(self._engine, request)
        completion_tokens = 0
        first_token_s: float | None = None
        first_text_s: float | None = None
        gen_start = time.perf_counter()
        handle = self._worker.submit(
            prompt_ids,
            max_new_tokens=cfg.max_new_tokens,
            eos_token_ids=tuple(cfg.eos_token_ids),
            generation_config=cfg,
            stop_sequences=tuple(_encode_stop_strings(self._engine, request)),
            include_stop_str_in_output=bool(request.sampling.extra.get("include_stop_str_in_output", False)),
            request_id=request.request_id,
        )
        emitted_ids: list[int] = []
        try:
            for token_id in handle.iter_token_ids():
                if int(token_id) >= 0:
                    completion_tokens += 1
                    emitted_ids.append(int(token_id))
                    if first_token_s is None:
                        first_token_s = time.perf_counter() - t0
                text = self._engine.tokenizer.decode([int(token_id)], skip_special_tokens=True)
                if text:
                    if first_text_s is None:
                        first_text_s = time.perf_counter() - t0
                    yield EngineChatChunk(text=text, model=self.spec.public_name)
        finally:
            gen_s = time.perf_counter() - gen_start
            e2e_s = time.perf_counter() - t0
            handle_metrics = handle.metrics()
            metrics = {
                "stream": True,
                "request_id": request.request_id,
                "model": self.spec.public_name,
                "prompt_tokens": len(prompt_ids),
                "message_chars": _message_chars(request.messages),
                "completion_tokens": completion_tokens,
                "encode_s": encode_s,
                "queue_wait_s": handle_metrics.get("queue_wait_s"),
                "first_token_s": first_token_s,
                "first_text_s": first_text_s,
                "generate_s": gen_s,
                "e2e_s": e2e_s,
                "tok_s": completion_tokens / max(gen_s, 1e-9),
                **_cuda_memory_snapshot(),
            }
            _profile_log(metrics)
        yield EngineChatChunk(
            text="",
            model=self.spec.public_name,
            finish_reason="stop",
            usage=EngineUsage(prompt_tokens=len(prompt_ids), completion_tokens=completion_tokens),
            raw=metrics,
        )


def _native_generation_config(engine, request: EngineChatRequest):
    eos = engine.eos_token_ids()
    return GenerationConfig(
        max_new_tokens=request.sampling.max_tokens,
        min_new_tokens=request.sampling.min_tokens,
        temperature=request.sampling.temperature,
        top_k=max(0, request.sampling.top_k),
        top_p=request.sampling.top_p,
        seed=request.sampling.seed,
        eos_token_ids=eos,
        stop_token_ids=request.sampling.stop_token_ids,
        ignore_eos=request.sampling.ignore_eos,
    )


def _encode_stop_strings(engine, request: EngineChatRequest) -> tuple[tuple[int, ...], ...]:
    out: list[tuple[int, ...]] = []
    for stop in request.sampling.stop:
        ids = engine.tokenizer.encode(str(stop), add_special_tokens=False)
        if ids:
            out.append(tuple(int(t) for t in ids))
    return tuple(out)


def _profile_enabled() -> bool:
    value = os.environ.get("LANGBURST_REQUEST_PROFILE", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _content_chars(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray, str)):
        total = 0
        for item in content:
            if isinstance(item, dict):
                total += _content_chars(item.get("text", item.get("content")))
            else:
                total += _content_chars(item)
        return total
    return len(str(content))


def _message_chars(messages: Sequence[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        total += _content_chars(msg.get("content"))
    return total


def _cuda_memory_snapshot() -> dict[str, float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        device = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(device)
        return {
            "cuda_free_gib": free / (1024**3),
            "cuda_total_gib": total / (1024**3),
            "cuda_allocated_gib": torch.cuda.memory_allocated(device) / (1024**3),
            "cuda_reserved_gib": torch.cuda.memory_reserved(device) / (1024**3),
        }
    except Exception:
        return {}


def _profile_log(metrics: dict[str, Any]) -> None:
    if not _profile_enabled():
        return
    ordered = [
        "request_id",
        "model",
        "stream",
        "prompt_tokens",
        "message_chars",
        "completion_tokens",
        "encode_s",
        "lock_wait_s",
        "first_token_s",
        "first_text_s",
        "generate_s",
        "e2e_s",
        "tok_s",
        "cuda_free_gib",
        "cuda_allocated_gib",
        "cuda_reserved_gib",
    ]
    parts: list[str] = []
    for key in ordered:
        value = metrics.get(key)
        if value is None:
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    print("[langburst][native][profile] " + " ".join(parts), flush=True)


class NativeProvider:
    descriptor = NativeBackend.descriptor

    def create(self, spec: EngineModelSpec) -> EngineBackend:
        return NativeBackend(spec)


__all__ = [
    "AdmissionController",
    "BatchGenerationHandle",
    "BatchGenerationWorker",
    "BatchedModelRunner",
    "ContinuousBatchScheduler",
    "EngineManager",
    "EngineResourcePolicy",
    "GenerationConfig",
    "KVBlockTable",
    "ModelResourceSpec",
    "NativeBackend",
    "NativeProvider",
    "RuntimeEngine",
    "load_model_specs",
    "sample_next",
    "sample_next_tensor",
]
