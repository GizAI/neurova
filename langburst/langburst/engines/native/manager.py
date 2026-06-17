from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import gc
import time
from pathlib import Path
import json
import threading
from typing import Any

import torch

from ...core.adapter import adapter_registry
from ...core.defaults import (
    serving_recent_window_default,
)
from ...core.features import RuntimeFeatures, RuntimePlan, resolve_runtime_plan
from .batch_worker import BatchGenerationWorker
from .model_runner import BatchedModelRunner
from .runtime import RuntimeEngine
from .scheduler import AdmissionController, ContinuousBatchScheduler
from .block_table import KVBlockTable
from .cuda_graph import CudaGraphBucketPlanner
from .resource_policy import EngineResourcePolicy


@dataclass(frozen=True)
class ModelResourceSpec:
    """Declarative model entry for multi-model serving."""

    model_name: str
    adapter_id: str
    hf_model: Path
    qb_model: Path
    device: str = "cuda"
    recent_window: int = field(default_factory=serving_recent_window_default)
    weight_device: str = "auto"
    cpu_embed: bool = False
    estimated_vram_mib: int | None = None
    runtime_features: RuntimeFeatures = field(default_factory=lambda: RuntimeFeatures.from_profile("stateful"))

    @classmethod
    def from_mapping(cls, data: dict[str, Any], default_features: RuntimeFeatures) -> "ModelResourceSpec":
        profile = data.get("runtime_profile")
        features = (
            RuntimeFeatures.from_profile(profile).with_overrides_from_mapping(data)
            if profile
            else default_features.with_overrides_from_mapping(data)
        )
        adapter_id = data.get("adapter_id", data.get("adapter"))
        if adapter_id is None:
            raise ValueError("model spec requires explicit adapter or adapter_id")
        return cls(
            model_name=str(data["model_name"]),
            adapter_id=str(adapter_id),
            hf_model=Path(data["hf_model"]),
            qb_model=Path(data["qb_model"]),
            device=str(data.get("device", "cuda")),
            recent_window=int(data.get("recent_window", serving_recent_window_default())),
            weight_device=str(data.get("weight_device", "auto")),
            cpu_embed=bool(data.get("cpu_embed", False)),
            estimated_vram_mib=(int(data["estimated_vram_mib"]) if data.get("estimated_vram_mib") is not None else None),
            runtime_features=features,
        )

    @classmethod
    def from_args(cls, args: Any, *, features: RuntimeFeatures) -> "ModelResourceSpec":
        if args.hf_model is None or args.qb_model is None:
            raise ValueError("hf_model and qb_model are required for a single model spec")
        adapter = adapter_registry.get(args.adapter)
        return cls(
            model_name=args.model_name or adapter.descriptor.default_model_name,
            adapter_id=args.adapter,
            hf_model=args.hf_model,
            qb_model=args.qb_model,
            device=args.device,
            recent_window=args.recent_window,
            weight_device=args.weight_device,
            cpu_embed=bool(args.cpu_embed),
            runtime_features=features,
        )


@dataclass
class ModelRuntimeStatus:
    model_name: str
    state: str = "unloaded"
    loaded_unix: float | None = None
    last_used_unix: float | None = None
    load_count: int = 0
    unload_count: int = 0
    last_error: str | None = None

    def summary(self) -> dict[str, object]:
        return {
            "state": self.state,
            "loaded_unix": self.loaded_unix,
            "last_used_unix": self.last_used_unix,
            "load_count": self.load_count,
            "unload_count": self.unload_count,
            "last_error": self.last_error,
        }


class EngineManager:
    """Lazy multi-model manager with bounded loaded-engine residency."""

    def __init__(self, specs: list[ModelResourceSpec], policy: EngineResourcePolicy | None = None) -> None:
        if not specs:
            raise ValueError("EngineManager requires at least one model spec")
        self.specs = {spec.model_name: spec for spec in specs}
        if len(self.specs) != len(specs):
            raise ValueError("duplicate model_name in model specs")
        self.policy = policy or EngineResourcePolicy()
        self._engines: OrderedDict[str, RuntimeEngine] = OrderedDict()
        self._status = {name: ModelRuntimeStatus(model_name=name) for name in self.specs}
        self._lock = threading.RLock()
        self.admission = AdmissionController(
            max_active_requests=self.policy.max_active_requests,
            max_queued_requests=self.policy.max_queued_requests,
            admission_timeout_s=self.policy.admission_timeout_s,
        )
        paged_kv_enabled = _env_enabled("LANGBURST_PAGED_KV", default=True)
        self.kv_block_table = (
            KVBlockTable(
                num_blocks=self.policy.kv_blocks,
                block_size=self.policy.kv_block_size,
            )
            if paged_kv_enabled
            else None
        )
        self.cuda_graph_planner = CudaGraphBucketPlanner()
        self.batch_scheduler = ContinuousBatchScheduler(
            max_num_requests=self.policy.max_active_requests,
            max_num_batched_tokens=self.policy.max_num_batched_tokens,
            prefill_chunk_size=self.policy.prefill_chunk_size,
            max_prefill_rows_per_batch=self.policy.max_prefill_rows_per_batch,
            decode_prefill_interleave_steps=self.policy.decode_prefill_interleave_steps,
            block_table=self.kv_block_table,
            cuda_graph_planner=self.cuda_graph_planner,
        )
        self._batch_runners: dict[tuple[str, tuple[tuple[str, object], ...]], BatchedModelRunner] = {}
        self._batch_workers: dict[tuple[str, tuple[tuple[str, object], ...]], BatchGenerationWorker] = {}

    @classmethod
    def from_engine(cls, engine: RuntimeEngine, policy: EngineResourcePolicy | None = None) -> "EngineManager":
        spec = ModelResourceSpec(
            model_name=engine.model_name,
            adapter_id=engine.adapter.descriptor.adapter_id,
            hf_model=engine.hf_model,
            qb_model=engine.qb_model,
            device=engine.device,
            recent_window=engine.recent_window,
            weight_device=engine.weight_device,
            cpu_embed=engine.cpu_embed,
            runtime_features=engine.features,
        )
        manager = cls([spec], policy=policy)
        manager._engines[engine.model_name] = engine
        return manager

    def list_models(self) -> list[dict[str, Any]]:
        out = []
        for spec in self.specs.values():
            adapter = adapter_registry.get(spec.adapter_id)
            status = self._status[spec.model_name]
            out.append(
                {
                    "id": spec.model_name,
                    "adapter": spec.adapter_id,
                    "family": adapter.descriptor.family,
                    "estimated_vram_mib": spec.estimated_vram_mib,
                    "loaded": spec.model_name in self._engines,
                    "status": status.summary(),
                    "capabilities": self._serving_capability_summary(adapter.descriptor.capabilities),
                }
            )
        return out

    def _serving_capability_summary(self, capabilities) -> dict[str, Any]:
        summary = dict(capabilities.summary())
        summary["max_concurrency"] = self.policy.max_active_requests
        return summary

    def summary(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": self.list_models(),
            "policy": self.policy.summary(),
            "admission": self.admission.stats().summary(),
            "batch_scheduler": self.batch_scheduler.stats().summary(),
            "batch_runners": self.batch_runner_summary(),
            "batch_workers": self.batch_worker_summary(),
            "resource": self.resource_summary(),
        }

    def health(self) -> dict[str, Any]:
        failed = [status.model_name for status in self._status.values() if status.state == "failed"]
        return {
            "ok": not failed,
            "failed_models": failed,
            "models": self.list_models(),
            "policy": self.policy.summary(),
            "admission": self.admission.stats().summary(),
            "batch_scheduler": self.batch_scheduler.stats().summary(),
            "batch_runners": self.batch_runner_summary(),
            "batch_workers": self.batch_worker_summary(),
            "resource": self.resource_summary(),
            "loaded_tensor_memory": self.loaded_tensor_memory_summary(),
            "kv_blocks": self.kv_block_table.summary() if self.kv_block_table is not None else None,
            "state_pools": {
                name: engine.state_pool_summary()
                for name, engine in self._engines.items()
            },
        }

    def loaded_tensor_memory_summary(self) -> dict[str, object]:
        out: dict[str, object] = {}
        for name, engine in self._engines.items():
            store = getattr(getattr(engine, "model", None), "store", None)
            summary = getattr(store, "loaded_tensor_summary", None)
            if callable(summary):
                out[name] = summary()
        return out

    def resource_summary(self) -> dict[str, Any]:
        loaded = list(self._engines)
        out: dict[str, Any] = {
            "loaded_models": loaded,
            "loaded_model_count": len(loaded),
        }
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            device = torch.cuda.current_device()
            out.update(
                {
                    "cuda_free_mib": free_bytes // (1024 * 1024),
                    "cuda_total_mib": total_bytes // (1024 * 1024),
                    "cuda_allocated_mib": torch.cuda.memory_allocated(device) // (1024 * 1024),
                    "cuda_reserved_mib": torch.cuda.memory_reserved(device) // (1024 * 1024),
                    "reserve_free_vram_mib": self.policy.reserve_free_vram_mib,
                }
            )
        return out

    def default_model_name(self) -> str:
        return next(iter(self.specs))

    def get(self, model_name: str | None = None) -> RuntimeEngine:
        name = model_name or self.default_model_name()
        with self._lock:
            if name in self._engines:
                engine = self._engines.pop(name)
                self._engines[name] = engine
                self._status[name].last_used_unix = time.time()
                return engine
            spec = self.specs.get(name)
            if spec is None:
                known = ", ".join(sorted(self.specs))
                raise KeyError(f"unknown model {name!r}; known models: {known}")
            while len(self._engines) >= self.policy.max_loaded_models:
                evicted_name, evicted_engine = self._engines.popitem(last=False)
                evicted_engine.clear_state_pool()
                self._drop_batch_runners(evicted_name)
                status = self._status[evicted_name]
                status.state = "unloaded"
                status.unload_count += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            adapter = adapter_registry.get(spec.adapter_id)
            status = self._status[name]
            status.state = "loading"
            status.last_error = None
            try:
                self._check_load_admission(spec)
                engine = RuntimeEngine(
                    adapter=adapter,
                    hf_model=spec.hf_model,
                    qb_model=spec.qb_model,
                    device=spec.device,
                    recent_window=spec.recent_window,
                    weight_device=spec.weight_device,
                    cpu_embed=spec.cpu_embed,
                    model_name=spec.model_name,
                    features=spec.runtime_features,
                    max_state_pool_size=self.policy.max_state_pool_size,
                )
            except torch.cuda.OutOfMemoryError as exc:
                status.state = "failed"
                status.last_error = str(exc)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                raise
            except Exception as exc:
                status.state = "failed"
                status.last_error = str(exc)
                raise
            now = time.time()
            status.state = "loaded"
            status.loaded_unix = now
            status.last_used_unix = now
            status.load_count += 1
            self._engines[name] = engine
            return engine

    def _check_load_admission(self, spec: ModelResourceSpec) -> None:
        if spec.estimated_vram_mib is None or not str(spec.device).startswith("cuda") or not torch.cuda.is_available():
            return
        free_bytes, _ = torch.cuda.mem_get_info()
        free_mib = free_bytes // (1024 * 1024)
        required_mib = spec.estimated_vram_mib + self.policy.reserve_free_vram_mib
        if free_mib < required_mib:
            raise RuntimeError(
                f"insufficient free VRAM for {spec.model_name}: "
                f"free={free_mib} MiB required={required_mib} MiB "
                f"(estimate={spec.estimated_vram_mib} reserve={self.policy.reserve_free_vram_mib})"
            )

    def unload(self, model_name: str | None = None) -> bool:
        name = model_name or self.default_model_name()
        with self._lock:
            if name not in self._engines:
                return False
            engine = self._engines.pop(name)
            engine.clear_state_pool()
            self._drop_batch_runners(name)
            status = self._status[name]
            status.state = "unloaded"
            status.unload_count += 1
            del engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            return True

    def resolve_plan(self, model_name: str | None = None, features: RuntimeFeatures | None = None) -> RuntimePlan:
        name = model_name or self.default_model_name()
        spec = self.specs.get(name)
        if spec is None:
            known = ", ".join(sorted(self.specs))
            raise KeyError(f"unknown model {name!r}; known models: {known}")
        adapter = adapter_registry.get(spec.adapter_id)
        return resolve_runtime_plan(features or spec.runtime_features, adapter.descriptor.capabilities)

    def acquire_request(self):
        return self.admission.acquire()

    def create_batch_runner(self, model_name: str | None = None, features: RuntimeFeatures | None = None) -> BatchedModelRunner:
        engine = self.get(model_name)
        resolved = engine.resolve_plan(engine.features).effective
        key = (engine.model_name, self._batch_resource_key(resolved))
        with self._lock:
            runner = self._batch_runners.get(key)
            if runner is None:
                runner = BatchedModelRunner(
                    engine=engine,
                    scheduler=self.batch_scheduler,
                    features=resolved,
                    max_state_pool_size=self.policy.max_state_pool_size,
                )
                self._batch_runners[key] = runner
            return runner

    def create_batch_worker(self, model_name: str | None = None, features: RuntimeFeatures | None = None) -> BatchGenerationWorker:
        engine = self.get(model_name)
        resolved = engine.resolve_plan(engine.features).effective
        key = (engine.model_name, self._batch_resource_key(resolved))
        with self._lock:
            worker = self._batch_workers.get(key)
            if worker is None:
                worker = BatchGenerationWorker(
                    runner=self.create_batch_runner(engine.model_name, resolved),
                    device=engine.device,
                    max_wait_s=float(self.policy.batch_wait_ms) / 1000.0,
                )
                self._batch_workers[key] = worker
            return worker

    def _batch_resource_key(self, features: RuntimeFeatures) -> tuple[tuple[str, object], ...]:
        """Return the resource-shape key for a model runner.

        Request-level flags such as prefix cache or speculative decoding must
        not allocate another state arena on 16GB deployments. The runner owns a
        model's physical state/KV resources; per-request behavior is carried by
        request rows and generation config.
        """

        return (
            ("kv_cache_dtype", features.kv_cache_dtype),
            ("kv_window_policy", features.kv_window_policy),
            ("state_pool", features.state_pool),
        )

    def batch_runner_summary(self) -> dict[str, object]:
        return {
            "runners": len(self._batch_runners),
            "state_stores": {
                model_name: {
                    **runner.state_store.stats().summary(),
                    "arena": runner.state_store.arena_summary(),
                    "reuse_pool": runner.state_store.reuse_pool_summary(),
                    "prefix_cache": runner.prefix_cache_summary(),
                    "memory_policy": runner.memory_policy_summary(),
                    "speculative": runner.speculative_summary(),
                }
                for (model_name, _), runner in self._batch_runners.items()
            },
        }

    def batch_worker_summary(self) -> dict[str, object]:
        return {
            "workers": len(self._batch_workers),
            "worker_stats": {
                model_name: worker.stats()
                for (model_name, _), worker in self._batch_workers.items()
            },
        }

    def _drop_batch_runners(self, model_name: str) -> None:
        for key in [key for key in self._batch_workers if key[0] == model_name]:
            worker = self._batch_workers.pop(key)
            worker.shutdown()
            del worker
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for key in [key for key in self._batch_runners if key[0] == model_name]:
            runner = self._batch_runners.pop(key)
            runner.clear()
            del runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def validate_generation_request(self, *, prompt_tokens: int, generation_tokens: int) -> None:
        if prompt_tokens < 1:
            raise ValueError("prompt must contain at least one token")
        if generation_tokens < 1:
            raise ValueError("max generation tokens must be >= 1")
        if self.policy.max_prompt_tokens is not None and prompt_tokens > self.policy.max_prompt_tokens:
            raise ValueError(
                f"prompt too long: tokens={prompt_tokens} max_prompt_tokens={self.policy.max_prompt_tokens}"
            )
        if self.policy.max_generation_tokens is not None and generation_tokens > self.policy.max_generation_tokens:
            raise ValueError(
                f"generation too long: max_new_tokens={generation_tokens} "
                f"max_generation_tokens={self.policy.max_generation_tokens}"
            )

    def validate_runtime_memory(self, engine: RuntimeEngine) -> None:
        if not str(engine.device).startswith("cuda") or not torch.cuda.is_available():
            return
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        props = torch.cuda.get_device_properties(torch.device(engine.device))
        total_mib = props.total_memory // (1024 * 1024)
        free_bytes, _ = torch.cuda.mem_get_info()
        free_mib = free_bytes // (1024 * 1024)
        weight_mib = engine.estimated_weight_bytes() // (1024 * 1024)
        estimate_arena = getattr(engine.adapter, "estimate_arena_state_bytes", None)
        if callable(estimate_arena):
            state_bytes = int(estimate_arena(
                engine.cfg,
                max_slots=min(self.policy.max_state_pool_size, self.policy.max_active_requests),
                kv_num_blocks=self.policy.kv_blocks,
                kv_block_size=self.policy.kv_block_size,
                features=engine.features,
            ))
        else:
            state_bytes = engine.estimated_state_bytes()
        state_mib = state_bytes // (1024 * 1024)
        loaded_weight = engine.model_name in self._engines
        arena_allocated = any(key[0] == engine.model_name for key in self._batch_runners)
        if arena_allocated:
            # Once the model and runtime arena are already allocated, low free
            # VRAM is expected on 16GB cards.  Do not reject a request merely
            # because the reserve watermark is below target; the admission
            # controller already serializes active requests and the paged/ring
            # KV contract bounds per-request memory.  Real OOMs are handled at
            # execution time where the runtime can clear pools and recover.
            if self.policy.reserve_free_vram_mib > 0 and free_mib < self.policy.reserve_free_vram_mib:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            return
        if loaded_weight:
            required_mib = state_mib + self.policy.runtime_overhead_mib + self.policy.reserve_free_vram_mib
            if required_mib > free_mib:
                raise RuntimeError(
                    f"insufficient free GPU memory for runtime arena: free={free_mib} MiB "
                    f"required={required_mib} MiB state={state_mib} MiB "
                    f"overhead={self.policy.runtime_overhead_mib} MiB "
                    f"reserve={self.policy.reserve_free_vram_mib} MiB. "
                    "Lower LANGBURST_CONTEXT_WINDOW, reduce LANGBURST_MAX_ACTIVE_REQUESTS, or wait for current requests."
                )
            return
        required_mib = weight_mib + state_mib + self.policy.runtime_overhead_mib + self.policy.reserve_free_vram_mib
        if required_mib > total_mib:
            raise RuntimeError(
                f"configuration exceeds GPU memory budget: total={total_mib} MiB "
                f"required={required_mib} MiB weights={weight_mib} MiB "
                f"state={state_mib} MiB overhead={self.policy.runtime_overhead_mib} MiB "
                f"reserve={self.policy.reserve_free_vram_mib} MiB. "
                "Lower LANGBURST_CONTEXT_WINDOW or use a smaller checkpoint."
            )

    def clear_runtime_pools(self, model_name: str | None = None) -> int:
        with self._lock:
            names = [model_name] if model_name else list(self._engines)
            cleared = 0
            for name in names:
                engine = self._engines.get(name)
                if engine is None:
                    continue
                engine.clear_state_pool()
                cleared += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return cleared

    def recover_runtime_oom(self, model_name: str | None = None) -> bool:
        name = model_name or self.default_model_name()
        recovered = self.unload(name)
        self._drop_batch_runners(name)
        self.batch_scheduler.clear()
        if self.kv_block_table is not None:
            self.kv_block_table.clear()
        status = self._status.get(name)
        if status is not None:
            status.state = "unloaded"
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        return recovered

    def mark_runtime_error(self, model_name: str, exc: BaseException) -> None:
        with self._lock:
            status = self._status.get(model_name)
            if status is not None:
                status.last_error = str(exc)
                if status.state != "loaded":
                    status.state = "unloaded"


def load_model_specs(path: Path, default_features: RuntimeFeatures) -> list[ModelResourceSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else payload.get("models", None)
    if not isinstance(items, list):
        raise ValueError("models json must be a list or an object with a 'models' list")
    specs: list[ModelResourceSpec] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("each model spec must be an object")
        specs.append(ModelResourceSpec.from_mapping(item, default_features))
    return specs


def _env_enabled(name: str, *, default: bool) -> bool:
    import os

    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "off", "no"}
