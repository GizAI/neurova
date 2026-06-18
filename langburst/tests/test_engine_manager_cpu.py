from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import langburst.adapters  # noqa: F401
import torch
from langburst.core.adapter import adapter_registry
from langburst.core.features import RuntimeFeatures
from langburst.engines.native.manager import EngineManager, EngineResourcePolicy, ModelResourceSpec, RuntimeMemoryPressure
from langburst.engines.native.runtime import GenerationConfig
from langburst.server import create_app

from test_adapter_runtime_cpu import ToyAdapter


def test_engine_manager_lazy_loads_and_evicts(tmp_path: Path):
    adapter = ToyAdapter()
    try:
        adapter_registry.register(adapter)
    except ValueError:
        pass
    manager = EngineManager(
        [
            ModelResourceSpec("toy-a", "toy", tmp_path, tmp_path, device="cpu", weight_device="cpu"),
            ModelResourceSpec("toy-b", "toy", tmp_path, tmp_path, device="cpu", weight_device="cpu"),
        ],
        policy=EngineResourcePolicy(max_loaded_models=1, max_active_requests=1),
    )
    first = manager.get("toy-a")
    assert first.model_name == "toy-a"
    assert manager.list_models()[0]["loaded"]
    assert manager.list_models()[0]["status"]["state"] == "loaded"
    second = manager.get("toy-b")
    assert second.model_name == "toy-b"
    assert "toy-a" not in manager._engines
    assert "toy-b" in manager._engines
    models = {row["id"]: row for row in manager.list_models()}
    assert models["toy-a"]["status"]["state"] == "unloaded"
    assert models["toy-a"]["status"]["unload_count"] == 1


def test_server_models_endpoint_uses_manager(tmp_path: Path):
    adapter = ToyAdapter()
    try:
        adapter_registry.register(adapter)
    except ValueError:
        pass
    manager = EngineManager(
        [ModelResourceSpec("toy-a", "toy", tmp_path, tmp_path, device="cpu", weight_device="cpu")],
        policy=EngineResourcePolicy(max_loaded_models=1, max_active_requests=1),
    )
    app = create_app(manager)
    route_paths = {route.path for route in app.routes}
    assert "/v1/models" in route_paths
    assert "/v1/langburst/models" in route_paths
    assert "/v1/langburst/health" in route_paths
    assert "/v1/langburst/models/{model_name}" in route_paths


def test_models_endpoint_reports_serving_concurrency(tmp_path: Path):
    adapter = ToyAdapter()
    try:
        adapter_registry.register(adapter)
    except ValueError:
        pass
    manager = EngineManager(
        [ModelResourceSpec("toy-a", "toy", tmp_path, tmp_path, device="cpu", weight_device="cpu")],
        policy=EngineResourcePolicy(max_loaded_models=1, max_active_requests=2),
    )

    model = manager.list_models()[0]

    assert model["capabilities"]["max_concurrency"] == 2
    assert manager.health()["models"][0]["capabilities"]["max_concurrency"] == 2


def test_server_request_model_can_fall_back_to_manager_default(tmp_path: Path):
    adapter = ToyAdapter()
    try:
        adapter_registry.register(adapter)
    except ValueError:
        pass
    manager = EngineManager(
        [ModelResourceSpec("custom-default", "toy", tmp_path, tmp_path, device="cpu", weight_device="cpu")],
        policy=EngineResourcePolicy(max_loaded_models=1, max_active_requests=1),
    )
    app = create_app(manager)
    assert app is not None
    engine = manager.get(None)
    assert engine.model_name == "custom-default"


def test_server_non_stream_greedy_uses_batch_worker(tmp_path: Path):
    from fastapi.testclient import TestClient

    adapter = ToyAdapter()
    try:
        adapter_registry.register(adapter)
    except ValueError:
        pass
    manager = EngineManager(
        [ModelResourceSpec("toy-a", "toy", tmp_path, tmp_path, device="cpu", weight_device="cpu")],
        policy=EngineResourcePolicy(max_active_requests=2, max_num_batched_tokens=4, prefill_chunk_size=2),
    )
    app = create_app(manager)
    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "toy-a",
            "messages": [{"role": "user", "content": "ab"}],
            "max_tokens": 2,
            "temperature": 0,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "23"
    assert payload["usage"]["prompt_tokens"] >= 1
    assert payload["usage"]["completion_tokens"] == 2
    assert payload["usage"]["total_tokens"] == payload["usage"]["prompt_tokens"] + 2
    assert payload["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    assert payload["usage"]["completion_tokens_details"]["reasoning_tokens"] == 0
    assert payload["usage"]["performance"]["e2e_tok_s"] is not None
    assert manager.health()["batch_workers"]["workers"] == 1
    assert manager.health()["batch_scheduler"]["total_scheduled_batches"] >= 1


def test_server_reports_prefix_cache_usage_tokens(tmp_path: Path):
    from fastapi.testclient import TestClient

    adapter = ToyAdapter()
    try:
        adapter_registry.register(adapter)
    except ValueError:
        pass
    manager = EngineManager(
        [ModelResourceSpec("toy-a", "toy", tmp_path, tmp_path, device="cpu", weight_device="cpu")],
        policy=EngineResourcePolicy(max_active_requests=2, max_num_batched_tokens=4, prefill_chunk_size=2, kv_block_size=2),
    )
    client = TestClient(create_app(manager))
    body = {
        "model": "toy-a",
        "messages": [{"role": "user", "content": "abc"}],
        "max_tokens": 1,
        "temperature": 0,
        "prompt_cache_key": "shared-test",
        "prefix_cache": True,
    }
    first = client.post("/v1/chat/completions", json=body)
    second = client.post("/v1/chat/completions", json=body)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    assert second.json()["usage"]["prompt_tokens_details"]["cached_tokens"] >= 2


def test_server_generation_options_are_applied_end_to_end(tmp_path: Path):
    from fastapi.testclient import TestClient

    adapter = ToyAdapter()
    try:
        adapter_registry.register(adapter)
    except ValueError:
        pass
    manager = EngineManager(
        [ModelResourceSpec("toy-a", "toy", tmp_path, tmp_path, device="cpu", weight_device="cpu")],
        policy=EngineResourcePolicy(max_active_requests=2, max_num_batched_tokens=4, prefill_chunk_size=2),
    )
    client = TestClient(create_app(manager))

    biased = client.post(
        "/v1/chat/completions",
        json={
            "model": "toy-a",
            "messages": [{"role": "user", "content": "ab"}],
            "max_tokens": 1,
            "temperature": 0,
            "logit_bias": {"5": 3000},
        },
    )
    assert biased.status_code == 200
    assert biased.json()["choices"][0]["message"]["content"] == "5"

    stopped = client.post(
        "/v1/chat/completions",
        json={
            "model": "toy-a",
            "messages": [{"role": "user", "content": "a"}],
            "max_tokens": 3,
            "temperature": 0,
            "stop": "2",
        },
    )
    assert stopped.status_code == 200
    assert stopped.json()["choices"][0]["message"]["content"] == "1"


def test_server_rejects_removed_native_sessions_before_generation(tmp_path: Path):
    from fastapi.testclient import TestClient

    adapter = ToyAdapter()
    try:
        adapter_registry.register(adapter)
    except ValueError:
        pass
    manager = EngineManager(
        [ModelResourceSpec("toy-a", "toy", tmp_path, tmp_path, device="cpu", weight_device="cpu")],
        policy=EngineResourcePolicy(max_active_requests=2, max_num_batched_tokens=4, prefill_chunk_size=2),
    )
    client = TestClient(create_app(manager))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "toy-a",
            "messages": [{"role": "user", "content": "a"}],
            "max_tokens": 1,
            "session_id": "sess-test",
        },
    )

    assert response.status_code == 413
    assert "sessions were removed" in response.json()["detail"]


def test_server_history_messages_replace_removed_sessions(tmp_path: Path):
    from fastapi.testclient import TestClient

    adapter = ToyAdapter()
    try:
        adapter_registry.register(adapter)
    except ValueError:
        pass
    manager = EngineManager(
        [ModelResourceSpec("toy-a", "toy", tmp_path, tmp_path, device="cpu", weight_device="cpu")],
        policy=EngineResourcePolicy(max_active_requests=2, max_num_batched_tokens=4, prefill_chunk_size=2),
    )
    client = TestClient(create_app(manager))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "toy-a",
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "1"},
                {"role": "user", "content": "b"},
            ],
            "max_tokens": 1,
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "4"


def test_server_stream_greedy_uses_batch_worker(tmp_path: Path):
    from fastapi.testclient import TestClient

    adapter = ToyAdapter()
    try:
        adapter_registry.register(adapter)
    except ValueError:
        pass
    manager = EngineManager(
        [ModelResourceSpec("toy-a", "toy", tmp_path, tmp_path, device="cpu", weight_device="cpu")],
        policy=EngineResourcePolicy(max_active_requests=2, max_num_batched_tokens=4, prefill_chunk_size=2),
    )
    app = create_app(manager)
    with TestClient(app).stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "toy-a",
            "messages": [{"role": "user", "content": "ab"}],
            "max_tokens": 2,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert '"content": "2"' in body
    assert '"content": "3"' in body
    assert '"usage"' in body
    assert '"completion_tokens": 2' in body
    assert "data: [DONE]" in body
    assert manager.health()["batch_workers"]["workers"] == 1
    assert manager.health()["batch_scheduler"]["total_scheduled_batches"] >= 1


def test_server_rejects_over_limit_prompt_before_generation(tmp_path: Path):
    from fastapi.testclient import TestClient

    adapter = ToyAdapter()
    try:
        adapter_registry.register(adapter)
    except ValueError:
        pass
    manager = EngineManager(
        [ModelResourceSpec("toy-a", "toy", tmp_path, tmp_path, device="cpu", weight_device="cpu")],
        policy=EngineResourcePolicy(max_prompt_tokens=2, max_generation_tokens=4, allow_context_overflow=False),
    )
    app = create_app(manager)
    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "toy-a",
            "messages": [{"role": "user", "content": "abcdef"}],
            "max_tokens": 1,
            "temperature": 0,
        },
    )
    assert response.status_code == 413
    assert "prompt too long" in response.text
    assert manager.admission.stats().total_admitted == 0


def test_resolve_plan_does_not_lazy_load_engine(tmp_path: Path):
    adapter = ToyAdapter()
    try:
        adapter_registry.register(adapter)
    except ValueError:
        pass
    manager = EngineManager(
        [ModelResourceSpec("toy-a", "toy", tmp_path, tmp_path, device="cpu", weight_device="cpu")],
        policy=EngineResourcePolicy(max_loaded_models=1, max_active_requests=1),
    )
    plan = manager.resolve_plan("toy-a")
    assert plan.effective.block_prefill
    assert manager._engines == {}


def test_engine_manager_unload(tmp_path: Path):
    adapter = ToyAdapter()
    try:
        adapter_registry.register(adapter)
    except ValueError:
        pass
    manager = EngineManager(
        [ModelResourceSpec("toy-a", "toy", tmp_path, tmp_path, device="cpu", weight_device="cpu")],
        policy=EngineResourcePolicy(max_loaded_models=1, max_active_requests=1),
    )
    manager.get("toy-a")
    assert manager.unload("toy-a")
    assert not manager.unload("toy-a")
    row = manager.list_models()[0]
    assert not row["loaded"]
    assert row["status"]["state"] == "unloaded"


def test_engine_manager_clears_state_pool_on_unload(tmp_path: Path):
    adapter = ToyAdapter()
    try:
        adapter_registry.register(adapter)
    except ValueError:
        pass
    manager = EngineManager(
        [ModelResourceSpec("toy-a", "toy", tmp_path, tmp_path, device="cpu", weight_device="cpu")],
        policy=EngineResourcePolicy(max_loaded_models=1, max_active_requests=1, max_state_pool_size=1),
    )
    engine = manager.get("toy-a")
    engine.completion_ids_greedy_gpu([{"role": "user", "content": "ab"}], GenerationConfig(max_new_tokens=1, eos_token_ids=()))
    assert engine.state_pool_summary()["pooled_states"] == 1
    assert manager.unload("toy-a")
    assert manager.health()["state_pools"] == {}


def test_engine_manager_summary_is_server_contract(tmp_path: Path):
    adapter = ToyAdapter()
    try:
        adapter_registry.register(adapter)
    except ValueError:
        pass
    manager = EngineManager(
        [ModelResourceSpec("toy-a", "toy", tmp_path, tmp_path, device="cpu", weight_device="cpu")],
        policy=EngineResourcePolicy(max_loaded_models=1, max_active_requests=1, max_queued_requests=2),
    )
    summary = manager.summary()
    assert summary["data"][0]["id"] == "toy-a"
    assert summary["policy"]["max_queued_requests"] == 2
    from langburst.core.defaults import DEFAULT_MAX_GENERATION_TOKENS, max_prompt_tokens_default, max_state_pool_size_default

    assert summary["policy"]["max_state_pool_size"] == max_state_pool_size_default()
    assert summary["policy"]["max_prompt_tokens"] == max_prompt_tokens_default()
    assert summary["policy"]["max_generation_tokens"] == DEFAULT_MAX_GENERATION_TOKENS
    assert summary["policy"]["max_num_batched_tokens"] == 256
    assert summary["policy"]["prefill_chunk_size"] == 64
    assert summary["admission"]["max_active_requests"] == 1
    assert summary["batch_scheduler"]["max_num_batched_tokens"] == 256
    assert summary["resource"]["loaded_model_count"] == 0


def test_engine_manager_owns_vllm_style_batch_resources(tmp_path: Path):
    adapter = ToyAdapter()
    try:
        adapter_registry.register(adapter)
    except ValueError:
        pass
    manager = EngineManager(
        [ModelResourceSpec("toy-a", "toy", tmp_path, tmp_path, device="cpu", weight_device="cpu")],
        policy=EngineResourcePolicy(
            max_active_requests=2,
            max_num_batched_tokens=8,
            prefill_chunk_size=4,
            kv_block_size=4,
            kv_blocks=8,
        ),
    )

    row = manager.batch_scheduler.add_request("r1", [1, 2, 3, 4, 5])
    batch = manager.batch_scheduler.schedule()

    assert row.request_id == "r1"
    assert batch is not None
    assert batch.num_scheduled_tokens == [4]
    assert manager.kv_block_table.summary()["used_blocks"] == 2
    health = manager.health()
    assert health["batch_scheduler"]["prefill_chunk_size"] == 4
    assert health["kv_blocks"]["block_size"] == 4


def test_engine_manager_creates_batch_runner(tmp_path: Path):
    adapter = ToyAdapter()
    try:
        adapter_registry.register(adapter)
    except ValueError:
        pass
    manager = EngineManager(
        [ModelResourceSpec("toy-a", "toy", tmp_path, tmp_path, device="cpu", weight_device="cpu")],
        policy=EngineResourcePolicy(max_active_requests=2, max_num_batched_tokens=4, prefill_chunk_size=2),
    )

    runner = manager.create_batch_runner("toy-a")
    row = runner.add_request("r1", [1, 2])
    step = runner.execute_step(device="cpu")

    assert row.computed_tokens == 2
    assert step is not None
    assert step.batch.request_ids == ["r1"]
    assert manager.summary()["batch_runners"]["runners"] == 1


def test_engine_manager_reuses_batch_runner_per_model_and_feature_plan(tmp_path: Path):
    adapter = ToyAdapter()
    try:
        adapter_registry.register(adapter)
    except ValueError:
        pass
    manager = EngineManager(
        [ModelResourceSpec("toy-a", "toy", tmp_path, tmp_path, device="cpu", weight_device="cpu")],
        policy=EngineResourcePolicy(max_active_requests=2, max_num_batched_tokens=4, prefill_chunk_size=2),
    )

    first = manager.create_batch_runner("toy-a")
    second = manager.create_batch_runner("toy-a")

    assert first is second
    row = first.add_request("r1", [1, 2])
    assert manager.health()["batch_runners"]["state_stores"]["toy-a"]["allocated_states"] == 1
    assert first.finish_request(row.request_id) is row
    assert manager.health()["batch_runners"]["state_stores"]["toy-a"]["allocated_states"] == 0


def test_engine_manager_does_not_duplicate_runner_for_request_level_feature_flags(tmp_path: Path):
    adapter = ToyAdapter()
    try:
        adapter_registry.register(adapter)
    except ValueError:
        pass
    manager = EngineManager(
        [ModelResourceSpec("toy-a", "toy", tmp_path, tmp_path, device="cpu", weight_device="cpu")],
        policy=EngineResourcePolicy(max_active_requests=2, max_num_batched_tokens=4, prefill_chunk_size=2),
    )
    base = RuntimeFeatures.from_profile("stateful")

    first = manager.create_batch_worker("toy-a", base.with_overrides(prefix_cache=False, speculative_decoding=False))
    second = manager.create_batch_worker("toy-a", base.with_overrides(prefix_cache=True, speculative_decoding=True))

    assert first is second
    assert first.runner.features.prefix_cache is False
    assert first.runner.features.speculative_decoding is False
    assert manager.health()["batch_runners"]["runners"] == 1
    assert manager.health()["batch_workers"]["workers"] == 1


def test_batch_worker_honors_request_prefix_cache_flag_without_new_runner(tmp_path: Path):
    adapter = ToyAdapter()
    try:
        adapter_registry.register(adapter)
    except ValueError:
        pass
    manager = EngineManager(
        [ModelResourceSpec("toy-a", "toy", tmp_path, tmp_path, device="cpu", weight_device="cpu")],
        policy=EngineResourcePolicy(max_active_requests=2, max_num_batched_tokens=4, prefill_chunk_size=2, kv_block_size=2),
    )
    base = RuntimeFeatures.from_profile("stateful")
    worker = manager.create_batch_worker("toy-a", base.with_overrides(prefix_cache=True))
    first = worker.submit([1, 2, 3], max_new_tokens=1, prompt_cache_key="shared", prefix_cache_enabled=True, request_id="r1")
    assert first.wait_ids(timeout=2.0)
    second = worker.submit([1, 2, 4], max_new_tokens=1, prompt_cache_key="shared", prefix_cache_enabled=False, request_id="r2")
    assert second.wait_ids(timeout=2.0)

    assert first.cached_input_tokens == 0
    assert second.cached_input_tokens == 0
    assert manager.health()["batch_runners"]["runners"] == 1


def test_engine_manager_creates_cached_batch_worker(tmp_path: Path):
    adapter = ToyAdapter()
    try:
        adapter_registry.register(adapter)
    except ValueError:
        pass
    manager = EngineManager(
        [ModelResourceSpec("toy-a", "toy", tmp_path, tmp_path, device="cpu", weight_device="cpu")],
        policy=EngineResourcePolicy(max_active_requests=2, max_num_batched_tokens=4, prefill_chunk_size=2),
    )

    first = manager.create_batch_worker("toy-a")
    second = manager.create_batch_worker("toy-a")
    handle = first.submit([1, 2], max_new_tokens=1, request_id="r1")

    assert first is second
    assert handle.wait_ids(timeout=2.0) == [2]
    assert manager.health()["batch_workers"]["workers"] == 1
    manager.unload("toy-a")
    assert manager.health()["batch_workers"]["workers"] == 0


def test_engine_manager_generation_admission_limits(tmp_path: Path):
    adapter = ToyAdapter()
    try:
        adapter_registry.register(adapter)
    except ValueError:
        pass
    manager = EngineManager(
        [ModelResourceSpec("toy-a", "toy", tmp_path, tmp_path, device="cpu", weight_device="cpu")],
        policy=EngineResourcePolicy(max_prompt_tokens=4, max_generation_tokens=2, allow_context_overflow=False),
    )
    manager.validate_generation_request(prompt_tokens=4, generation_tokens=2)
    try:
        manager.validate_generation_request(prompt_tokens=5, generation_tokens=2)
        raise AssertionError("prompt limit should fail")
    except ValueError as exc:
        assert "prompt too long" in str(exc)
    try:
        manager.validate_generation_request(prompt_tokens=4, generation_tokens=3)
        raise AssertionError("generation limit should fail")
    except ValueError as exc:
        assert "generation too long" in str(exc)


def test_engine_manager_allows_overflow_prompt_with_bounded_admission_tokens(tmp_path: Path):
    adapter = ToyAdapter()
    try:
        adapter_registry.register(adapter)
    except ValueError:
        pass
    manager = EngineManager(
        [ModelResourceSpec("toy-overflow", "toy", tmp_path, tmp_path, device="cpu", weight_device="cpu")],
        policy=EngineResourcePolicy(
            max_prompt_tokens=16,
            max_generation_tokens=4,
            context_tiers=(4, 16),
            context_tier_slots=(1, 1),
            allow_context_overflow=True,
        ),
    )

    manager.validate_generation_request(prompt_tokens=100, generation_tokens=4)
    assert manager.effective_admission_tokens(prompt_tokens=100, generation_tokens=4) == 20


def test_active_runtime_memory_pressure_is_caught_before_generation(monkeypatch, tmp_path: Path):
    adapter = ToyAdapter()
    try:
        adapter_registry.register(adapter)
    except ValueError:
        pass
    manager = EngineManager(
        [ModelResourceSpec("toy-a", "toy", tmp_path, tmp_path, device="cuda", weight_device="cpu")],
        policy=EngineResourcePolicy(reserve_free_vram_mib=512),
    )
    engine = SimpleNamespace(
        device="cuda",
        model_name="toy-a",
        adapter=SimpleNamespace(estimate_arena_state_bytes=None),
        cfg=object(),
        features=object(),
        estimated_weight_bytes=lambda: 0,
        estimated_state_bytes=lambda: 0,
    )
    manager._engines["toy-a"] = engine
    manager._batch_runners[("toy-a", ())] = object()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "ipc_collect", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args, **kwargs: (19 * 1024 * 1024, 16 * 1024 * 1024 * 1024))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *args, **kwargs: SimpleNamespace(total_memory=16 * 1024 * 1024 * 1024))

    try:
        manager.validate_runtime_memory(engine)
        raise AssertionError("low active runtime headroom should fail before model forward")
    except RuntimeMemoryPressure as exc:
        assert "active runtime" in str(exc)


def test_active_runtime_memory_validation_clears_model_runtime_cache(monkeypatch, tmp_path: Path):
    adapter = ToyAdapter()
    try:
        adapter_registry.register(adapter)
    except ValueError:
        pass
    manager = EngineManager(
        [ModelResourceSpec("toy-a", "toy", tmp_path, tmp_path, device="cuda", weight_device="cpu")],
        policy=EngineResourcePolicy(reserve_free_vram_mib=512),
    )
    cleared = {"count": 0}

    def clear_runtime_caches():
        cleared["count"] += 1

    engine = SimpleNamespace(
        device="cuda",
        model_name="toy-a",
        model=SimpleNamespace(clear_runtime_caches=clear_runtime_caches),
        adapter=SimpleNamespace(estimate_arena_state_bytes=None),
        cfg=object(),
        features=object(),
        estimated_weight_bytes=lambda: 0,
        estimated_state_bytes=lambda: 0,
    )
    manager._engines["toy-a"] = engine
    manager._batch_runners[("toy-a", ())] = object()
    free_values = iter([19 * 1024 * 1024, 768 * 1024 * 1024])

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "ipc_collect", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args, **kwargs: (next(free_values), 16 * 1024 * 1024 * 1024))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *args, **kwargs: SimpleNamespace(total_memory=16 * 1024 * 1024 * 1024))

    manager.validate_runtime_memory(engine)

    assert cleared["count"] == 1


def test_active_runtime_memory_reserve_scales_with_active_requests(monkeypatch, tmp_path: Path):
    adapter = ToyAdapter()
    try:
        adapter_registry.register(adapter)
    except ValueError:
        pass
    manager = EngineManager(
        [ModelResourceSpec("toy-a", "toy", tmp_path, tmp_path, device="cuda", weight_device="cpu")],
        policy=EngineResourcePolicy(reserve_free_vram_mib=512),
    )
    engine = SimpleNamespace(
        device="cuda",
        model_name="toy-a",
        adapter=SimpleNamespace(estimate_arena_state_bytes=None),
        cfg=object(),
        features=object(),
        estimated_weight_bytes=lambda: 0,
        estimated_state_bytes=lambda: 0,
    )
    manager._engines["toy-a"] = engine
    manager._batch_runners[("toy-a", ())] = object()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "ipc_collect", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args, **kwargs: (768 * 1024 * 1024, 16 * 1024 * 1024 * 1024))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *args, **kwargs: SimpleNamespace(total_memory=16 * 1024 * 1024 * 1024))

    manager.validate_runtime_memory(engine, active_requests=1)
    try:
        manager.validate_runtime_memory(engine, active_requests=2)
        raise AssertionError("two active requests must reserve two execution headrooms")
    except RuntimeMemoryPressure as exc:
        assert "active_requests=2" in str(exc)


def test_memory_aware_request_lease_retries_while_another_request_is_active(monkeypatch, tmp_path: Path):
    adapter = ToyAdapter()
    try:
        adapter_registry.register(adapter)
    except ValueError:
        pass
    manager = EngineManager(
        [ModelResourceSpec("toy-a", "toy", tmp_path, tmp_path, device="cuda", weight_device="cpu")],
        policy=EngineResourcePolicy(
            max_active_requests=2,
            max_queued_requests=2,
            admission_timeout_s=1.0,
            reserve_free_vram_mib=512,
            context_tiers=(4, 16),
            context_tier_slots=(1, 1),
        ),
    )
    engine = SimpleNamespace(
        device="cuda",
        model_name="toy-a",
        adapter=SimpleNamespace(estimate_arena_state_bytes=None),
        cfg=object(),
        features=object(),
        estimated_weight_bytes=lambda: 0,
        estimated_state_bytes=lambda: 0,
    )
    manager._engines["toy-a"] = engine
    manager._batch_runners[("toy-a", ())] = object()

    free_values = iter([
        19 * 1024 * 1024,
        19 * 1024 * 1024,
        1024 * 1024 * 1024,
    ])

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "ipc_collect", lambda: None)
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda *args, **kwargs: (next(free_values), 16 * 1024 * 1024 * 1024),
    )
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *args, **kwargs: SimpleNamespace(total_memory=16 * 1024 * 1024 * 1024))

    with manager.acquire_request(prompt_tokens=12):
        with manager.acquire_request(prompt_tokens=4, engine=engine):
            stats = manager.admission.stats()
            assert stats.active_requests == 2


def test_memory_aware_request_lease_does_not_wait_on_itself(monkeypatch, tmp_path: Path):
    adapter = ToyAdapter()
    try:
        adapter_registry.register(adapter)
    except ValueError:
        pass
    manager = EngineManager(
        [ModelResourceSpec("toy-a", "toy", tmp_path, tmp_path, device="cuda", weight_device="cpu")],
        policy=EngineResourcePolicy(
            max_active_requests=2,
            max_queued_requests=2,
            admission_timeout_s=1.0,
            reserve_free_vram_mib=512,
        ),
    )
    engine = SimpleNamespace(
        device="cuda",
        model_name="toy-a",
        adapter=SimpleNamespace(estimate_arena_state_bytes=None),
        cfg=object(),
        features=object(),
        estimated_weight_bytes=lambda: 0,
        estimated_state_bytes=lambda: 0,
    )
    manager._engines["toy-a"] = engine
    manager._batch_runners[("toy-a", ())] = object()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "ipc_collect", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args, **kwargs: (19 * 1024 * 1024, 16 * 1024 * 1024 * 1024))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *args, **kwargs: SimpleNamespace(total_memory=16 * 1024 * 1024 * 1024))

    try:
        with manager.acquire_request(prompt_tokens=4, engine=engine):
            raise AssertionError("self-only pressure should not be admitted")
    except RuntimeMemoryPressure:
        pass

    assert manager.admission.stats().active_requests == 0


def test_resource_policy_validates_limits():
    try:
        EngineResourcePolicy(max_loaded_models=0)
        raise AssertionError("invalid policy should fail")
    except ValueError as exc:
        assert "max_loaded_models" in str(exc)
    try:
        EngineResourcePolicy(reserve_free_vram_mib=-1)
        raise AssertionError("invalid reserve should fail")
    except ValueError as exc:
        assert "reserve_free_vram_mib" in str(exc)
    try:
        EngineResourcePolicy(max_state_pool_size=-1)
        raise AssertionError("invalid state pool should fail")
    except ValueError as exc:
        assert "max_state_pool_size" in str(exc)
    try:
        EngineResourcePolicy(max_prompt_tokens=0)
        raise AssertionError("invalid prompt limit should fail")
    except ValueError as exc:
        assert "max_prompt_tokens" in str(exc)
    try:
        EngineResourcePolicy(max_generation_tokens=0)
        raise AssertionError("invalid generation limit should fail")
    except ValueError as exc:
        assert "max_generation_tokens" in str(exc)
