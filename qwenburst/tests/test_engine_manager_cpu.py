from __future__ import annotations

from pathlib import Path

import qwenburst.adapters  # noqa: F401
from qwenburst.core.adapter import adapter_registry
from qwenburst.core.manager import EngineManager, EngineResourcePolicy, ModelResourceSpec
from qwenburst.core.runtime import GenerationConfig
from qwenburst.server import create_app

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
    assert "/v1/qwenburst/models" in route_paths
    assert "/v1/qwenburst/health" in route_paths
    assert "/v1/qwenburst/models/{model_name}" in route_paths


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
    assert response.json()["choices"][0]["message"]["content"] == "23"
    assert manager.health()["batch_workers"]["workers"] == 1
    assert manager.health()["batch_scheduler"]["total_scheduled_batches"] >= 1


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
        },
    ) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert '"content": "2"' in body
    assert '"content": "3"' in body
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
        policy=EngineResourcePolicy(max_prompt_tokens=2, max_generation_tokens=4),
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
    assert manager.scheduler.stats().total_admitted == 0


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
    assert summary["policy"]["max_state_pool_size"] == 1
    assert summary["policy"]["max_prompt_tokens"] == 4096
    assert summary["policy"]["max_generation_tokens"] == 1024
    assert summary["policy"]["max_num_batched_tokens"] == 256
    assert summary["policy"]["prefill_chunk_size"] == 64
    assert summary["scheduler"]["max_active_requests"] == 1
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
        policy=EngineResourcePolicy(max_prompt_tokens=4, max_generation_tokens=2),
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
