from __future__ import annotations

import pytest

from langburst.engines import ensure_engines_loaded, engine_registry
from langburst.engines.base import EngineFeatureRequest, EngineModelSpec
from langburst.engines.vllm.bridge import vllm_engine_extra_kwargs


def test_default_engine_is_native():
    ensure_engines_loaded()
    assert engine_registry.default_engine_id() == "native"
    assert {"vllm", "sglang", "exl3", "native"}.issubset(set(engine_registry.ids()))


def test_engine_descriptors_have_capabilities():
    ensure_engines_loaded()
    rows = {descriptor.engine_id: descriptor for descriptor in engine_registry.list()}
    assert rows["native"].default
    assert rows["vllm"].capabilities.continuous_batching
    assert rows["vllm"].capabilities.paged_kv
    assert rows["sglang"].capabilities.structured_output
    assert rows["exl3"].capabilities.quantization == ("exl3",)
    assert rows["native"].capabilities.custom_model


def test_sglang_and_exl3_are_registered_but_explicitly_unavailable():
    ensure_engines_loaded()
    for engine_id in ("sglang", "exl3"):
        backend = engine_registry.create(EngineModelSpec(model="dummy"), engine_id=engine_id)
        with pytest.raises(RuntimeError):
            backend.start()


def test_vllm_backend_fails_fast_when_dependency_missing():
    ensure_engines_loaded()
    backend = engine_registry.create(EngineModelSpec(model="dummy"), engine_id="vllm")
    try:
        import vllm  # noqa: F401
    except Exception:
        with pytest.raises(RuntimeError, match="vLLM"):
            backend.start()


def test_langburst_feature_plan_is_shared_across_vllm_and_native():
    ensure_engines_loaded()
    features = EngineFeatureRequest(
        qwen36_lowbit=True,
        ring_kv=True,
        recurrent_state=True,
        infinite_context=True,
        episodic_memory=True,
        ttt_sidecar=True,
        stateful_sessions=True,
    )
    vllm_backend = engine_registry.create(EngineModelSpec(model="dummy", features=features), engine_id="vllm")
    native_backend = engine_registry.create(
        EngineModelSpec(model="dummy", features=features, extra={"qb_model": "dummy-qb"}),
        engine_id="native",
    )
    assert vllm_backend.feature_plan.requested == features
    assert native_backend.feature_plan.requested == features
    assert not vllm_backend.feature_plan.summary()["unsupported"]
    assert not native_backend.feature_plan.summary()["unsupported"]
    assert vllm_backend.feature_plan.summary()["support"]["qwen36_lowbit"] == "bridge"
    assert vllm_backend.feature_plan.summary()["support"]["ring_kv"] == "bridge"
    assert vllm_backend.feature_plan.summary()["support"]["recurrent_state"] == "bridge"
    assert vllm_backend.feature_plan.summary()["support"]["stateful_sessions"] == "host"
    assert native_backend.feature_plan.summary()["support"]["qwen36_lowbit"] == "engine"


def test_vllm_bridge_carries_langburst_native_feature_metadata():
    ensure_engines_loaded()
    features = EngineFeatureRequest(
        qwen36_lowbit=True,
        ring_kv=True,
        recurrent_state=True,
        infinite_context=True,
        episodic_memory=True,
        ttt_sidecar=True,
        stateful_sessions=True,
    )
    backend = engine_registry.create(
        EngineModelSpec(
            model="dummy-hf",
            features=features,
            extra={
                "qb_model": "dummy-qb",
                "langburst_quantization_config": {"format": "langburst-lowbit", "bits": 4},
            },
        ),
        engine_id="vllm",
    )
    bridge = backend.bridge.summary()
    kwargs = bridge["engine_kwargs"]

    assert kwargs["enable_prefix_caching"]
    assert kwargs["load_format"] == "langburst_lowbit"
    assert kwargs["quantization"] == "langburst_lowbit"
    assert kwargs["model_loader_extra_config"] == {"qb_model": "dummy-qb"}
    assert "architectures" not in kwargs["hf_overrides"]
    assert kwargs["hf_overrides"]["langburst_quantization_config"] == {"format": "langburst-lowbit", "bits": 4}
    assert kwargs["hf_overrides"]["quantization_config"]["quant_method"] == "langburst_lowbit"
    assert kwargs["hf_overrides"]["langburst_qb_model"] == "dummy-qb"
    assert kwargs["hf_overrides"]["langburst_qwen36_lowbit"]
    assert kwargs["hf_overrides"]["langburst_kv_policy"] == "ring"
    assert kwargs["hf_overrides"]["langburst_recurrent_state"]
    assert kwargs["hf_overrides"]["langburst_sidecars"] == {"episodic_memory": True, "ttt_sidecar": True}
    assert not bridge["requires_custom_model"]
    assert "RuntimeEngine" in bridge["metadata"]["excluded_native_runtime"]
    assert "scheduler" in bridge["metadata"]["vllm_owned"]
    assert "qwen36_lowbit_checkpoint_loader" in bridge["metadata"]["langburst_qwen36_bridge"]


def test_vllm_mtp_is_explicit_opt_in():
    ensure_engines_loaded()
    features = EngineFeatureRequest(qwen36_lowbit=True, recurrent_state=True)
    backend = engine_registry.create(
        EngineModelSpec(
            model="dummy-hf",
            features=features,
            extra={"qb_model": "dummy-qb", "enable_mtp": True, "mtp_speculative_tokens": 3},
        ),
        engine_id="vllm",
    )
    kwargs = backend.bridge.summary()["engine_kwargs"]

    assert kwargs["speculative_config"] == {"method": "mtp", "num_speculative_tokens": 3}


def test_vllm_engine_kwargs_do_not_leak_native_runtime_extras():
    mixed = {
        "adapter": "qwen36",
        "qb_model": "/tmp/qb",
        "runtime_profile": "stateful",
        "vllm_custom_model": "LangBurstQwen36ForCausalLM",
        "max_num_batched_tokens": 4096,
        "vllm_arg_enable_prefix_caching": True,
    }
    assert vllm_engine_extra_kwargs(mixed) == {
        "max_num_batched_tokens": 4096,
        "enable_prefix_caching": True,
    }
