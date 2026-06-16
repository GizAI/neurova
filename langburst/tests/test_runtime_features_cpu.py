from __future__ import annotations

import argparse

from langburst.cli_features import add_model_path_args, add_runtime_feature_args, runtime_feature_override_from_args, runtime_features_from_args
from langburst.core.features import RuntimeCapabilities, RuntimeFeatureOverride, RuntimeFeatures, resolve_runtime_plan
from langburst.tuning import batch_conv_kernels_enabled, batch_gdn_kernels_enabled, batch_state_kernels_enabled


def parse_features(args: list[str]) -> RuntimeFeatures:
    parser = argparse.ArgumentParser()
    add_runtime_feature_args(parser)
    return runtime_features_from_args(parser.parse_args(args))


def test_runtime_profile_original_disables_stateful_extras():
    features = RuntimeFeatures.from_profile("original")
    assert features.kv_window_policy == "error"
    assert not features.stateful_chat
    assert features.state_pool
    assert not features.snapshots
    assert features.gpu_sampling
    assert not features.speculative_decoding
    assert not features.cuda_graph
    assert features.block_prefill
    assert not features.infinite_streaming
    assert not features.episodic_memory
    assert not features.ttt_sidecar
    assert features.prefill_chunk_size == 64


def test_batch_kernel_debug_flags_inherit_parent_and_can_split(monkeypatch):
    monkeypatch.delenv("LANGBURST_BATCH_STATE_KERNELS", raising=False)
    monkeypatch.delenv("LANGBURST_BATCH_CONV_KERNELS", raising=False)
    monkeypatch.delenv("LANGBURST_BATCH_GDN_KERNELS", raising=False)
    assert batch_state_kernels_enabled()
    assert batch_conv_kernels_enabled()
    assert batch_gdn_kernels_enabled()


def test_model_path_args_use_env_defaults(monkeypatch):
    monkeypatch.setenv("LANGBURST_HF_MODEL", "/models/hf")
    monkeypatch.setenv("LANGBURST_QB_MODEL", "/models/qb")
    parser = argparse.ArgumentParser()
    add_model_path_args(parser)
    args = parser.parse_args([])

    assert str(args.hf_model) == "/models/hf"
    assert str(args.qb_model) == "/models/qb"


def test_model_path_args_accept_explicit_paths_without_env(monkeypatch):
    monkeypatch.delenv("LANGBURST_HF_MODEL", raising=False)
    monkeypatch.delenv("LANGBURST_QB_MODEL", raising=False)
    parser = argparse.ArgumentParser()
    add_model_path_args(parser)
    args = parser.parse_args(["--hf-model", "/models/hf", "--qb-model", "/models/qb"])

    assert str(args.hf_model) == "/models/hf"
    assert str(args.qb_model) == "/models/qb"


def test_batch_kernel_debug_flags_can_split(monkeypatch):
    monkeypatch.delenv("LANGBURST_BATCH_STATE_KERNELS", raising=False)
    monkeypatch.delenv("LANGBURST_BATCH_CONV_KERNELS", raising=False)
    monkeypatch.delenv("LANGBURST_BATCH_GDN_KERNELS", raising=False)
    monkeypatch.setenv("LANGBURST_BATCH_STATE_KERNELS", "1")
    assert batch_state_kernels_enabled()
    assert batch_conv_kernels_enabled()
    assert batch_gdn_kernels_enabled()

    monkeypatch.setenv("LANGBURST_BATCH_CONV_KERNELS", "0")
    monkeypatch.setenv("LANGBURST_BATCH_GDN_KERNELS", "1")
    assert batch_state_kernels_enabled()
    assert not batch_conv_kernels_enabled()
    assert batch_gdn_kernels_enabled()


def test_runtime_profile_stateful_matches_current_default():
    features = RuntimeFeatures.from_profile("stateful")
    assert features.kv_window_policy == "ring"
    assert features.stateful_chat
    assert features.state_pool
    assert not features.snapshots
    assert features.gpu_sampling
    assert features.speculative_decoding
    assert not features.cuda_graph
    assert features.block_prefill


def test_runtime_profile_research_enables_speculation():
    features = RuntimeFeatures.from_profile("research")
    assert features.speculative_decoding
    assert features.infinite_streaming
    assert features.episodic_memory
    assert features.ttt_sidecar


def test_runtime_feature_cli_overrides_are_single_source():
    features = parse_features([
        "--runtime-profile",
        "original",
        "--kv-window-policy",
        "ring",
        "--stateful-chat",
        "on",
        "--snapshots",
        "on",
        "--state-pool",
        "off",
        "--gpu-sampling",
        "off",
        "--boundary-decay",
        "0.5",
        "--block-prefill",
        "off",
        "--infinite-streaming",
        "on",
        "--ttt-sidecar",
        "on",
        "--prefill-chunk-size",
        "16",
    ])
    assert features.profile == "original"
    assert features.kv_window_policy == "ring"
    assert features.stateful_chat
    assert features.snapshots
    assert not features.state_pool
    assert not features.gpu_sampling
    assert features.boundary_decay == 0.5
    assert not features.block_prefill
    assert features.infinite_streaming
    assert features.ttt_sidecar
    assert features.prefill_chunk_size == 16


def test_runtime_feature_override_args_can_exclude_profile():
    parser = argparse.ArgumentParser()
    add_runtime_feature_args(parser, include_profile=False)
    args = parser.parse_args(["--gpu-sampling", "off", "--prefill-chunk-size", "8"])
    override = runtime_feature_override_from_args(args)
    assert override.gpu_sampling is False
    assert override.prefill_chunk_size == 8
    assert not hasattr(args, "runtime_profile")


def test_runtime_feature_override_from_request_like_object():
    class Request:
        kv_window_policy = "ring"
        stateful_chat = True
        state_pool = False
        snapshots = True
        boundary_decay = 0.25
        gpu_sampling = None
        speculative_decoding = None
        cuda_graph = None
        block_prefill = None
        infinite_streaming = True
        episodic_memory = None
        ttt_sidecar = True
        prefill_chunk_size = 32

    features = RuntimeFeatures.from_profile("original").with_overrides(RuntimeFeatureOverride.from_obj(Request()))
    assert features.kv_window_policy == "ring"
    assert features.stateful_chat
    assert features.snapshots
    assert not features.state_pool
    assert features.boundary_decay == 0.25
    assert features.infinite_streaming
    assert features.ttt_sidecar
    assert features.prefill_chunk_size == 32


def test_runtime_plan_disables_unsupported_features():
    requested = RuntimeFeatures.from_profile("research").with_overrides(kv_window_policy="ring", kv_cache_dtype="int4_bdr")
    plan = resolve_runtime_plan(requested, RuntimeCapabilities())
    assert plan.effective.kv_window_policy == "error"
    assert plan.effective.kv_cache_dtype == "fp16"
    assert not plan.effective.stateful_chat
    assert plan.effective.state_pool
    assert not plan.effective.snapshots
    assert plan.effective.gpu_sampling
    assert not plan.effective.infinite_streaming
    assert not plan.effective.episodic_memory
    assert not plan.effective.ttt_sidecar
    assert "kv_window_policy" in plan.disabled
    assert "kv_cache_dtype" in plan.disabled
    assert "stateful_chat" in plan.disabled
    assert "infinite_streaming" in plan.disabled
    assert "episodic_memory" in plan.disabled
    assert "ttt_sidecar" in plan.disabled


def test_qwen_capabilities_keep_stateful_profile():
    requested = RuntimeFeatures.from_profile("stateful").with_overrides(kv_cache_dtype="int4_bdr")
    plan = resolve_runtime_plan(requested, RuntimeCapabilities.stateful_hybrid())
    assert plan.effective == requested
    assert plan.disabled == ()


def test_transformer_decoder_caps_disable_model_specific_features():
    requested = RuntimeFeatures.from_profile("research").with_overrides(kv_window_policy="ring", kv_cache_dtype="int4")
    plan = resolve_runtime_plan(requested, RuntimeCapabilities.transformer_decoder())

    assert plan.effective.stateful_chat
    assert plan.effective.state_pool
    assert plan.effective.snapshots
    assert plan.effective.block_prefill
    assert plan.effective.kv_window_policy == "error"
    assert plan.effective.kv_cache_dtype == "fp16"
    assert not plan.effective.speculative_decoding
    assert not plan.effective.infinite_streaming
    assert not plan.effective.episodic_memory
    assert not plan.effective.ttt_sidecar
    assert "kv_window_policy" in plan.disabled
    assert "kv_cache_dtype" in plan.disabled
    assert "speculative_decoding" in plan.disabled
    assert "infinite_streaming" in plan.disabled
    assert "episodic_memory" in plan.disabled
    assert "ttt_sidecar" in plan.disabled
