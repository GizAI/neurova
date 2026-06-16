from __future__ import annotations

import argparse

from qwenburst.cli_features import add_runtime_feature_args, runtime_feature_override_from_args, runtime_features_from_args
from qwenburst.core.features import RuntimeCapabilities, RuntimeFeatureOverride, RuntimeFeatures, resolve_runtime_plan


def parse_features(args: list[str]) -> RuntimeFeatures:
    parser = argparse.ArgumentParser()
    add_runtime_feature_args(parser)
    return runtime_features_from_args(parser.parse_args(args))


def test_runtime_profile_original_disables_stateful_extras():
    features = RuntimeFeatures.from_profile("original")
    assert features.kv_window_policy == "error"
    assert not features.stateful_chat
    assert not features.infinite_streaming
    assert features.state_pool
    assert not features.snapshots
    assert not features.episodic_memory
    assert not features.ttt_sidecar
    assert features.gpu_sampling
    assert not features.speculative_decoding
    assert not features.cuda_graph
    assert features.block_prefill
    assert features.prefill_chunk_size == 64


def test_runtime_profile_stateful_matches_current_default():
    features = RuntimeFeatures.from_profile("stateful")
    assert features.kv_window_policy == "ring"
    assert features.stateful_chat
    assert features.infinite_streaming
    assert features.state_pool
    assert not features.snapshots
    assert features.gpu_sampling
    assert features.speculative_decoding
    assert not features.cuda_graph
    assert features.block_prefill


def test_runtime_profile_research_enables_speculation():
    features = RuntimeFeatures.from_profile("research")
    assert features.speculative_decoding


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
        infinite_streaming = None
        state_pool = False
        snapshots = True
        boundary_decay = 0.25
        episodic_memory = None
        ttt_sidecar = False
        gpu_sampling = None
        speculative_decoding = None
        cuda_graph = None
        block_prefill = None
        prefill_chunk_size = 32

    features = RuntimeFeatures.from_profile("original").with_overrides(RuntimeFeatureOverride.from_obj(Request()))
    assert features.kv_window_policy == "ring"
    assert features.stateful_chat
    assert features.snapshots
    assert not features.state_pool
    assert features.boundary_decay == 0.25
    assert not features.ttt_sidecar
    assert features.prefill_chunk_size == 32


def test_runtime_plan_disables_unsupported_features():
    requested = RuntimeFeatures.from_profile("research").with_overrides(kv_window_policy="ring")
    plan = resolve_runtime_plan(requested, RuntimeCapabilities())
    assert plan.effective.kv_window_policy == "error"
    assert not plan.effective.stateful_chat
    assert not plan.effective.infinite_streaming
    assert plan.effective.state_pool
    assert not plan.effective.snapshots
    assert not plan.effective.episodic_memory
    assert not plan.effective.ttt_sidecar
    assert plan.effective.gpu_sampling
    assert "kv_window_policy" in plan.disabled
    assert "stateful_chat" in plan.disabled


def test_qwen_capabilities_keep_stateful_profile():
    requested = RuntimeFeatures.from_profile("stateful")
    plan = resolve_runtime_plan(requested, RuntimeCapabilities.qwen_hybrid_gdn())
    assert plan.effective == requested
    assert plan.disabled == ()
