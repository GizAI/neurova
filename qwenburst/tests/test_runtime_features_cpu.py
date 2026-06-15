from __future__ import annotations

import argparse

from qwenburst.cli_features import add_runtime_feature_args, runtime_features_from_args
from qwenburst.core.features import RuntimeFeatureOverride, RuntimeFeatures


def parse_features(args: list[str]) -> RuntimeFeatures:
    parser = argparse.ArgumentParser()
    add_runtime_feature_args(parser)
    return runtime_features_from_args(parser.parse_args(args))


def test_runtime_profile_original_disables_stateful_extras():
    features = RuntimeFeatures.from_profile("original")
    assert features.kv_window_policy == "error"
    assert not features.stateful_chat
    assert not features.infinite_streaming
    assert not features.snapshots
    assert not features.episodic_memory
    assert not features.ttt_sidecar
    assert not features.speculative_mtp
    assert not features.cuda_graph
    assert features.block_prefill
    assert features.prefill_chunk_size == 64


def test_runtime_profile_stateful_matches_current_default():
    features = RuntimeFeatures.from_profile("stateful")
    assert features.kv_window_policy == "ring"
    assert features.stateful_chat
    assert features.infinite_streaming
    assert not features.snapshots
    assert not features.speculative_mtp
    assert not features.cuda_graph
    assert features.block_prefill


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
    assert features.boundary_decay == 0.5
    assert not features.block_prefill
    assert features.prefill_chunk_size == 16


def test_runtime_feature_override_from_request_like_object():
    class Request:
        kv_window_policy = "ring"
        stateful_chat = True
        infinite_streaming = None
        snapshots = True
        boundary_decay = 0.25
        episodic_memory = None
        ttt_sidecar = False
        speculative_mtp = None
        cuda_graph = None
        block_prefill = None
        prefill_chunk_size = 32

    features = RuntimeFeatures.from_profile("original").with_overrides(RuntimeFeatureOverride.from_obj(Request()))
    assert features.kv_window_policy == "ring"
    assert features.stateful_chat
    assert features.snapshots
    assert features.boundary_decay == 0.25
    assert not features.ttt_sidecar
    assert features.prefill_chunk_size == 32
