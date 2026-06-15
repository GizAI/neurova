from __future__ import annotations

import argparse

from .core.features import BOOL_FEATURE_KEYS, KV_POLICIES, RUNTIME_PROFILES, RuntimeFeatureOverride, RuntimeFeatures


def add_runtime_feature_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runtime-profile",
        choices=RUNTIME_PROFILES,
        default="stateful",
        help="feature preset: original disables QwenBurst stateful extras; stateful is the default streaming-state runtime",
    )
    parser.add_argument("--kv-window-policy", choices=KV_POLICIES, default=None)
    for key in BOOL_FEATURE_KEYS:
        parser.add_argument(f"--{key.replace('_', '-')}", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--boundary-decay", type=float, default=None)
    parser.add_argument("--prefill-chunk-size", type=int, default=None)


def _tri(value: str) -> bool | None:
    if value == "auto":
        return None
    return value == "on"


def runtime_features_from_args(args: argparse.Namespace) -> RuntimeFeatures:
    override = RuntimeFeatureOverride(
        kv_window_policy=args.kv_window_policy,
        boundary_decay=args.boundary_decay,
        prefill_chunk_size=args.prefill_chunk_size,
        **{key: _tri(getattr(args, key)) for key in BOOL_FEATURE_KEYS},
    )
    return RuntimeFeatures.from_profile(args.runtime_profile).with_overrides(override)
