from __future__ import annotations

import argparse
from pathlib import Path

from .adapters import ensure_adapters_loaded
from .core.adapter import adapter_registry
from .core.features import BOOL_FEATURE_KEYS, KV_POLICIES, RUNTIME_PROFILES, RuntimeFeatureOverride, RuntimeFeatures
from .core.platform import ENV_PREFIX, env


def add_adapter_arg(
    parser: argparse.ArgumentParser,
    *,
    default: str | None = None,
    adapter_ids: tuple[str, ...] | None = None,
) -> None:
    ensure_adapters_loaded()
    choices = adapter_ids or adapter_registry.ids()
    resolved_default = default or env("ADAPTER")
    if resolved_default is None and "qwen36" in choices:
        resolved_default = "qwen36"
    parser.add_argument(
        "--adapter",
        default=resolved_default or (choices[0] if choices else None),
        choices=choices or None,
        required=not choices,
        help="model adapter id registered in builtin adapters or entry points",
    )


def add_model_path_args(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    """Add common HF/runtime model path arguments.

    The CLI surface should not hardcode model-family paths.  Callers can pass
    explicit paths or set LANGBURST_HF_MODEL / LANGBURST_QB_MODEL.
    """

    hf_default = env("HF_MODEL")
    qb_default = env("QB_MODEL")
    parser.add_argument(
        "--hf-model",
        type=Path,
        default=Path(hf_default) if hf_default else None,
        required=required and hf_default is None,
        help=f"HF model dir, or {ENV_PREFIX}_HF_MODEL",
    )
    parser.add_argument(
        "--qb-model",
        type=Path,
        default=Path(qb_default) if qb_default else None,
        required=required and qb_default is None,
        help=f"converted runtime model dir, or {ENV_PREFIX}_QB_MODEL",
    )


def add_runtime_feature_args(parser: argparse.ArgumentParser, *, include_profile: bool = True) -> None:
    if include_profile:
        parser.add_argument(
            "--runtime-profile",
            choices=RUNTIME_PROFILES,
            default="stateful",
            help="feature preset: original disables stateful extras; stateful is the default streaming-state runtime",
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


def runtime_feature_override_from_args(args: argparse.Namespace) -> RuntimeFeatureOverride:
    return RuntimeFeatureOverride(
        kv_window_policy=args.kv_window_policy,
        boundary_decay=args.boundary_decay,
        prefill_chunk_size=args.prefill_chunk_size,
        **{key: _tri(getattr(args, key)) for key in BOOL_FEATURE_KEYS},
    )


def runtime_features_from_args(args: argparse.Namespace) -> RuntimeFeatures:
    return RuntimeFeatures.from_profile(args.runtime_profile).with_overrides(runtime_feature_override_from_args(args))
