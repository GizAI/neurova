from __future__ import annotations

import argparse
from pathlib import Path

from .adapters import ensure_adapters_loaded
from .core.adapter import adapter_registry
from .core.features import (
    BOOL_FEATURE_KEYS,
    CORE_BOOL_FEATURE_KEYS,
    KV_CACHE_DTYPES,
    KV_POLICIES,
    RESEARCH_FEATURE_KEYS,
    RUNTIME_PROFILES,
    RuntimeFeatureOverride,
    RuntimeFeatures,
)
from .core.defaults import kv_cache_dtype_default
from .core.platform import ENV_PREFIX, env
from .engines.base import EngineFeatureRequest, EngineModelSpec


def add_adapter_arg(
    parser: argparse.ArgumentParser,
    *,
    default: str | None = None,
    adapter_ids: tuple[str, ...] | None = None,
) -> None:
    ensure_adapters_loaded()
    choices = adapter_ids or adapter_registry.ids()
    resolved_default = default or env("ADAPTER")
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
    parser.add_argument("--kv-cache-dtype", choices=KV_CACHE_DTYPES, default=env("KV_CACHE_DTYPE", kv_cache_dtype_default()))
    core_group = parser.add_argument_group("runtime feature toggles")
    for key in CORE_BOOL_FEATURE_KEYS:
        core_group.add_argument(f"--{key.replace('_', '-')}", choices=("auto", "on", "off"), default="auto")
    research_group = parser.add_argument_group("research feature toggles")
    for key in RESEARCH_FEATURE_KEYS:
        research_group.add_argument(f"--{key.replace('_', '-')}", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--boundary-decay", type=float, default=None)
    parser.add_argument("--prefill-chunk-size", type=int, default=None)


def _tri(value: str) -> bool | None:
    if value == "auto":
        return None
    return value == "on"


def runtime_feature_override_from_args(args: argparse.Namespace) -> RuntimeFeatureOverride:
    return RuntimeFeatureOverride(
        kv_window_policy=args.kv_window_policy,
        kv_cache_dtype=args.kv_cache_dtype,
        boundary_decay=args.boundary_decay,
        prefill_chunk_size=args.prefill_chunk_size,
        **{key: _tri(getattr(args, key)) for key in BOOL_FEATURE_KEYS},
    )


def runtime_features_from_args(args: argparse.Namespace) -> RuntimeFeatures:
    features = RuntimeFeatures.from_profile(args.runtime_profile).with_overrides(runtime_feature_override_from_args(args))
    enable_mtp = getattr(args, "enable_mtp", None)
    if enable_mtp is not None:
        features = features.with_overrides(speculative_decoding=bool(enable_mtp))
    return features


def runtime_features_from_obj(obj: object, *, base: RuntimeFeatures | None = None) -> RuntimeFeatures:
    profile = getattr(obj, "runtime_profile", None) or (base.profile if base is not None else "stateful")
    features = base if base is not None and profile == base.profile else RuntimeFeatures.from_profile(profile)
    if base is not None and profile != base.profile:
        features = features.with_overrides(
            kv_cache_dtype=base.kv_cache_dtype,
            prefill_chunk_size=base.prefill_chunk_size,
        )
    return features.with_overrides(RuntimeFeatureOverride.from_obj(obj))


def engine_feature_request_from_features(
    features: RuntimeFeatures,
    *,
    custom_model_bridge: bool = False,
    recurrent_state: bool = False,
) -> EngineFeatureRequest:
    return EngineFeatureRequest.from_mapping(
        {
            **features.summary(),
            "custom_model_bridge": bool(custom_model_bridge),
            "ring_kv": features.kv_window_policy == "ring",
            "recurrent_state": bool(recurrent_state or custom_model_bridge),
            "infinite_context": bool(features.infinite_streaming),
        }
    )


def engine_feature_request_from_args(args: argparse.Namespace) -> EngineFeatureRequest:
    features = runtime_features_from_args(args)
    custom_model_bridge = bool(getattr(args, "custom_model_bridge", False) or getattr(args, "qb_model", None))
    return engine_feature_request_from_features(
        features,
        custom_model_bridge=custom_model_bridge,
        recurrent_state=bool(getattr(args, "recurrent_state", False)),
    )


def engine_model_spec_from_args(args: argparse.Namespace, *, served_model_name: str | None = None) -> EngineModelSpec:
    model = args.model or (str(args.hf_model) if args.hf_model is not None else None)
    if model is None:
        raise ValueError("--model or --hf-model is required")
    extra = {
        "adapter": args.adapter,
        "qb_model": str(args.qb_model) if args.qb_model is not None else None,
        "device": args.device,
        "recent_window": args.recent_window,
        "weight_device": args.weight_device,
        "cpu_embed": bool(args.cpu_embed),
        "runtime_profile": args.runtime_profile,
        "vllm_custom_model": args.vllm_custom_model,
        "mtp_speculative_tokens": args.mtp_speculative_tokens,
    }
    if args.enable_mtp is not None:
        extra["enable_mtp"] = bool(args.enable_mtp)
    return EngineModelSpec(
        model=model,
        served_model_name=served_model_name or args.served_model_name,
        tokenizer=args.tokenizer,
        dtype=env("DTYPE", "auto"),
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len or args.recent_window,
        quantization=args.quantization,
        trust_remote_code=not args.no_trust_remote_code,
        features=engine_feature_request_from_args(args),
        extra={k: v for k, v in extra.items() if v is not None},
    )


def create_runtime_engine_from_args(
    args: argparse.Namespace,
    *,
    features: RuntimeFeatures | None = None,
    model_name: str | None = None,
    max_state_pool_size: int = 0,
):
    """Build a `RuntimeEngine` from the shared CLI contract.

    CLI, bench, and correctness tools should not each decide adapter lookup,
    path fields, CPU embedding, or feature defaults. The server uses
    `EngineManager` because it has multi-model policy, but single-engine tools
    should come through this factory.
    """

    from .engines.native import RuntimeEngine

    return RuntimeEngine(
        adapter=adapter_registry.get(args.adapter),
        hf_model=args.hf_model,
        qb_model=args.qb_model,
        device=args.device,
        recent_window=args.recent_window,
        weight_device=args.weight_device,
        cpu_embed=bool(getattr(args, "cpu_embed", False)),
        model_name=model_name,
        features=features if features is not None else runtime_features_from_args(args),
        max_state_pool_size=max_state_pool_size,
    )
