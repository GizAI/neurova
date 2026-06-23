from __future__ import annotations

import argparse
import copy
import os
import shlex
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except Exception as exc:  # pragma: no cover - dependency is declared.
    yaml = None
    _YAML_IMPORT_ERROR = exc
else:
    _YAML_IMPORT_ERROR = None


_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "ml-dmc8-q4.yaml"


DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "hf_model": "/home/user/models/Qwen3.6-27B",
        "qb_model": "/home/user/models/Qwen3.6-27B-qb4-marlin-fused",
        "name": "langburst-qwen3.6-27b-q4",
        "adapter": "qwen36",
        "engine": "native",
        "device": "cuda",
        "weight_device": "cuda",
        "cpu_embed": False,
    },
    "server": {
        "host": "0.0.0.0",
        "port": 8008,
        "restart": True,
        "log_dir": "/tmp",
        "request_timeout_s": 300,
        "prefix_cache": "on",
    },
    "serving": {
        "context_window": 65536,
        "context_tiers": [4096, 65536],
        "context_tier_slots": [2, 1],
        "exclusive_prefill_tokens": None,
        "allow_context_overflow": True,
        "max_active_requests": 3,
        "max_queued_requests": 8,
        "max_state_pool_size": 3,
        "max_generation_tokens": 8192,
        "default_max_tokens": 8192,
        "default_min_new_tokens": 0,
        "min_completion_budget": 0,
        "message_context_policy": "auto_long_user",
        "standalone_user_min_tokens": 1024,
        "reserve_free_vram_mib": 256,
        "batch_wait_ms": 8,
        "max_num_batched_tokens": 256,
        "prefill_chunk_size": 64,
        "max_prefill_rows_per_batch": 1,
        "decode_prefill_interleave_steps": 16,
    },
    "kv": {
        "cache_dtype": "int4_bdr",
        "paged": True,
        "paged_mirror": False,
        "paged_shadow": True,
        "paged_attention_kernels": True,
        "paged_prefill_block": True,
        "short_prefill_sdpa_tokens": 2048,
        "short_prefill_sdpa_min_free_mib": 384,
        "attention_recent_tokens": 32,
        "paged_attention_backend": "flash",
        "int4_layout": "tiled",
        "raw_prefill_block_tokens": None,
    },
    "marlin": {
        "out_cache_policy": "all",
        "out_cache_max_batch": 4,
        "cache_max_mib": 32,
        "cache_min_free_mib": 256,
        "direct_max_batch": 256,
        "lowbit_rows_per_cta": 4,
        "internal_argmax": True,
        "max_par": None,
    },
    "mtp": {
        "enabled": True,
        "speculative_tokens": 4,
        "max_draft": 4,
        "draft_candidates": 4,
        "max_draft_by_active": "2:1",
        "legacy_list_cache": False,
        "local_tkh_attention": False,
        "batch_proposer": True,
        "fc_split": False,
        "skip_after_reject": False,
        "adaptive": False,
        "min_free_vram_mib": 128,
    },
    "kernels": {
        "cuda_graph": False,
        "verify_full_logits": False,
        "gdn_recurrent_norm_gate_fused": False,
        "gdn_spec_norm_gate_fused": True,
        "gdn_ba_lowbit_pair": True,
        "mlp_tensorcore_down_silu_a": True,
        "mlp_scalar_streaming_debug": False,
        "spec_trajectory_copy_cuda": True,
        "batch_state_arena": "auto",
    },
    "runtime": {
        "serve_batch": True,
        "suppress_think_tokens": False,
        "repetition_stop_min_ngram_size": 32,
        "repetition_stop_ngram_size": 96,
        "repetition_stop_repeats": 2,
        "suppress_chat_control_tokens": False,
        "trim_cache_after_request": True,
        "trim_cache_free_below_mib": 1024,
        "trim_model_cache_before_prefill": True,
        "trim_model_cache_prefill_free_below_mib": 1024,
        "prefix_cache_max_entries": 2,
        "prefix_cache_max_tokens": 16384,
        "prefix_cache_min_free_mib": 768,
        "prefix_cache_min_free_blocks": 512,
        "pytorch_cuda_alloc_conf": "expandable_segments:True",
    },
}


SERVER_ENV_KEYS: tuple[str, ...] = (
    "LANGBURST_REQUIRE_CUDA_EXT",
    "LANGBURST_CONTEXT_WINDOW",
    "LANGBURST_CONTEXT_TIERS",
    "LANGBURST_CONTEXT_TIER_SLOTS",
    "LANGBURST_EXCLUSIVE_PREFILL_TOKENS",
    "LANGBURST_ALLOW_CONTEXT_OVERFLOW",
    "LANGBURST_PREFILL_CHUNK_SIZE",
    "LANGBURST_MAX_PREFILL_ROWS_PER_BATCH",
    "LANGBURST_DECODE_PREFILL_INTERLEAVE_STEPS",
    "LANGBURST_BATCH_WAIT_MS",
    "LANGBURST_MAX_NUM_BATCHED_TOKENS",
    "LANGBURST_MAX_ACTIVE_REQUESTS",
    "LANGBURST_MAX_QUEUED_REQUESTS",
    "LANGBURST_MAX_STATE_POOL_SIZE",
    "LANGBURST_MAX_GENERATION_TOKENS",
    "LANGBURST_DEFAULT_MIN_NEW_TOKENS",
    "LANGBURST_MIN_COMPLETION_TOKENS",
    "LANGBURST_MESSAGE_CONTEXT_POLICY",
    "LANGBURST_STANDALONE_USER_MIN_TOKENS",
    "LANGBURST_SERVE_BATCH",
    "LANGBURST_BATCH_STATE_ARENA",
    "LANGBURST_KV_CACHE_DTYPE",
    "LANGBURST_PAGED_KV",
    "LANGBURST_PAGED_KV_MIRROR",
    "LANGBURST_PAGED_KV_SHADOW",
    "LANGBURST_PAGED_ATTENTION_KERNELS",
    "LANGBURST_PAGED_PREFILL_BLOCK",
    "LANGBURST_SHORT_PREFILL_SDPA_TOKENS",
    "LANGBURST_SHORT_PREFILL_SDPA_MIN_FREE_MIB",
    "LANGBURST_ATTENTION_RECENT_TOKENS",
    "LANGBURST_PAGED_ATTENTION_BACKEND",
    "LANGBURST_INT4_KV_LAYOUT",
    "LANGBURST_MARLIN_OUT_CACHE_POLICY",
    "LANGBURST_MARLIN_OUT_CACHE_MAX_BATCH",
    "LANGBURST_MARLIN_CACHE_MAX_MIB",
    "LANGBURST_MARLIN_CACHE_MIN_FREE_MIB",
    "LANGBURST_MARLIN_DIRECT_MAX_BATCH",
    "LANGBURST_LOWBIT_ROWS_PER_CTA",
    "LANGBURST_CUDA_GRAPH",
    "LANGBURST_MARLIN_INTERNAL_ARGMAX",
    "LANGBURST_VERIFY_FULL_LOGITS",
    "LANGBURST_GDN_RECURRENT_NORM_GATE_FUSED",
    "LANGBURST_GDN_SPEC_NORM_GATE_FUSED",
    "LANGBURST_GDN_BA_LOWBIT_PAIR",
    "LANGBURST_MLP_TENSORCORE_DOWN_SILU_A",
    "LANGBURST_MLP_SCALAR_STREAMING_DEBUG",
    "LANGBURST_SPEC_TRAJECTORY_COPY_CUDA",
    "LANGBURST_MTP_MAX_DRAFT",
    "LANGBURST_MTP_DRAFT_CANDIDATES",
    "LANGBURST_MTP_MAX_DRAFT_BY_ACTIVE",
    "LANGBURST_MTP_LEGACY_LIST_CACHE",
    "LANGBURST_MTP_LOCAL_TKH_ATTENTION",
    "LANGBURST_MTP_BATCH_PROPOSER",
    "LANGBURST_MTP_FC_SPLIT",
    "LANGBURST_MTP_SKIP_AFTER_REJECT",
    "LANGBURST_MTP_ADAPTIVE",
    "LANGBURST_MTP_MIN_FREE_VRAM_MIB",
    "LANGBURST_REQUEST_TIMEOUT_S",
    "LANGBURST_DEFAULT_MAX_TOKENS",
    "LANGBURST_SUPPRESS_THINK_TOKENS",
    "LANGBURST_SUPPRESS_CHAT_CONTROL_TOKENS",
    "LANGBURST_REPETITION_STOP_MIN_NGRAM_SIZE",
    "LANGBURST_REPETITION_STOP_NGRAM_SIZE",
    "LANGBURST_REPETITION_STOP_REPEATS",
    "LANGBURST_TRIM_CACHE_AFTER_REQUEST",
    "LANGBURST_TRIM_CACHE_FREE_BELOW_MIB",
    "LANGBURST_TRIM_MODEL_CACHE_BEFORE_PREFILL",
    "LANGBURST_TRIM_MODEL_CACHE_PREFILL_FREE_BELOW_MIB",
    "LANGBURST_PREFIX_CACHE_MAX_ENTRIES",
    "LANGBURST_PREFIX_CACHE_MAX_TOKENS",
    "LANGBURST_PREFIX_CACHE_MIN_FREE_MIB",
    "LANGBURST_PREFIX_CACHE_MIN_FREE_BLOCKS",
    "LANGBURST_RAW_PREFILL_BLOCK_TOKENS",
    "PYTORCH_CUDA_ALLOC_CONF",
)


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load LangBurst YAML configs") from _YAML_IMPORT_ERROR
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _bool_value(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _BOOL_TRUE:
            return "1"
        if text in _BOOL_FALSE:
            return "0"
        raise ValueError(f"invalid boolean value {value!r}")
    return "1" if bool(value) else "0"


def _csv(values: Any) -> str:
    if values is None:
        return ""
    if isinstance(values, str):
        return values
    return ",".join(str(int(v)) for v in values)


def _value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return _bool_value(value)
    if isinstance(value, (list, tuple)):
        return _csv(value)
    return str(value)


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def load_serving_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    config = _deep_merge(DEFAULT_CONFIG, _read_yaml(config_path))
    validate_serving_config(config)
    return config


def validate_serving_config(config: Mapping[str, Any]) -> None:
    serving = _section(config, "serving")
    tiers = serving.get("context_tiers") or []
    slots = serving.get("context_tier_slots") or []
    if bool(tiers) != bool(slots) or len(tiers) != len(slots):
        raise ValueError("serving.context_tiers and serving.context_tier_slots must have the same length")
    for key in ("context_window", "max_active_requests", "max_queued_requests", "max_generation_tokens"):
        if int(serving[key]) < 1:
            raise ValueError(f"serving.{key} must be >= 1")
    if int(serving.get("min_completion_budget", 0) or 0) < 0:
        raise ValueError("serving.min_completion_budget must be >= 0")
    if tiers and max(int(v) for v in tiers) > int(serving["context_window"]):
        raise ValueError("serving.context_window must be >= max(serving.context_tiers)")
    for key in ("hf_model", "qb_model", "name", "adapter", "engine"):
        if not str(_section(config, "model").get(key, "")).strip():
            raise ValueError(f"model.{key} is required")


def resolved_env(config: Mapping[str, Any]) -> dict[str, str]:
    model = _section(config, "model")
    server = _section(config, "server")
    serving = _section(config, "serving")
    kv = _section(config, "kv")
    marlin = _section(config, "marlin")
    mtp = _section(config, "mtp")
    kernels = _section(config, "kernels")
    runtime = _section(config, "runtime")

    raw_prefill = kv.get("raw_prefill_block_tokens")
    if raw_prefill in {None, ""}:
        raw_prefill = serving["prefill_chunk_size"]
    exclusive_prefill = serving.get("exclusive_prefill_tokens")
    if exclusive_prefill in {None, ""} and len(serving.get("context_tiers") or []) > 1:
        exclusive_prefill = sorted(int(v) for v in serving["context_tiers"])[0] + 1

    server_env_keys = list(SERVER_ENV_KEYS)
    env = {
        "MODEL_DIR": model["hf_model"],
        "QB_DIR": model["qb_model"],
        "MODEL_NAME": model["name"],
        "LANGBURST_ENGINE": model["engine"],
        "LANGBURST_ADAPTER": model["adapter"],
        "LANGBURST_DEVICE": model["device"],
        "LANGBURST_WEIGHT_DEVICE": model["weight_device"],
        "SERVER_HOST": server["host"],
        "SERVER_PORT": server["port"],
        "RESTART_LANGBURST": server["restart"],
        "LOG_DIR": server["log_dir"],
        "ENABLE_MTP": mtp["enabled"],
        "MTP_SPECULATIVE_TOKENS": mtp["speculative_tokens"],
        "CONTEXT_WINDOW": serving["context_window"],
        "CONTEXT_TIERS": serving["context_tiers"],
        "CONTEXT_TIER_SLOTS": serving["context_tier_slots"],
        "EXCLUSIVE_PREFILL_TOKENS": exclusive_prefill,
        "ALLOW_CONTEXT_OVERFLOW": serving["allow_context_overflow"],
        "KV_CACHE_DTYPE": kv["cache_dtype"],
        "LANGBURST_KV_CACHE_DTYPE": kv["cache_dtype"],
        "PREFILL_CHUNK_SIZE": serving["prefill_chunk_size"],
        "DECODE_PREFILL_INTERLEAVE_STEPS": serving["decode_prefill_interleave_steps"],
        "BATCH_WAIT_MS": serving["batch_wait_ms"],
        "MAX_NUM_BATCHED_TOKENS": serving["max_num_batched_tokens"],
        "RAW_PREFILL_BLOCK_TOKENS": raw_prefill,
        "CPU_EMBED": model["cpu_embed"],
        "MAX_ACTIVE_REQUESTS": serving["max_active_requests"],
        "MAX_PREFILL_ROWS_PER_BATCH": serving["max_prefill_rows_per_batch"],
        "MAX_QUEUED_REQUESTS": serving["max_queued_requests"],
        "MAX_STATE_POOL_SIZE": serving["max_state_pool_size"],
        "MAX_GENERATION_TOKENS": serving["max_generation_tokens"],
        "DEFAULT_MAX_TOKENS": serving["default_max_tokens"],
        "DEFAULT_MIN_NEW_TOKENS": serving["default_min_new_tokens"],
        "MIN_COMPLETION_TOKENS": serving["min_completion_budget"],
        "MESSAGE_CONTEXT_POLICY": serving["message_context_policy"],
        "STANDALONE_USER_MIN_TOKENS": serving["standalone_user_min_tokens"],
        "RESERVE_FREE_VRAM_MIB": serving["reserve_free_vram_mib"],
        "PREFIX_CACHE": server["prefix_cache"],
        "LANGBURST_REQUIRE_CUDA_EXT": 1,
        "LANGBURST_CONTEXT_WINDOW": serving["context_window"],
        "LANGBURST_CONTEXT_TIERS": serving["context_tiers"],
        "LANGBURST_CONTEXT_TIER_SLOTS": serving["context_tier_slots"],
        "LANGBURST_EXCLUSIVE_PREFILL_TOKENS": exclusive_prefill,
        "LANGBURST_ALLOW_CONTEXT_OVERFLOW": serving["allow_context_overflow"],
        "LANGBURST_PREFILL_CHUNK_SIZE": serving["prefill_chunk_size"],
        "LANGBURST_MAX_PREFILL_ROWS_PER_BATCH": serving["max_prefill_rows_per_batch"],
        "LANGBURST_DECODE_PREFILL_INTERLEAVE_STEPS": serving["decode_prefill_interleave_steps"],
        "LANGBURST_BATCH_WAIT_MS": serving["batch_wait_ms"],
        "LANGBURST_MAX_NUM_BATCHED_TOKENS": serving["max_num_batched_tokens"],
        "LANGBURST_MAX_ACTIVE_REQUESTS": serving["max_active_requests"],
        "LANGBURST_MAX_QUEUED_REQUESTS": serving["max_queued_requests"],
        "LANGBURST_MAX_STATE_POOL_SIZE": serving["max_state_pool_size"],
        "LANGBURST_MAX_GENERATION_TOKENS": serving["max_generation_tokens"],
        "LANGBURST_DEFAULT_MAX_TOKENS": serving["default_max_tokens"],
        "LANGBURST_DEFAULT_MIN_NEW_TOKENS": serving["default_min_new_tokens"],
        "LANGBURST_MIN_COMPLETION_TOKENS": serving["min_completion_budget"],
        "LANGBURST_MESSAGE_CONTEXT_POLICY": serving["message_context_policy"],
        "LANGBURST_STANDALONE_USER_MIN_TOKENS": serving["standalone_user_min_tokens"],
        "LANGBURST_SERVE_BATCH": runtime["serve_batch"],
        "LANGBURST_BATCH_STATE_ARENA": kernels["batch_state_arena"],
        "LANGBURST_PAGED_KV": kv["paged"],
        "LANGBURST_PAGED_KV_MIRROR": kv["paged_mirror"],
        "LANGBURST_PAGED_KV_SHADOW": kv["paged_shadow"],
        "LANGBURST_PAGED_ATTENTION_KERNELS": kv["paged_attention_kernels"],
        "LANGBURST_PAGED_PREFILL_BLOCK": kv["paged_prefill_block"],
        "LANGBURST_SHORT_PREFILL_SDPA_TOKENS": kv["short_prefill_sdpa_tokens"],
        "LANGBURST_SHORT_PREFILL_SDPA_MIN_FREE_MIB": kv["short_prefill_sdpa_min_free_mib"],
        "LANGBURST_ATTENTION_RECENT_TOKENS": kv["attention_recent_tokens"],
        "LANGBURST_PAGED_ATTENTION_BACKEND": kv["paged_attention_backend"],
        "LANGBURST_INT4_KV_LAYOUT": kv["int4_layout"],
        "LANGBURST_MARLIN_OUT_CACHE_POLICY": marlin["out_cache_policy"],
        "LANGBURST_MARLIN_OUT_CACHE_MAX_BATCH": marlin["out_cache_max_batch"],
        "LANGBURST_MARLIN_CACHE_MAX_MIB": marlin["cache_max_mib"],
        "LANGBURST_MARLIN_CACHE_MIN_FREE_MIB": marlin["cache_min_free_mib"],
        "LANGBURST_MARLIN_DIRECT_MAX_BATCH": marlin["direct_max_batch"],
        "LANGBURST_LOWBIT_ROWS_PER_CTA": marlin["lowbit_rows_per_cta"],
        "LANGBURST_CUDA_GRAPH": kernels["cuda_graph"],
        "LANGBURST_MARLIN_INTERNAL_ARGMAX": marlin["internal_argmax"],
        "LANGBURST_VERIFY_FULL_LOGITS": kernels["verify_full_logits"],
        "LANGBURST_GDN_RECURRENT_NORM_GATE_FUSED": kernels["gdn_recurrent_norm_gate_fused"],
        "LANGBURST_GDN_SPEC_NORM_GATE_FUSED": kernels["gdn_spec_norm_gate_fused"],
        "LANGBURST_GDN_BA_LOWBIT_PAIR": kernels["gdn_ba_lowbit_pair"],
        "LANGBURST_MLP_TENSORCORE_DOWN_SILU_A": kernels["mlp_tensorcore_down_silu_a"],
        "LANGBURST_MLP_SCALAR_STREAMING_DEBUG": kernels["mlp_scalar_streaming_debug"],
        "LANGBURST_SPEC_TRAJECTORY_COPY_CUDA": kernels["spec_trajectory_copy_cuda"],
        "LANGBURST_MTP_MAX_DRAFT": mtp["max_draft"],
        "LANGBURST_MTP_DRAFT_CANDIDATES": mtp["draft_candidates"],
        "LANGBURST_MTP_MAX_DRAFT_BY_ACTIVE": mtp["max_draft_by_active"],
        "LANGBURST_MTP_LEGACY_LIST_CACHE": mtp["legacy_list_cache"],
        "LANGBURST_MTP_LOCAL_TKH_ATTENTION": mtp["local_tkh_attention"],
        "LANGBURST_MTP_BATCH_PROPOSER": mtp["batch_proposer"],
        "LANGBURST_MTP_FC_SPLIT": mtp["fc_split"],
        "LANGBURST_MTP_SKIP_AFTER_REJECT": mtp["skip_after_reject"],
        "LANGBURST_MTP_ADAPTIVE": mtp["adaptive"],
        "LANGBURST_MTP_MIN_FREE_VRAM_MIB": mtp["min_free_vram_mib"],
        "LANGBURST_REQUEST_TIMEOUT_S": server["request_timeout_s"],
        "LANGBURST_SUPPRESS_THINK_TOKENS": runtime["suppress_think_tokens"],
        "LANGBURST_SUPPRESS_CHAT_CONTROL_TOKENS": runtime["suppress_chat_control_tokens"],
        "LANGBURST_REPETITION_STOP_MIN_NGRAM_SIZE": runtime["repetition_stop_min_ngram_size"],
        "LANGBURST_REPETITION_STOP_NGRAM_SIZE": runtime["repetition_stop_ngram_size"],
        "LANGBURST_REPETITION_STOP_REPEATS": runtime["repetition_stop_repeats"],
        "LANGBURST_TRIM_CACHE_AFTER_REQUEST": runtime["trim_cache_after_request"],
        "LANGBURST_TRIM_CACHE_FREE_BELOW_MIB": runtime["trim_cache_free_below_mib"],
        "LANGBURST_TRIM_MODEL_CACHE_BEFORE_PREFILL": runtime["trim_model_cache_before_prefill"],
        "LANGBURST_TRIM_MODEL_CACHE_PREFILL_FREE_BELOW_MIB": runtime["trim_model_cache_prefill_free_below_mib"],
        "LANGBURST_PREFIX_CACHE_MAX_ENTRIES": runtime["prefix_cache_max_entries"],
        "LANGBURST_PREFIX_CACHE_MAX_TOKENS": runtime["prefix_cache_max_tokens"],
        "LANGBURST_PREFIX_CACHE_MIN_FREE_MIB": runtime["prefix_cache_min_free_mib"],
        "LANGBURST_PREFIX_CACHE_MIN_FREE_BLOCKS": runtime["prefix_cache_min_free_blocks"],
        "LANGBURST_RAW_PREFILL_BLOCK_TOKENS": raw_prefill,
        "PYTORCH_CUDA_ALLOC_CONF": runtime["pytorch_cuda_alloc_conf"],
    }
    if marlin.get("max_par") not in {None, ""}:
        env["LANGBURST_MARLIN_MAX_PAR"] = marlin["max_par"]
        server_env_keys.append("LANGBURST_MARLIN_MAX_PAR")
    env["LANGBURST_SERVER_ENV_KEYS"] = " ".join(server_env_keys)
    return {key: _value(value) for key, value in env.items()}


def export_shell(config: Mapping[str, Any]) -> str:
    env = resolved_env(config)
    return "\n".join(f"export {key}={shlex.quote(value)}" for key, value in sorted(env.items()))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Resolve LangBurst YAML serving config")
    parser.add_argument("command", choices=("export-shell", "print"))
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("LANGBURST_CONFIG", DEFAULT_CONFIG_PATH)))
    args = parser.parse_args(argv)
    config = load_serving_config(args.config)
    if args.command == "export-shell":
        print(export_shell(config))
    else:
        if yaml is None:
            raise RuntimeError("PyYAML is required to print LangBurst YAML configs") from _YAML_IMPORT_ERROR
        print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
