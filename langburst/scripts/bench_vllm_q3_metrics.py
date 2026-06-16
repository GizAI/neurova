from __future__ import annotations

import argparse
import os

from langburst.engines import ensure_engines_loaded, engine_registry
from langburst.engines.base import (
    EngineChatRequest,
    EngineFeatureRequest,
    EngineModelSpec,
    EngineSamplingParams,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LangBurst q3 through the vLLM engine path.")
    parser.add_argument("--model", default="/home/user/models/Qwen3.6-27B")
    parser.add_argument("--tokenizer", default="/home/user/models/Qwen3.6-27B")
    parser.add_argument("--qb-model", default="/home/user/models/Qwen3.6-27B-langburst-q3")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.965)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--kv-cache-dtype", default=None)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--enable-mtp", action="store_true")
    parser.add_argument("--mtp-speculative-tokens", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = _args()
    ensure_engines_loaded()
    features = EngineFeatureRequest(
        qwen36_lowbit=True,
        ring_kv=True,
        recurrent_state=True,
        stateful_sessions=True,
    )
    spec = EngineModelSpec(
        model=args.model,
        tokenizer=args.tokenizer,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        features=features,
        extra={
            "qb_model": args.qb_model,
            "vllm_arg_disable_log_stats": False,
        },
    )
    if args.max_num_batched_tokens is not None:
        spec.extra["max_num_batched_tokens"] = args.max_num_batched_tokens
    elif os.environ.get("LANGBURST_VLLM_MAX_NUM_BATCHED_TOKENS"):
        spec.extra["max_num_batched_tokens"] = int(os.environ["LANGBURST_VLLM_MAX_NUM_BATCHED_TOKENS"])
    if args.max_num_seqs is not None:
        spec.extra["max_num_seqs"] = args.max_num_seqs
    if args.kv_cache_dtype is not None:
        spec.extra["kv_cache_dtype"] = args.kv_cache_dtype
    if args.kv_cache_memory_bytes is not None:
        spec.extra["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes
    if args.enable_mtp:
        spec.extra["enable_mtp"] = True
        spec.extra["mtp_speculative_tokens"] = args.mtp_speculative_tokens
        spec.extra.setdefault("kv_cache_memory_bytes", int(os.environ.get("LANGBURST_VLLM_MTP_KV_CACHE_MEMORY_BYTES", "760000000")))
    backend = engine_registry.create(spec, engine_id="vllm")
    req = EngineChatRequest(
        request_id="bench",
        model=spec.public_name,
        messages=[
            {"role": "system", "content": "Answer directly. Do not explain reasoning."},
            {"role": "user", "content": "Write one short greeting."},
        ],
        sampling=EngineSamplingParams(max_tokens=args.max_tokens, temperature=0.0),
    )
    backend.generate_chat(req)
    result = backend.generate_chat(req)
    raw = result.raw
    metrics = raw.metrics
    out = raw.outputs[0]
    prompt_tokens = len(raw.prompt_token_ids or [])
    completion_tokens = len(out.token_ids or [])
    scheduled = float(getattr(metrics, "scheduled_ts", 0.0) or 0.0)
    first = float(getattr(metrics, "first_token_ts", 0.0) or 0.0)
    last = float(getattr(metrics, "last_token_ts", 0.0) or 0.0)
    prefill_s = max(first - scheduled, 0.0)
    decode_s = max(last - first, 0.0)
    decode_after_first = max(completion_tokens - 1, 0)
    print("TEXT", repr(result.text))
    print(
        "TOKENS",
        {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "decode_after_first": decode_after_first,
        },
    )
    print(
        "TIMING",
        {
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "scheduled_ts": scheduled,
            "first_token_ts": first,
            "last_token_ts": last,
        },
    )
    print(
        "TOK_S",
        {
            "prefill_tok_s": prompt_tokens / prefill_s if prefill_s > 0 else None,
            "decode_tok_s_after_first": decode_after_first / decode_s if decode_s > 0 else None,
            "completion_tok_s_all": completion_tokens / decode_s if decode_s > 0 else None,
        },
    )
    print("METRICS_DICT", getattr(metrics, "__dict__", {}))
    backend.shutdown()


if __name__ == "__main__":
    main()
