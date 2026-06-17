from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from typing import Sequence

import torch

from .cli_features import add_adapter_arg, add_model_path_args, create_runtime_engine_from_args
from .core.features import RUNTIME_PROFILES
from .core.defaults import kv_block_size_default, kv_blocks_default, serving_recent_window_default
from .core.features import RuntimeFeatures
from .engines.native.resource_policy import EngineResourcePolicy
from .engines.native import (
    BatchGenerationHandle,
    BatchGenerationWorker,
    BatchedModelRunner,
    ContinuousBatchScheduler,
    GenerationConfig,
    KVBlockTable,
    RuntimeEngine,
)


@dataclass(frozen=True)
class ServingBenchCase:
    requests: int
    prompt_tokens: int
    max_new_tokens: int
    max_num_batched_tokens: int
    prefill_chunk_size: int
    max_prefill_rows_per_batch: int
    max_wait_ms: float
    paged_kv: bool = True


def _mean(values: Sequence[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(statistics.fmean(vals))


def _p95(values: Sequence[float | None]) -> float | None:
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    idx = min(len(vals) - 1, int(round(0.95 * (len(vals) - 1))))
    return vals[idx]


def _prompt_ids_near(engine: RuntimeEngine, target_tokens: int) -> list[int]:
    if target_tokens < 1:
        raise ValueError("prompt token target must be >= 1")
    seed = (
        "Serving benchmark. Explain low-bit LLM inference, continuous batching, "
        "paged cache management, recurrent state, and CUDA kernels. "
    )
    text = seed
    ids = engine.encode_prompt(text)
    while len(ids) < target_tokens:
        text += seed
        ids = engine.encode_prompt(text)
    return ids[:target_tokens]


def run_serving_case(
    engine: RuntimeEngine,
    case: ServingBenchCase,
    *,
    recent_window: int,
    kv_block_size: int,
    kv_blocks: int,
    timeout_s: float,
) -> dict[str, object]:
    prompt_ids = _prompt_ids_near(engine, case.prompt_tokens)
    block_table = (
        KVBlockTable(num_blocks=kv_blocks, block_size=kv_block_size)
        if case.paged_kv
        else None
    )
    scheduler = ContinuousBatchScheduler(
        max_num_requests=case.requests,
        max_num_batched_tokens=case.max_num_batched_tokens,
        prefill_chunk_size=case.prefill_chunk_size,
        max_prefill_rows_per_batch=case.max_prefill_rows_per_batch,
        block_table=block_table,
    )
    runner = BatchedModelRunner(engine=engine, scheduler=scheduler)
    worker = BatchGenerationWorker(
        runner=runner,
        device=engine.device,
        max_wait_s=case.max_wait_ms / 1000.0,
    )
    handles: list[BatchGenerationHandle] = []
    if torch.cuda.is_available() and str(engine.device).startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        for i in range(case.requests):
            handles.append(
                worker.submit(
                    prompt_ids,
                    max_new_tokens=case.max_new_tokens,
                    eos_token_ids=(),
                    request_id=f"bench-{case.requests}-{i}",
                )
            )
        for handle in handles:
            handle.wait_ids(timeout=timeout_s)
        if torch.cuda.is_available() and str(engine.device).startswith("cuda"):
            torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - t0
    finally:
        worker.shutdown()
    metrics = [handle.metrics() for handle in handles]
    output_tokens = sum(int(row["output_tokens"] or 0) for row in metrics)
    first_tokens = [
        handle.first_token_monotonic
        for handle in handles
        if handle.first_token_monotonic is not None
    ]
    finishes = [
        handle.finished_monotonic
        for handle in handles
        if handle.finished_monotonic is not None
    ]
    decode_wall_s = (
        max(finishes) - min(first_tokens)
        if first_tokens and finishes
        else None
    )
    scheduler_stats = scheduler.stats().summary()
    worker_stats = worker.stats()
    return {
        "requests": case.requests,
        "prompt_tokens": len(prompt_ids),
        "max_new_tokens": case.max_new_tokens,
        "max_num_batched_tokens": case.max_num_batched_tokens,
        "prefill_chunk_size": case.prefill_chunk_size,
        "max_prefill_rows_per_batch": case.max_prefill_rows_per_batch,
        "max_wait_ms": case.max_wait_ms,
        "paged_kv": case.paged_kv,
        "recent_window": recent_window,
        "kv_block_size": kv_block_size,
        "kv_blocks": kv_blocks,
        "elapsed_s": elapsed_s,
        "output_tokens": output_tokens,
        "aggregate_output_tok_s": output_tokens / max(elapsed_s, 1e-9),
        "aggregate_decode_tok_s": (
            output_tokens / max(decode_wall_s, 1e-9)
            if decode_wall_s is not None and decode_wall_s > 0
            else None
        ),
        "decode_wall_s": decode_wall_s,
        "mean_request_e2e_tok_s": _mean([row["e2e_tok_s"] for row in metrics]),  # type: ignore[list-item]
        "mean_request_decode_tok_s": _mean([row["decode_tok_s"] for row in metrics]),  # type: ignore[list-item]
        "mean_ttft_s": _mean([row["ttft_s"] for row in metrics]),  # type: ignore[list-item]
        "p95_ttft_s": _p95([row["ttft_s"] for row in metrics]),  # type: ignore[list-item]
        "mean_itl_s": _mean([row["mean_itl_s"] for row in metrics]),  # type: ignore[list-item]
        "scheduled_batches": scheduler_stats["total_scheduled_batches"],
        "scheduled_tokens": scheduler_stats["total_scheduled_tokens"],
        "avg_scheduled_tokens_per_batch": (
            scheduler_stats["total_scheduled_tokens"]
            / max(1, scheduler_stats["total_scheduled_batches"])
        ),
        "scheduler": scheduler_stats,
        "worker": worker_stats,
        "requests_detail": metrics,
    }


def run_single_decode_case(
    engine: RuntimeEngine,
    *,
    prompt_tokens: int,
    max_new_tokens: int,
    recent_window: int,
    timeout_s: float,
) -> dict[str, object]:
    del timeout_s
    prompt_ids = _prompt_ids_near(engine, prompt_tokens)
    cfg = GenerationConfig.greedy(max_new_tokens=max_new_tokens)
    if torch.cuda.is_available() and str(engine.device).startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = engine.generate_ids_greedy_gpu(prompt_ids, cfg)
    if torch.cuda.is_available() and str(engine.device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - t0
    return {
        "requests": 1,
        "prompt_tokens": len(prompt_ids),
        "max_new_tokens": max_new_tokens,
        "recent_window": recent_window,
        "elapsed_s": elapsed_s,
        "output_tokens": len(out),
        "aggregate_output_tok_s": len(out) / max(elapsed_s, 1e-9),
        "path": "single_generate_end_to_end",
    }


def _parse_int_list(raw: str) -> list[int]:
    vals = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not vals:
        raise ValueError("expected at least one integer")
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark adapter runtime serving throughput")
    add_model_path_args(parser)
    add_adapter_arg(parser)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weight-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--runtime-profile", choices=RUNTIME_PROFILES, default="stateful")
    mtp_group = parser.add_mutually_exclusive_group()
    mtp_group.add_argument("--enable-mtp", dest="enable_mtp", action="store_true", default=None)
    mtp_group.add_argument("--disable-mtp", dest="enable_mtp", action="store_false")
    parser.add_argument("--recent-window", type=int, default=serving_recent_window_default())
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--requests", default="1,2,4")
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--prefill-chunk-size", type=int, default=64)
    parser.add_argument("--max-prefill-rows-per-batch", type=int, default=EngineResourcePolicy.from_env().max_prefill_rows_per_batch)
    parser.add_argument("--kv-block-size", type=int, default=kv_block_size_default())
    parser.add_argument("--kv-blocks", type=int, default=kv_blocks_default())
    parser.add_argument("--max-wait-ms", type=float, default=2.0)
    parser.add_argument("--no-paged-kv", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--include-single", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    features = RuntimeFeatures.from_profile(args.runtime_profile)
    if args.enable_mtp is not None:
        features = features.with_overrides(speculative_decoding=bool(args.enable_mtp))
    engine = create_runtime_engine_from_args(args, features=features, max_state_pool_size=0)
    results: list[dict[str, object]] = []
    if args.include_single:
        results.append(
            run_single_decode_case(
                engine,
                prompt_tokens=args.prompt_tokens,
                max_new_tokens=args.max_new_tokens,
                recent_window=args.recent_window,
                timeout_s=args.timeout_s,
            )
        )
    for requests in _parse_int_list(args.requests):
        policy = EngineResourcePolicy(
            max_active_requests=requests,
            max_num_batched_tokens=args.max_num_batched_tokens,
            prefill_chunk_size=args.prefill_chunk_size,
            max_prefill_rows_per_batch=args.max_prefill_rows_per_batch,
            kv_block_size=args.kv_block_size,
            kv_blocks=args.kv_blocks,
        )
        case = ServingBenchCase(
            requests=requests,
            prompt_tokens=args.prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            max_num_batched_tokens=policy.max_num_batched_tokens,
            prefill_chunk_size=policy.prefill_chunk_size,
            max_prefill_rows_per_batch=policy.max_prefill_rows_per_batch,
            max_wait_ms=args.max_wait_ms,
            paged_kv=not args.no_paged_kv,
        )
        results.append(
            run_serving_case(
                engine,
                case,
                recent_window=args.recent_window,
                kv_block_size=args.kv_block_size,
                kv_blocks=args.kv_blocks,
                timeout_s=args.timeout_s,
            )
        )

    if args.json:
        print(json.dumps({"results": results}, indent=2, sort_keys=True))
        return
    print(
        "path,paged_kv,requests,prompt_tokens,max_new_tokens,elapsed_s,output_tokens,"
        "aggregate_output_tok_s,aggregate_decode_tok_s,mean_ttft_s,mean_itl_s,"
        "avg_scheduled_tokens_per_batch"
    )
    for row in results:
        path = row.get("path", "continuous_batching")
        print(
            f"{path},{row.get('paged_kv', '')},{row['requests']},"
            f"{row['prompt_tokens']},{row['max_new_tokens']},"
            f"{float(row['elapsed_s']):.3f},{row['output_tokens']},"
            f"{float(row['aggregate_output_tok_s']):.2f},"
            f"{_fmt(row.get('aggregate_decode_tok_s'))},"
            f"{_fmt(row.get('mean_ttft_s'))},{_fmt(row.get('mean_itl_s'))},"
            f"{_fmt(row.get('avg_scheduled_tokens_per_batch'))}"
        )


def _fmt(value: object) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
