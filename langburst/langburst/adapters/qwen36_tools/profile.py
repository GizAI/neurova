from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator

import torch
import torch.nn.functional as F

from ...cli_features import (
    add_adapter_arg,
    add_model_path_args,
    add_runtime_feature_args,
    create_runtime_engine_from_args,
    runtime_features_from_args,
)
from ...engines.native.runtime import GenerationConfig, sample_next_tensor
from ...engines.native import (
    BatchGenerationWorker,
    BatchedModelRunner,
    ContinuousBatchScheduler,
    KVBlockTable,
)
from ...loader import LowBitMarlinTensor, LowBitTensor
from ...ops import cuda_ops
from ...speculative_batch import DecodeBatchPlan
from ...tuning import marlin_direct_max_batch
from langburst.adapters.qwen36_impl import model as model_mod


def projection_category(name: str) -> str:
    if "lm_head" in name or "output.weight" in name:
        return "lm_head"
    if "embed_tokens" in name or "tok_embeddings" in name:
        return "embedding"
    if ".mlp.gate_up_proj" in name:
        return "mlp_gate_up"
    if ".mlp.gate_proj" in name or ".mlp.up_proj" in name:
        return "mlp_gate_or_up"
    if ".mlp.down_proj" in name:
        return "mlp_down"
    if ".linear_attn.in_proj_qkvz" in name or ".linear_attention.in_proj_qkvz" in name:
        return "gdn_qkvz"
    if ".linear_attn.in_proj_qkv" in name or ".linear_attention.in_proj_qkv" in name:
        return "gdn_qkv"
    if ".linear_attn.in_proj_z" in name or ".linear_attention.in_proj_z" in name:
        return "gdn_z"
    if ".linear_attn.in_proj_ba" in name or ".linear_attention.in_proj_ba" in name:
        return "gdn_ba"
    if ".linear_attn.in_proj_a" in name or ".linear_attention.in_proj_a" in name:
        return "gdn_a"
    if ".linear_attn.in_proj_b" in name or ".linear_attention.in_proj_b" in name:
        return "gdn_b"
    if ".linear_attn.out_proj" in name or ".linear_attention.out_proj" in name:
        return "gdn_out"
    if ".self_attn.qkv_proj" in name:
        return "attn_qkv"
    if ".self_attn.q_proj" in name or ".self_attn.k_proj" in name or ".self_attn.v_proj" in name:
        return "attn_qkv_split"
    if ".self_attn.o_proj" in name:
        return "attn_o"
    if name.startswith("mtp."):
        return "mtp_projection"
    return "other_projection"


@dataclass
class TimedCall:
    category: str
    start: torch.cuda.Event
    end: torch.cuda.Event


class DecodeProfiler:
    def __init__(self) -> None:
        self.calls: list[TimedCall] = []
        self.cpu_counts: defaultdict[str, int] = defaultdict(int)
        self.path_counts: defaultdict[str, int] = defaultdict(int)
        self.batch_counts: defaultdict[str, int] = defaultdict(int)
        self.batch_rows: defaultdict[str, int] = defaultdict(int)
        self.batch_tokens: defaultdict[str, int] = defaultdict(int)

    def record_cuda(self, category: str, fn: Callable, *args, **kwargs):
        if not torch.cuda.is_available():
            self.cpu_counts[category] += 1
            return fn(*args, **kwargs)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn(*args, **kwargs)
        end.record()
        self.calls.append(TimedCall(category, start, end))
        return out

    def table(self) -> list[tuple[str, int, float, float, float]]:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        by_cat: dict[str, list[float]] = defaultdict(list)
        for call in self.calls:
            by_cat[call.category].append(call.start.elapsed_time(call.end))
        for cat, count in self.cpu_counts.items():
            by_cat.setdefault(cat, [])
            if count and not by_cat[cat]:
                by_cat[cat] = [0.0] * count
        rows = []
        for cat, vals in by_cat.items():
            if not vals:
                continue
            total = float(sum(vals))
            count = len(vals)
            rows.append((cat, count, total, total / count, max(vals)))
        rows.sort(key=lambda row: row[2], reverse=True)
        return rows

    def count_path(self, category: str, path: str, *, batch: int, bits: int | None = None) -> None:
        bit_label = "na" if bits is None else str(int(bits))
        self.path_counts[f"{category},batch={int(batch)},bits={bit_label},{path}"] += 1

    def path_table(self) -> list[tuple[str, int]]:
        return sorted(self.path_counts.items(), key=lambda item: (-item[1], item[0]))

    def record_batch(self, label: str, *, rows: int, tokens: int) -> None:
        self.batch_counts[label] += 1
        self.batch_rows[label] += int(rows)
        self.batch_tokens[label] += int(tokens)

    def batch_table(self) -> list[tuple[str, int, float, float]]:
        rows: list[tuple[str, int, float, float]] = []
        for label, count in self.batch_counts.items():
            if count <= 0:
                continue
            rows.append(
                (
                    label,
                    count,
                    self.batch_rows[label] / count,
                    self.batch_tokens[label] / count,
                )
            )
        return sorted(rows, key=lambda row: row[0])


@contextmanager
def profile_decode() -> Iterator[DecodeProfiler]:
    profiler = DecodeProfiler()
    orig_marlin_gemm = LowBitMarlinTensor.gemm
    orig_lowbit_gemv = LowBitTensor.gemv
    orig_lowbit_gemm = LowBitTensor.gemm
    orig_row_dequant = LowBitTensor.row_dequant
    orig_rmsnorm = model_mod.qwen_rmsnorm
    orig_gdn_gate = model_mod.qwen_gdn_norm_silu_gate
    orig_gdn_gate_2d = model_mod.gdn_norm_silu_gate_2d
    orig_depthwise = model_mod.depthwise_conv_update
    orig_attention = cuda_ops().attention_decode_fp16
    orig_paged_fp16 = getattr(cuda_ops(), "attention_decode_paged_fp16", None)
    orig_paged_int4 = getattr(cuda_ops(), "attention_decode_paged_int4", None)
    orig_paged_int4_flash = getattr(cuda_ops(), "attention_paged_int4_flash", None)
    orig_append_int4 = getattr(cuda_ops(), "attention_append_paged_int4", None)
    orig_append_int4_spec = getattr(cuda_ops(), "attention_append_paged_int4_spec", None)
    orig_gdn_ab = cuda_ops().gdn_recurrent_ab
    orig_sdpa = F.scaled_dot_product_attention

    def marlin_gemm(self: LowBitMarlinTensor, x: torch.Tensor):
        batch = int(x.size(0)) if x.ndim >= 2 else 1
        category = projection_category(self.name)
        max_direct = marlin_direct_max_batch()
        if batch > max_direct:
            profiler.count_path(category, "marlin_row_loop_parent", batch=batch, bits=getattr(self, "exec_bits", None))
        else:
            cache_policy = os.environ.get("LANGBURST_MARLIN_OUT_CACHE_POLICY", "off").strip().lower()
            cache_out = cache_policy not in {"0", "false", "off", "none", "no_cache"} and (
                cache_policy != "decode_only" or batch == 1
            )
            if cache_out and batch in getattr(self, "_out_cache", {}):
                profiler.count_path(category, "marlin_direct_cache_hit", batch=batch, bits=getattr(self, "exec_bits", None))
            elif cache_out:
                profiler.count_path(category, "marlin_direct_cache_miss", batch=batch, bits=getattr(self, "exec_bits", None))
            else:
                profiler.count_path(category, "marlin_direct_no_cache", batch=batch, bits=getattr(self, "exec_bits", None))
        return profiler.record_cuda(projection_category(self.name), orig_marlin_gemm, self, x)

    def lowbit_gemv(self: LowBitTensor, x: torch.Tensor):
        profiler.count_path(projection_category(self.name), "lowbit_gemv", batch=1, bits=getattr(self, "bits", None))
        return profiler.record_cuda(projection_category(self.name), orig_lowbit_gemv, self, x)

    def lowbit_gemm(self: LowBitTensor, x: torch.Tensor):
        batch = int(x.size(0)) if x.ndim >= 2 else 1
        profiler.count_path(projection_category(self.name), "lowbit_gemm", batch=batch, bits=getattr(self, "bits", None))
        return profiler.record_cuda(projection_category(self.name), orig_lowbit_gemm, self, x)

    def row_dequant(self: LowBitTensor, row):
        return profiler.record_cuda("embedding", orig_row_dequant, self, row)

    def qwen_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float):
        return profiler.record_cuda("rmsnorm", orig_rmsnorm, x, weight, eps)

    def qwen_gdn_norm_silu_gate(core: torch.Tensor, weight: torch.Tensor, z: torch.Tensor, eps: float):
        return profiler.record_cuda("gdn_norm_gate", orig_gdn_gate, core, weight, z, eps)

    def gdn_norm_silu_gate_2d(core: torch.Tensor, weight: torch.Tensor, z: torch.Tensor, eps: float):
        return profiler.record_cuda("gdn_norm_gate", orig_gdn_gate_2d, core, weight, z, eps)

    def depthwise_conv_update(buf: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, bias=None):
        return profiler.record_cuda("gdn_conv", orig_depthwise, buf, x, weight, bias)

    def attention_decode_fp16(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, length: int, scale: float):
        return profiler.record_cuda("attention_decode", orig_attention, q, k_cache, v_cache, length, scale)

    def attention_decode_paged_fp16(*args, **kwargs):
        return profiler.record_cuda("attention_paged_fp16", orig_paged_fp16, *args, **kwargs)

    def attention_decode_paged_int4(*args, **kwargs):
        q = args[0] if args else None
        batch = int(q.size(0)) if torch.is_tensor(q) and q.ndim >= 1 else 0
        profiler.count_path("attention_paged_int4", "direct_kernel", batch=batch, bits=4)
        return profiler.record_cuda("attention_paged_int4", orig_paged_int4, *args, **kwargs)

    def attention_paged_int4_flash(*args, **kwargs):
        q = args[0] if args else None
        batch = int(q.size(0)) if torch.is_tensor(q) and q.ndim >= 1 else 0
        profiler.count_path("attention_paged_int4", "flash_contract_kernel", batch=batch, bits=4)
        return profiler.record_cuda("attention_paged_int4_flash", orig_paged_int4_flash, *args, **kwargs)

    def attention_append_paged_int4(*args, **kwargs):
        k = args[0] if args else None
        batch = int(k.size(0)) if torch.is_tensor(k) and k.ndim >= 1 else 0
        profiler.count_path("cache_update", "append_paged_int4", batch=batch, bits=4)
        return profiler.record_cuda("cache_update", orig_append_int4, *args, **kwargs)

    def attention_append_paged_int4_spec(*args, **kwargs):
        k = args[0] if args else None
        batch = int(k.size(0)) if torch.is_tensor(k) and k.ndim >= 1 else 0
        profiler.count_path("cache_update", "append_paged_int4_spec", batch=batch, bits=4)
        return profiler.record_cuda("cache_update", orig_append_int4_spec, *args, **kwargs)

    def gdn_recurrent_ab(q, k, v, a, b, A_log, dt_bias, state):
        return profiler.record_cuda("gdn_recurrent", orig_gdn_ab, q, k, v, a, b, A_log, dt_bias, state)

    def scaled_dot_product_attention(*args, **kwargs):
        return profiler.record_cuda("sdpa_attention", orig_sdpa, *args, **kwargs)

    LowBitMarlinTensor.gemm = marlin_gemm  # type: ignore[method-assign]
    LowBitTensor.gemv = lowbit_gemv  # type: ignore[method-assign]
    LowBitTensor.gemm = lowbit_gemm  # type: ignore[method-assign]
    LowBitTensor.row_dequant = row_dequant  # type: ignore[method-assign]
    model_mod.qwen_rmsnorm = qwen_rmsnorm
    model_mod.qwen_gdn_norm_silu_gate = qwen_gdn_norm_silu_gate
    model_mod.gdn_norm_silu_gate_2d = gdn_norm_silu_gate_2d
    model_mod.depthwise_conv_update = depthwise_conv_update
    try:
        setattr(cuda_ops(), "attention_decode_fp16", attention_decode_fp16)
        if callable(orig_paged_fp16):
            setattr(cuda_ops(), "attention_decode_paged_fp16", attention_decode_paged_fp16)
        if callable(orig_paged_int4):
            setattr(cuda_ops(), "attention_decode_paged_int4", attention_decode_paged_int4)
        if callable(orig_paged_int4_flash):
            setattr(cuda_ops(), "attention_paged_int4_flash", attention_paged_int4_flash)
        if callable(orig_append_int4):
            setattr(cuda_ops(), "attention_append_paged_int4", attention_append_paged_int4)
        if callable(orig_append_int4_spec):
            setattr(cuda_ops(), "attention_append_paged_int4_spec", attention_append_paged_int4_spec)
        setattr(cuda_ops(), "gdn_recurrent_ab", gdn_recurrent_ab)
        F.scaled_dot_product_attention = scaled_dot_product_attention
        yield profiler
    finally:
        LowBitMarlinTensor.gemm = orig_marlin_gemm  # type: ignore[method-assign]
        LowBitTensor.gemv = orig_lowbit_gemv  # type: ignore[method-assign]
        LowBitTensor.gemm = orig_lowbit_gemm  # type: ignore[method-assign]
        LowBitTensor.row_dequant = orig_row_dequant  # type: ignore[method-assign]
        model_mod.qwen_rmsnorm = orig_rmsnorm
        model_mod.qwen_gdn_norm_silu_gate = orig_gdn_gate
        model_mod.gdn_norm_silu_gate_2d = orig_gdn_gate_2d
        model_mod.depthwise_conv_update = orig_depthwise
        setattr(cuda_ops(), "attention_decode_fp16", orig_attention)
        if callable(orig_paged_fp16):
            setattr(cuda_ops(), "attention_decode_paged_fp16", orig_paged_fp16)
        if callable(orig_paged_int4):
            setattr(cuda_ops(), "attention_decode_paged_int4", orig_paged_int4)
        if callable(orig_paged_int4_flash):
            setattr(cuda_ops(), "attention_paged_int4_flash", orig_paged_int4_flash)
        if callable(orig_append_int4):
            setattr(cuda_ops(), "attention_append_paged_int4", orig_append_int4)
        if callable(orig_append_int4_spec):
            setattr(cuda_ops(), "attention_append_paged_int4_spec", orig_append_int4_spec)
        setattr(cuda_ops(), "gdn_recurrent_ab", orig_gdn_ab)
        F.scaled_dot_product_attention = orig_sdpa


def _make_single_token_batch_plan(
    *,
    input_ids: torch.Tensor,
    states: list[object],
    device: torch.device,
) -> DecodeBatchPlan:
    batch = int(input_ids.numel())
    positions = torch.tensor([int(getattr(state, "pos")) for state in states], dtype=torch.long, device=device)
    query_start = torch.arange(0, batch + 1, dtype=torch.int32, device=device)
    seq_lens = (positions + 1).to(dtype=torch.int32)
    logits_indices = torch.arange(0, batch, dtype=torch.long, device=device)
    cu_num_logits = torch.arange(0, batch + 1, dtype=torch.int32, device=device)
    return DecodeBatchPlan(
        request_ids=[f"profile-{row}" for row in range(batch)],
        state_indices=torch.arange(0, batch, dtype=torch.int32, device=device),
        input_ids=input_ids.to(device=device, dtype=torch.long).reshape(-1).contiguous(),
        positions=positions,
        query_start_loc=query_start,
        seq_lens=seq_lens,
        logits_indices=logits_indices,
        cu_num_logits=cu_num_logits,
        row_spans=tuple((row, row + 1) for row in range(batch)),
        num_scheduled_tokens=[1] * batch,
        num_draft_tokens_per_request=[0] * batch,
        is_prefill=[False] * batch,
    )


def _print_profile_result(
    *,
    label: str,
    prompt_tokens: int,
    generated: int,
    elapsed: float,
    profiler: DecodeProfiler,
) -> None:
    print(
        f"profile={label} prompt_tokens={prompt_tokens} generated={generated} "
        f"elapsed_s={elapsed:.3f} tok_s={generated/max(elapsed, 1e-9):.2f}"
    )
    print("category,calls,total_ms,avg_us,max_us,pct_measured")
    rows = profiler.table()
    total = sum(row[2] for row in rows) or 1.0
    for cat, count, total_ms, avg_ms, max_ms in rows:
        print(f"{cat},{count},{total_ms:.3f},{avg_ms*1000:.2f},{max_ms*1000:.2f},{total_ms/total*100:.2f}")
    if profiler.path_counts:
        print("path,count")
        for path, count in profiler.path_table():
            print(f"{path},{count}")
    if profiler.batch_counts:
        print("batch_kind,calls,avg_rows,avg_tokens")
        for label, count, avg_rows, avg_tokens in profiler.batch_table():
            print(f"{label},{count},{avg_rows:.2f},{avg_tokens:.2f}")


def _parse_batch_sizes(raw: str) -> list[int]:
    out: list[int] = []
    if not raw.strip():
        return out
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 1:
            raise ValueError("batch sizes must be >= 1")
        out.append(value)
    return out or [1]


def _profile_direct_single(engine, prompt_ids: list[int], cfg: GenerationConfig, features, args) -> None:
    state = engine.new_state()
    prefill_logits = engine.prefill(prompt_ids, state, features)
    prefill_next = sample_next_tensor(prefill_logits, cfg)
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.synchronize()
    with profile_decode() as profiler:
        t0 = time.perf_counter()
        next_token = prefill_next
        generated = 0
        for i in range(args.max_new_tokens):
            generated += 1
            if i == args.max_new_tokens - 1:
                break
            logits = engine.forward_one(next_token, state, return_logits=True)
            next_token = profiler.record_cuda("sampling", sample_next_tensor, logits, cfg)
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    _print_profile_result(
        label="direct_single",
        prompt_tokens=len(prompt_ids),
        generated=generated,
        elapsed=elapsed,
        profiler=profiler,
    )


def _profile_decode_batch(engine, prompt_ids: list[int], cfg: GenerationConfig, features, args, *, batch_size: int) -> None:
    device = torch.device(args.device)
    states = []
    next_tokens: list[torch.Tensor] = []
    for _ in range(batch_size):
        state = engine.new_state()
        logits = engine.prefill(prompt_ids, state, features)
        token = sample_next_tensor(logits, cfg)
        states.append(state)
        next_tokens.append(token.reshape(()))
    next_input = torch.stack(next_tokens, dim=0).to(device=device, dtype=torch.long).contiguous()
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.synchronize()
    with profile_decode() as profiler:
        t0 = time.perf_counter()
        generated = 0
        for i in range(args.max_new_tokens):
            generated += batch_size
            if i == args.max_new_tokens - 1:
                break
            plan = _make_single_token_batch_plan(input_ids=next_input, states=states, device=device)
            logits_rows = engine.forward_batch(plan, states, return_logits=True)
            logits = torch.stack([row.contiguous() for row in logits_rows if row is not None], dim=0).contiguous()
            next_input = profiler.record_cuda("sampling", cuda_ops().argmax_many, logits).to(device=device, dtype=torch.long)
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    _print_profile_result(
        label=f"decode_batch_B{batch_size}",
        prompt_tokens=len(prompt_ids),
        generated=generated,
        elapsed=elapsed,
        profiler=profiler,
    )


def _profile_serving_worker(engine, prompt_ids: list[int], cfg: GenerationConfig, args, *, batch_size: int) -> None:
    prefill_chunk_size = int(args.prefill_chunk_size) if args.prefill_chunk_size is not None else 64
    block_table = KVBlockTable(num_blocks=int(args.worker_kv_blocks), block_size=int(args.worker_kv_block_size))
    scheduler = ContinuousBatchScheduler(
        max_num_requests=batch_size,
        max_num_batched_tokens=max(1, int(args.worker_max_num_batched_tokens)),
        prefill_chunk_size=max(1, prefill_chunk_size),
        max_prefill_rows_per_batch=max(0, int(args.worker_max_prefill_rows_per_batch)),
        block_table=block_table,
    )
    runner = BatchedModelRunner(engine=engine, scheduler=scheduler)
    worker = BatchGenerationWorker(runner=runner, device=engine.device, max_wait_s=float(args.worker_max_wait_ms) / 1000.0)
    try:
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
        with profile_decode() as profiler:
            orig_schedule = scheduler.schedule

            def schedule_with_profile(*, device: str = "cpu"):
                batch = orig_schedule(device=device)
                if batch is not None:
                    if any(bool(v) for v in batch.is_prefill):
                        label = "prefill"
                    elif batch.spec_decode_metadata is not None:
                        label = "spec_decode"
                    else:
                        label = "decode"
                    profiler.record_batch(label, rows=len(batch.request_ids), tokens=int(batch.num_tokens))
                return batch

            scheduler.schedule = schedule_with_profile  # type: ignore[method-assign]
            t0 = time.perf_counter()
            handles = [
                worker.submit(
                    prompt_ids,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_ids=(),
                    generation_config=cfg,
                    request_id=f"profile-worker-{batch_size}-{row}",
                    prefix_cache_enabled=False,
                )
                for row in range(batch_size)
            ]
            for handle in handles:
                handle.wait_ids(timeout=max(30.0, args.max_new_tokens * 2.0))
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
        generated = sum(len(handle.generated) for handle in handles)
        _print_profile_result(
            label=f"serving_worker_paged_B{batch_size}",
            prompt_tokens=len(prompt_ids),
            generated=generated,
            elapsed=elapsed,
            profiler=profiler,
        )
    finally:
        worker.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Qwen3.6 adapter target-only decode bottlenecks")
    add_model_path_args(parser)
    add_adapter_arg(parser, adapter_ids=("qwen36", "qwen36-a3b"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weight-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--recent-window", type=int, default=256)
    parser.add_argument("--prompt", default="Write a concise technical note about quantized LLM inference.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--include-prefill", action="store_true", help="profile prompt prefill as well as decode")
    parser.add_argument(
        "--batch-sizes",
        default="1",
        help="comma-separated single-token decode batch sizes to profile through RuntimeEngine.forward_batch",
    )
    parser.add_argument(
        "--skip-direct-single",
        action="store_true",
        help="skip the direct engine.forward_one single-request profile",
    )
    parser.add_argument(
        "--worker-batch-sizes",
        default="",
        help="comma-separated serving worker batch sizes to profile through the arena/paged KV path",
    )
    parser.add_argument("--worker-kv-block-size", type=int, default=16)
    parser.add_argument("--worker-kv-blocks", type=int, default=768)
    parser.add_argument("--worker-max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--worker-max-prefill-rows-per-batch", type=int, default=1)
    parser.add_argument("--worker-max-wait-ms", type=float, default=2.0)
    add_runtime_feature_args(parser)
    args = parser.parse_args()
    if args.adapter not in {"qwen36", "qwen36-a3b"}:
        parser.error("langburst-qwen-profile currently supports only qwen36 adapters")
    features = runtime_features_from_args(args)

    engine = create_runtime_engine_from_args(args, features=features)
    prompt_ids = engine.encode_prompt(args.prompt)
    cfg = GenerationConfig.greedy(max_new_tokens=args.max_new_tokens)
    if args.include_prefill:
        with profile_decode() as profiler:
            t0 = time.perf_counter()
            out = engine.generate_ids_greedy_gpu(prompt_ids, cfg)
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
        _print_profile_result(
            label="direct_generate_include_prefill",
            prompt_tokens=len(prompt_ids),
            generated=len(out),
            elapsed=elapsed,
            profiler=profiler,
        )
        return

    if not args.skip_direct_single:
        _profile_direct_single(engine, prompt_ids, cfg, features, args)
    for batch_size in _parse_batch_sizes(args.batch_sizes):
        _profile_decode_batch(engine, prompt_ids, cfg, features, args, batch_size=batch_size)
    if args.worker_batch_sizes.strip():
        for batch_size in _parse_batch_sizes(args.worker_batch_sizes):
            _profile_serving_worker(engine, prompt_ids, cfg, args, batch_size=batch_size)


if __name__ == "__main__":
    main()
