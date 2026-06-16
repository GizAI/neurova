from __future__ import annotations

import argparse
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import torch
import torch.nn.functional as F

from .adapters import Qwen36Adapter  # noqa: F401 - registers qwen36
from .cli_features import add_runtime_feature_args, runtime_features_from_args
from .core.adapter import adapter_registry
from .core.runtime import GenerationConfig, RuntimeEngine, sample_next_tensor
from .loader import LowBitMarlinTensor, LowBitTensor
from .ops import cuda_ops
import qwenburst.model as model_mod


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
    orig_gdn_ab = cuda_ops().gdn_recurrent_ab
    orig_sdpa = F.scaled_dot_product_attention

    def marlin_gemm(self: LowBitMarlinTensor, x: torch.Tensor):
        return profiler.record_cuda(projection_category(self.name), orig_marlin_gemm, self, x)

    def lowbit_gemv(self: LowBitTensor, x: torch.Tensor):
        return profiler.record_cuda(projection_category(self.name), orig_lowbit_gemv, self, x)

    def lowbit_gemm(self: LowBitTensor, x: torch.Tensor):
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
        setattr(cuda_ops(), "gdn_recurrent_ab", orig_gdn_ab)
        F.scaled_dot_product_attention = orig_sdpa


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile QwenBurst target-only decode bottlenecks")
    parser.add_argument("--hf-model", type=Path, default=Path("/home/user/models/Qwen3.6-27B"))
    parser.add_argument("--qb-model", type=Path, default=Path("/home/user/models/Qwen3.6-27B-qb4-marlin-fused"))
    parser.add_argument("--adapter", default="qwen36", choices=("qwen36",))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weight-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--recent-window", type=int, default=256)
    parser.add_argument("--prompt", default="Write a concise technical note about quantized LLM inference.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--include-prefill", action="store_true", help="profile prompt prefill as well as decode")
    add_runtime_feature_args(parser)
    args = parser.parse_args()
    features = runtime_features_from_args(args)

    engine = RuntimeEngine(
        adapter=adapter_registry.get(args.adapter),
        hf_model=args.hf_model,
        qb_model=args.qb_model,
        device=args.device,
        recent_window=args.recent_window,
        weight_device=args.weight_device,
        features=features,
    )
    prompt_ids = engine.encode_prompt(args.prompt)
    cfg = GenerationConfig(max_new_tokens=args.max_new_tokens, temperature=0.0, top_k=0, eos_token_ids=())
    state = engine.new_state()
    prefill_logits: torch.Tensor | None = None
    prefill_next: torch.Tensor | None = None
    if not args.include_prefill:
        prefill_logits = engine.prefill(prompt_ids, state, features)
        prefill_next = sample_next_tensor(prefill_logits, cfg)
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
    with profile_decode() as profiler:
        t0 = time.perf_counter()
        if args.include_prefill:
            out = engine.generate_ids_greedy_gpu(prompt_ids, cfg)
        else:
            assert prefill_next is not None
            next_token = prefill_next
            out = []
            for i in range(args.max_new_tokens):
                out.append(next_token)
                if i == args.max_new_tokens - 1:
                    break
                logits = engine.forward_one(next_token, state, return_logits=True)
                next_token = sample_next_tensor(logits, cfg)
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    print(
        f"prompt_tokens={len(prompt_ids)} generated={len(out)} "
        f"max_new_tokens={args.max_new_tokens} include_prefill={args.include_prefill} "
        f"elapsed_s={elapsed:.3f} tok_s={len(out)/max(elapsed, 1e-9):.2f}"
    )
    print("category,calls,total_ms,avg_us,max_us,pct_measured")
    rows = profiler.table()
    total = sum(row[2] for row in rows) or 1.0
    for cat, count, total_ms, avg_ms, max_ms in rows:
        print(f"{cat},{count},{total_ms:.3f},{avg_ms*1000:.2f},{max_ms*1000:.2f},{total_ms/total*100:.2f}")


if __name__ == "__main__":
    main()
