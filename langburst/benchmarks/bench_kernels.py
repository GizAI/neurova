from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from langburst.ops import cuda_ops
from langburst.adapters.qwen36_impl.config import Qwen36_27B_TextConfig
from langburst.adapters.qwen36_tools.quantize import quantize_symmetric_lowbit


def bench(fn, warmup=20, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--lowbit-rows", type=int, default=5120)
    ap.add_argument("--lowbit-cols", type=int, default=5120)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--rows-per-cta", type=int, default=8)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; run this benchmark on the RTX 4080 box")
    ops = cuda_ops()
    cfg = Qwen36_27B_TextConfig()

    # GDN recurrent kernel benchmark: one layer, one token.
    q = torch.randn(cfg.linear_num_key_heads, 128, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn(cfg.linear_num_value_heads, 128, device="cuda", dtype=torch.float16)
    g = torch.full((cfg.linear_num_value_heads,), -0.1, device="cuda", dtype=torch.float32)
    beta = torch.full((cfg.linear_num_value_heads,), 0.5, device="cuda", dtype=torch.float16)
    state = torch.zeros(cfg.linear_num_value_heads, 128, 128, device="cuda", dtype=torch.float16)
    dt = bench(lambda: ops.gdn_recurrent(q, k, v, g, beta, state), iters=args.iters)
    print(f"gdn_recurrent one layer: {dt*1e6:.2f} us")
    print(f"gdn_recurrent 48 layers projected: {dt*48*1e3:.2f} ms")

    # RMSNorm hidden=5120.
    x = torch.randn(cfg.hidden_size, device="cuda", dtype=torch.float16)
    w = torch.ones(cfg.hidden_size, device="cuda", dtype=torch.float16)
    dt = bench(lambda: ops.rmsnorm(x, w, cfg.rms_norm_eps), iters=args.iters)
    print(f"rmsnorm 5120: {dt*1e6:.2f} us")

    # Low-bit GEMV shape close to core hidden projection.
    torch.manual_seed(0)
    host_w = torch.randn(args.lowbit_rows, args.lowbit_cols, dtype=torch.float32) * 0.02
    packed, scales, meta = quantize_symmetric_lowbit(host_w, group_size=128, bits=args.bits)
    qw = torch.from_numpy(np.asarray(packed)).cuda().contiguous()
    sc = torch.from_numpy(np.asarray(scales)).cuda().contiguous()
    xv = torch.randn(args.lowbit_cols, device="cuda", dtype=torch.float16)
    dt = bench(
        lambda: ops.lowbit_gemv(qw, sc, xv, meta["cols"], meta["group_size"], meta["bits"], args.rows_per_cta),
        iters=max(50, args.iters // 5),
    )
    bytes_read = qw.numel() + sc.numel() * 2 + xv.numel() * 2
    print(
        f"lowbit_gemv bits={meta['bits']} rows_per_cta={args.rows_per_cta} "
        f"{args.lowbit_rows}x{args.lowbit_cols}: {dt*1e6:.2f} us, approx read {bytes_read/1e6:.2f} MB"
    )
    host_w2 = torch.randn(args.lowbit_rows, args.lowbit_cols, dtype=torch.float32) * 0.02
    packed2, scales2, _ = quantize_symmetric_lowbit(host_w2, group_size=128, bits=args.bits)
    qw2 = torch.from_numpy(np.asarray(packed2)).cuda().contiguous()
    sc2 = torch.from_numpy(np.asarray(scales2)).cuda().contiguous()
    dt_two = bench(
        lambda: (
            ops.lowbit_gemv(qw, sc, xv, meta["cols"], meta["group_size"], meta["bits"], args.rows_per_cta),
            ops.lowbit_gemv(qw2, sc2, xv, meta["cols"], meta["group_size"], meta["bits"], args.rows_per_cta),
        ),
        iters=max(50, args.iters // 5),
    )
    dt_pair = bench(
        lambda: ops.lowbit_gemv_pair(qw, sc, qw2, sc2, xv, meta["cols"], meta["group_size"], meta["bits"], args.rows_per_cta),
        iters=max(50, args.iters // 5),
    )
    print(f"lowbit_gemv_pair two singles: {dt_two*1e6:.2f} us; pair: {dt_pair*1e6:.2f} us")

    # GPU argmax over padded Qwen vocab.
    logits = torch.randn(4, cfg.vocab_size_padded, device="cuda", dtype=torch.float16)
    out = torch.empty(4, device="cuda", dtype=torch.long)
    dt = bench(lambda: ops.argmax_many_out(logits, out), iters=args.iters)
    print(f"argmax_many_out 4 x {cfg.vocab_size_padded}: {dt*1e6:.2f} us")


if __name__ == "__main__":
    main()
