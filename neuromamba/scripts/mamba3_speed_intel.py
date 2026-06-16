#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mamba_ssm.utils.generation import decode
from neuromamba.cli import load_or_new, setup_perf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure Mamba-3 inference bottlenecks.")
    parser.add_argument("--mode", default="mamba3-siso-fast-0.3b")
    parser.add_argument("--tokenizer", default="llama31")
    parser.add_argument("--checkpoint", type=Path, default=Path("neuromamba/runs/mamba3_siso_fast_0_3b_v1/model.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--prompt", default="Instruction: Who are you?\nAnswer:")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--batches", type=str, default="1,2,4,8,16,32")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cuda-graph", action="store_true")
    return parser.parse_args()


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


@torch.inference_mode()
def bench_decode(model, tokenizer, ids: list[int], batch_size: int, args: argparse.Namespace) -> dict:
    base = torch.tensor(ids[-args.seq_len:], device=args.device, dtype=torch.long)
    input_ids = base.unsqueeze(0).expand(batch_size, -1).contiguous()
    runs = []
    for idx in range(args.repeats + 1):
        sync(args.device)
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        out = decode(
            input_ids,
            model,
            max_length=input_ids.shape[1] + args.max_new,
            top_k=1,
            top_p=0.0,
            temperature=1.0,
            repetition_penalty=1.0,
            eos_token_id=tokenizer.eos_id,
            vocab_size=tokenizer.vocab_size,
            output_scores=False,
            cg=args.cuda_graph,
        )
        sync(args.device)
        elapsed = max(time.time() - t0, 1e-9)
        new_tokens_each = max(0, out.sequences.shape[1] - input_ids.shape[1])
        total_new_tokens = new_tokens_each * batch_size
        runs.append(
            {
                "kind": "warmup" if idx == 0 else "measure",
                "batch_size": batch_size,
                "new_tokens_each": int(new_tokens_each),
                "total_new_tokens": int(total_new_tokens),
                "elapsed_sec": round(elapsed, 6),
                "aggregate_tok_s": round(total_new_tokens / elapsed, 2),
                "per_request_tok_s": round(new_tokens_each / elapsed, 2),
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3)
                if args.device.startswith("cuda")
                else None,
            }
        )
    measured = runs[1:]
    return {
        "batch_size": batch_size,
        "runs": runs,
        "avg_aggregate_tok_s": round(
            sum(run["aggregate_tok_s"] for run in measured) / max(1, len(measured)), 2
        ),
        "avg_per_request_tok_s": round(
            sum(run["per_request_tok_s"] for run in measured) / max(1, len(measured)), 2
        ),
    }


@torch.inference_mode()
def bench_forward_split(model, tokenizer, ids: list[int], batch_size: int, args: argparse.Namespace) -> dict:
    base = torch.tensor(ids[-args.seq_len:], device=args.device, dtype=torch.long)
    input_ids = base.unsqueeze(0).expand(batch_size, -1).contiguous()
    # Prefill-style full sequence: separate trunk and LM head. This estimates whether vocab projection dominates.
    sync(args.device)
    t0 = time.time()
    hidden = model.backbone(input_ids)
    sync(args.device)
    trunk_elapsed = max(time.time() - t0, 1e-9)
    last_hidden = hidden[:, -1:]
    sync(args.device)
    t1 = time.time()
    logits = model.lm_head(last_hidden)
    _ = torch.argmax(logits[:, -1, : tokenizer.vocab_size], dim=-1)
    sync(args.device)
    head_elapsed = max(time.time() - t1, 1e-9)
    return {
        "batch_size": batch_size,
        "trunk_fullseq_elapsed_ms": round(trunk_elapsed * 1000, 3),
        "lm_head_last_token_elapsed_ms": round(head_elapsed * 1000, 3),
        "lm_head_vocab_size": tokenizer.vocab_size,
    }


def main() -> None:
    args = parse_args()
    setup_perf(args.device)
    load_args = SimpleNamespace(
        cmd="fast-generate",
        mode=args.mode,
        tokenizer=args.tokenizer,
        checkpoint=args.checkpoint,
        device=args.device,
        dtype=args.dtype,
        activation_checkpointing=False,
    )
    model, tokenizer, _ = load_or_new(load_args)
    model.eval()
    ids = tokenizer.encode(args.prompt, add_eos=False)
    batches = [int(item) for item in args.batches.split(",") if item.strip()]
    results = []
    for batch_size in batches:
        try:
            results.append(
                {
                    "decode": bench_decode(model, tokenizer, ids, batch_size, args),
                    "forward_split": bench_forward_split(model, tokenizer, ids, batch_size, args),
                }
            )
        except torch.cuda.OutOfMemoryError as exc:
            results.append({"batch_size": batch_size, "oom": str(exc)})
            torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "mode": args.mode,
                "checkpoint": str(args.checkpoint),
                "cuda_graph": bool(args.cuda_graph),
                "seq_len": args.seq_len,
                "max_new": args.max_new,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
