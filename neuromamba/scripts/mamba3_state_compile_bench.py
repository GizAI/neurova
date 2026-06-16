#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mamba_ssm.utils.generation import InferenceParams
from neuromamba.cli import load_or_new, setup_perf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark state-compiled Mamba-3 SISO prefill reuse.")
    parser.add_argument("--mode", default="mamba3-siso-fast-0.3b-ds128")
    parser.add_argument("--tokenizer", default="llama31")
    parser.add_argument("--checkpoint", type=Path, default=Path("neuromamba/runs/mamba3_siso_fast_0_3b_ds128_v1/model.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--prefix-tokens", type=int, default=4096)
    parser.add_argument(
        "--prefix-text",
        default="The important fact is Neurova should keep speed, intelligence, and context.",
        help="Text repeated until --prefix-tokens is reached.",
    )
    parser.add_argument("--question", default="Instruction: What is the important fact?\nAnswer:")
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=4)
    return parser.parse_args()


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def clone_cache(cache: dict) -> dict:
    out = {}
    for key, value in cache.items():
        if isinstance(value, tuple):
            out[key] = tuple(item.clone() if torch.is_tensor(item) else copy.deepcopy(item) for item in value)
        elif torch.is_tensor(value):
            out[key] = value.clone()
        else:
            out[key] = copy.deepcopy(value)
    return out


def build_repeated_prefix_ids(tokenizer, text: str, target_tokens: int) -> list[int]:
    unit = tokenizer.encode(text.strip() + " ", add_eos=False)
    if not unit:
        raise ValueError("--prefix-text produced no tokens")
    repeats = (target_tokens + len(unit) - 1) // len(unit)
    return (unit * repeats)[:target_tokens]


@torch.inference_mode()
def greedy_from_state(model, tokenizer, params: InferenceParams, first_token_id: int, args: argparse.Namespace) -> tuple[str, int, float]:
    ids = [first_token_id]
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    sync(args.device)
    t0 = time.time()
    token = torch.tensor([[first_token_id]], device=args.device, dtype=torch.long)
    for _ in range(args.max_new):
        logits = model(token, inference_params=params, num_last_tokens=1).logits[:, -1, : tokenizer.vocab_size]
        params.seqlen_offset += token.shape[1]
        next_id = int(torch.argmax(logits, dim=-1).item())
        ids.append(next_id)
        if next_id == tokenizer.eos_id:
            break
        token = torch.tensor([[next_id]], device=args.device, dtype=torch.long)
    sync(args.device)
    return tokenizer.decode(ids), len(ids), max(time.time() - t0, 1e-9)


@torch.inference_mode()
def run_tokens_from_state(model, token_ids: list[int], params: InferenceParams, device: str, vocab_size: int) -> torch.Tensor:
    logits = None
    for token_id in token_ids:
        token = torch.tensor([[token_id]], device=device, dtype=torch.long)
        logits = model(token, inference_params=params, num_last_tokens=1).logits[:, -1, :vocab_size]
        params.seqlen_offset += 1
    if logits is None:
        raise ValueError("token_ids must not be empty")
    return logits


@torch.inference_mode()
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

    prefix_ids = build_repeated_prefix_ids(tokenizer, args.prefix_text, args.prefix_tokens)
    question_ids = tokenizer.encode(args.question, add_eos=False)
    max_seqlen = len(prefix_ids) + len(question_ids) + args.max_new + 8

    prefix = torch.tensor([prefix_ids], device=args.device, dtype=torch.long)
    question = torch.tensor([question_ids], device=args.device, dtype=torch.long)
    full = torch.tensor([prefix_ids + question_ids], device=args.device, dtype=torch.long)

    sync(args.device)
    compile_t0 = time.time()
    base_params = InferenceParams(max_seqlen=max_seqlen, max_batch_size=1)
    _ = model(prefix, inference_params=base_params, num_last_tokens=1)
    base_params.seqlen_offset += prefix.shape[1]
    sync(args.device)
    compile_elapsed = max(time.time() - compile_t0, 1e-9)
    compiled_cache = clone_cache(base_params.key_value_memory_dict)

    runs = []
    for idx in range(args.repeats):
        sync(args.device)
        t0 = time.time()
        full_logits = model(full, num_last_tokens=1).logits[:, -1, : tokenizer.vocab_size]
        full_first = int(torch.argmax(full_logits, dim=-1).item())
        sync(args.device)
        full_elapsed = max(time.time() - t0, 1e-9)

        params = InferenceParams(
            max_seqlen=max_seqlen,
            max_batch_size=1,
            seqlen_offset=len(prefix_ids),
            key_value_memory_dict=clone_cache(compiled_cache),
        )
        sync(args.device)
        q0 = time.time()
        compiled_logits = run_tokens_from_state(
            model,
            question_ids,
            params,
            args.device,
            tokenizer.vocab_size,
        )
        compiled_first = int(torch.argmax(compiled_logits, dim=-1).item())
        sync(args.device)
        compiled_question_elapsed = max(time.time() - q0, 1e-9)
        text, tokens, decode_elapsed = greedy_from_state(model, tokenizer, params, compiled_first, args)
        runs.append(
            {
                "repeat": idx,
                "full_prefill_elapsed_sec": round(full_elapsed, 6),
                "compiled_question_elapsed_sec": round(compiled_question_elapsed, 6),
                "speedup_excluding_compile": round(full_elapsed / compiled_question_elapsed, 2),
                "first_token_parity": full_first == compiled_first,
                "compiled_first_token": tokenizer.decode([compiled_first]),
                "decode_tokens": tokens,
                "decode_elapsed_sec": round(decode_elapsed, 6),
                "decode_tok_s": round(tokens / decode_elapsed, 2),
                "sample": text[:160],
            }
        )

    print(
        json.dumps(
            {
                "mode": args.mode,
                "checkpoint": str(args.checkpoint),
                "prefix_tokens": len(prefix_ids),
                "question_tokens": len(question_ids),
                "compile_elapsed_sec": round(compile_elapsed, 6),
                "cache_layers": len(compiled_cache),
                "runs": runs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
