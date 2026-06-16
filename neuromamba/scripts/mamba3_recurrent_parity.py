#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mamba_ssm.utils.generation import InferenceParams
from neuromamba.cli import _top_tokens, load_or_new, setup_perf


PROMPTS = [
    "Instruction: Who are you? Answer:",
    "Instruction: Where is Korea? Answer:",
    "Instruction: What is 2 + 5? Answer:",
    "Who are you?",
    "The main idea is",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Mamba recurrent cache/full-forward parity")
    parser.add_argument("--mode", default="mimo-r4-tiny")
    parser.add_argument("--tokenizer", default="llama31")
    parser.add_argument("--checkpoint", type=Path, default=Path("neuromamba/runs/mamba3_current/model.pt"))
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--exact-cache", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def check_prompt(model, tokenizer, args: argparse.Namespace, prompt: str) -> dict:
    ids = tokenizer.encode(prompt, add_eos=False)
    ids = ids[-args.seq_len:]
    inference_params = InferenceParams(max_seqlen=args.seq_len + args.steps + 4, max_batch_size=1)
    generated: list[int] = []

    x = torch.tensor([ids], device=args.device, dtype=torch.long)
    cache_logits = model(x, inference_params=inference_params, num_last_tokens=1).logits[0, -1, : tokenizer.vocab_size]
    full_logits = model(x, num_last_tokens=1).logits[0, -1, : tokenizer.vocab_size]
    inference_params.seqlen_offset += x.shape[1]

    rows = []
    for step in range(args.steps):
        if args.exact_cache:
            cache_prefix = torch.tensor([ids + generated], device=args.device, dtype=torch.long)
            cache_logits = model(cache_prefix[:, -args.seq_len:], num_last_tokens=1).logits[0, -1, : tokenizer.vocab_size]
        cache_arg = int(torch.argmax(cache_logits).item())
        full_arg = int(torch.argmax(full_logits).item())
        diff = (cache_logits.float() - full_logits.float()).abs()
        rows.append(
            {
                "step": step,
                "ok": cache_arg == full_arg,
                "cache_argmax": {"id": cache_arg, "text": tokenizer.decode([cache_arg])},
                "full_argmax": {"id": full_arg, "text": tokenizer.decode([full_arg])},
                "max_abs": round(float(diff.max().item()), 6),
                "mean_abs": round(float(diff.mean().item()), 6),
                "cache_top": _top_tokens(tokenizer, cache_logits, 5),
                "full_top": _top_tokens(tokenizer, full_logits, 5),
            }
        )
        generated.append(full_arg)
        if full_arg == tokenizer.eos_id:
            break
        step_ids = torch.tensor([[full_arg]], device=args.device, dtype=torch.long)
        cache_logits = model(step_ids, inference_params=inference_params, num_last_tokens=1).logits[0, -1, : tokenizer.vocab_size]
        inference_params.seqlen_offset += 1
        full_ids = torch.tensor([ids + generated], device=args.device, dtype=torch.long)
        full_logits = model(full_ids, num_last_tokens=1).logits[0, -1, : tokenizer.vocab_size]

    return {
        "prompt": prompt,
        "ok": all(row["ok"] for row in rows),
        "generated_with_full_argmax": tokenizer.decode(generated),
        "rows": rows,
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
    prompts = args.prompt or PROMPTS
    reports = [check_prompt(model, tokenizer, args, prompt) for prompt in prompts]
    payload = {
        "ok": all(item["ok"] for item in reports),
        "mode": args.mode,
        "tokenizer": args.tokenizer,
        "checkpoint": str(args.checkpoint),
        "steps": args.steps,
        "reports": reports,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
