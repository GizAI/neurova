#!/usr/bin/env python3
"""Collect sparse-MoE router/expert diagnostics for a Mamba-3 checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuromamba.cli import load_or_new, setup_perf
from neuromamba.data import iter_packed_token_batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="mimo-r4-moe-2.4b")
    parser.add_argument("--tokenizer", default="llama31")
    parser.add_argument("--checkpoint", type=Path, default=Path("neuromamba/runs/mamba3_clean_doc_base_moe24_v1/base.pt"))
    parser.add_argument("--data", nargs="+", type=Path, default=[Path("neuromamba/data/splits/base_doc_cont_v3_valid.jsonl")])
    parser.add_argument("--out", type=Path, default=Path("neuromamba/runs/mamba3_clean_doc_base_moe24_v1/router_diagnostics/latest.json"))
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-text-chars", type=int, default=65536)
    parser.add_argument("--max-text-tokens", type=int, default=120000)
    return parser.parse_args()


def module_name_is_sparse_moe(module: torch.nn.Module) -> bool:
    return (
        hasattr(module, "router")
        and hasattr(module, "num_experts")
        and hasattr(module, "top_k")
        and hasattr(module, "experts")
    )


def empty_stats(num_experts: int) -> dict[str, Any]:
    return {
        "num_experts": int(num_experts),
        "tokens": 0,
        "counts": [0 for _ in range(int(num_experts))],
        "entropy_sum": 0.0,
        "top_prob_sum": 0.0,
        "max_logit_sum": 0.0,
        "batches": 0,
    }


def update_router_stats(stats: dict[str, Any], module: torch.nn.Module, x: torch.Tensor) -> None:
    flat = x.reshape(-1, x.shape[-1])
    router_logits = module.router(flat).float()
    probs = torch.softmax(router_logits, dim=-1)
    top_prob, expert_id = probs.max(dim=-1)
    counts = torch.bincount(expert_id.detach().cpu(), minlength=int(module.num_experts))
    entropy = -(probs * torch.clamp(probs, min=1e-12).log()).sum(dim=-1)
    stats["tokens"] += int(flat.shape[0])
    stats["counts"] = [int(a + b) for a, b in zip(stats["counts"], counts.tolist())]
    stats["entropy_sum"] += float(entropy.sum().detach().cpu())
    stats["top_prob_sum"] += float(top_prob.sum().detach().cpu())
    stats["max_logit_sum"] += float(router_logits.max(dim=-1).values.sum().detach().cpu())
    stats["batches"] += 1


def finalize_layer(name: str, stats: dict[str, Any]) -> dict[str, Any]:
    tokens = max(1, int(stats["tokens"]))
    num_experts = int(stats["num_experts"])
    counts = [int(item) for item in stats["counts"]]
    total = max(1, sum(counts))
    shares = [count / total for count in counts]
    top_share = max(shares) if shares else 0.0
    nonzero = sum(1 for count in counts if count > 0)
    expected = total / num_experts if num_experts else 0.0
    expert_skew = (max(counts) / expected) if expected > 0 else 0.0
    entropy = float(stats["entropy_sum"]) / tokens
    normalized_entropy = entropy / math.log(num_experts) if num_experts > 1 else 0.0
    return {
        "layer": name,
        "num_experts": num_experts,
        "tokens": total,
        "counts": counts,
        "shares": [round(item, 6) for item in shares],
        "nonzero_experts": nonzero,
        "top_expert": int(max(range(len(counts)), key=lambda idx: counts[idx])) if counts else None,
        "top_expert_share": round(top_share, 6),
        "expert_skew_vs_uniform": round(expert_skew, 6),
        "router_entropy": round(entropy, 6),
        "router_entropy_normalized": round(normalized_entropy, 6),
        "mean_top_route_prob": round(float(stats["top_prob_sum"]) / tokens, 6),
        "mean_max_router_logit": round(float(stats["max_logit_sum"]) / tokens, 6),
        "hook_calls": int(stats["batches"]),
    }


def estimate_active_params(model: torch.nn.Module, moe_layers: list[tuple[str, torch.nn.Module]]) -> dict[str, Any]:
    total = sum(param.numel() for param in model.parameters())
    expert_total = 0
    selected_expert = 0
    for _, module in moe_layers:
        expert_params = [sum(param.numel() for param in expert.parameters()) for expert in module.experts]
        expert_total += sum(expert_params)
        selected_expert += max(expert_params) * int(getattr(module, "top_k", 1))
    active = total - expert_total + selected_expert
    return {
        "total_parameters": int(total),
        "expert_parameters_total": int(expert_total),
        "estimated_active_parameters_per_token": int(active),
        "estimated_active_parameter_ratio": round(active / total, 6) if total else 0.0,
    }


def main() -> None:
    args = parse_args()
    setup_perf(args.device)
    load_args = SimpleNamespace(
        cmd="eval-loss",
        mode=args.mode,
        tokenizer=args.tokenizer,
        checkpoint=args.checkpoint,
        device=args.device,
        dtype=args.dtype,
        activation_checkpointing=False,
    )
    model, tokenizer, _ = load_or_new(load_args)
    model.eval()

    moe_layers = [(name, module) for name, module in model.named_modules() if module_name_is_sparse_moe(module)]
    if not moe_layers:
        raise SystemExit("no sparse MoE layers found")

    stats: dict[str, dict[str, Any]] = {
        name: empty_stats(int(module.num_experts)) for name, module in moe_layers
    }
    hooks = []
    for name, module in moe_layers:
        def hook(mod, inputs, layer_name=name):
            if inputs:
                update_router_stats(stats[layer_name], mod, inputs[0].detach())
        hooks.append(module.register_forward_pre_hook(hook))

    batches = iter_packed_token_batches(
        tokenizer,
        args.data,
        args.seq_len,
        args.batch_size,
        args.device,
        max_text_chars=args.max_text_chars,
        max_text_tokens=args.max_text_tokens,
    )

    losses: list[float] = []
    with torch.no_grad():
        for _ in range(args.batches):
            batch = next(batches)
            logits = model(batch[:, :-1]).logits
            labels = batch[:, 1:]
            if logits.shape[1] != labels.shape[1]:
                logits = logits[:, -labels.shape[1]:]
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            losses.append(float(loss.detach().cpu()))

    for handle in hooks:
        handle.remove()

    layer_reports = [finalize_layer(name, stats[name]) for name, _ in moe_layers]
    worst = max(layer_reports, key=lambda item: item["top_expert_share"])
    payload = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": args.mode,
        "tokenizer": args.tokenizer,
        "checkpoint": str(args.checkpoint),
        "data": [str(path) for path in args.data],
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "batches": args.batches,
        "eval_loss_mean": round(sum(losses) / len(losses), 6) if losses else None,
        "moe_layers": len(layer_reports),
        "parameter_estimate": estimate_active_params(model, moe_layers),
        "summary": {
            "max_top_expert_share": worst["top_expert_share"],
            "worst_layer": worst["layer"],
            "min_nonzero_experts": min(item["nonzero_experts"] for item in layer_reports),
            "mean_router_entropy_normalized": round(
                sum(item["router_entropy_normalized"] for item in layer_reports) / len(layer_reports),
                6,
            ),
        },
        "layers": layer_reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
