#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saneflow.model import SaneFlowConfig, SaneFlowLM
from saneflow.tokenizer import SaneFlowBPETokenizer


PROBES = [
    {
        "id": "add_apples",
        "kind": "arithmetic",
        "prompt": "Question: If you have 2 apples and get 3 more, how many apples do you have? Answer:",
        "answers": ["5", "five"],
    },
    {
        "id": "larger_number",
        "kind": "comparison",
        "prompt": "Question: Which is larger, 9 or 12? Answer:",
        "answers": ["12", "twelve"],
    },
    {
        "id": "causal_push",
        "kind": "causal",
        "prompt": "Question: A cup fell from a table because it was pushed. Why did it fall? Answer:",
        "answers": ["pushed", "gravity", "because it was pushed"],
    },
    {
        "id": "capital_france",
        "kind": "knowledge",
        "prompt": "Question: Paris is the capital of which country? Answer:",
        "answers": ["france"],
    },
    {
        "id": "even_pattern",
        "kind": "pattern",
        "prompt": "Question: Complete the pattern: 2, 4, 6, 8,",
        "answers": ["10", "ten"],
    },
    {
        "id": "ordering_youngest",
        "kind": "ordering",
        "prompt": "Question: If Tom is older than Sam, and Sam is older than Leo, who is the youngest? Answer:",
        "answers": ["leo"],
    },
    {
        "id": "copy_code",
        "kind": "copy",
        "prompt": "Remember this code: AX-917. Question: What is the code? Answer:",
        "answers": ["ax-917", "ax 917"],
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exact-match reasoning and language probes for SaneFlow checkpoints.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--max-new", type=int, default=40)
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16" if torch.cuda.is_available() else "fp32")
    return p.parse_args()


def filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    logits = logits.clone()
    if top_k > 0 and top_k < logits.numel():
        cutoff = torch.topk(logits, top_k).values[-1]
        logits[logits < cutoff] = -torch.inf
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        keep = torch.cumsum(probs, dim=-1) <= top_p
        keep[0] = True
        filtered = torch.full_like(logits, -torch.inf)
        filtered[sorted_idx[keep]] = logits[sorted_idx[keep]]
        logits = filtered
    return logits


def repeated_ngram_max(text: str, n: int = 4) -> int:
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    if len(toks) < n:
        return 0
    runs = 1
    best = 1
    prev = None
    for i in range(len(toks) - n + 1):
        gram = tuple(toks[i : i + n])
        if gram == prev:
            runs += 1
        else:
            runs = 1
            prev = gram
        best = max(best, runs)
    return best


@torch.no_grad()
def generate(model: SaneFlowLM, tokenizer: SaneFlowBPETokenizer, prompt: str, args: argparse.Namespace) -> tuple[str, int, float]:
    ids = tokenizer.encode(prompt, add_special_tokens=False) or [tokenizer.bos_token_id]
    generated: list[int] = []
    cache = None
    x = torch.tensor([ids[-args.context :]], dtype=torch.long, device=args.device)
    logits, cache = model.forward_step(x, cache)
    logits = logits[0]
    start = time.perf_counter()
    for _ in range(args.max_new):
        logits = filter_logits(logits, args.top_k, args.top_p)
        probs = F.softmax(logits / max(args.temperature, 1e-4), dim=-1)
        next_id = int(torch.multinomial(probs, 1).item())
        if next_id == tokenizer.eos_token_id:
            break
        generated.append(next_id)
        logits, cache = model.forward_step(torch.tensor([next_id], dtype=torch.long, device=args.device), cache)
        logits = logits[0]
    elapsed = max(time.perf_counter() - start, 1e-9)
    return tokenizer.decode(generated, skip_special_tokens=True), len(generated), len(generated) / elapsed


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def main() -> None:
    args = parse_args()
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    payload = torch.load(Path(args.ckpt), map_location=args.device, weights_only=True)
    cfg = SaneFlowConfig(**payload["config"])
    tokenizer = SaneFlowBPETokenizer(cfg.tokenizer_path)
    model = SaneFlowLM(cfg).to(device=args.device, dtype=dtype)
    model.load_state_dict(payload["model"])
    model.eval()

    rows = []
    for probe in PROBES:
        text, tokens, tok_s = generate(model, tokenizer, probe["prompt"], args)
        norm = normalize(text)
        passed = any(answer in norm for answer in probe["answers"])
        rows.append({
            "id": probe["id"],
            "kind": probe["kind"],
            "prompt": probe["prompt"],
            "output": text,
            "passed": passed,
            "tokens": tokens,
            "tok_s": round(tok_s, 2),
            "repeated_4gram_max": repeated_ngram_max(text, 4),
        })

    summary = {
        "ckpt": args.ckpt,
        "step": payload.get("step"),
        "passed": sum(1 for row in rows if row["passed"]),
        "total": len(rows),
        "pass_rate": sum(1 for row in rows if row["passed"]) / max(1, len(rows)),
        "avg_tok_s": round(sum(row["tok_s"] for row in rows) / max(1, len(rows)), 2),
    }
    result = {"summary": summary, "rows": rows}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
