#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saneflow.data import TokenStreamDataset
from saneflow.decoding import filter_logits
from saneflow.model import SaneFlowConfig, SaneFlowLM
from saneflow.tokenizer import SaneFlowBPETokenizer


DEFAULT_PROMPTS = [
    "Hi. Who are you?",
    "Tell me a short story about a robot and a garden.",
    "Explain what a computer is in simple English.",
    "What is the capital of France?",
    "Write one sentence about the moon.",
    "Once upon a time",
    "Lily found a small",
]

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "for", "from", "i",
    "in", "is", "it", "me", "my", "of", "on", "or", "the", "to", "what",
    "who", "why", "with", "you", "your", "tell", "write", "one", "simple",
    "english", "sentence", "about", "explain",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SaneFlow checkpoint quality, speed, and context gates.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--valid-data", nargs="*", default=[])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    p.add_argument("--max-new", type=int, default=96)
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--context-probes", default="128,256,512,1024,2048,4096")
    p.add_argument("--temperature", type=float, default=0.75)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.08)
    p.add_argument("--no-repeat-ngram-size", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16" if torch.cuda.is_available() else "fp32")
    p.add_argument("--seed", type=int, default=20260614)
    return p.parse_args()


def pick_dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def repeated_ngram_max(tokens: list[str], n: int = 4) -> int:
    if len(tokens) < n:
        return 0
    counts = Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return max(counts.values(), default=0)


def distinct_n(tokens: list[str], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / max(1, len(grams))


def avg_sentence_len(text: str) -> float:
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        return 0.0
    return statistics.mean(len(words(s)) for s in sentences)


def topic_adherence(prompt: str, output: str) -> float:
    p = {w for w in words(prompt) if w not in STOP_WORDS and len(w) > 2}
    if not p:
        return 0.0
    o = set(words(output))
    return len(p & o) / len(p)


@torch.no_grad()
def eval_loss(model: SaneFlowLM, valid: TokenStreamDataset, cfg: SaneFlowConfig, batch_size: int, batches: int, device: torch.device) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = valid.batch(batch_size, device)
        logits = model(x)
        losses.append(F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1)).item())
    return sum(losses) / max(1, len(losses))


@torch.inference_mode()
def generate(
    model: SaneFlowLM,
    tokenizer: SaneFlowBPETokenizer,
    prompt: str,
    *,
    max_new: int,
    context: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: torch.device,
    decode: str,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> dict[str, Any]:
    ids = tokenizer.encode(prompt, add_special_tokens=False) or [tokenizer.bos_token_id]
    generated = []
    eos = False
    start = time.perf_counter()
    prefill_elapsed = 0.0
    decode_elapsed = 0.0
    if decode == "cache":
        prefill_start = time.perf_counter()
        x = torch.tensor([ids[-context:]], dtype=torch.long, device=device)
        logits, cache = model.forward_step(x, None)
        logits = logits[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        prefill_elapsed = max(time.perf_counter() - prefill_start, 1e-9)
        decode_start = time.perf_counter()
        for _ in range(max_new):
            logits = filter_logits(
                logits,
                top_k=top_k,
                top_p=top_p,
                generated_ids=generated,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            probs = F.softmax(logits / max(temperature, 1e-4), dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())
            if next_id == tokenizer.eos_token_id:
                eos = True
                break
            ids.append(next_id)
            generated.append(next_id)
            logits, cache = model.forward_step(torch.tensor([next_id], dtype=torch.long, device=device), cache)
            logits = logits[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        decode_elapsed = max(time.perf_counter() - decode_start, 1e-9)
    else:
        decode_start = time.perf_counter()
        for _ in range(max_new):
            x = torch.tensor([ids[-context:]], dtype=torch.long, device=device)
            logits = model(x)[0, -1]
            logits = filter_logits(
                logits,
                top_k=top_k,
                top_p=top_p,
                generated_ids=generated,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            probs = F.softmax(logits / max(temperature, 1e-4), dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())
            if next_id == tokenizer.eos_token_id:
                eos = True
                break
            ids.append(next_id)
            generated.append(next_id)
        if device.type == "cuda":
            torch.cuda.synchronize()
        decode_elapsed = max(time.perf_counter() - decode_start, 1e-9)
    elapsed = max(time.perf_counter() - start, 1e-9)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    toks = words(text)
    return {
        "text": text,
        "tokens": len(generated),
        "tok_s": len(generated) / elapsed,
        "prefill_sec": prefill_elapsed,
        "decode_sec": decode_elapsed,
        "decode_tok_s": len(generated) / max(decode_elapsed, 1e-9),
        "eos": eos,
        "empty": len(text.strip()) == 0,
        "invalid_chars": text.count("\ufffd") + text.count("�"),
        "repeated_4gram_max": repeated_ngram_max(toks, 4),
        "distinct_1": distinct_n(toks, 1),
        "distinct_2": distinct_n(toks, 2),
        "avg_sentence_words": avg_sentence_len(text),
    }


@torch.no_grad()
def cache_parity(model: SaneFlowLM, input_ids: torch.Tensor) -> dict[str, float]:
    full = model(input_ids)
    cache = None
    step_logits = []
    for t in range(input_ids.shape[1]):
        logits, cache = model.forward_step(input_ids[:, t], cache)
        step_logits.append(logits)
    stepped = torch.stack(step_logits, dim=1)
    diff = (full - stepped).float().abs()
    argmax_match = (full.argmax(dim=-1) == stepped.argmax(dim=-1)).float().mean()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "argmax_match": float(argmax_match.item()),
    }


@torch.no_grad()
def context_probe(model: SaneFlowLM, cfg: SaneFlowConfig, tokenizer: SaneFlowBPETokenizer, context: int, device: torch.device) -> dict[str, Any]:
    vocab_high = max(5, cfg.vocab_size - 1)
    x = torch.randint(4, vocab_high, (1, context), dtype=torch.long, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    start = time.perf_counter()
    logits = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - start, 1e-9)
    out = {
        "context": context,
        "forward_ok": bool(torch.isfinite(logits).all().item()),
        "forward_tok_s": context / elapsed,
    }
    if device.type == "cuda":
        out["max_vram_mb"] = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
    prompt = "Once upon a time " + ("the garden " * max(1, context // 8))
    gen = generate(
        model,
        tokenizer,
        prompt,
        max_new=16,
        context=context,
        temperature=0.75,
        top_k=40,
        top_p=0.9,
        device=device,
        decode="cache",
        repetition_penalty=1.08,
        no_repeat_ngram_size=4,
    )
    out["generation_tok_s"] = gen["tok_s"]
    out["generation_decode_tok_s"] = gen["decode_tok_s"]
    out["generation_repeated_4gram_max"] = gen["repeated_4gram_max"]
    out["generation_empty"] = gen["empty"]
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = pick_dtype(args.dtype)
    payload = torch.load(Path(args.ckpt), map_location="cpu", weights_only=True)
    cfg = SaneFlowConfig(**payload["config"])
    tokenizer = SaneFlowBPETokenizer(cfg.tokenizer_path)
    model = SaneFlowLM(cfg).to(device=device, dtype=dtype)
    model.load_state_dict(payload["model"])
    model.eval()

    result: dict[str, Any] = {
        "checkpoint": args.ckpt,
        "config": cfg.to_dict(),
        "quality_gate_version": 1,
        "device": str(device),
        "dtype": args.dtype,
    }

    if args.valid_data:
        valid = TokenStreamDataset(
            tokenizer=tokenizer,
            paths=[Path(p) for p in args.valid_data],
            seq_len=min(args.context, 512),
            max_records=2000,
        )
        loss = eval_loss(model, valid, cfg, args.batch_size, args.eval_batches, device)
        result["valid_loss"] = loss
        result["valid_ppl"] = math.exp(min(20.0, loss))

    prompt_rows = []
    full_speeds = []
    cache_speeds = []
    cache_decode_speeds = []
    for prompt in args.prompts:
        cache_gen = generate(
            model,
            tokenizer,
            prompt,
            max_new=args.max_new,
            context=args.context,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
            decode="cache",
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        full_gen = generate(
            model,
            tokenizer,
            prompt,
            max_new=min(args.max_new, 32),
            context=args.context,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
            decode="full",
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        cache_gen["topic_adherence"] = topic_adherence(prompt, cache_gen["text"])
        full_speeds.append(full_gen["tok_s"])
        cache_speeds.append(cache_gen["tok_s"])
        cache_decode_speeds.append(cache_gen["decode_tok_s"])
        prompt_rows.append({"prompt": prompt, "cache": cache_gen, "full_short": full_gen})
    result["prompts"] = prompt_rows
    result["summary"] = {
        "cache_avg_total_tok_s": sum(cache_speeds) / max(1, len(cache_speeds)),
        "cache_avg_decode_tok_s": sum(cache_decode_speeds) / max(1, len(cache_decode_speeds)),
        "full_short_avg_tok_s": sum(full_speeds) / max(1, len(full_speeds)),
        "empty_output_rate": sum(1 for r in prompt_rows if r["cache"]["empty"]) / max(1, len(prompt_rows)),
        "eos_rate": sum(1 for r in prompt_rows if r["cache"]["eos"]) / max(1, len(prompt_rows)),
        "invalid_output_rate": sum(1 for r in prompt_rows if r["cache"]["invalid_chars"] > 0) / max(1, len(prompt_rows)),
        "mean_topic_adherence": statistics.mean(r["cache"]["topic_adherence"] for r in prompt_rows),
        "max_repeated_4gram": max((r["cache"]["repeated_4gram_max"] for r in prompt_rows), default=0),
        "mean_distinct_1": statistics.mean(r["cache"]["distinct_1"] for r in prompt_rows),
        "mean_distinct_2": statistics.mean(r["cache"]["distinct_2"] for r in prompt_rows),
        "mean_sentence_words": statistics.mean(r["cache"]["avg_sentence_words"] for r in prompt_rows),
    }

    sample_ids = tokenizer.encode(args.prompts[0], add_special_tokens=False) or [tokenizer.bos_token_id]
    sample_ids = torch.tensor([sample_ids[: min(len(sample_ids), args.context)]], dtype=torch.long, device=device)
    result["cache_parity"] = cache_parity(model, sample_ids)

    probes = []
    for context in [int(x) for x in args.context_probes.split(",") if x.strip()]:
        try:
            probes.append(context_probe(model, cfg, tokenizer, context, device))
        except RuntimeError as exc:
            probes.append({"context": context, "forward_ok": False, "error": str(exc).splitlines()[0]})
            if device.type == "cuda":
                torch.cuda.empty_cache()
    result["context_probes"] = probes
    if device.type == "cuda":
        result["vram_allocated_mb"] = round(torch.cuda.memory_allocated(device) / (1024 ** 2), 2)
        result["vram_reserved_mb"] = round(torch.cuda.memory_reserved(device) / (1024 ** 2), 2)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "valid_loss": result.get("valid_loss"),
        "summary": result["summary"],
        "cache_parity": result["cache_parity"],
    }, indent=2))


if __name__ == "__main__":
    main()
