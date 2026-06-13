#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mamba3_kr.cli import _fast_decode_once, load_or_new


DEFAULT_PROMPTS = [
    '<doc source="probe" domain="science"> The main idea is',
    '<doc source="probe" domain="history"> The event happened because',
    '<doc source="probe" domain="code"> The function returns',
    '<doc source="probe" domain="math"> To solve the problem,',
]


def collapse_metrics(prompt: str, text: str, new_tokens: int, max_repeated_word_run: int, min_distinct_words: int) -> dict:
    continuation = text[len(prompt) :].strip() if text.startswith(prompt) else text.strip()
    words = re.findall(r"[A-Za-z][A-Za-z']*", continuation.lower())
    longest_run = 0
    current_run = 0
    previous = None
    for word in words:
        if word == previous:
            current_run += 1
        else:
            previous = word
            current_run = 1
        longest_run = max(longest_run, current_run)
    distinct_words = len(set(words))
    collapsed = longest_run > max_repeated_word_run or (new_tokens >= 16 and distinct_words < min_distinct_words)
    return {
        "collapsed": collapsed,
        "distinct_words": distinct_words,
        "longest_repeated_word_run": longest_run,
        "max_repeated_word_run": max_repeated_word_run,
        "min_distinct_words": min_distinct_words,
        "repetition_scope": "continuation_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate raw document continuation collapse over several prompts.")
    parser.add_argument("--mode", default="mimo-r4-moe-2.4b")
    parser.add_argument("--tokenizer", default="llama31")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/mamba3_clean_doc_base_moe24_v1/base.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-repeated-word-run", type=int, default=12)
    parser.add_argument("--min-distinct-words", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("runs/mamba3_clean_doc_base_moe24_v1/raw_decode_set/latest.jsonl"))
    parser.add_argument("--prompt", action="append", default=None)
    args = parser.parse_args()

    prompts = args.prompt or DEFAULT_PROMPTS
    args.out.parent.mkdir(parents=True, exist_ok=True)

    load_args = SimpleNamespace(
        cmd="fast-generate",
        mode=args.mode,
        tokenizer=args.tokenizer,
        checkpoint=args.checkpoint,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        state_out=Path("/tmp/mamba3_unused_state.pt"),
        cuda_graph=False,
    )
    model, tokenizer, _ = load_or_new(load_args)
    model.eval()

    reports = []
    with torch.inference_mode():
        for index, prompt in enumerate(prompts):
            decode_args = SimpleNamespace(
                seq_len=args.seq_len,
                max_new=args.max_new,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                device=args.device,
                cuda_graph=False,
                safe_decode=True,
            )
            started = time.time()
            result = _fast_decode_once(model, tokenizer, decode_args, prompt)
            elapsed = time.time() - started
            text = result.get("text", "")
            metrics = collapse_metrics(
                prompt=prompt,
                text=text,
                new_tokens=int(result.get("new_tokens", 0)),
                max_repeated_word_run=args.max_repeated_word_run,
                min_distinct_words=args.min_distinct_words,
            )
            report = {
                "index": index,
                "prompt": prompt,
                "text": text,
                "new_tokens": int(result.get("new_tokens", 0)),
                "new_tokens_per_sec": result.get("new_tokens_per_sec"),
                "elapsed_sec": round(elapsed, 4),
                **metrics,
            }
            reports.append(report)
            print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)

    summary = {
        "mode": args.mode,
        "tokenizer": args.tokenizer,
        "checkpoint": str(args.checkpoint),
        "prompts": len(reports),
        "collapsed_prompts": sum(1 for item in reports if item["collapsed"]),
        "passed": all(not item["collapsed"] and item["new_tokens"] >= 16 for item in reports),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with args.out.open("w", encoding="utf-8") as fh:
        for report in reports:
            fh.write(json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n")
        fh.write(json.dumps({"summary": summary}, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"summary": summary}, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
