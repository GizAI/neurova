#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mamba3_kr.cli import load_or_new, setup_perf


STOP_MARKERS = [
    "\nQuestion:",
    "\nInstruction:",
    "\nUser:",
    "\nAssistant:",
    "\nAnswer in",
]

def clean_plain_answer(answer: str) -> str:
    for marker in STOP_MARKERS:
        pos = answer.find(marker)
        if pos >= 0:
            answer = answer[:pos]
    answer = re.sub(r"\s+", " ", answer.strip())
    return answer


def make_prompt(text: str) -> str:
    text = text.strip()
    if "Answer:" in text:
        return text
    return f"Instruction: {text}\nAnswer:"


def clean_answer(full_text: str, prompt: str) -> str:
    answer = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
    answer = answer.strip()
    answer = re.sub(r"^\?+\s*", "", answer)
    answer = re.sub(r"^(Answer:|A:)\s*", "", answer, flags=re.IGNORECASE)
    for marker in STOP_MARKERS:
        pos = answer.find(marker)
        if pos >= 0:
            answer = answer[:pos]
    return clean_plain_answer(answer)


def clean_generated_answer(answer: str) -> str:
    answer = answer.strip()
    answer = re.sub(r"^\?+\s*", "", answer)
    answer = re.sub(r"^(Answer:|A:)\s*", "", answer, flags=re.IGNORECASE)
    return clean_plain_answer(answer)


def should_stop_after_sentence(answer: str, min_chars: int) -> bool:
    if len(answer) < min_chars:
        return False
    if re.search(r"[.!?。？！]($|\s)", answer):
        return True
    return False


def truncate_after_sentence(answer: str, min_chars: int) -> str:
    answer = answer.strip()
    if len(answer) < min_chars:
        return answer
    match = re.search(r"^(.{%d,}?[.!?。？！])(?:\s|$)" % max(1, min_chars), answer)
    if match:
        return match.group(1).strip()
    return answer


def _next_token(model, ids: list[int], tokenizer, args: argparse.Namespace) -> int:
    seq = torch.tensor(ids[-args.seq_len:], device=args.device, dtype=torch.long).unsqueeze(0)
    logits = model(seq, num_last_tokens=1).logits[:, -1, : tokenizer.vocab_size].float()
    return int(torch.argmax(logits, dim=-1).item())


@torch.inference_mode()
def generate(model, tokenizer, args: argparse.Namespace, user_text: str) -> tuple[str, int, float]:
    prompt = make_prompt(user_text)
    ids = tokenizer.encode(prompt, add_eos=False)
    ids = ids[-args.seq_len:]
    start_len = len(ids)
    raw_answer = ""
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.max_new):
        next_id = _next_token(model, ids, tokenizer, args)
        ids.append(next_id)
        if next_id == tokenizer.eos_id:
            break
        raw_answer += tokenizer.decode([next_id])
        partial = clean_generated_answer(raw_answer)
        if any(marker in raw_answer for marker in STOP_MARKERS) and partial:
            break
        if args.stop_after_sentence and should_stop_after_sentence(partial, args.min_sentence_chars):
            break
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = max(time.time() - t0, 1e-9)
    return clean_generated_answer(raw_answer), max(0, len(ids) - start_len), elapsed


@torch.inference_mode()
def iter_generate(model, tokenizer, args: argparse.Namespace, user_text: str):
    prompt = make_prompt(user_text)
    ids = tokenizer.encode(prompt, add_eos=False)
    ids = ids[-args.seq_len:]
    start_len = len(ids)
    emitted = ""
    raw_answer = ""
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    tokens = 0
    for _ in range(args.max_new):
        next_id = _next_token(model, ids, tokenizer, args)
        ids.append(next_id)
        tokens = len(ids) - start_len
        if next_id == tokenizer.eos_id:
            break
        raw_answer += tokenizer.decode([next_id])
        partial = clean_generated_answer(raw_answer)
        if partial.startswith(emitted):
            delta = partial[len(emitted):]
            if delta:
                emitted = partial
                yield delta, tokens, 0.0
        if any(marker in raw_answer for marker in STOP_MARKERS) and partial:
            break
        if args.stop_after_sentence and should_stop_after_sentence(partial, args.min_sentence_chars):
            break
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = max(time.time() - t0, 1e-9)
    yield "", tokens, elapsed


@torch.inference_mode()
def stream_generate(model, tokenizer, args: argparse.Namespace, user_text: str) -> tuple[int, float]:
    prompt = make_prompt(user_text)
    ids = tokenizer.encode(prompt, add_eos=False)
    ids = ids[-args.seq_len:]
    start_len = len(ids)
    emitted = ""
    raw_answer = ""
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.max_new):
        next_id = _next_token(model, ids, tokenizer, args)
        ids.append(next_id)
        if next_id == tokenizer.eos_id:
            break
        raw_answer += tokenizer.decode([next_id])
        partial = clean_generated_answer(raw_answer)
        if partial.startswith(emitted):
            delta = partial[len(emitted):]
            if delta:
                print(delta, end="", flush=True)
                emitted = partial
        if any(marker in raw_answer for marker in STOP_MARKERS) and partial:
            break
        if args.stop_after_sentence and should_stop_after_sentence(partial, args.min_sentence_chars):
            break
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = max(time.time() - t0, 1e-9)
    return max(0, len(ids) - start_len), elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="mimo-r4-tiny")
    parser.add_argument("--tokenizer", default="llama31")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/mamba3_neurova_speak_v1/sft.pt"))
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--stop-after-sentence", action="store_true", default=True)
    parser.add_argument("--min-sentence-chars", type=int, default=18)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--warmup-new", type=int, default=2)
    parser.add_argument("--stream", action="store_true")
    return parser.parse_args()


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

    if not args.no_warmup and args.warmup_new > 0:
        warm_args = copy.copy(args)
        warm_args.max_new = min(args.max_new, args.warmup_new)
        warm_args.stop_after_sentence = False
        generate(model, tokenizer, warm_args, "Can you help me?")

    if args.prompt:
        if args.stream:
            tokens, elapsed = stream_generate(model, tokenizer, args, args.prompt)
            print()
        else:
            answer, tokens, elapsed = generate(model, tokenizer, args, args.prompt)
            print(answer)
        print(f"({tokens / elapsed:.1f} tok/s)")
        return

    print(f"Neurova safe chat ready ({args.mode}). Type /q to quit.", flush=True)
    while True:
        try:
            user = input("you> ")
        except EOFError:
            break
        user = user.strip()
        if not user:
            continue
        if user in {"/q", "/quit", "quit", "exit"}:
            break
        print("neurova> ", end="", flush=True)
        if args.stream:
            tokens, elapsed = stream_generate(model, tokenizer, args, user)
            print(f" ({tokens / elapsed:.1f} tok/s)", flush=True)
        else:
            answer, tokens, elapsed = generate(model, tokenizer, args, user)
            print(f"{answer} ({tokens / elapsed:.1f} tok/s)", flush=True)


if __name__ == "__main__":
    main()
